"""
bridge_api.py - Minimal externalized hidden 29-arg API for SkyrimNet UI

mapping (29 args → config-aware values → generate_audio call). Protects pipeline from refactors.

Call setup_bridge_api(demo, audio_output, api_status_md) inside gr.Blocks() to register for /api/generate_audio.
Self-contained: Imports/creates CONFIG = SkyrimNetConfig().
Uses safe_to_* from ui_helpers only for type safety (lists/None) if imported—assume valid remote inputs.
Extends original by calling generate_audio (13 args) instead of generate (12 args).
29-param signature & hidden elements exact match from original.
"""
import asyncio
import functools
import tempfile
from pathlib import Path

import gradio as gr
import torch
import torchaudio
from loguru import logger
from .config import get_config, get_config_value

config = get_config()

# FIX: Load TTS via singleton (essential; supports multilingual)
from src.tts_model import ModelManager

### SkyrimNet Zonos Emulated
@functools.cache
def cpp_uuid_to_seed(uuid_64: int) -> int:
    """
    Convert a 64-bit UUID to a valid PyTorch seed (0 to 2^32 - 1).
    Uses hash() for better distribution across the seed space.
    """
    return abs(hash(uuid_64)) % (2 ** 32)

def generate_audio_ui(  # REVERT/PATCH: Sync def (Gradio calls without await → no coroutine error)
        model_choice=None,
        text="On that first day from Saturalia, My missus gave for me, A big bowl of moon sugar!",
        language="en",
        speaker_audio=None,
        prefix_audio=None,
        e1=None,
        e2=None,
        e3=None,
        e4=None,
        e5=None,
        e6=None,
        e7=None,
        e8=None,
        vq_single=None,
        fmax=None,
        pitch_std=None,
        speaking_rate=None,
        dnsmos_ovrl=None,
        speaker_noised: bool = None,
        cfg_scale=0.3,
        top_p=1.0,
        top_k=None,
        min_p=0.5,
        linear=None,
        confidence=None,
        quadratic=None,
        uuid=-1,
        randomize_seed: bool = False,
        unconditional_keys: list = None
):
    """Generate audio using configurable parameter system (sync wrapper for inner async)"""
    from .config import get_config_value
    api_temperature = linear if linear is not None else None
    api_min_p = min_p if min_p is not None else None
    api_top_p = top_p if top_p is not None else None
    api_repetition_penalty = confidence if confidence is not None else None
    api_cfg_weight = cfg_scale if cfg_scale is not None else None
    api_exaggeration = quadratic if quadratic is not None else None

    final_temperature = get_config_value('globalstemperature')
    final_min_p = get_config_value('min_p')
    final_top_p = get_config_value('top_p')
    final_repetition_penalty = get_config_value('repetition_penalty')
    final_cfg_weight = get_config_value('cfg_weight')
    final_exaggeration = get_config_value('exaggeration')

    # Merge voice params (globals + overrides like serena temp=0.9)
    voice_params = get_config().get_merged_audio_params('default')  # Default voice

    # Fill None from merged (e.g., if UI None, use 0.9 for serena temp)
    params = {
        'temperature': api_temperature or voice_params.get('temperature', voice_params['tts'].get('temperature', 0.7)),
        'min_p': api_min_p or voice_params.get('min_p', voice_params['tts'].get('min_p', 0.07)),
        'top_p': api_top_p or voice_params.get('top_p', voice_params['tts'].get('top_p', 1.0)),
        'repetition_penalty': api_repetition_penalty or voice_params.get('repetition_penalty',
                                                                     voice_params['tts'].get('repetition_penalty',
                                                                                             2.0)),
        'cfg_weight': api_cfg_weight or voice_params.get('cfg_weight', voice_params['tts'].get('cfg_weight', 0.45)),
        'exaggeration': api_exaggeration or voice_params.get('exaggeration', voice_params['tts'].get('exaggeration', 0.7)),
        # Add more (e.g., 'speaking_rate': ... from audio)
        'sr': voice_params.get('sr', 24000),  # Global
        'device': config.app_config.globals.device
    }

    logger.info(
        f"UI provided parameters - cache_uuid: {uuid}, temp: {params['temperature']}, min_p: {params['min_p']}, top_p: {params['top_p']}, rep_penalty: {params['repetition_penalty']}, cfg_weight: {params['cfg_weight']}, exaggeration: {params['exaggeration']}")

    # Important - use server supplied uuid for consistent seed
    seed_num = cpp_uuid_to_seed(uuid)

    # NEW: Temp event loop for inner async (safe in sync Gradio context; ~0.01s overhead)
    # Creates/runs/closes loop around generate_audio call (prepares for async generate_audio)
    loop = None
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Lazy import + call (via kwargs for sig safety; model=MODEL from global)
        from src.generate_audio import generate_audio  # Assume will be async (returns coroutine)
        from .config import get_config_value
        gen_coroutine = generate_audio(  # Call async fn → coroutine
            model=config.app_config.globals.model,
            text=text,
            audio_prompt_path=speaker_audio,  # None ok (used internally for stem/post)
            seed_num=seed_num,
            cache_uuid=uuid,
            exaggeration=params['exaggeration'],
            temperature=params['temperature'],
            cfgw=params['cfg_weight'],
            min_p=params['min_p'],
            top_p=params['top_p'],
            repetition_penalty=params['repetition_penalty'],
            enable_memory_cache=get_config_value('app_config.globals.enable_memory_cache', True),
            enable_disk_cache=get_config_value('app_config.globals.enable_disk_cache', True),
            language_id=language  # kwarg-safe
        )

        # Await in temp loop → gets actual result (tensor/path; no coroutine return to Gradio)
        result = loop.run_until_complete(gen_coroutine)
        logger.debug(
            f"Inner async generate_audio complete via wrapper: result type {type(result)} | shape {getattr(result, 'shape', 'N/A') if hasattr(result, 'shape') else 'N/A'}")

    except Exception as inner_e:
        logger.error(f"Inner async wrapper error in generate_audio_ui: {inner_e} – fallback to silence path")
        if loop:
            loop.close()
        # FIXED: 2D mono silence (Gradio-safe: [1, samples], float32 CPU) + save to temp path (str return like HIT)
        from .config import get_config_value  # Absolute for fallback (sr/dtype/device)
        silence_2d = torch.zeros(1, get_config_value('globals.sr', 24000) * 2, dtype=torch.float32, device='cpu')  # 2D [1, 48000]; float32 CPU
        # Temp save (mimic save_and_cache_output; sr=CONFIG.sr)
        with tempfile.TemporaryDirectory() as tmpdir:
            fallback_path = Path(tmpdir) / f"fallback_silence_{uuid}.wav"
            torchaudio.save(str(fallback_path), silence_2d, get_config_value('globals.sr', 24000) )
            result = str(fallback_path)  # Str path (Gradio Audio handles; delete on close)
            logger.warning(f"Fallback silence path created: {result} (2s; error: {str(inner_e)[:100]})")

    finally:
        if loop and not loop.is_closed():
            loop.close()

    return result, uuid  # Matches outputs (str path from inner/fallback + uuid; Gradio Audio ok)


# Builder: Creates 29 hidden components + hidden_btn.click (exact from original working code)
def setup_bridge_api(demo, audio_output, api_status_md):
    """
    Registers 29 hidden gr components + button/.click for /api/generate_audio (triggers generate_audio).
    Exact match from original: visible=False, default values (cfg_scale=0.3, min_p=0.5, linear=None, etc.).
    No States in inputs (safe for remote API).
    Call inside gr.Blocks() in ui.py.
    """
    from .config import get_config_value
    model_type = 'multilingual' if get_config_value('multilingual') else 'english'
    ModelManager.get_instance().load_model(model_type)  # Lazy-loads


    # 29 hidden components (exact from original working code signature/order/defaults)
    model_choice = gr.Textbox(visible=False, value=None, label="Model Choice")
    text_input_hidden = gr.Textbox(visible=False, value="", label="Text Hidden")  # text
    language = gr.Textbox(visible=False, value="en", label="Language")
    speaker_audio = gr.Audio(visible=False, type="filepath", value=None, label="Speaker Audio")
    prefix_audio = gr.Audio(visible=False, type="filepath", value=None, label="Prefix Audio")
    emotion1 = gr.Number(visible=False, value=0, label="Emotion 1")  # e1
    emotion2 = gr.Number(visible=False, value=0, label="Emotion 2")
    emotion3 = gr.Number(visible=False, value=0, label="Emotion 3")
    emotion4 = gr.Number(visible=False, value=0, label="Emotion 4")
    emotion5 = gr.Number(visible=False, value=0, label="Emotion 5")
    emotion6 = gr.Number(visible=False, value=0, label="Emotion 6")
    emotion7 = gr.Number(visible=False, value=0, label="Emotion 7")
    emotion8 = gr.Number(visible=False, value=0, label="Emotion 8")
    vq_single = gr.Number(visible=False, value=None, label="VQ Single")
    fmax = gr.Number(visible=False, value=None, label="FMax")
    pitch_std = gr.Number(visible=False, value=None, label="Pitch Std")
    speaking_rate_param = gr.Number(visible=False, value=None, label="Speaking Rate")
    dnsmos_ovrl = gr.Number(visible=False, value=None, label="DNSMOS Ovrl")
    speaker_noised = gr.Checkbox(visible=False, value=False, label="Speaker Noised")
    cfg_scale = gr.Number(visible=False, value=0.3, label="CFG Scale")
    top_p_param = gr.Number(visible=False, value=1.0, label="Top P")
    top_k = gr.Number(visible=False, value=None, label="Top K")
    min_p_param = gr.Number(visible=False, value=0.5, label="Min P")
    linear_temp = gr.Number(visible=False, value=None, label="Linear (Temp)")
    confidence_rep = gr.Number(visible=False, value=None, label="Confidence (Rep)")
    quadratic_exagg = gr.Number(visible=False, value=None, label="Quadratic (Exagg)")
    uuid_seed = gr.Number(visible=False, value=-1, label="UUID")
    randomize_seed_toggle = gr.Checkbox(visible=False, value=False, label="Randomize Seed")
    unconditional_keys_list = gr.Textbox(visible=False, value="", label="Unconditional Keys")
    hidden_api_btn = gr.Button(visible=False, value="Hidden API Trigger")

    # .click event (exact 29 inputs from original working code; no States; api_name exposes remote endpoint)
    hidden_api_btn.click(
        fn=generate_audio_ui,  # FIXED: Now async (auto-awaited by Gradio)
        inputs=[  # Exact 29 unchanged
            model_choice, text_input_hidden, language, speaker_audio, prefix_audio,
            emotion1, emotion2, emotion3, emotion4, emotion5, emotion6, emotion7, emotion8,
            vq_single, fmax, pitch_std, speaking_rate_param, dnsmos_ovrl, speaker_noised,
            cfg_scale, top_p_param, top_k, min_p_param, linear_temp, confidence_rep, quadratic_exagg,
            uuid_seed, randomize_seed_toggle, unconditional_keys_list
        ],
        outputs=[audio_output, api_status_md],
        api_name="generate_audio",  # Unchanged: POST /api/generate_audio (29-arg JSON)
        show_progress=True,  # Spinner during async await
        concurrency_limit=4  # FIXED: 4 concurrent API calls (GPU overlap on I/O/post)
    )
    logger.info("Minimal hidden API registered (29 original components + .click; CONFIG 'api'/value mode; calls generate_audio)")

# Expose for ui.py import (minimal)
__all__ = ['generate_audio_ui', 'setup_bridge_api']