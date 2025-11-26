# src/ui.py
import os
import gradio as gr
from loguru import logger
from pathlib import Path

# CONFIG singleton
from src.config import get_config, get_config_value
from src.tts_model import get_model, ModelManager

# Path helpers
from src.ui_helpers import (
    update_api_status, stub_wav_path, test_voice_generation,
)

# Hidden API supports communication with SkyrimNet via a Zonos bridge
from src.generate.ui_interface import generate_audio_ui, setup_bridge_api
from src.normalize_stem import normalize_stem
import shutil
import json
from datetime import datetime

# Warnings cleanup
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", message="torchaudio._backend.utils.info has been deprecated")

# Required for proper config handling
from src.config.models import AppConfig

def create_ui(
    cache_manager=None,
    pipeline=None,
    config: AppConfig = None
):
    """Unified Generate/Test tab: upload/select voice, set params, generate audio, and optionally persist per-voice overrides.
    Note: generate_audio_ui signature/behavior remains untouched (bridge-compatible)."""
    # Ensure config is available
    config = config or get_config()

    def list_voice_wavs() -> list:
        """Collect available voice reference wavs.
        Looks in voices_cache_dir and its 'resampled' subfolder.
        """
        try:
            voices_dir = config.app_config.globals.voices_cache_dir
            if not voices_dir:
                return []
            paths = []
            vdir = Path(voices_dir)
            # Top-level wavs
            for p in vdir.glob('*.wav'):
                try:
                    paths.append(str(p))
                except Exception:
                    continue
            # Resampled subfolder wavs
            resampled_dir = vdir / 'resampled'
            if resampled_dir.exists():
                for p in resampled_dir.glob('*.wav'):
                    try:
                        paths.append(str(p))
                    except Exception:
                        continue
            return sorted(paths)
        except Exception as e:
            logger.warning(f"Failed listing voice wavs: {e}")
            return []

    def get_default_params():
        # UI defaults should mirror a normal generate_audio_ui call defaults
        # irrespective of any per-voice overrides. Hidden API will use config overrides.
        return {
            'temperature': 0.8,
            'exaggeration': 0.5,
            'cfg_weight': 0.3,
            'min_p': 0.5,
            'top_p': 1.0,
            'repetition_penalty': 1.2,
            'language_id': get_config_value('globals.language_id', 'en')
        }

    with gr.Blocks(title="SkyrimNet Chatterbox", theme=gr.themes.Soft()) as demo:
        api_status_md = gr.Markdown(value=update_api_status())
        gr.Markdown("# SkyrimNet Chatterbox TTS UI\nUnified generation and testing.", elem_id="title-md")

        defaults = get_default_params()

        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(label="Input Text", placeholder="Enter text to generate speech...", lines=3)

                with gr.Group():
                    gr.Markdown("### Voice Reference")
                    ref_upload = gr.Audio(label="Upload .wav", sources="upload", type="filepath", format="wav")
                    voice_choices = gr.Dropdown(label="Or select from voices directory", choices=list_voice_wavs(), value=None, allow_custom_value=False)
                    refresh_btn = gr.Button("Refresh Voices List", variant="secondary")

                with gr.Group():
                    gr.Markdown("### TTS Parameters")
                    seed = gr.Number(value=42, label="Seed", precision=0)
                    temperature = gr.Slider(0.1, 1.5, defaults['temperature'], step=0.01, label="Temperature")
                    cfg_scale = gr.Slider(0.0, 2.0, defaults['cfg_weight'], step=0.01, label="CFG Scale")
                    min_p = gr.Slider(0.0, 1.0, defaults['min_p'], step=0.01, label="Min P")
                    top_p = gr.Slider(0.0, 1.0, defaults['top_p'], step=0.01, label="Top P")
                    repetition_penalty = gr.Slider(0.5, 3.0, defaults['repetition_penalty'], step=0.01, label="Repetition Penalty")
                    exaggeration = gr.Slider(0.0, 2.0, defaults['exaggeration'], step=0.01, label="Exaggeration")
                    language = gr.Textbox(value=defaults['language_id'], label="Language (id)")
                    use_audio_cache = gr.Checkbox(value=True, label="Use audio cache (exact)")
                    use_fuzzy_cache = gr.Checkbox(value=True, label="Use fuzzy cache")

                with gr.Row():
                    generate_btn = gr.Button("Generate Audio", variant="primary")
                    save_overrides_btn = gr.Button("Save Voice Overrides", variant="secondary")
                    persist_note = gr.Markdown("Will save per-voice overrides only and make a timestamped backup of config.json", elem_id="persist-note")

            with gr.Column(scale=1):
                audio_output = gr.Audio(label="Generated Speech", type="filepath")
                generate_status = gr.Textbox(label="Status", interactive=False, value="Ready")

        # Handlers
        def on_refresh():
            return gr.update(choices=list_voice_wavs())

        refresh_btn.click(fn=on_refresh, inputs=[], outputs=[voice_choices])

        def resolve_audio_choice(upload_path: str, selected_path: str) -> str:
            return upload_path or selected_path or ""

        async def do_generate(text, upload_path, selected_path, seed_val, temp, cfg, minpv, toppv, rep, exagg, lang, use_audio_flag, use_fuzzy_flag):
            try:
                audio_path = resolve_audio_choice(upload_path, selected_path)
                if not audio_path:
                    return None, "Please upload or select a .wav reference first."

                # Map to generate_audio_ui parameters without changing its signature
                # Pass a sentinel in unconditional_keys_list to control cache usage without changing signature
                # Supported tokens: "skip_audio" and/or "skip_fuzzy" (default: use both caches)
                sentinel_tokens = []
                if use_audio_flag is False:
                    sentinel_tokens.append("skip_audio")
                if use_fuzzy_flag is False:
                    sentinel_tokens.append("skip_fuzzy")
                cache_sentinel = ",".join(sentinel_tokens) if sentinel_tokens else None
                out_path, status = await generate_audio_ui(
                    model_choice=None,
                    text=text or "",
                    language=lang or "en",
                    speaker_audio=audio_path,
                    prefix_audio=None,
                    e1=None, e2=None, e3=None, e4=None, e5=None, e6=None, e7=None, e8=None,
                    vq_single=None, fmax=None, pitch_std=None, speaking_rate=None, dnsmos_ovrl=None, speaker_noised=None,
                    cfg_scale=cfg,
                    top_p_param=toppv,
                    top_k=None,
                    min_p_param=minpv,
                    linear_temp=temp,
                    confidence_rep=rep,
                    quadratic_exagg=exagg,
                    uuid_seed=int(seed_val) if seed_val is not None else -1,
                    randomize_seed_toggle=False,
                    unconditional_keys_list=cache_sentinel
                )
                return out_path, status
            except Exception as e:
                logger.exception("Generation failed")
                return None, f"Error: {e}"

        generate_btn.click(
            fn=do_generate,
            inputs=[text_input, ref_upload, voice_choices, seed, temperature, cfg_scale, min_p, top_p, repetition_penalty, exaggeration, language, use_audio_cache, use_fuzzy_cache],
            outputs=[audio_output, generate_status],
            show_progress=True,
            concurrency_limit=1
        )

        def do_save_overrides(upload_path, selected_path, temp, cfg, minpv, toppv, rep, exagg, lang):
            try:
                audio_path = resolve_audio_choice(upload_path, selected_path)
                if not audio_path:
                    return "Cannot save: select or upload a voice .wav to derive voice name."
                voice_name = normalize_stem(audio_path) or 'default'

                # Write only per-voice overrides
                to_set = {
                    'temperature': temp,
                    'cfg_weight': cfg,
                    'min_p': minpv,
                    'top_p': toppv,
                    'repetition_penalty': rep,
                    'exaggeration': exagg,
                }
                for k, v in to_set.items():
                    config.set_value(k, v, voice=voice_name)

                # Also allow language override if provided
                if lang:
                    config.set_value('language_id', lang, voice=voice_name)

                # Timestamped backup then save
                cfg_path = Path('config.json')
                backups_dir = Path('config_backups')
                backups_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                if cfg_path.exists():
                    shutil.copy2(cfg_path, backups_dir / f"config_{ts}.json")
                config.save_config(create_backup=False, filename=str(cfg_path))
                return f"Saved overrides for voice '{voice_name}'. Backup created at config_backups/config_{ts}.json"
            except Exception as e:
                logger.exception("Save overrides failed")
                return f"Save failed: {e}"

        save_overrides_btn.click(
            fn=do_save_overrides,
            inputs=[ref_upload, voice_choices, temperature, cfg_scale, min_p, top_p, repetition_penalty, exaggeration, language],
            outputs=[generate_status]
        )

        # Attach Hidden API for bridge (use our generated components)
        setup_bridge_api(demo, audio_output, generate_status, config=config)

        logger.info("Unified tab UI initialized. Bridge API connected.")
        return demo

# Tab Load Functions - simplified to match new UI structure
def load_test_tab(config):
    """Load Voice Test tab components from CONFIG (lazy on btn click)."""
    voice_choices = config.get_all_voices() or ['default']
    if 'default' not in voice_choices:
        voice_choices.append('default')

    # Get audio parameters from config with proper hierarchy
    audio_cfg = config.app_config.audio
    txs_cfg = config.app_config.tts

    # Defaults from CONFIG (global, as voice-specific needs select first)
    params = {
        'speaking_rate': audio_cfg.speaking_rate,
        'eq_gain_db': audio_cfg.eq_gain_db,
        'max_gain': audio_cfg.max_gain,
        'target_max': audio_cfg.target_max,
        'noise_floor_db': audio_cfg.noise_floor_db,
        'trim_threshold_db': audio_cfg.trim_threshold_db,
        'notch_enabled': audio_cfg.notch_enabled,
        'hp_enabled': audio_cfg.hp_enabled,
    }

    return (
        gr.update(choices=voice_choices),
        "Testing shared config voice parameters.",
        42,
        params['speaking_rate'], params['eq_gain_db'], params['max_gain'], params['target_max'],
        params['noise_floor_db'], params['trim_threshold_db'], params['notch_enabled'], params['hp_enabled'],
        {},  # test_params
        "Values loaded from CONFIG – test with custom params if needed."
    )

def update_test_voice_choices(config, voice):
    """Handle voice dropdown changes with proper config reference."""
    choices = config.get_all_voices() or ['default']
    if 'default' not in choices:
        choices.append('default')
    return gr.update(choices=choices, value=voice or 'default')

def test_voice_generation(config, voice_name, test_text, seed):
    """
    Handle voice test generation with proper config reference.
    """
    try:
        logger.info(f"Voice test requested: voice='{voice_name}', text='{test_text[:50]}...', seed={seed}")

        # Get config values using hierarchical structure
        txs_cfg = config.app_config.tts
        audio_cfg = config.app_config.audio

        # Generate UUID for cache (simple deterministic)
        cache_uuid = abs(hash(f"{voice_name}_{test_text}_{seed}")) % (2**31)

        # Call the UX-compatible audio generation interface
        result = generate_audio_ui(
            model_choice=None,
            text=test_text,
            language="en",
            speaker_audio=None,
            prefix_audio=None,
            e1=None, e2=None, e3=None, e4=None, e5=None, e6=None, e7=None, e8=None,
            vq_single=None, fmax=None, pitch_std=None,
            speaking_rate=None,
            dnsmos_ovrl=None,
            speaker_noised=False,
            cfg_scale=txs_cfg.cfg_weight,
            top_p_param=txs_cfg.top_p,
            top_k=None,
            min_p_param=txs_cfg.min_p,
            linear_temp=txs_cfg.temperature,
            confidence_rep=txs_cfg.repetition_penalty,
            quadratic_exagg=txs_cfg.exaggeration,
            uuid_seed=cache_uuid,
            randomize_seed_toggle=False,
            unconditional_keys_list=None
        )

        # Extract the path from result
        audio_path = result[0] if isinstance(result, (list, tuple)) and result else None

        # Prepare used params for display
        actual_params = {
            'voice': voice_name,
            'text': test_text,
            'seed': seed,
            'temperature': txs_cfg.temperature,
            'cfg_weight': txs_cfg.cfg_weight,
            'exaggeration': txs_cfg.exaggeration,
            'cache_uuid': cache_uuid,
            'status': result[1] if isinstance(result, (list, tuple)) and len(result) > 1 else "Unknown status"
        }

        status = "✅ Test generated successfully" if audio_path and Path(audio_path).exists() else "⚠️ Generated but audio not found"
        return audio_path, status, actual_params

    except Exception as e:
        logger.exception("Voice test failed")
        fallback_path, fallback_status = stub_wav_path()
        return None, f"❌ Test generation failed: {str(e)} | {fallback_status}", {}

if __name__ == "__main__":
    # For standalone testing
    demo = create_ui()
    demo.queue(
        concurrency_limit=3,
        max_size=9
    ).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        enable_queue=True,
        max_threads=40,
        show_error=True,
        api_open=True
    )