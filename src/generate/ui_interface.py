"""
ui_interface.py - Entry point for the UI with exactly 29 parameters.

This file acts as the bridge between the UI and the internal generation pipeline.
All generation logic has been moved to the pipeline, this function is
strictly responsible for:
1. Loading model, cache_manager, config (if not available)
2. Creating the context object with UI-provided parameters + loaded internals
3. Executing the pipeline
4. Formatting the result for Gradio
"""

import asyncio
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import tempfile
import time

import gradio as gr
import torch
from loguru import logger

from src.config import get_config
from src.tts_model import ModelManager, GEN_ACTIVE_LOCK
from src.generate.pipeline.context import AudioGenerationContext
from src.generate.cache.cache_manager import CacheManager, get_cache_manager
from src.seeding import cpp_uuid_to_seed

# Cache for hot reloads (loaded model/config)
PIPELINE_CACHE = None
CONFIG_CACHE = None
CACHE_MANAGER_CACHE = None

def _ensure_valid_return(result: Any, error_msg: Optional[str] = None) -> list:
    """Ensure return value matches Gradio's expected output structure."""
    status_text = f"Error: {error_msg[:100]}" if error_msg else "Audio generated successfully"

    # Format valid result
    if isinstance(result, (list, tuple)) and len(result) >= 2:
        return [str(result[0]), status_text]
    elif isinstance(result, str) and Path(result).exists():
        return [result, status_text]

    # Try fallbacks in order of reliability
    try:
        sr = get_config().app_config.globals.sr  # Fixed: Use config.globals.sr
        fallback_path = _create_raw_silence_fallback(0, sr)
        return [fallback_path, status_text]
    except:
        try:
            fallback_path = _create_raw_silence_fallback(0, 24000)
            return [fallback_path, status_text]
        except:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                return [tmp.name, f"Critical fallback: {error_msg}" if error_msg else "Status"]

def _create_raw_silence_fallback(uuid: int, sr: int = 24000) -> str:
    """Create static silence fallback WAV file once for reuse."""
    import numpy as np
    from scipy.io import wavfile

    temp_dir = Path(tempfile.gettempdir())
    temp_dir.mkdir(parents=True, exist_ok=True)  # Use system temp as fallback

    # Use predictable name for reuse across requests
    silence_path = temp_dir / f"silence_{sr}.wav"

    if not silence_path.exists():
        duration = 0.5  # seconds
        t = np.linspace(0, duration, int(sr * duration))
        silence = np.zeros_like(t, dtype=np.float32)
        wavfile.write(str(silence_path), sr, silence)
        logger.debug(f"Created silence fallback: {silence_path}")

    return str(silence_path)

async def generate_audio_ui(
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
    top_p_param=1.0,
    top_k=None,
    min_p_param=0.5,
    linear_temp=None,
    confidence_rep=None,
    quadratic_exagg=None,
    uuid_seed=-1,
    randomize_seed_toggle: bool = False,
    unconditional_keys_list=None
):
    """Entry point with exactly 29 parameters for UI compatibility."""
    result = None
    error_msg = None
    global CONFIG_CACHE, CACHE_MANAGER_CACHE, PIPELINE_CACHE

    try:
        # Load config (global)
        config = get_config()
        if CONFIG_CACHE is None:
            CONFIG_CACHE = config  # Cache for reuse

        # Load model (via singleton)
        model_manager = ModelManager.get_instance()
        model_type = 'multilingual' if config.app_config.globals.multilingual else 'english'  # Fixed: config.globals
        if not model_manager.is_loaded(model_type):
            model_manager.load_model(model_type)
        model = model_manager.get_model()
        if model is None:
            error_msg = "No TTS model available"
            return _ensure_valid_return(None, error_msg)

        # Initialize cache_manager (lazy load)
        if CACHE_MANAGER_CACHE is None:
            CACHE_MANAGER_CACHE = get_cache_manager(config)
        cache_manager = CACHE_MANAGER_CACHE

        # Create generation context with params + loaded internals
        context = _create_generation_context(
            text=text,
            audio_prompt_path=speaker_audio,  # UI-provided audio
            cache_uuid=uuid_seed,
            exaggeration=quadratic_exagg or 0.5,
            temperature=linear_temp or 0.8,
            cfgw=cfg_scale,
            min_p=min_p_param,
            top_p=top_p_param,
            repetition_penalty=confidence_rep or 1.2,
            language_id=language,
            seed_num=cpp_uuid_to_seed(uuid_seed) if randomize_seed_toggle else None,
            enable_memory_cache=True,
            enable_disk_cache=True,
            model=model,  # FIXED: Pass the loaded model
            config=config,  # FIXED: Pass the loaded config
            cache_manager=cache_manager  # FIXED: Pass the cache_manager
        )



        if PIPELINE_CACHE is None:
            from .pipeline.coordinator import GenerationCoordinator
            PIPELINE_CACHE = GenerationCoordinator()
        pipeline = PIPELINE_CACHE

        # Execute the pipeline within concurrency control
        start_time = time.time()
        with GEN_ACTIVE_LOCK:
            # Run sync pipeline in threadpool (non-blocking for async UI)
            result = await asyncio.to_thread(pipeline.run, context)

        # Process result
        if result and result.output_path and Path(result.output_path).exists():
            return _ensure_valid_return(result.output_path)

        error_msg = "No valid output path generated"
        return _ensure_valid_return(None, error_msg)

    except Exception as e:
        error_msg = f"UI generation error: {str(e)}"
        logger.exception(error_msg)
        return _ensure_valid_return(None, error_msg)

def _create_generation_context(
    text: str,
    audio_prompt_path: Optional[str],
    cache_uuid: int,
    exaggeration: Optional[float],  # UI param (or None)
    temperature: Optional[float],  # UI param (or None)
    cfgw: float,
    min_p: float,
    top_p: float,
    repetition_penalty: float,
    language_id: str,
    seed_num: int,
    enable_memory_cache: bool = True,
    enable_disk_cache: bool = True,
    model: Optional[Any] = None,
    config: Optional["AppConfig"] = None,
    cache_manager: Optional["CacheManager"] = None
) -> AudioGenerationContext:
    """Creates context object with UI-provided parameters + injected loaders."""
    if config is None:
        config = get_config()
    if model is None:
        model = ModelManager.get_instance().get_model()

    # Validate and normalize audio path first (needed for voice stem)
    if audio_prompt_path and not Path(audio_prompt_path).exists():
        logger.warning(f"Provided audio path does not exist: {audio_prompt_path}")
        audio_prompt_path = None

    # Determine voice stem
    voice_stem = None
    if audio_prompt_path:
        from src.normalize_stem import normalize_stem
        voice_stem = normalize_stem(audio_path=audio_prompt_path)

    # FIXED: Use get_voice_params (sole merger; replaces deprecated)
    voice_params = config.get_voice_params(voice_name=voice_stem) if voice_stem else {}

    # Coerce seeds
    seed = cpp_uuid_to_seed(cache_uuid) if seed_num is None else seed_num

    # Create and initialize context
    context = AudioGenerationContext(
        text=text,
        audio_prompt_path=audio_prompt_path,
        cache_uuid=cache_uuid,
        exaggeration=exaggeration or voice_params.get('exaggeration'),  # From merged (or None -> model default)
        temperature=temperature or voice_params.get('temperature'),    # From merged (or None -> model default)
        cfgw=cfgw or voice_params.get('cfg_weight', 0.45),
        min_p=min_p or voice_params.get('min_p', 0.05),
        top_p=top_p or voice_params.get('top_p', 1.0),
        repetition_penalty=repetition_penalty or voice_params.get('repetition_penalty', 1.2),
        language_id=language_id,
        seed=seed,
        enable_memory_cache=enable_memory_cache,
        enable_disk_cache=enable_disk_cache,
        voice_stem=voice_stem or "default",
        voice_params=voice_params,  # Now from get_voice_params (merged + cached)
        t3_params={
            "generate_token_backend": "cudagraphs-manual",
            "stride_length": 4,
            "skip_when_1": True
        },
        device=torch.device(config.app_config.globals.device if config else "cuda" if torch.cuda.is_available() else "cpu"),
        dtype=config.app_config.globals.dtype if config else torch.bfloat16,
        model=model,
        config=config,
        cache_manager=cache_manager,
        sr=config.app_config.globals.sr if config else 24000,
        multilingual=config.app_config.globals.multilingual if config else False
    )


    logger.debug(f"Context created: voice_stem={voice_stem}, model present={model is not None}, cache_manager={cache_manager is not None}")
    return context

def setup_bridge_api(demo, audio_output, api_status_md, config=None):
    """
    Registers 29 hidden components + button/.click for /api/generate_audio.
    Args:
        demo: Gradio demo instance
        audio_output: Audio output component for integration
        api_status_md: Status component for API updates
        config: Optional config object (faster to inject than global lookup)
    """
    # Get config safely (either injected or global)
    if config is None:
        from src.config import get_config
        config = get_config()

    # Extract model type from properly structured config
    model_type = 'multilingual' if config.app_config.globals.multilingual else 'english'

    # Load model using dependency-injected config
    from src.tts_model import ModelManager
    model_manager = ModelManager.get_instance()

    # Check if model already loaded before loading again
    if not model_manager.is_loaded(model_type):
        model_manager.load_model(model_type)

    # 29 hidden components (exactly matching UI requirements)
    model_choice = gr.Textbox(visible=False, value=None, label="Model Choice")
    text_input_hidden = gr.Textbox(visible=False, value="", label="Text Hidden")
    language = gr.Textbox(visible=False, value="en", label="Language")
    speaker_audio = gr.Audio(visible=False, type="filepath", value=None, label="Speaker Audio")
    prefix_audio = gr.Audio(visible=False, type="filepath", value=None, label="Prefix Audio")
    emotion1 = gr.Number(visible=False, value=0, label="Emotion 1")
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

    # Click event with exactly 29 inputs
    hidden_api_btn.click(
        fn=generate_audio_ui,
        inputs=[
            model_choice, text_input_hidden, language, speaker_audio, prefix_audio,
            emotion1, emotion2, emotion3, emotion4, emotion5, emotion6, emotion7, emotion8,
            vq_single, fmax, pitch_std, speaking_rate_param, dnsmos_ovrl, speaker_noised,
            cfg_scale, top_p_param, top_k, min_p_param, linear_temp, confidence_rep, quadratic_exagg,
            uuid_seed, randomize_seed_toggle, unconditional_keys_list
        ],
        outputs=[audio_output, api_status_md],
        api_name="generate_audio",
        show_progress=True,
        concurrency_limit=4
    )

    logger.info(
        "Hidden API bridge connected: "
        f"29 parameters correctly wired to generate_audio_ui | "
        f"Model: {model_type}"
    )

# Expose for ui.py import (minimal)
__all__ = ['generate_audio_ui', 'setup_bridge_api']