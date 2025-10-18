# src/ui_helpers.py
import json
from loguru import logger
import asyncio
import os
import torchaudio
from pathlib import Path
import tempfile
import torch
from .config import get_config, get_config_value
from .tts_model import ModelManager, GEN_ACTIVE_LOCK
from .generate.ui_interface import generate_audio_ui

def update_api_status():
    """Return API/model status string with proper model check."""
    try:
        model_ready = "Ready" if ModelManager.get_instance().get_model() else "Not Loaded"
        return f"API Status: Model={model_ready} | CONFIG loaded | Pipeline ready"
    except Exception as e:
        logger.error(f"Status update failed: {e}")
        return "Status: Error (check logs)"

def test_voice_generation(voice_name, test_text, seed):
    """
    Handle voice test generation with proper error handling and parameter mapping.
    Now fully uses the new pipeline through generate_audio_ui.
    """
    try:
        logger.info(f"Voice test requested: voice='{voice_name}', text='{test_text[:50]}...', seed={seed}")

        # Validate inputs
        if not test_text.strip():
            return None, "Please enter test text", {}

        # Generate UUID for cache (simple deterministic)
        cache_uuid = abs(hash(f"{voice_name}_{test_text}_{seed}")) % (2**31)

        # Call the UX-compatible audio generation interface
        # Note: All other parameters filled with defaults
        result = generate_audio_ui(
            model_choice=None,
            text=test_text,
            language="en",
            speaker_audio=None,  # Voice selected by name, not audio
            prefix_audio=None,
            e1=None, e2=None, e3=None, e4=None, e5=None, e6=None, e7=None, e8=None,
            vq_single=None, fmax=None, pitch_std=None,
            speaking_rate=None,  # Using default from config
            dnsmos_ovrl=None,
            speaker_noised=False,
            cfg_scale=get_config_value('tts.cfg_weight', 0.45),
            top_p_param=get_config_value('tts.top_p', 1.0),
            top_k=None,
            min_p_param=get_config_value('tts.min_p', 0.05),
            linear_temp=get_config_value('tts.temperature', 0.8),
            confidence_rep=get_config_value('tts.repetition_penalty', 1.2),
            quadratic_exagg=get_config_value('tts.exaggeration', 0.5),
            uuid_seed=cache_uuid,
            randomize_seed_toggle=False,
            unconditional_keys_list=None
        )

        # Extract the path from result (it's a tuple with path and status)
        audio_path = result[0] if isinstance(result, (list, tuple)) and result else None

        # Prepare used params for display
        actual_params = {
            'voice': voice_name,
            'text': test_text,
            'seed': seed,
            'temperature': get_config_value('tts.temperature', 0.8),
            'cfg_weight': get_config_value('tts.cfg_weight', 0.45),
            'exaggeration': get_config_value('tts.exaggeration', 0.5),
            'cache_uuid': cache_uuid,
            'status': result[1] if isinstance(result, (list, tuple)) and len(result) > 1 else "Unknown status"
        }

        status = "✅ Test generated successfully" if audio_path and Path(audio_path).exists() else "⚠️ Generated but audio not found"
        return audio_path, status, actual_params

    except Exception as e:
        logger.exception("Voice test failed")
        return None, f"❌ Test generation failed: {str(e)}", {}

# STUB WAV PATH - Simplified to use Pipeline-Ready Config
STUB_DIR = Path(get_config().app_config.globals.cache_dir) / "temp"
STUB_PATH = STUB_DIR / "stub_silent.wav"

def stub_wav_path(empty_dict=False):
    """Generate/reuse a persistent silent WAV file for UI fallbacks (3s silence @ 24kHz)."""
    try:
        # Ensure dir exists
        STUB_DIR.mkdir(parents=True, exist_ok=True)

        # If stub exists and is valid, reuse it
        if STUB_PATH.exists() and STUB_PATH.stat().st_size > 0:
            sr = get_config_value('globals.sr', 24000)
            try:
                info = torchaudio.info(str(STUB_PATH))
                if abs(info.sample_rate - sr) < 100 and info.num_frames > 0:
                    status = f"Reused stub: {STUB_PATH.name}"
                    return (str(STUB_PATH), status, {}) if empty_dict else (str(STUB_PATH), status)
            except Exception as e:
                logger.warning(f"Invalid stub file {STUB_PATH}: {e} - regenerating")

        # Create new stub
        sr = get_config_value('globals.sr', 24000)
        duration = 3.0  # 3s silence
        samples = int(sr * duration)
        silence = torch.zeros(1, samples, dtype=torch.float32)

        torchaudio.save(str(STUB_PATH), silence, sr, format='wav')
        status = f"Created stub: {STUB_PATH.name}"
        logger.info(status)

        return (str(STUB_PATH), status, {}) if empty_dict else (str(STUB_PATH), status)

    except Exception as e:
        logger.error(f"Stub WAV creation failed: {e}")
        fallback_path = tempfile.mktemp(suffix=".wav")
        return (fallback_path, "Audio unavailable", {}) if empty_dict else (fallback_path, "Audio unavailable")