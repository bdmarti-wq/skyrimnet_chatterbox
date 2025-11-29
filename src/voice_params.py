"""
Canonical voice parameter merging utilities.

Merge order (lowest → highest precedence):
1) Sane model defaults from models.py (TtsConfig/AudioConfig class defaults)
2) Global config values from config.json (globals.tts / globals.audio)
3) Voice-specific overrides in config.json (config.voices[voice])
4) Call-site overrides dict (explicit per-request UI/API overrides)

Only keys present in the upstream models are pre-populated; any extra keys
supplied by voice-specific overrides or the `overrides` dict are passed through
verbatim so callers can introduce new experimental knobs without changing this
module.
"""
from __future__ import annotations

from typing import Dict, Optional, Any
from loguru import logger

try:
    # Local imports (no heavy deps)
    from src.config.models import TtsConfig, AudioConfig, AppConfig, VoiceConfig
except Exception as _e:  # pragma: no cover - defensive
    TtsConfig = AudioConfig = AppConfig = VoiceConfig = object  # type: ignore
    logger.warning(f"voice_params: fallback types due to import error: {_e}")


def _defaults_from_models() -> Dict[str, Any]:
    """Return a dict of sane defaults derived from model classes' default values.

    We instantiate the Pydantic models with no arguments to obtain class defaults,
    which represent the "sane" baseline independent of user config.
    """
    try:
        tts = TtsConfig()  # class defaults
        aud = AudioConfig()
        base = {
            # Core generation params
            'temperature': tts.temperature,
            'min_p': tts.min_p,
            'top_p': tts.top_p,
            'repetition_penalty': tts.repetition_penalty,
            'cfg_weight': tts.cfg_weight,
            'exaggeration': tts.exaggeration,
            # A few commonly used audio/post-processing defaults that may be referenced
            'speaking_rate': aud.speaking_rate,
            'eq_gain_db': aud.eq_gain_db,
            'eq_cutoff_hz': aud.eq_cutoff_hz,
            'notch_gain_db': aud.notch_gain_db,
            'notch_low_hz': aud.notch_low_hz,
            'notch_high_hz': aud.notch_high_hz,
            'fade_ms': aud.fade_ms,
            'gain_max_limit': aud.gain_max_limit,
            'trailing_silence_db': aud.trailing_silence_db,
            'max_short_word_len': aud.max_short_word_len,
            'enable_post_processing': aud.enable_post_processing,
        }
        return base
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"voice_params: failed to derive model defaults: {e}")
        return {}


def _merge(dst: Dict[str, Any], src: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Shallow merge where non-None values in src override dst."""
    if not src:
        return dst
    for k, v in src.items():
        if v is not None:
            dst[k] = v
    return dst


def get_voice_params(config: 'AppConfig', voice_name: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Collect and merge voice parameters following the defined precedence.

    Args:
        config: Loaded AppConfig instance.
        voice_name: Optional voice key; when provided, voice-specific overrides are applied.
        overrides: Final explicit overrides that win over all others.

    Returns:
        Dict[str, Any]: A flat dict of voice parameters ready for use.
    """
    # 1) Start with sane model defaults
    merged: Dict[str, Any] = _defaults_from_models()

    # 2) Overlay global config values (globals.tts + selected audio fields)
    try:
        gtts = getattr(config.globals, 'tts', None)
        gaud = getattr(config.globals, 'audio', None)
        if gtts is not None:
            merged = _merge(merged, {
                'temperature': gtts.temperature,
                'min_p': gtts.min_p,
                'top_p': gtts.top_p,
                'repetition_penalty': gtts.repetition_penalty,
                'cfg_weight': gtts.cfg_weight,
                'exaggeration': gtts.exaggeration,
            })
        if gaud is not None:
            merged = _merge(merged, {
                'speaking_rate': gaud.speaking_rate,
                'eq_gain_db': gaud.eq_gain_db,
                'eq_cutoff_hz': gaud.eq_cutoff_hz,
                'notch_gain_db': gaud.notch_gain_db,
                'notch_low_hz': gaud.notch_low_hz,
                'notch_high_hz': gaud.notch_high_hz,
                'fade_ms': gaud.fade_ms,
                'gain_max_limit': gaud.gain_max_limit,
                'trailing_silence_db': gaud.trailing_silence_db,
                'max_short_word_len': gaud.max_short_word_len,
                'enable_post_processing': gaud.enable_post_processing,
            })
    except Exception as e:  # pragma: no cover
        logger.warning(f"voice_params: failed to apply global config values: {e}")

    # 3) Apply voice-specific overrides if available
    try:
        if voice_name:
            vcfg: Optional['VoiceConfig'] = config.voices.get(str(voice_name), None)
            if vcfg is not None:
                # Convert to dict while keeping only non-None values
                # Pydantic's model_dump excludes None with exclude_none=True
                vdict = vcfg.model_dump(exclude_none=True)
                merged = _merge(merged, vdict)
    except Exception as e:  # pragma: no cover
        logger.warning(f"voice_params: failed to apply per-voice overrides for '{voice_name}': {e}")

    # 4) Final explicit overrides win
    if isinstance(overrides, dict) and overrides:
        merged = _merge(merged, overrides)

    return merged


__all__ = [
    'get_voice_params'
]
