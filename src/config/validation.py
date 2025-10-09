"""
Shared Validation Helpers: Clamping, parsing for Config getters/setters.
Uses local CAPS copy. Returns validated value or warns/fallback on invalid.
"""

from typing import Any, Union, List, Dict, Optional
from loguru import logger
import torch  # For type checks



# Local copy of CAPS (duplicated from defaults.py—no import; breaks cycle while keeping clamps)
CAPS = {
    # Token Caps
    'MAX_NEW_TOKENS': 1499,
    'MIN_NEW_TOKENS_MIN': 100, 'MIN_NEW_TOKENS_MAX': 500,
    'MAX_CACHE_LEN_MIN': 1024, 'MAX_CACHE_LEN_MAX': 8192,
    'STRIDE_LENGTH_MIN': 4, 'STRIDE_LENGTH_MAX': 16,

    # TTS Params Caps
    'TEMPERATURE_MIN': 0.0, 'TEMPERATURE_MAX': 2.0,
    'MIN_P_MIN': 0.0, 'MIN_P_MAX': 1.0,
    'TOP_P_MIN': 0.0, 'TOP_P_MAX': 1.0,
    'REPETITION_PENALTY_MIN': 1.0, 'REPETITION_PENALTY_MAX': 2.0,
    'CFG_WEIGHT_MIN': 0.0, 'CFG_WEIGHT_MAX': 1.0,
    'EXAGGERATION_MIN': 0.0, 'EXAGGERATION_MAX': 2.0,

    # Audio Caps (all from original)
    'SPEAKING_RATE_MIN': 0.5, 'SPEAKING_RATE_MAX': 2.0,
    'EQ_GAIN_DB_MIN': -12, 'EQ_GAIN_DB_MAX': 6,
    'EQ_CUTOFF_HZ_MIN': 2000, 'EQ_CUTOFF_HZ_MAX': 5000,
    'NOTCH_GAIN_DB_MIN': -24, 'NOTCH_GAIN_DB_MAX': 0,
    'NOTCH_LOW_HZ_MIN': 4000, 'NOTCH_LOW_HZ_MAX': 10000,
    'NOTCH_HIGH_HZ_MIN': 8000, 'NOTCH_HIGH_HZ_MAX': 16000,
    'GAIN_TARGET_MAX_MIN': 0.1, 'GAIN_TARGET_MAX_MAX': 1.0,
    'GAIN_MAX_LIMIT_MIN': 0.5, 'GAIN_MAX_LIMIT_MAX': 3.0,
    'TRIM_THRESHOLD_DB_MIN': -100, 'TRIM_THRESHOLD_DB_MAX': -20,
    'FADE_MS_MIN': 0, 'FADE_MS_MAX': 100,
    'NOISE_FLOOR_DB_MIN': -80, 'NOISE_FLOOR_DB_MAX': -20,
    'MIN_POST_DURATION_SEC': 0.01, 'MAX_POST_DURATION_SEC': 10.0,
    'TRIM_FRAME_LENGTH_FACTOR_MIN': 2, 'TRIM_FRAME_LENGTH_FACTOR_MAX': 8,
    'MAX_N_FFT_FOR_TRIM_MIN': 512, 'MAX_N_FFT_FOR_TRIM_MAX': 4096,
    'MIN_SAMPLES_FOR_DENOISE_MIN': 50, 'MIN_SAMPLES_FOR_DENOISE_MAX': 1000,

    'N_FFT_DENOISE_MIN': 512, 'N_FFT_DENOISE_MAX': 2048,
    'FADE_MS_TRAIL_MIN': 20, 'FADE_MS_TRAIL_MAX': 200,
    'TRAILING_SILENCE_DB_MIN': -60, 'TRAILING_SILENCE_DB_MAX': -30,
    'FUZZY_ARTIFACT_THRESHOLD_HZ_MIN': 5000, 'FUZZY_ARTIFACT_THRESHOLD_HZ_MAX': 10000,
    'DENOISE_HIGHPASS_HZ_MIN': 50, 'DENOISE_HIGHPASS_HZ_MAX': 200,
    'DENOISE_MEDIAN_KSIZE_MIN': 1, 'DENOISE_MEDIAN_KSIZE_MAX': 5,
    'DENOISE_TARGET_BAND_LOW_MIN': 3000, 'DENOISE_TARGET_BAND_LOW_MAX': 8000,
    'DENOISE_TARGET_BAND_HIGH_MIN': 8000, 'DENOISE_TARGET_BAND_HIGH_MAX': 15000,

    # Cache Caps
    'MAX_MEMORY_ENTRIES_MIN': 10, 'MAX_MEMORY_ENTRIES_MAX': 100,

    # Fuzzy Caps
    'FUZZY_CACHE_LIMIT_MIN': 100, 'FUZZY_CACHE_LIMIT_MAX': 5000,
    'FUZZY_THRESHOLD_MIN': 0.50, 'FUZZY_THRESHOLD_MAX': 0.95,
    'AUDIO_PAD_SEC_MIN': 0.0, 'AUDIO_PAD_SEC_MAX': 0.5,
    'TINY_PAD_MULTIPLIER_MIN': 1.0, 'TINY_PAD_MULTIPLIER_MAX': 3.0,
    'TINY_THRESHOLD_SEC_MIN': 0.1, 'TINY_THRESHOLD_SEC_MAX': 1.0,
    'TEXT_ELLIPSES_COUNT_MIN': 0, 'TEXT_ELLIPSES_COUNT_MAX': 5,
    'SHORT_WORD_LEN_MIN': 1, 'SHORT_WORD_LEN_MAX': 5,
    'FUZZY_BOOST_WORDS': ['ahh', 'mmm', 'ooh', 'gasp'],  # Default list for parse
    'COMPRESS_LEVEL_MIN': 1, 'COMPRESS_LEVEL_MAX': 9,
}

def clamp_numeric(param_name: str, value: Any, caps: Optional[Dict] = None) -> Union[int, float]:
    """
    Clamp numeric value using CAPS (min/max keys derived from param_name).
    FIXED: Handle None/non-numeric by fallback to DEFAULTS (if available) or CAPS mean/0.0.
    Ensures numeric return (int/float) for consumers like frame_length = hop * frame_factor.
    """
    from src.config.defaults import DEFAULTS
    caps = caps or CAPS  # Local CAPS copy (self-contained)
    if value is None:
        # Fallback to known default if in DEFAULTS (avoids None for numerics)
        if hasattr(DEFAULTS, '__getitem__') and param_name in DEFAULTS:  # Check if DEFAULTS accessible (import if needed; here assume passed or global)
            value = DEFAULTS[param_name]  # Use stored default (int/float)
            logger.debug(f"None for {param_name} → DEFAULTS {value}")
        else:
            # Infer from CAPS min/max (mean for balanced fallback, e.g., trim_frame: (2+8)/2=5)
            min_key = param_name.upper() + '_MIN'
            max_key = param_name.upper() + '_MAX'
            if min_key in caps and max_key in caps:
                value = (caps[min_key] + caps[max_key]) / 2  # Numeric mean (float)
                logger.debug(f"None for {param_name} → CAPS mean {value}")
            else:
                # Safe numeric fallback (0.0 for most; or 1 for factors like frame)
                value = 4 if 'FRAME' in param_name.upper() or 'FACTOR' in param_name.upper() else 0.0
                logger.debug(f"No CAPS/DEFAULTS for {param_name} → fallback {value}")
        # Now value is numeric; proceed to clamp it
    if not isinstance(value, (int, float)):
        logger.warning(f"Non-numeric {param_name}: {value} → 0.0")
        value = 0.0  # Force numeric return

    # Standard clamping (unchanged)
    upper_name = param_name.upper()
    min_key = f"{upper_name}_MIN"
    max_key = f"{upper_name}_MAX"
    if min_key in caps and max_key in caps:
        min_val, max_val = caps[min_key], caps[max_key]
        clamped = max(min_val, min(max_val, value))
        if clamped != value:
            logger.debug(f"Clamped {param_name}: {value} → {clamped} (caps: {min_val}-{max_val})")
        return float(clamped)  # Always float for consistency (or int if original int)
    return float(value)  # No CAPS: Return casted float

def parse_bool(value: Any) -> bool:
    """Parse string/different types to bool (case-insensitive)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if hasattr(value, 'lower'):
        return value.lower() in ['true', 'yes', '1', 'on']
    return False  # Fallback

def parse_list(value: Any, delimiter: str = ',', key: str = 'fuzzy_boost_words') -> List[str]:
    """Parse string to list (comma-separated, stripped/lowercased). For fuzzy_boost_words etc."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [w.strip().lower() for w in value.split(delimiter) if w.strip()]
    logger.warning(f"Invalid list value for {key}: {value}, using empty list")
    return []

def validate_and_clamp(param_name: str, value: Any, defaults: Optional[Dict[str, Any]] = None) -> Any:
    """Wrapper: Validate/clamp based on type (numeric clamp, bool parse, list parse). Returns value or fallback."""
    defaults = defaults or {}  # Optional; avoids import if not passed
    if isinstance(value, (int, float)):
        return clamp_numeric(param_name, value)
    elif isinstance(value, str):
        # For lists (e.g., fuzzy_boost_words)
        if param_name == 'fuzzy_boost_words':
            return parse_list(value)
        # Bool flags handled by setters in config.py (no heuristic needed here)
        return value  # String as-is
    elif isinstance(value, bool):
        return value
    # Fallback (use passed defaults or warn/None)
    fallback = defaults.get(param_name, None)
    if fallback is None:
        logger.warning(f"Invalid type for {param_name}: {type(value)}, using None")
        return None
    logger.warning(f"Invalid type for {param_name}: {type(value)}, using default {fallback}")
    return fallback