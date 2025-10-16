"""
Pydantic Models for SkyrimNet Config: Integrated with CAPS for Validation and UI Bounds.
- globals: Nested categories (tts, audio, fuzzy) with direct CAPS references in Fields.
- voices: Flat dict per voice (as before), with overrides for any category field.
- CAPS Integration: Fields use CAPS for ge/le constraints. get_field_bounds() exposes min/max for UI (e.g., sliders).
- Merging: In config.py, merge globals categories + voice overrides (flat keys map to categories).
- Validation: Inline via Field; custom validators for clamping/warnings fallback.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from loguru import logger
from pathlib import Path
import torch

# Global CAPS dict (constraints referenced in Fields)
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
    # Add these for completeness (based on defaults; adjust values)
    'FUZZY_BOOST_AMOUNT_MIN': 0.0, 'FUZZY_BOOST_AMOUNT_MAX': 0.3,
    'N_MELS_MIN': 64, 'N_MELS_MAX': 128,
    'EBU_POST_GAIN_DB_MIN': -12, 'EBU_POST_GAIN_DB_MAX': 6,  # Reuse EQ
    'EBU_TRUE_PEAK_MIN': -1.0, 'EBU_TRUE_PEAK_MAX': 1.0,
    'MAX_GAIN_MIN': 0.0, 'MAX_GAIN_MAX': 5.0,  # Reasonable
}

# Helper: Get bounds for a field (for UI queries, e.g., slider min/max/default)
def _get_field_bounds(field_name: str) -> Optional[Dict[str, Any]]:
    """Map field_name to CAPS keys (e.g., 'min_p' → 'MIN_P_MIN'/'MIN_P_MAX'). Returns {'min': val, 'max': val} or None."""
    # Map: field_name → (min_key, max_key)
    field_map = {
        # Globals (top-level)
        'compress_level': ('COMPRESS_LEVEL_MIN', 'COMPRESS_LEVEL_MAX'),
        # TTS
        'temperature': ('TEMPERATURE_MIN', 'TEMPERATURE_MAX'),
        'min_p': ('MIN_P_MIN', 'MIN_P_MAX'),
        'top_p': ('TOP_P_MIN', 'TOP_P_MAX'),
        'repetition_penalty': ('REPETITION_PENALTY_MIN', 'REPETITION_PENALTY_MAX'),
        'cfg_weight': ('CFG_WEIGHT_MIN', 'CFG_WEIGHT_MAX'),
        'exaggeration': ('EXAGGERATION_MIN', 'EXAGGERATION_MAX'),
        'max_new_tokens': ('MIN_NEW_TOKENS_MIN', 'MAX_NEW_TOKENS'),  # Reuse for int
        'min_new_tokens': ('MIN_NEW_TOKENS_MIN', 'MIN_NEW_TOKENS_MAX'),
        'max_cache_len': ('MAX_CACHE_LEN_MIN', 'MAX_CACHE_LEN_MAX'),
        'stride_length': ('STRIDE_LENGTH_MIN', 'STRIDE_LENGTH_MAX'),
        # Audio
        'speaking_rate': ('SPEAKING_RATE_MIN', 'SPEAKING_RATE_MAX'),
        'eq_gain_db': ('EQ_GAIN_DB_MIN', 'EQ_GAIN_DB_MAX'),
        'eq_cutoff_hz': ('EQ_CUTOFF_HZ_MIN', 'EQ_CUTOFF_HZ_MAX'),
        'notch_gain_db': ('NOTCH_GAIN_DB_MIN', 'NOTCH_GAIN_DB_MAX'),
        'notch_low_hz': ('NOTCH_LOW_HZ_MIN', 'NOTCH_LOW_HZ_MAX'),
        'notch_high_hz': ('NOTCH_HIGH_HZ_MIN', 'NOTCH_HIGH_HZ_MAX'),
        'gain_max_limit': ('GAIN_MAX_LIMIT_MIN', 'GAIN_MAX_LIMIT_MAX'),
        'trim_threshold_db': ('TRIM_THRESHOLD_DB_MIN', 'TRIM_THRESHOLD_DB_MAX'),
        'fade_ms': ('FADE_MS_MIN', 'FADE_MS_MAX'),
        'noise_floor_db': ('NOISE_FLOOR_DB_MIN', 'NOISE_FLOOR_DB_MAX'),
        'n_fft': ('MAX_N_FFT_FOR_TRIM_MIN', 'MAX_N_FFT_FOR_TRIM_MAX'),  # Reuse
        'hop_length': ('MIN_SAMPLES_FOR_DENOISE_MIN', 'MIN_SAMPLES_FOR_DENOISE_MAX'),  # Approximate
        'base_audio_pad_sec': ('AUDIO_PAD_SEC_MIN', 'AUDIO_PAD_SEC_MAX'),
        'tiny_audio_pad_multiplier': ('TINY_PAD_MULTIPLIER_MIN', 'TINY_PAD_MULTIPLIER_MAX'),
        'tiny_threshold_sec': ('TINY_THRESHOLD_SEC_MIN', 'TINY_THRESHOLD_SEC_MAX'),
        'n_fft_denoise': ('N_FFT_DENOISE_MIN', 'N_FFT_DENOISE_MAX'),
        'denoise_median_ksize': ('DENOISE_MEDIAN_KSIZE_MIN', 'DENOISE_MEDIAN_KSIZE_MAX'),
        'denoise_target_band_low': ('DENOISE_TARGET_BAND_LOW_MIN', 'DENOISE_TARGET_BAND_LOW_MAX'),
        'denoise_target_band_high': ('DENOISE_TARGET_BAND_HIGH_MIN', 'DENOISE_TARGET_BAND_HIGH_MAX'),
        'trailing_silence_db': ('TRAILING_SILENCE_DB_MIN', 'TRAILING_SILENCE_DB_MAX'),
        'denoise_highpass_hz': ('DENOISE_HIGHPASS_HZ_MIN', 'DENOISE_HIGHPASS_HZ_MAX'),
        'gain_target_max': ('GAIN_TARGET_MAX_MIN', 'GAIN_TARGET_MAX_MAX'),
        'max_gain': ('MAX_GAIN_MIN', 'MAX_GAIN_MAX'),
        'n_mels': ('N_MELS_MIN', 'N_MELS_MAX'),
        'ebu_post_gain_db': ('EBU_POST_GAIN_DB_MIN', 'EBU_POST_GAIN_DB_MAX'),
        'ebu_true_peak': ('EBU_TRUE_PEAK_MIN', 'EBU_TRUE_PEAK_MAX'),
        # Fuzzy
        'fuzzy_threshold': ('FUZZY_THRESHOLD_MIN', 'FUZZY_THRESHOLD_MAX'),
        'fuzzy_boost_amount': ('FUZZY_BOOST_AMOUNT_MIN', 'FUZZY_BOOST_AMOUNT_MAX'),
        'fuzzy_index_size': ('FUZZY_CACHE_LIMIT_MIN', 'FUZZY_CACHE_LIMIT_MAX'),
        'fuzzy_artifact_threshold_hz': ('FUZZY_ARTIFACT_THRESHOLD_HZ_MIN', 'FUZZY_ARTIFACT_THRESHOLD_HZ_MAX'),
    }
    if (bounds_tuple := field_map.get(field_name)) is None:
        return None  # No bounds
    min_val = CAPS.get(bounds_tuple[0], 0.0)
    max_val = CAPS.get(bounds_tuple[1], 1.0)
    return {'min': min_val, 'max': max_val}

# TTS Category: Generation params
class TtsConfig(BaseModel):
    model_config = ConfigDict(extra='ignore', populate_by_name=True)

    temperature: float = Field(default=0.7, ge=CAPS['TEMPERATURE_MIN'], le=CAPS['TEMPERATURE_MAX'])
    min_p: float = Field(default=0.07, ge=CAPS['MIN_P_MIN'], le=CAPS['MIN_P_MAX'])
    top_p: float = Field(default=1.0, ge=CAPS['TOP_P_MIN'], le=CAPS['TOP_P_MAX'])
    repetition_penalty: float = Field(default=2.0, ge=CAPS['REPETITION_PENALTY_MIN'], le=CAPS['REPETITION_PENALTY_MAX'])
    cfg_weight: float = Field(default=0.45, ge=CAPS['CFG_WEIGHT_MIN'], le=CAPS['CFG_WEIGHT_MAX'])
    exaggeration: float = Field(default=0.7, ge=CAPS['EXAGGERATION_MIN'], le=CAPS['EXAGGERATION_MAX'])
    max_new_tokens: int = Field(default=1499, ge=CAPS['MIN_NEW_TOKENS_MIN'], le=CAPS['MAX_NEW_TOKENS'])
    min_new_tokens: int = Field(default=224, ge=CAPS['MIN_NEW_TOKENS_MIN'], le=CAPS['MIN_NEW_TOKENS_MAX'])
    max_cache_len: int = Field(default=4096, ge=CAPS['MAX_CACHE_LEN_MIN'], le=CAPS['MAX_CACHE_LEN_MAX'])
    stride_length: int = Field(default=8, ge=CAPS['STRIDE_LENGTH_MIN'], le=CAPS['STRIDE_LENGTH_MAX'])
    compile_t3: bool = True  # Enable T3 compile
    warmup_t3: bool = True  # Enable warmup
    re_optimize_on_reload: bool = False  # Re-apply opts on get_model re-load

    def get_field_bounds(self, field_name: str) -> Optional[Dict[str, Any]]:
        """UI helper: Return min/max for field from CAPS."""
        return _get_field_bounds(field_name)

# Audio Category: Post-processing params
class AudioConfig(BaseModel):
    model_config = ConfigDict(extra='ignore', populate_by_name=True)

    enable_post_processing: bool = Field(default=True)
    enable_post_resample: bool = Field(default=False)
    enable_post_jit_gain: bool = Field(default=True)
    enable_post_voice_processing: bool = Field(default=True)
    enable_pre_adjustment: bool = Field(default=True)
    eq_gain_db: float = Field(default=0.0, ge=CAPS['EQ_GAIN_DB_MIN'], le=CAPS['EQ_GAIN_DB_MAX'])
    eq_cutoff_hz: float = Field(default=3000.0, ge=CAPS['EQ_CUTOFF_HZ_MIN'], le=CAPS['EQ_CUTOFF_HZ_MAX'])
    notch_gain_db: float = Field(default=0.0, ge=CAPS['NOTCH_GAIN_DB_MIN'], le=CAPS['NOTCH_GAIN_DB_MAX'])
    notch_low_hz: float = Field(default=8000.0, ge=CAPS['NOTCH_LOW_HZ_MIN'], le=CAPS['NOTCH_LOW_HZ_MAX'])
    notch_high_hz: float = Field(default=11000.0, ge=CAPS['NOTCH_HIGH_HZ_MIN'], le=CAPS['NOTCH_HIGH_HZ_MAX'])
    fade_ms: float = Field(default=0.0, ge=CAPS['FADE_MS_MIN'], le=CAPS['FADE_MS_MAX'])
    speaking_rate: float = Field(default=1.0, ge=CAPS['SPEAKING_RATE_MIN'], le=CAPS['SPEAKING_RATE_MAX'])
    normalize_method: str = Field(default='peak')  # No CAPS; pattern if needed
    gain_max_limit: float = Field(default=1.0, ge=CAPS['GAIN_MAX_LIMIT_MIN'], le=CAPS['GAIN_MAX_LIMIT_MAX'])
    noise_floor_db: float = Field(default=-60.0, ge=CAPS['NOISE_FLOOR_DB_MIN'], le=CAPS['NOISE_FLOOR_DB_MAX'])
    trim_threshold_db: float = Field(default=-100.0, ge=CAPS['TRIM_THRESHOLD_DB_MIN'], le=CAPS['TRIM_THRESHOLD_DB_MAX'])
    enable_denoise_normalize: bool = Field(default=False)
    enable_denoising: bool = Field(default=False)
    enable_audio_padding: bool = Field(default=False)
    base_audio_pad_sec: float = Field(default=0.15, ge=CAPS['AUDIO_PAD_SEC_MIN'], le=CAPS['AUDIO_PAD_SEC_MAX'])
    tiny_audio_pad_multiplier: float = Field(default=2.0, ge=CAPS['TINY_PAD_MULTIPLIER_MIN'], le=CAPS['TINY_PAD_MULTIPLIER_MAX'])
    tiny_threshold_sec: float = Field(default=0.5, ge=CAPS['TINY_THRESHOLD_SEC_MIN'], le=CAPS['TINY_THRESHOLD_SEC_MAX'])
    n_fft: int = Field(default=2048, ge=CAPS['MAX_N_FFT_FOR_TRIM_MIN'], le=CAPS['MAX_N_FFT_FOR_TRIM_MAX'])
    hop_length: int = Field(default=256, ge=CAPS['MIN_SAMPLES_FOR_DENOISE_MIN'], le=CAPS['MIN_SAMPLES_FOR_DENOISE_MAX'])  # Approximate reuse
    highpass_cutoff_hz: float = Field(default=50.0, ge=CAPS['DENOISE_HIGHPASS_HZ_MIN'], le=CAPS['DENOISE_HIGHPASS_HZ_MAX'])
    n_fft_denoise: int = Field(default=1024, ge=CAPS['N_FFT_DENOISE_MIN'], le=CAPS['N_FFT_DENOISE_MAX'])
    denoise_median_ksize: int = Field(default=3, ge=CAPS['DENOISE_MEDIAN_KSIZE_MIN'], le=CAPS['DENOISE_MEDIAN_KSIZE_MAX'])
    denoise_target_band_low: float = Field(default=5000.0, ge=CAPS['DENOISE_TARGET_BAND_LOW_MIN'], le=CAPS['DENOISE_TARGET_BAND_LOW_MAX'])
    denoise_target_band_high: float = Field(default=12000.0, ge=CAPS['DENOISE_TARGET_BAND_HIGH_MIN'], le=CAPS['DENOISE_TARGET_BAND_HIGH_MAX'])
    trailing_silence_db: float = Field(default=-45.0, ge=CAPS['TRAILING_SILENCE_DB_MIN'], le=CAPS['TRAILING_SILENCE_DB_MAX'])
    gain_target_max: float = Field(default=1.0, ge=CAPS['GAIN_TARGET_MAX_MIN'], le=CAPS['GAIN_TARGET_MAX_MAX'])
    max_gain: float = Field(default=2.0, ge=CAPS['MAX_GAIN_MIN'], le=CAPS['MAX_GAIN_MAX'])
    n_mels: int = Field(default=80, ge=CAPS['N_MELS_MIN'], le=CAPS['N_MELS_MAX'])
    ebu_post_gain_db: float = Field(default=0.0, ge=CAPS['EBU_POST_GAIN_DB_MIN'], le=CAPS['EBU_POST_GAIN_DB_MAX'])
    ebu_true_peak: float = Field(default=0.0, ge=CAPS['EBU_TRUE_PEAK_MIN'], le=CAPS['EBU_TRUE_PEAK_MAX'])

    # Validator fallback for any missed clamps (e.g., complex logic)
    @model_validator(mode='after')
    def validate_audio_params(self):
        # Example: Ensure notch_low_hz < notch_high_hz if needed
        if self.notch_low_hz >= self.notch_high_hz:
            logger.warning("notch_low_hz >= notch_high_hz; adjusting notch_high_hz")
            self.notch_high_hz = self.notch_low_hz + 1000  # Minimal adjustment
        return self

    def get_field_bounds(self, field_name: str) -> Optional[Dict[str, Any]]:
        """UI helper: Return min/max for field from CAPS."""
        return _get_field_bounds(field_name)


# Fuzzy Category: Fuzzy cache params
class FuzzyConfig(BaseModel):
    model_config = ConfigDict(extra='ignore', populate_by_name=True)

    enable_fuzzy_cache: bool = Field(default=True)
    fuzzy_threshold: float = Field(default=0.70, ge=CAPS['FUZZY_THRESHOLD_MIN'], le=CAPS['FUZZY_THRESHOLD_MAX'])
    fuzzy_boost_amount: float = Field(default=0.15, ge=CAPS['FUZZY_BOOST_AMOUNT_MIN'], le=CAPS['FUZZY_BOOST_AMOUNT_MAX'])
    fuzzy_boost_words: List[str] = Field(default_factory=lambda: CAPS['FUZZY_BOOST_WORDS'])  # List from CAPS
    fuzzy_index_size: int = Field(default=1000, ge=CAPS['FUZZY_CACHE_LIMIT_MIN'], le=CAPS['FUZZY_CACHE_LIMIT_MAX'])
    fuzzy_artifact_threshold_hz: float = Field(default=7000.0, ge=CAPS['FUZZY_ARTIFACT_THRESHOLD_HZ_MIN'], le=CAPS['FUZZY_ARTIFACT_THRESHOLD_HZ_MAX'])

    # Validator for input parsing (str/list → normalized list[str])
    @field_validator('fuzzy_boost_words', mode='before')
    @classmethod
    def validate_fuzzy_boost_words(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            return [w.strip().lower() for w in v.split(',') if w.strip()]
        elif isinstance(v, list):
            return [str(w).strip().lower() for w in v]
        logger.warning(f"Invalid fuzzy_boost_words {v}; using default from CAPS")
        return CAPS['FUZZY_BOOST_WORDS']

    def get_field_bounds(self, field_name: str) -> Optional[Dict[str, Any]]:
        """UI helper: Return min/max for field from CAPS."""
        return _get_field_bounds(field_name)


# Globals: Nested categories (tts, audio, fuzzy, etc.)
# Globals: Nested categories (tts, audio, fuzzy, etc.)
class Globals(BaseModel):
    """
    Top-level globals: Nested by category for clarity (tts, audio, fuzzy).
    Each category is a sub-model.
    """
    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True)  # Allow torch.dtype

    tts: TtsConfig = Field(default_factory=TtsConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    fuzzy: FuzzyConfig = Field(default_factory=FuzzyConfig)

    # Other fields (logging, etc.)
    dtype: torch.dtype = Field(default=torch.bfloat16)
    logging_level: str = Field(default='INFO')
    compress_pt_saves: bool = Field(default=True)
    compress_level: int = Field(default=6, ge=CAPS['COMPRESS_LEVEL_MIN'], le=CAPS['COMPRESS_LEVEL_MAX'])
    enable_deferred_cleanup: bool = Field(default=True)
    enable_memory_cache: bool = Field(default=True)
    enable_disk_cache: bool = Field(default=True)
    tts_name: str = Field(default='Chatterbox')

    # Core fields
    device: str = Field(default='cuda')
    multilingual: bool = Field(default=False)
    sr: int = Field(default=24000)
    model: Optional[Any] = Field(default=None)

    # Core validators (coercion for device/dtype)
    @field_validator('device', mode='before')
    @classmethod
    def parse_device(cls, v):
        if isinstance(v, str):
            v = v.lower()
            return 'cuda' if torch.cuda.is_available() and v == 'cuda' else 'cpu'
        logger.warning(f"Invalid device {v}; using 'cuda' if available")
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    @field_validator('dtype', mode='before')
    @classmethod
    def parse_dtype(cls, v):
        if isinstance(v, str):
            v = v.lower()
            if v == 'bfloat16' and torch.cuda.is_available():
                return torch.bfloat16
            elif v == 'float32':
                return torch.float32
            else:
                logger.warning(f"Invalid dtype '{v}'; using bfloat16 on CUDA if available")
                return torch.bfloat16 if torch.cuda.is_available() else torch.float32
        elif isinstance(v, torch.dtype):
            return v
        logger.warning(f"Invalid dtype {v}; using float32")
        return torch.float32

    def get_field_bounds(self, field_name: str) -> Optional[Dict[str, Any]]:
        """UI helper: Return min/max for field from CAPS (delegates to sub-model if nested)."""
        # Handle direct top-level fields (e.g., 'compress_level')
        if field_name == 'compress_level':
            return _get_field_bounds(field_name)
        # Delegate to sub-models for nested (e.g., 'tts.temperature' not handled here; use full path in callers)
        if field_name in ['tts', 'audio', 'fuzzy']:
            return None  # Nested; use sub.get_field_bounds
        return _get_field_bounds(field_name)


# VoiceConfig: Flat overrides (as before)
class VoiceConfig(BaseModel):
    """
    Per-voice overrides: Flat dict (matches original voices.json).
    Optional; uses globals categories during merge (None means inherit from globals).
    Bounds enforced only if value is not None (via validators; Pydantic skips for None).
    """
    model_config = ConfigDict(extra='ignore', populate_by_name=True)

    # TTS overrides (optional; inherit from globals.tts)
    temperature: Optional[float] = Field(default=None, ge=CAPS['TEMPERATURE_MIN'], le=CAPS['TEMPERATURE_MAX'])
    min_p: Optional[float] = Field(default=None, ge=CAPS['MIN_P_MIN'], le=CAPS['MIN_P_MAX'])
    top_p: Optional[float] = Field(default=None, ge=CAPS['TOP_P_MIN'], le=CAPS['TOP_P_MAX'])
    repetition_penalty: Optional[float] = Field(default=None, ge=CAPS['REPETITION_PENALTY_MIN'], le=CAPS['REPETITION_PENALTY_MAX'])
    cfg_weight: Optional[float] = Field(default=None, ge=CAPS['CFG_WEIGHT_MIN'], le=CAPS['CFG_WEIGHT_MAX'])
    exaggeration: Optional[float] = Field(default=None, ge=CAPS['EXAGGERATION_MIN'], le=CAPS['EXAGGERATION_MAX'])
    max_new_tokens: Optional[int] = Field(default=None, ge=CAPS['MIN_NEW_TOKENS_MIN'], le=CAPS['MAX_NEW_TOKENS'])
    min_new_tokens: Optional[int] = Field(default=None, ge=CAPS['MIN_NEW_TOKENS_MIN'], le=CAPS['MIN_NEW_TOKENS_MAX'])
    max_cache_len: Optional[int] = Field(default=None, ge=CAPS['MAX_CACHE_LEN_MIN'], le=CAPS['MAX_CACHE_LEN_MAX'])
    stride_length: Optional[int] = Field(default=None, ge=CAPS['STRIDE_LENGTH_MIN'], le=CAPS['STRIDE_LENGTH_MAX'])

    # Audio overrides (optional; inherit from globals.audio)
    enable_post_processing: Optional[bool] = Field(default=None)
    enable_post_resample: Optional[bool] = Field(default=None)
    enable_post_jit_gain: Optional[bool] = Field(default=None)
    enable_post_voice_processing: Optional[bool] = Field(default=None)
    enable_pre_adjustment: Optional[bool] = Field(default=None)
    eq_gain_db: Optional[float] = Field(default=None, ge=CAPS['EQ_GAIN_DB_MIN'], le=CAPS['EQ_GAIN_DB_MAX'])
    eq_cutoff_hz: Optional[float] = Field(default=None, ge=CAPS['EQ_CUTOFF_HZ_MIN'], le=CAPS['EQ_CUTOFF_HZ_MAX'])
    notch_gain_db: Optional[float] = Field(default=None, ge=CAPS['NOTCH_GAIN_DB_MIN'], le=CAPS['NOTCH_GAIN_DB_MAX'])
    notch_low_hz: Optional[float] = Field(default=None, ge=CAPS['NOTCH_LOW_HZ_MIN'], le=CAPS['NOTCH_LOW_HZ_MAX'])
    notch_high_hz: Optional[float] = Field(default=None, ge=CAPS['NOTCH_HIGH_HZ_MIN'], le=CAPS['NOTCH_HIGH_HZ_MAX'])
    fade_ms: Optional[float] = Field(default=None, ge=CAPS['FADE_MS_MIN'], le=CAPS['FADE_MS_MAX'])
    speaking_rate: Optional[float] = Field(default=None, ge=CAPS['SPEAKING_RATE_MIN'], le=CAPS['SPEAKING_RATE_MAX'])
    normalize_method: Optional[str] = Field(default=None)
    gain_max_limit: Optional[float] = Field(default=None, ge=CAPS['GAIN_MAX_LIMIT_MIN'], le=CAPS['GAIN_MAX_LIMIT_MAX'])
    noise_floor_db: Optional[float] = Field(default=None, ge=CAPS['NOISE_FLOOR_DB_MIN'], le=CAPS['NOISE_FLOOR_DB_MAX'])
    trim_threshold_db: Optional[float] = Field(default=None, ge=CAPS['TRIM_THRESHOLD_DB_MIN'], le=CAPS['TRIM_THRESHOLD_DB_MAX'])
    enable_denoise_normalize: Optional[bool] = Field(default=None)
    enable_denoising: Optional[bool] = Field(default=None)
    enable_audio_padding: Optional[bool] = Field(default=None)
    base_audio_pad_sec: Optional[float] = Field(default=None, ge=CAPS['AUDIO_PAD_SEC_MIN'], le=CAPS['AUDIO_PAD_SEC_MAX'])
    tiny_audio_pad_multiplier: Optional[float] = Field(default=None, ge=CAPS['TINY_PAD_MULTIPLIER_MIN'], le=CAPS['TINY_PAD_MULTIPLIER_MAX'])
    tiny_threshold_sec: Optional[float] = Field(default=None, ge=CAPS['TINY_THRESHOLD_SEC_MIN'], le=CAPS['TINY_THRESHOLD_SEC_MAX'])
    n_fft: Optional[int] = Field(default=None, ge=CAPS['MAX_N_FFT_FOR_TRIM_MIN'], le=CAPS['MAX_N_FFT_FOR_TRIM_MAX'])
    hop_length: Optional[int] = Field(default=None, ge=CAPS['MIN_SAMPLES_FOR_DENOISE_MIN'], le=CAPS['MIN_SAMPLES_FOR_DENOISE_MAX'])
    highpass_cutoff_hz: Optional[float] = Field(default=None, ge=CAPS['DENOISE_HIGHPASS_HZ_MIN'], le=CAPS['DENOISE_HIGHPASS_HZ_MAX'])
    n_fft_denoise: Optional[int] = Field(default=None, ge=CAPS['N_FFT_DENOISE_MIN'], le=CAPS['N_FFT_DENOISE_MAX'])
    denoise_median_ksize: Optional[int] = Field(default=None, ge=CAPS['DENOISE_MEDIAN_KSIZE_MIN'], le=CAPS['DENOISE_MEDIAN_KSIZE_MAX'])
    denoise_target_band_low: Optional[float] = Field(default=None, ge=CAPS['DENOISE_TARGET_BAND_LOW_MIN'], le=CAPS['DENOISE_TARGET_BAND_LOW_MAX'])
    denoise_target_band_high: Optional[float] = Field(default=None, ge=CAPS['DENOISE_TARGET_BAND_HIGH_MIN'], le=CAPS['DENOISE_TARGET_BAND_HIGH_MAX'])
    trailing_silence_db: Optional[float] = Field(default=None, ge=CAPS['TRAILING_SILENCE_DB_MIN'], le=CAPS['TRAILING_SILENCE_DB_MAX'])
    gain_target_max: Optional[float] = Field(default=None, ge=CAPS['GAIN_TARGET_MAX_MIN'], le=CAPS['GAIN_TARGET_MAX_MAX'])
    max_gain: Optional[float] = Field(default=None, ge=CAPS['MAX_GAIN_MIN'], le=CAPS['MAX_GAIN_MAX'])
    n_mels: Optional[int] = Field(default=None, ge=CAPS['N_MELS_MIN'], le=CAPS['N_MELS_MAX'])
    ebu_post_gain_db: Optional[float] = Field(default=None, ge=CAPS['EBU_POST_GAIN_DB_MIN'], le=CAPS['EBU_POST_GAIN_DB_MAX'])
    ebu_true_peak: Optional[float] = Field(default=None, ge=CAPS['EBU_TRUE_PEAK_MIN'], le=CAPS['EBU_TRUE_PEAK_MAX'])

    # Fuzzy overrides (optional; inherit from globals.fuzzy)
    enable_fuzzy_cache: Optional[bool] = Field(default=True)
    fuzzy_threshold: Optional[float] = Field(default=None, ge=CAPS['FUZZY_THRESHOLD_MIN'], le=CAPS['FUZZY_THRESHOLD_MAX'])
    fuzzy_boost_amount: Optional[float] = Field(default=None, ge=CAPS['FUZZY_BOOST_AMOUNT_MIN'], le=CAPS['FUZZY_BOOST_AMOUNT_MAX'])
    fuzzy_index_size: Optional[int] = Field(default=None, ge=CAPS['FUZZY_CACHE_LIMIT_MIN'], le=CAPS['FUZZY_CACHE_LIMIT_MAX'])
    fuzzy_artifact_threshold_hz: Optional[float] = Field(default=None, ge=CAPS['FUZZY_ARTIFACT_THRESHOLD_HZ_MIN'], le=CAPS['FUZZY_ARTIFACT_THRESHOLD_HZ_MAX'])

    # Validators (fallback; only validate if not None, since Optional)
    @model_validator(mode='after')
    def validate_overrides(self):
        # Log overrides: Iterate set, get values via getattr
        for field in self.model_fields_set:
            value = getattr(self, field)
            if value is not None:
                logger.debug(f"Voice override set: {field}={value}")
        # Custom checks (e.g., ensure notch_low < high if both set)
        if self.notch_low_hz is not None and self.notch_high_hz is not None and self.notch_low_hz >= self.notch_high_hz:
            logger.warning("notch_low_hz >= notch_high_hz in voice overrides; adjusting notch_high_hz")
            self.notch_high_hz = self.notch_low_hz + 1000  # Minimal adjustment
        return self

    def get_field_bounds(self, field_name: str) -> Optional[Dict[str, Any]]:
        """UI helper: Return min/max for field from CAPS (same as globals)."""
        return _get_field_bounds(field_name)

# AppConfig: Top-level with globals + voices
class AppConfig(BaseModel):
    """
    Full app config: globals (nested categories) + voices (dict of flat overrides).
    """
    model_config = ConfigDict(extra='ignore')

    globals: Globals = Field(default_factory=Globals)
    voices: Dict[str, 'VoiceConfig'] = Field(default_factory=dict)  # Forward ref for nesting

    def get_field_bounds(self, field_name: str) -> Optional[Dict[str, Any]]:
        """UI helper: Return min/max for field from CAPS (delegates to globals or sub-models)."""
        # Handle top-level (rare; e.g., 'globals.device' → pass to globals)
        if field_name.startswith('globals.'):
            sub_field = field_name.split('.', 1)[1]
            return self.globals.get_field_bounds(sub_field)
        # Direct globals fields
        return self.globals.get_field_bounds(field_name)