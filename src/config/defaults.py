"""
Defaults: Contains DEFAULTS, CAPS, and globals. Provides _sync_globals() for syncing globals to Config attrs.
"""
from loguru import logger
from typing import Dict, Any, List

from .validation import clamp_numeric  # Relative import (breaks absolute path issue)

# Core globals (preserved for backward compat; synced from Config in config.py)
DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"
DTYPE = None  # Set after import
MODEL = None
MULTILINGUAL = False
ENABLE_DISK_CACHE = True
ENABLE_MEMORY_CACHE = True
FUZZY_CACHE_LIMIT = 1000
_CONFIG_CACHE = None
_CONFIG_FILE = "skyrimnet_config.txt"
_USE_API_MODE = False

# DEFAULTS: Adjustable defaults (exact copy from original; used as base for Config)
DEFAULTS = {
    # Generation Tokens (adjustable)
    'max_new_tokens': 1499,
    'min_new_tokens': 224,
    'max_cache_len': 4096,
    'stride_length': 8,

    # TTS Params (adjustable)
    'temperature': 0.7,
    'min_p': 0.07,
    'top_p': 1.0,
    'repetition_penalty': 2.0,
    'cfg_weight': 0.45,  # From logs example
    'exaggeration': 0.7,

    # Audio No-Ops: Skip/Identity (most voices pass-through; override per-voice)
    'speaking_rate': 1.0,  # Identity
    'eq_gain_db': 0.0,  # Skip EQ
    'eq_cutoff_hz': 3000,  # Default (skipped)
    'notch_gain_db': 0.0,  # Skip notch (None in apply_notch)
    'notch_low_hz': 8000,
    'notch_high_hz': 11000,
    'gain_target_max': 1.0,  # Skip normalize target
    'gain_max_limit': 1.0,  # Identity clamp
    'trim_threshold_db': -100,  # Skip trim (None in trim_silence)
    'fade_ms': 0,  # Skip fade
    'enable_denoise_normalize': False,  # Skip normalize
    'noise_floor_db': -60.0,  # Default (skipped)
    'normalize_method': 'peak',  # Default (skipped)
    'n_fft': 2048,
    'hop_length': 256,

    'max_gain': 2.0,
    'target_max': 0.6,
    'n_mels': 80,
    'ebu_post_gain_db': 0,
    'ebu_true_peak': 0,

    'base_audio_pad_sec': 0.15,                    # Base pad per side (s)
    'tiny_audio_pad_multiplier': 2.0,              # Extra x for tiny audio (< tiny_threshold_sec)
    'tiny_threshold_sec': 0.5,                     # Detect short audio (post-trim dur < this)

    # Boolean Flags (adjustable)
    'enable_pre_adjustment': True,
    'enable_post_processing': True,
    'enable_denoising': False,
    'enable_smoothing': False,  # Assuming from original context
    'enable_quantization': True,
    'enable_resample': False,
    'enable_spectral_gating': True,
    'notch_enabled': False,
    'hp_enabled': False,
    'enable_disk_cache': True,
    'enable_memory_cache': True,
    'force_local_refs': True,
    'auto_update_refs': True,
    'timings_enabled': False,
    'auto_reset_timings': False,
    'enable_post_resample': False,
    'enable_post_jit_gain': True,
    'enable_post_voice_processing': True,
    'enable_deferred_cleanup': True,
    'enable_audio_padding': False,

    'denoise_highpass_hz': 80,        # High-pass before denoise (cut rumble/breaths)
    'denoise_median_ksize': 3,         # Median kernel (smooths bursts; odd size)
    'denoise_target_band_low': 5000,   # Gate high-freq (chirps >5kHz)
    'denoise_target_band_high': 12000, # Gate up to 12kHz (chirp range)

    # Cache/Other
    'max_memory_entries': 100,  # Cache size (voices/conds in RAM; up from 50)
    'save_queue_max': 20,       # Disk save queue limit (prevents backlog)
    'fuzzy_index_size': 1000,   # Max fuzzy entries per stem (DB size)
    'fuzzy_cache_limit': 1000,
    'fuzzy_boost_amount': 0.15, # boost given to short word matches to increase cache hits
    'fuzzy_threshold': 0.70,    # level of string match required to use audio cache
    'memory_cache_enable': True, # Toggle memory cache (instead of env)
    'disk_cache_enable': True,  # Toggle disk cache
    'fuzzy_enable': True,       # Toggle fuzzy audio cache
    'compress_pt_saves': True,  # Toggle gzip compression on .pt files
    'compress_level': 6,        # Gzip compression level (1=fast, 9=max small)
    'n_fft_denoise': 1024,               # Faster STFT (half 2048)
    'fade_ms_trail': 50,                 # Longer for breath trails (use if fade_ms=None)
    'trailing_silence_db': -45.0,        # Cut post-rate trails below this (new step)
    'fuzzy_artifact_threshold_hz': 7000.0,  # For is_artifact_laden

    # Strings (adjustable defaults)
    'logging_level': 'INFO',
    'tts_name': 'Chatterbox',
    'dtype': 'bfloat16',

    # Lists (adjustable)
    'vocalise_patterns': [],  # Empty list by default
    # Fuzzy boost words example (parsed as list in validation)
    'fuzzy_boost_words': 'ahh,mmm,ooh,gasp',  # Default as string; parsed to list
}

def _sync_globals(config_instance):
    """Sync CONFIG storage → globals (direct access, no properties)."""
    global DEVICE, DTYPE, MODEL, MULTILINGUAL, ENABLE_DISK_CACHE, ENABLE_MEMORY_CACHE, FUZZY_CACHE_LIMIT, _USE_API_MODE
    # Cores (direct attrs)
    DEVICE = str(config_instance.device)
    DTYPE = config_instance.dtype
    MODEL = config_instance.model
    MULTILINGUAL = config_instance.multilingual
    _USE_API_MODE = config_instance._use_api_mode
    # Flags/defaults (from storage, not properties)
    ENABLE_DISK_CACHE = config_instance._flags.get('enable_disk_cache', DEFAULTS.get('enable_disk_cache', True))
    ENABLE_MEMORY_CACHE = config_instance._flags.get('enable_memory_cache', DEFAULTS.get('enable_memory_cache', True))
    FUZZY_CACHE_LIMIT = clamp_numeric('fuzzy_cache_limit', config_instance._defaults.get('fuzzy_cache_limit', DEFAULTS.get('fuzzy_cache_limit', 100)))
    # Add other globals as needed (direct from _defaults/_flags)
    logger.trace("Globals synced from CONFIG storage")