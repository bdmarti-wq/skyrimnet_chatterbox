# skyrimnet_config.py - Refactored for modularity: load_config broken into sharable helpers (_load_txt_config, _load_voices_json, _merge_defaults).
# Removed voice_overrides from txt parsing (separate voices.json). Enhanced for post-processing (DEFAULTS/CAPS/audio_defaults/get_merged_audio_params).
# Backward compat preserved (facades route to class). Patched for audio hierarchy (defaults < global < voice.json < overrides; no-op safe).

import threading
from pathlib import Path
import json
from loguru import logger
import os
import torch
from typing import Dict, Any, Tuple, List, Union, Optional

# Core globals (preserved for backward compat; set from class in load_skyrimnet_config)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MODEL = None
MULTILINGUAL = False
ENABLE_DISK_CACHE = True
ENABLE_MEMORY_CACHE = True
FUZZY_CACHE_LIMIT = 1000
_CONFIG_CACHE = None
_CONFIG_FILE = "skyrimnet_config.txt"
_USE_API_MODE = False  # Testing flag

# SkyrimNetConfig Class (singleton; modular load/save for sharability)
class SkyrimNetConfig:
    # DEFAULTS: Adjustable defaults (numerics/strings/booleans from config.txt; override via parser)
    DEFAULTS = {
        # Generation Tokens (adjustable)
        'max_new_tokens': 1499,
        'min_new_tokens': 224,
        'max_cache_len': 4096,
        'stride_length': 8,

        # TTS Params (adjustable)
        'temperature': 0.8,
        'min_p': 0.07,
        'top_p': 1.0,
        'repetition_penalty': 2.0,
        'cfg_weight': 0.0,
        'exaggeration': 0.7,

        # Audio No-Ops: Skip/Identity (most voices pass-through; override per-voice)
        'speaking_rate': 1.0,  # Identity
        'eq_gain_db': 0.0,  # Skip EQ
        'eq_cutoff_hz': 3000,  # Default (skipped)
        'notch_gain_db': None,  # Skip notch (None in apply_notch)
        'notch_low_hz': 8000,
        'notch_high_hz': 11000,
        'gain_target_max': None,  # Skip normalize target
        'gain_max_limit': 1.0,  # Identity clamp
        'trim_threshold_db': None,  # Skip trim (None in trim_silence)
        'fade_ms': None,  # Skip fade
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
        'tiny_audio_pad_multiplier': 2.0,              # Extra x for tiny (< tiny_threshold_sec)
        'tiny_threshold_sec': 0.5,                     # Detect short audio (post-trim dur <

        # Strings (adjustable defaults; new from alt)
        'logging_level': 'INFO',
        'tts_name': 'Chatterbox',
        'dtype': 'bfloat16',

        # Booleans (adjustable flags; new from alt)
        'enable_pre_adjustment': True,  # New: Opt-in pad/mel align (alt; default False for non-breaking)
        'enable_post_processing': True,  # New (gate for post; set False for slowly)
        'enable_denoising': False,  # New
        'enable_smoothing': False,  # New
        'enable_quantization': True,  # New
        'enable_resample': False,
        'enable_spectral_gating': True,  # New
        'notch_enabled': False,  # New
        'hp_enabled': False,  # New
        'enable_memory_cache': True,
        'enable_disk_cache': True,
        'force_local_refs': True,  # New: Alt flag
        'auto_update_refs': True,  # New: Alt flag
        'timings_enabled': False,  # New
        'auto_reset_timings': False,  # New
        'enable_post_resample': False,  # New: Alt (review)
        'enable_post_jit_gain': True,  # New: Alt (review)
        'enable_post_voice_processing': True,  # New: Alt (review)
        'enable_deferred_cleanup': True,  # New
        'enable_audio_padding': False,  # Gate silence front/back

        'denoise_highpass_hz': 80,        # High-pass before denoise (cut rumble/breaths)
        'denoise_median_ksize': 3,         # Median kernel (smooths bursts; odd size)
        'denoise_target_band_low': 5000,   # Gate high-freq (chirps >5kHz)
        'denoise_target_band_high': 12000, # Gate up to 12kHz (chirp range)

        'max_memory_entries': 100,  # Cache size (voices/conds in RAM; up from 50)
        'save_queue_max': 20,       # Disk save queue limit (prevents backlog)
        'fuzzy_index_size': 1000,   # Max fuzzy entries per stem (DB size)
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
    }

    # CAPS: Hard constants/immutable limits for clamping (numerics only; from alt; used in get_value)
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

        # Audio Caps (existing + new guards)
        'SPEAKING_RATE_MIN': 0.5, 'SPEAKING_RATE_MAX': 2.0,
        'EQ_GAIN_DB_MIN': -12, 'EQ_GAIN_DB_MAX': 6,
        'EQ_CUTOFF_HZ_MIN': 2000, 'EQ_CUTOFF_HZ_MAX': 5000,
        'NOTCH_GAIN_DB_MIN': -24, 'NOTCH_GAIN_DB_MAX': 0,
        'NOTCH_LOW_HZ_MIN': 4000, 'NOTCH_LOW_HZ_MAX': 10000,
        'NOTCH_HIGH_HZ_MIN': 8000, 'NOTCH_HIGH_HZ_MAX': 16000,
        'GAIN_TARGET_MAX_MIN': 0.1, 'GAIN_TARGET_MAX_MAX': 1.0,
        'GAIN_MAX_LIMIT_MIN': 0.5, 'GAIN_MAX_LIMIT_MAX': 3.0,
        'TRIM_THRESHOLD_DB_MIN': -60, 'TRIM_THRESHOLD_DB_MAX': -20,  # Wider for mild/no-op (-30 center)
        'FADE_MS_MIN': 0, 'FADE_MS_MAX': 100,  # 0=no-op
        'NOISE_FLOOR_DB_MIN': -80, 'NOISE_FLOOR_DB_MAX': -20,  # -60 center (mild)
        'MIN_POST_DURATION_SEC': 0.01, 'MAX_POST_DURATION_SEC': 10.0,  # Guards (0.05 default)
        'TRIM_FRAME_LENGTH_FACTOR_MIN': 2, 'TRIM_FRAME_LENGTH_FACTOR_MAX': 8,  # 4 default
        'MAX_N_FFT_FOR_TRIM_MIN': 512, 'MAX_N_FFT_FOR_TRIM_MAX': 4096,  # Cap for shorts
        'MIN_SAMPLES_FOR_DENOISE_MIN': 50, 'MIN_SAMPLES_FOR_DENOISE_MAX': 1000,  # 100 default

        'N_FFT_DENOISE_MIN': 512, 'N_FFT_DENOISE_MAX': 2048,
        'FADE_MS_TRAIL_MIN': 20, 'FADE_MS_TRAIL_MAX': 200,
        'TRAILING_SILENCE_DB_MIN': -60, 'TRAILING_SILENCE_DB_MAX': -30,
        'FUZZY_ARTIFACT_THRESHOLD_HZ_MIN': 5000, 'FUZZY_ARTIFACT_THRESHOLD_HZ_MAX': 10000,
        'DENOISE_HIGHPASS_HZ_MIN': 50, 'DENOISE_HIGHPASS_HZ_MAX': 200,
        'DENOISE_MEDIAN_KSIZE_MIN': 1, 'DENOISE_MEDIAN_KSIZE_MAX': 5,
        'DENOISE_TARGET_BAND_LOW_MIN': 3000, 'DENOISE_TARGET_BAND_LOW_MAX': 8000,
        'DENOISE_TARGET_BAND_HIGH_MIN': 8000, 'DENOISE_TARGET_BAND_HIGH_MAX': 15000,

        # Cache Caps (new from alt)
        'COND_CACHE_MAX_ENTRIES_MIN': 10, 'COND_CACHE_MAX_ENTRIES_MAX': 100,
        'FUZZY_CACHE_LIMIT_MIN': 100, 'FUZZY_CACHE_LIMIT_MAX': 5000,
        'FUZZY_THRESHOLD_MIN': 0.50, 'FUZZY_THRESHOLD_MAX': 0.95,
        'AUDIO_PAD_SEC_MIN': 0.0, 'AUDIO_PAD_SEC_MAX': 0.5,
        'TINY_PAD_MULTIPLIER_MIN': 1.0, 'TINY_PAD_MULTIPLIER_MAX': 3.0,
        'TINY_THRESHOLD_SEC_MIN': 0.1, 'TINY_THRESHOLD_SEC_MAX': 1.0,
        'TEXT_ELLIPSES_COUNT_MIN': 0, 'TEXT_ELLIPSES_COUNT_MAX': 5,
        'SHORT_WORD_LEN_MIN': 1, 'SHORT_WORD_LEN_MAX': 5,
    }

    _instance = None
    _lock = threading.Lock()
    _config_file_path = _CONFIG_FILE  # "skyrimnet_config.txt"
    _voices_file_path = Path(__file__).parent.parent / "voices.json"  # Root dir

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        """Early init (before load; sets core globals)."""
        # Core attrs (preserved globals)
        global DEVICE, DTYPE, MODEL, MULTILINGUAL, ENABLE_DISK_CACHE, ENABLE_MEMORY_CACHE, FUZZY_CACHE_LIMIT
        self.device = DEVICE
        self.dtype = DTYPE
        self.model = MODEL
        self.multilingual = MULTILINGUAL
        self.enable_disk_cache = ENABLE_DISK_CACHE
        self.enable_memory_cache = ENABLE_MEMORY_CACHE
        self.fuzzy_cache_limit = FUZZY_CACHE_LIMIT

        # Private storage (from DEFAULTS; loaded full in load_config)
        self._defaults = self.DEFAULTS.copy()
        self._flags = {k: v for k, v in self.DEFAULTS.items() if isinstance(v, bool)}
        self._strings = {k: v for k, v in self.DEFAULTS.items() if isinstance(v, str) and k in ['normalize_method', 'logging_level', 'tts_name', 'dtype']}
        self.voice_overrides = {}  # Dict[str, Dict] for voices (from voices.json)
        self._is_modified = False
        self.sr = 24000  # Core

        logger.debug("SkyrimNetConfig initialized (core globals set)")

        # Load full config
        self.load_config()

    # Sharable Helper: Load globals/flags from config.txt (INI-style; no voice block)
    def _load_txt_config(self) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, bool]]:
        """Sharable: Parse config.txt for globals/flags/modes. Returns (default_config, config_mode, global_flags)."""
        default_config = {
            'temperature': 0.8,
            'min_p': 0.07,
            'top_p': 1.0,
            'repetition_penalty': 2.0,
            'cfg_weight': 0.0,
            'exaggeration': 0.7
        }
        config_mode = {k: 'default' for k in default_config}
        global_flags = {
            'enable_memory_cache': self.DEFAULTS.get('enable_memory_cache', True),
            'enable_disk_cache': self.DEFAULTS.get('enable_disk_cache', True)
        }

        config_path = Path(self._config_file_path)
        if not config_path.exists():
            logger.warning(f"Config file {self._config_file_path} not found, using hardcoded defaults")
            return default_config, config_mode, global_flags

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Parser: Globals/flags only (ignore old voice_overrides block)
            in_voice_block = False
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('voice_overrides = {'):  # Deprecate: Ignore
                    in_voice_block = True
                    logger.warning("Ignoring old voice_overrides block in config.txt – use voices.json")
                    continue
                if in_voice_block and line == '}':
                    in_voice_block = False
                    continue
                if in_voice_block:  # Skip block lines
                    continue

                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    # Global boolean flags (preserve old logic)
                    if key in global_flags:
                        if value.lower() in ['true', 'yes', '1', 'on']:
                            global_flags[key] = True
                            if key == 'enable_memory_cache':
                                ENABLE_MEMORY_CACHE = True
                            elif key == 'enable_disk_cache':
                                ENABLE_DISK_CACHE = True
                            logger.info(f"Setting {key} to True")
                        elif value.lower() in ['false', 'no', '0', 'off']:
                            global_flags[key] = False
                            if key == 'enable_memory_cache':
                                ENABLE_MEMORY_CACHE = False
                            elif key == 'enable_disk_cache':
                                ENABLE_DISK_CACHE = False
                            logger.info(f"Setting {key} to False")
                        else:
                            logger.warning(f"Invalid boolean value '{value}' for {key}, using default")

                    # Parameter modes/values (preserve old; merge to default_config)
                    elif key in config_mode:
                        if value.lower() == 'default':
                            config_mode[key] = 'default'
                        elif value.lower() == 'api':
                            config_mode[key] = 'api'
                        else:
                            try:
                                custom_value = float(value)
                                config_mode[key] = 'custom'
                                default_config[key] = custom_value
                                logger.info(f"Using custom {key} value: {custom_value}")
                            except ValueError:
                                logger.warning(f"Invalid value '{value}' for {key}, using default")

            logger.debug(f"TXT config loaded: {len(default_config)} params, {len(global_flags)} flags")
            return default_config, config_mode, global_flags

        except Exception as e:
            logger.error(f"TXT config load failed: {e}, using hardcoded defaults")
            return default_config, config_mode, global_flags

    # Sharable Helper: Load voice overrides from voices.json
    # Enhanced: Multi-path load for voices.json (root/config/; log found/missing)
    def _load_voices_json(self) -> Dict[str, Dict]:
        """Load voices.json from root or config/ dir. Returns {voice: params} or {}. FIXED: Multi-path, debug log."""
        possible_paths = [
            Path(__file__).parent.parent / "voices.json",  # Root (skyrimnet_chatterbox/voices.json)
            Path(__file__).parent / "voices.json",         # config/voices.json (if src/config.py)
            Path("voices.json"),                           # Current dir fallback
            Path(__file__).parent.parent / "config" / "voices.json"  # config/ subdir
        ]
        self.voice_overrides = {}
        loaded_path = None
        for voices_path in possible_paths:
            if voices_path.exists():
                try:
                    with open(voices_path, 'r') as f:
                        voices_data = json.load(f)
                    # Validate/Flatten if list (optional)
                    if isinstance(voices_data, list):
                        voices_data = {item.get('voice', f'voice_{i}'): item for i, item in enumerate(voices_data)}
                    self.voice_overrides = voices_data
                    loaded_path = voices_path
                    logger.info(f"✓ Loaded voices.json: {loaded_path} ({len(self.voice_overrides)} voices: {list(self.voice_overrides.keys())})")
                    return voices_data
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid voices.json {voices_path}: {e} – skipping")
                except Exception as e:
                    logger.error(f"Load voices.json {voices_path} failed: {e} – trying next")
            else:
                logger.trace(f"Tried voices.json path (not found): {voices_path}")

        logger.warning("voices.json not found in expected paths – no per-voice overrides (add to root for [dlc1seranavoice/femaleuniquelydia])")
        self.voice_overrides = {'global': {}}  # Fallback
        return {}


    def load_config(self):
        """Load: Txt (globals/flags) → merge defaults → voices.json (overrides). FIXED: Always call _load_voices_json; full debug logs (multi-path)."""
        global _CONFIG_CACHE, ENABLE_MEMORY_CACHE, ENABLE_DISK_CACHE

        if _CONFIG_CACHE is not None:  # Cache hit (but re-load voices always for updates)
            defaults, modes, global_flags = _CONFIG_CACHE
            # Re-merge old cache with fresh voices (ensure always current)
            self._defaults.update(defaults)  # Preserve existing
            self.enable_memory_cache = global_flags.get('enable_memory_cache', ENABLE_MEMORY_CACHE)
            self.enable_disk_cache = global_flags.get('enable_disk_cache', ENABLE_DISK_CACHE)
            ENABLE_MEMORY_CACHE = self.enable_memory_cache
            ENABLE_DISK_CACHE = self.enable_disk_cache
            # Always refresh voices (no cache skip)
            self._load_voices_json()
            logger.debug("Config from cache (txt + fresh voices.json)")
            return _CONFIG_CACHE

        # Step 1: Load txt (globals/flags/modes; existing logic)
        default_config, config_mode, global_flags = self._load_txt_config()

        # Step 2: Merge defaults (input → _defaults; sharable)
        self._defaults.update({k: v for k, v in default_config.items() if k in self.DEFAULTS})
        for k in self.DEFAULTS:
            if k not in self._defaults:
                self._defaults[k] = self.DEFAULTS[k]

        # Step 3: Load voices.json (enhanced: Multi-path, always fresh, log raw)
        self.voice_overrides = self._load_voices_json()

        # Step 4: Sync flags/enables from globals (after txt)
        self.enable_memory_cache = global_flags.get('enable_memory_cache', self.DEFAULTS['enable_memory_cache'])
        self.enable_disk_cache = global_flags.get('enable_disk_cache', self.DEFAULTS['enable_disk_cache'])
        ENABLE_MEMORY_CACHE = self.enable_memory_cache
        ENABLE_DISK_CACHE = self.enable_disk_cache

        # Set self attrs from merged (easy access)
        for attr, val in self._defaults.items():
            if not hasattr(self, attr) or attr in ['device', 'dtype', 'model']:  # Avoid overwrite core
                setattr(self, attr, val)
        for attr, val in self._flags.items():
            setattr(self, attr, val)

        # Core sync (clamped where needed)
        self.device = torch.device(DEVICE if self.device == "cuda" else "cpu")
        dtype_str = self._defaults.get('dtype', 'bfloat16')
        self.dtype = torch.bfloat16 if dtype_str == 'bfloat16' else torch.float32
        DTYPE = self.dtype

        # Log summary (with voices count/debug)
        voices_count = len(self.voice_overrides)
        voices_keys = list(self.voice_overrides.keys()) if voices_count > 0 else []
        logger.info(
            f"Config loaded: {voices_count} voices (['global' + {len(voices_keys) - 1}] if global fallback; keys: {voices_keys}), {len(self._defaults)} params (logging_level={self.logging_level})")
        if voices_count > 1:  # Success log
            sample_voice = voices_keys[0] if voices_keys else 'none'
            logger.info(
                f"Voices loaded: {sample_voice} (trim={self.voice_overrides.get(sample_voice, {}).get('trim_threshold_db', 'N/A')}, eq={self.voice_overrides.get(sample_voice, {}).get('eq_gain_db', 'N/A')}, notch={self.voice_overrides.get(sample_voice, {}).get('notch_gain_db', 'N/A')}")

        # Cache result (but voices refreshed on call if needed)
        _CONFIG_CACHE = (default_config, config_mode, global_flags)
        return _CONFIG_CACHE


    def _merge_defaults(self, input_defaults: Dict, modes: Dict, flags: Dict):
        """Internal: Merge input to class storage (non-breaking). Sharable for custom merges."""
        # Merge defaults (input → _defaults; class DEFAULTS base)
        self._defaults.update({k: v for k, v in input_defaults.items() if k in self.DEFAULTS})
        for k in self.DEFAULTS:
            if k not in self._defaults:
                self._defaults[k] = self.DEFAULTS[k]
        # Merge flags (input → _flags)
        self._flags.update({k: v for k, v in flags.items() if isinstance(v, bool)})
        for k, v in self.DEFAULTS.items():
            if isinstance(v, bool) and k not in self._flags:
                self._flags[k] = v
        # Set self attrs from merged (easy access; e.g., self.temperature)
        for attr, val in self._defaults.items():
            if not hasattr(self, attr) or attr in ['device', 'dtype', 'model']:  # Avoid overwrite core
                setattr(self, attr, val)
        for attr, val in self._flags.items():
            setattr(self, attr, val)
        # Core (clamped)
        self.device = torch.device(DEVICE if self.device == "cuda" else "cpu")
        dtype_str = self._defaults.get('dtype', 'bfloat16')
        self.dtype = torch.bfloat16 if dtype_str == 'bfloat16' else torch.float32
        DTYPE = self.dtype

    # Sharable Helper: Save globals/flags to config.txt (INI-style)
    def _save_txt_config(self, config_path: Path):
        """Sharable: Save globals/flags to txt (no voices block)."""
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write("# Global Settings\n")
            f.write(f"enable_pre_adjustment = {self.enable_pre_adjustment}\n")
            f.write(f"speaking_rate = {self.speaking_rate}\n")
            f.write(f"eq_gain_db = {self.eq_gain_db}\n")
            f.write(f"eq_cutoff_hz = {self.eq_cutoff_hz}\n")
            f.write(f"max_gain = {self.max_gain}\n")
            f.write(f"target_max = {self.target_max}\n")
            f.write(f"noise_floor_db = {self.noise_floor_db}\n")
            f.write(f"trim_threshold_db = {self.trim_threshold_db}\n")
            f.write(f"notch_enabled = {self.notch_enabled}\n")
            f.write(f"hp_enabled = {self.hp_enabled}\n")
            f.write(f"temperature = {self.temperature}\n")
            f.write(f"exaggeration = {self.exaggeration}\n")
            f.write(f"cfg_weight = {self.cfg_weight}\n")
            f.write(f"min_p = {self.min_p}\n")
            f.write(f"top_p = {self.top_p}\n")
            f.write(f"repetition_penalty = {self.repetition_penalty}\n")
            f.write(f"max_new_tokens = {self.max_new_tokens}\n")
            f.write(f"min_new_tokens = {self.min_new_tokens}\n")
            f.write(f"max_cache_len = {self.max_cache_len}\n")
            f.write(f"n_fft = {self.n_fft}\n")
            f.write(f"hop_length = {self.hop_length}\n")
            f.write(f"stride_length = {self.stride_length}\n")
            f.write(f"fade_ms = {self.fade_ms}\n")
            f.write(f"enable_memory_cache = {self.enable_memory_cache}\n")
            f.write(f"enable_disk_cache = {self.enable_disk_cache}\n")
            f.write(f"force_local_refs = {self.force_local_refs}\n")
            f.write(f"auto_update_refs = {self.auto_update_refs}\n")
            f.write(f"tts_name = {self.tts_name}\n")
            f.write(f"dtype = {self._defaults.get('dtype', 'bfloat16')}\n")  # Use _defaults
            # Flags section
            f.write("\n# Flags\n")
            for flag, val in self._flags.items():
                if flag not in ['enable_pre_adjustment', 'enable_memory_cache', 'enable_disk_cache',
                                'force_local_refs', 'auto_update_refs']:
                    f.write(f"{flag} = {val}\n")

        logger.debug(f"TXT config saved: {config_path}")

    # Sharable Helper: Save voice overrides to voices.json
    def _save_voices_json(self, voices_file: Path):
        """Sharable: Save voice_overrides to JSON (indent=2)."""
        with open(voices_file, 'w', encoding='utf-8') as f:
            json.dump(self.voice_overrides, f, indent=2)
        logger.debug(f"Voices.json saved: {voices_file} ({len(self.voice_overrides)} voices)")

    def save_config(self, create_backup: bool = True) -> Tuple[bool, str]:
        """Save: Txt (globals/flags via _save_txt_config) + voices.json (_save_voices_json). Modular."""
        try:
            config_path = Path(self._config_file_path)
            if create_backup and config_path.exists():
                backup_path = config_path.with_suffix('.backup')
                config_path.rename(backup_path)
                logger.debug(f"Backup created: {backup_path}")

            # Step 1: Save txt (globals/flags; sharable)
            self._save_txt_config(config_path)

            # Step 2: Save voices.json (separate; sharable)
            voices_file = self._voices_file_path
            self._save_voices_json(voices_file)

            logger.info(f"Config saved: {config_path} + voices.json ({len(self.voice_overrides)} voices)")
            self._is_modified = False
            return True, f"Saved | {len(self.voice_overrides)} voices"

        except Exception as e:
            logger.error(f"Save failed: {e}")
            return False, str(e)

    def reload_config(self):
        """Reload: Clears cache; re-calls load_config (txt + voices.json)."""
        global _CONFIG_CACHE
        _CONFIG_CACHE = None
        self.voice_overrides = {}  # Clear prior
        self.load_config()  # Re-parses txt → voices.json
        logger.info("Config reloaded (txt + voices.json)")

    def get_value(self, param_name: str, api_value: Any = None, default: Any = None,
                  bypass_config: bool = False) -> Any:
        """Get value with clamping (numerics only; from alt). Backward compat with old get_config_value.
        Handles list parsing for fuzzy_boost_words (str → list from comma-sep)."""
        if bypass_config or _USE_API_MODE:
            # API mode: Use api_value or fallback to DEFAULTS (preserve old fallback_defaults)
            fallback_defaults = {
                'temperature': 0.9, 'min_p': 0.05, 'top_p': 1.0, 'repetition_penalty': 2.0,
                'cfg_weight': 0.0, 'exaggeration': 0.55, 'hop_length': 256, 'n_fft': 1024,  # New
                'fuzzy_threshold': 0.75, 'fuzzy_min_length': 3, 'fuzzy_boost_amount': 0.1,
                'fuzzy_boost_words': ['ahh', 'mmm', 'ooh', 'gasp']  # Add default list
            }
            val = api_value if api_value is not None else fallback_defaults.get(param_name, default or 0.0)
            # Parse lists in API mode too (if str provided)
            if param_name == 'fuzzy_boost_words' and isinstance(val, str):
                val = [w.strip().lower() for w in val.split(',') if w.strip()]
            return val

        # Prioritize self attrs (dynamic)
        if hasattr(self, param_name):
            val = getattr(self, param_name)
            if val is not None:
                logger.debug(f"Dynamic {param_name} from self = {val}")
                if isinstance(val, (int, float)):
                    return self.clamp_value(param_name, val)
                # Early parse for lists (if str attr)
                if param_name == 'fuzzy_boost_words' and isinstance(val, str):
                    val = [w.strip().lower() for w in val.split(',') if w.strip()]
                    setattr(self, param_name, val)  # Cache parsed
                return val  # Non-numeric: No clamp

        # Flags (bools first)
        if param_name in self._flags:
            val = self._flags[param_name]
            logger.debug(f"Flag {param_name} = {val}")
            return bool(val)

        # Strings (e.g., logging_level)
        if hasattr(self, '_strings') and param_name in self._strings:
            val = self._strings[param_name]
            logger.debug(f"String {param_name} = {val}")
            return val

        # Defaults (direct from _defaults; no modes here for simplicity)
        val = self._defaults.get(param_name, self.DEFAULTS.get(param_name, default or 0.0))

        # API override (if not bypass)
        if api_value is not None:
            val = api_value
            logger.debug(f"API override for {param_name} = {val}")

        # Parse bools from str (if mis-typed)
        bool_keys = [k for k, v in self.DEFAULTS.items() if isinstance(v, bool)]
        if param_name in bool_keys and isinstance(val, str):
            val = val.lower() in ['true', 'yes', '1', 'on']
            logger.debug(f"Parsed bool {param_name} = {val}")
            return bool(val)

        # Parse lists (e.g., fuzzy_boost_words: str → list)
        if param_name == 'fuzzy_boost_words' and isinstance(val, str):
            val = [w.strip().lower() for w in val.split(',') if w.strip()]
            logger.debug(f"Parsed {param_name}: {val}")
        # Defaults to list if requested (cached on self if needed)
        elif param_name == 'fuzzy_boost_words' and val is None:
            val = self.DEFAULTS.get(param_name, ['ahh', 'mmm', 'ooh', 'gasp'])
            logger.debug(f"Default {param_name}: {val}")

        # Clamp numerics (only if numeric)
        if isinstance(val, (int, float)):
            return self.clamp_value(param_name, val)
        return val

    def clamp_value(self, param_name: str, value: Any) -> Any:
        """Clamp numerics using CAPS (non-breaking; skip if no CAP). Sharable."""
        if isinstance(value, (int, float)):
            upper_name = param_name.upper()
            min_key = f"{upper_name}_MIN"
            max_key = f"{upper_name}_MAX"
            if min_key in self.CAPS and max_key in self.CAPS:
                min_val, max_val = self.CAPS[min_key], self.CAPS[max_key]
                clamped = max(min_val, min(max_val, value))
                if clamped != value:
                    logger.debug(f"Clamped {param_name}: {value} → {clamped} (caps: {min_val}-{max_val})")
                return clamped
        return value  # No clamp for non-numerics/missing CAPS


    @property
    def audio_defaults(self) -> Dict[str, Any]:
        """Audio params subset (merged; from alt; sharable property). FIXED: Added padding gates/tunables."""
        if not hasattr(self, '_audio_defaults'):
            self._audio_defaults = {
                # Existing core
                'enable_post_processing': self.get_value('enable_post_processing', default=False),
                'enable_post_resample': self.get_value('enable_post_resample', default=True),
                'enable_post_jit_gain': self.get_value('enable_post_jit_gain', default=True),
                'enable_post_voice_processing': self.get_value('enable_post_voice_processing', default=True),
                'enable_pre_adjustment': self.get_value('enable_pre_adjustment', default=False),
                'speaking_rate': self.get_value('speaking_rate'),
                'eq_gain_db': self.get_value('eq_gain_db'),
                'eq_cutoff_hz': self.get_value('eq_cutoff_hz'),
                'gain_max_limit': self.get_value('gain_max_limit'),
                'gain_target_max': self.get_value('gain_target_max'),
                'noise_floor_db': self.get_value('noise_floor_db'),
                'trim_threshold_db': self.get_value('trim_threshold_db'),
                'notch_gain_db': self.get_value('notch_gain_db'),
                'notch_low_hz': self.get_value('notch_low_hz'),
                'notch_high_hz': self.get_value('notch_high_hz'),
                'n_fft': self.get_value('n_fft'),
                'hop_length': self.get_value('hop_length'),
                'fade_ms': self.get_value('fade_ms'),  # From DEFAULTS None (skips unless voice sets)
                'normalize_method': self.get_value('normalize_method'),
                'enable_denoise_normalize': self.get_value('enable_denoise_normalize'),
                'enable_denoising': self.get_value('enable_denoising'),

                # NEW: Padding gates/tunables (merged like others)
                'enable_audio_padding': self.get_value('enable_audio_padding', default=True),
                'base_audio_pad_sec': self.get_value('base_audio_pad_sec'),
                'tiny_audio_pad_multiplier': self.get_value('tiny_audio_pad_multiplier'),
                'tiny_threshold_sec': self.get_value('tiny_threshold_sec'),
                'enable_text_padding': self.get_value('enable_text_padding', default=True),
                'text_ellipses_count': self.get_value('text_ellipses_count'),
                'max_short_word_len': self.get_value('max_short_word_len'),
                'vocalise_patterns': self.get_value('vocalise_patterns'),  # List
            }
        return self._audio_defaults


    def get_merged_audio_params(self, voice_name: Optional[str] = None, api_overrides: Optional[Dict] = None) -> Dict[str, Any]:
        """Merge: audio_defaults → Voice (JSON) → API. FIXED: Log all keys in merged (voice/raw + final); force voice get."""
        params = self.audio_defaults.copy()  # Subset with clamped globals
        params['voice_name'] = voice_name or 'default'

        # Force voice params (get even empty; log raw)
        voice_params_raw = self.get_voice_parameters(voice_name)  # Raw from JSON
        logger.debug(f"Raw voice params for '{voice_name}': {voice_params_raw}")  # e.g., {'trim_threshold_db': -40.0, ...} or {}

        # Merge voice → params (override defaults)
        if voice_params_raw:
            for key, value in voice_params_raw.items():
                params[key] = value  # e.g., trim=-40.0 overrides -30
                logger.debug(f"Voice override {key}={value} for {voice_name}")
        else:
            logger.debug(f"No voice-specific params for '{voice_name}' – using globals/defaults")

        # API/UI overrides (highest)
        if api_overrides:
            params.update({k: v for k, v in api_overrides.items() if k in params})
            logger.debug(f"API overrides for {voice_name}: {list(api_overrides.keys())}")

        # Log full merged (confirms load/apply)
        logger.debug(f"Merged audio params for '{voice_name}': { {k: v for k, v in params.items() if v is not None} }")  # Non-None only
        active_keys = [k for k, v in params.items() if v is not None and v != (0.0 if k.endswith('_db') else 0) and v != (1.0 if k == 'speaking_rate' else 1.0)]
        if active_keys:
            logger.info(f"Merged audio for {voice_name}: {len(active_keys)} active overrides ({sorted(active_keys)})")
        else:
            logger.info(f"Merged audio for {voice_name}: defaults (no overrides)")

        return params


    def get_voice_parameters(self, voice_name: str) -> Dict[str, Any]:
        """Voice overrides (from voices.json; dict or empty)."""
        return self.voice_overrides.get(voice_name, {})

    def get_all_voices(self) -> List[str]:
        """List voices from overrides."""
        return list(self.voice_overrides.keys())

# Backward Compat: Old Functions as Facades (Non-Breaking; Route to Class)
def load_skyrimnet_config():
    """Old API: Load (now via class; returns tuple for compat)."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        CONFIG.load_config()  # Init class
        # Return old format (defaults, modes, flags)
        default_config = CONFIG._defaults.copy()  # Shallow
        config_mode = {k: 'default' for k in default_config if
                       k in ['temperature', 'min_p', 'top_p', 'repetition_penalty', 'cfg_weight', 'exaggeration']}
        global_flags = {'enable_memory_cache': CONFIG.enable_memory_cache,
                        'enable_disk_cache': CONFIG.enable_disk_cache}
        _CONFIG_CACHE = (default_config, config_mode, global_flags)
    return _CONFIG_CACHE

def get_config_value(param_name: str, api_value, defaults=None, modes=None, bypass_config=False):
    """Old API: Get value (now via class get_value; preserves args)."""
    if defaults is None:
        defaults = {}
    if modes is None:
        modes = {}
    # Route to class (drops unused modes for now; compat)
    return CONFIG.get_value(param_name, api_value, default=defaults.get(param_name), bypass_config=bypass_config)

def reload_config():
    """Old API: Reload (now class method)."""
    CONFIG.reload_config()
    global _CONFIG_CACHE
    _CONFIG_CACHE = None

# Global Instance (Backward Compat: Use class attrs as globals)
CONFIG = SkyrimNetConfig()

# Set preserved globals from class (on load)
def _sync_globals():
    global DEVICE, DTYPE, MODEL, MULTILINGUAL, ENABLE_DISK_CACHE, ENABLE_MEMORY_CACHE, FUZZY_CACHE_LIMIT
    DEVICE = str(CONFIG.device)
    DTYPE = CONFIG.dtype
    MODEL = CONFIG.model
    MULTILINGUAL = CONFIG.multilingual
    ENABLE_DISK_CACHE = CONFIG.enable_disk_cache
    ENABLE_MEMORY_CACHE = CONFIG.enable_memory_cache
    FUZZY_CACHE_LIMIT = CONFIG.get_value('fuzzy_cache_limit', default=CONFIG.DEFAULTS.get('fuzzy_cache_limit', 1000))

# Initial sync (non-blocking)
load_skyrimnet_config()  # Triggers class init + _sync_globals implicit via properties
_sync_globals()