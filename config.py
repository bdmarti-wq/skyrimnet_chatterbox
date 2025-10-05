# skyrimnet_config.py - Fixed NameError in _initialize (use SkyrimNetConfig.DEFAULTS directly; cls not in scope)
# Minor: Added logging in _initialize for debug; ensured global sync after load.

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

# SkyrimNetConfig Class (singleton; from alternative, simplified for migration)
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
        'speaking_rate': 1.0,

        # Audio Defaults (adjustable; new from alt)
        'eq_gain_db': 0,
        'eq_cutoff_hz': 3000,
        'max_gain': 2.0,
        'target_max': 0.6,
        'noise_floor_db': -60,
        'trim_threshold_db': -28,
        'fade_ms': 20,
        'n_fft': 2048,  # New: For mel/STFT (alt)
        'n_mels': 80,
        'hop_length': 256,  # New: For mel/step (alt)
        'ebu_post_gain_db': 0,
        'ebu_true_peak': 0,
        'notch_low': 8000,
        'notch_high': 11000,
        'notch_gain_db': -12,

        # Strings (adjustable defaults; new from alt)
        'normalize_method': 'rms',
        'logging_level': 'INFO',
        'tts_name': 'Chatterbox',
        'dtype': 'bfloat16',

        # Booleans (adjustable flags; new from alt)
        'enable_pre_adjustment': False,  # New: Opt-in pad/mel align (alt; default False for non-breaking)
        'enable_post_processing': True,  # New
        'enable_denoising': False,  # New
        'enable_smoothing': False,  # New
        'enable_quantization': True,  # New
        'enable_spectral_gating': True,  # New
        'notch_enabled': False,  # New
        'hp_enabled': False,  # New
        'enable_memory_cache': True,
        'enable_disk_cache': True,
        'force_local_refs': True,  # New: Alt flag
        'auto_update_refs': True,  # New: Alt flag
        'timings_enabled': False,  # New
        'auto_reset_timings': False,  # New
        'enable_post_resample': True,  # New: Alt (review)
        'enable_post_jit_gain': True,  # New: Alt (review)
        'enable_post_voice_processing': True,  # New: Alt (review)
        'enable_deferred_cleanup': True,  # New

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
    }

    # CAPS: Hard constants/immutable limits for clamping (numerics only; from alt; used in get_value)
    # Add MIN/MAX for all DEFAULTS numerics (user wants ready; non-breaking as optional in get_value)
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
        'SPEAKING_RATE_MIN': 0.5, 'SPEAKING_RATE_MAX': 2.0,

        # Audio Caps
        'EQ_GAIN_DB_MIN': -12, 'EQ_GAIN_DB_MAX': 6,
        'EQ_CUTOFF_HZ_MIN': 2000, 'EQ_CUTOFF_HZ_MAX': 5000,
        'MAX_GAIN_MIN': 0.5, 'MAX_GAIN_MAX': 3.0,
        'TARGET_MAX_MIN': 0.1, 'TARGET_MAX_MAX': 1.0,
        'NOISE_FLOOR_DB_MIN': -60, 'NOISE_FLOOR_DB_MAX': -10,
        'TRIM_THRESHOLD_DB_MIN': -50, 'TRIM_THRESHOLD_DB_MAX': -10,
        'FADE_MS_MIN': 10, 'FADE_MS_MAX': 50,
        'N_FFT_MIN': 512, 'N_FFT_MAX': 2048,
        'HOP_LENGTH_MIN': 128, 'HOP_LENGTH_MAX': 512,
        'EBU_POST_GAIN_DB_MIN': -3, 'EBU_POST_GAIN_DB_MAX': 3,
        'EBU_TRUE_PEAK_MIN': -5, 'EBU_TRUE_PEAK_MAX': 0,
        'NOTCH_LOW_MIN': 4000, 'NOTCH_LOW_MAX': 10000,
        'NOTCH_HIGH_MIN': 8000, 'NOTCH_HIGH_MAX': 16000,
        'NOTCH_GAIN_DB_MIN': -24, 'NOTCH_GAIN_DB_MAX': 0,

        # Cache Caps (new from alt)
        'COND_CACHE_MAX_ENTRIES_MIN': 10, 'COND_CACHE_MAX_ENTRIES_MAX': 100,
        'FUZZY_CACHE_LIMIT_MIN': 100, 'FUZZY_CACHE_LIMIT_MAX': 5000,
        'FUZZY_THRESHOLD_MIN': 0.50, 'FUZZY_THRESHOLD_MAX': 0.95,
    }

    _instance = None
    _lock = threading.Lock()
    _config_file_path = _CONFIG_FILE  # "skyrimnet_config.txt"

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
        self.voice_overrides = {}  # Dict[str, Dict] for voices
        self._is_modified = False
        self.sr = 24000  # Core

        logger.debug("SkyrimNetConfig initialized (core globals set)")

        # Load full config
        self.load_config()

    def load_config(self):
        """Load config (simple INI + basic JSON blocks for overrides; merges with DEFAULTS). Non-blocking."""
        global _CONFIG_CACHE, ENABLE_MEMORY_CACHE, ENABLE_DISK_CACHE

        if _CONFIG_CACHE is not None:  # Preserve old cache
            defaults, modes, global_flags = _CONFIG_CACHE
            self._defaults.update(defaults)  # Merge old
            self.enable_memory_cache = global_flags.get('enable_memory_cache', ENABLE_MEMORY_CACHE)
            self.enable_disk_cache = global_flags.get('enable_disk_cache', ENABLE_DISK_CACHE)
            ENABLE_MEMORY_CACHE = self.enable_memory_cache
            ENABLE_DISK_CACHE = self.enable_disk_cache
            logger.debug("Config from cache (merged with class)")
            return _CONFIG_CACHE

        # Default config (preserved from rollback)
        default_config = {
            'temperature': 0.8,
            'min_p': 0.07,
            'top_p': 1.0,
            'repetition_penalty': 2.0,
            'cfg_weight': 0.0,
            'exaggeration': 0.7
        }
        config_mode = {k: 'default' for k in default_config}  # Modes (default/api/custom)
        global_flags = {
            'enable_memory_cache': ENABLE_MEMORY_CACHE,
            'enable_disk_cache': ENABLE_DISK_CACHE
        }

        try:
            config_path = Path(self._config_file_path)
            if not config_path.exists():
                logger.warning(f"Config file {self._config_file_path} not found, using hardcoded defaults")
                self._merge_defaults(default_config, config_mode, global_flags)
                _CONFIG_CACHE = (default_config, config_mode, global_flags)
                return _CONFIG_CACHE

            with open(config_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Simple parser (INI-like; basic JSON for voice_overrides block)
            in_json_block = False
            json_buffer = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                if line.startswith('voice_overrides = {'):
                    in_json_block = True
                    json_buffer = [line]
                    continue
                if in_json_block:
                    json_buffer.append(line)
                    if line == '}':
                        in_json_block = False
                        try:
                            # Parse JSON block for voice_overrides
                            json_str = '\n'.join(json_buffer)
                            self.voice_overrides = json.loads(json_str.replace('voice_overrides =', ''))
                            logger.debug(f"Parsed voice_overrides: {len(self.voice_overrides)} voices")
                        except json.JSONDecodeError as je:
                            logger.warning(f"Invalid voice_overrides JSON: {je} – Skipped")
                            self.voice_overrides = {}
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

                    # Handle parameter modes/values (preserve old; merge to _defaults)
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

            # Merge: Old defaults → class _defaults; flags → self attrs
            self._merge_defaults(default_config, config_mode, global_flags)
            self.enable_memory_cache = global_flags.get('enable_memory_cache', ENABLE_MEMORY_CACHE)
            self.enable_disk_cache = global_flags.get('enable_disk_cache', ENABLE_DISK_CACHE)
            ENABLE_MEMORY_CACHE = self.enable_memory_cache
            ENABLE_DISK_CACHE = self.enable_disk_cache

            voices_count = len(self.voice_overrides)
            logger.info(f"Config loaded: {voices_count} voices, {len(self._defaults)} params (logging_level={self.logging_level})")

            _CONFIG_CACHE = (default_config, config_mode, global_flags)
            return _CONFIG_CACHE


        except Exception as e:
            logger.error(f"Config load failed: {e}, using hardcoded defaults")
            self._merge_defaults(default_config, config_mode, global_flags)
            _CONFIG_CACHE = (default_config, config_mode, global_flags)
            return _CONFIG_CACHE

    def _merge_defaults(self, input_defaults: Dict, modes: Dict, flags: Dict):
        """Internal: Merge input to class storage (non-breaking)."""
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
        """Clamp numerics using CAPS (non-breaking; skip if no CAP)."""
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

        """Audio params subset (merged; from alt)."""

        if not hasattr(self, '_audio_defaults'):
            self._audio_defaults = {
                'enable_post_processing': self.get_value('enable_post_processing', default=False),
                'enable_post_resample': self.get_value('enable_post_resample', default=True),
                'enable_post_jit_gain': self.get_value('enable_post_jit_gain', default=True),
                'enable_post_voice_processing': self.get_value('enable_post_voice_processing', default=True),
                'enable_pre_adjustment': self.get_value('enable_pre_adjustment', default=False),
                'speaking_rate': self.speaking_rate,
                'eq_gain_db': self.eq_gain_db,
                'eq_cutoff_hz': self.eq_cutoff_hz,
                'max_gain': self.max_gain,
                'target_max': self.target_max,
                'noise_floor_db': self.noise_floor_db,
                'trim_threshold_db': self.trim_threshold_db,
                'notch_enabled': self.notch_enabled,
                'hp_enabled': self.hp_enabled,
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'fade_ms': self.fade_ms,
                'normalize_method': self.normalize_method,
                'enable_denoising': self.enable_denoising
            }

        return self._audio_defaults


    def get_merged_audio_params(self, voice_name: Optional[str] = None, api_overrides: Optional[Dict] = None) -> Dict[
        str, Any]:
        """Merge global + voice + api (from alt)."""
        params = self.audio_defaults.copy()
        if voice_name and voice_name in self.voice_overrides:
            voice_params = self.voice_overrides[voice_name]
            params.update({k: v for k, v in voice_params.items() if k in params})
            logger.debug(f"Merged audio for {voice_name}: enable_pre_adjustment={params['enable_pre_adjustment']}")
        if api_overrides:
            params.update({k: v for k, v in api_overrides.items() if k in params})
        return params


    def get_voice_parameters(self, voice_name: str) -> Dict[str, Any]:
        """Voice overrides (dict or empty)."""
        return self.voice_overrides.get(voice_name, {})


    def get_all_voices(self) -> List[str]:
        """List voices from overrides."""
        return list(self.voice_overrides.keys())

    def save_config(self, create_backup: bool = True) -> Tuple[bool, str]:
        """Save to txt (INI + JSON block; from alt, simplified)."""
        try:
            config_path = Path(self._config_file_path)
            if create_backup and config_path.exists():
                backup_path = config_path.with_suffix('.backup')
                config_path.rename(backup_path)
                logger.debug(f"Backup: {backup_path}")
            with open(config_path, 'w', encoding='utf-8') as f:
                # Global settings (INI)
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
                f.write(f"force_local_refs = {self.force_local_refs}\n")  # New
                f.write(f"auto_update_refs = {self.auto_update_refs}\n")  # New
                f.write(f"tts_name = {self.tts_name}\n")
                f.write(f"dtype = {self.dtype_str}\n")  # New
                # Flags section
                f.write("\n# Flags\n")

                for flag, val in self._flags.items():
                    if flag not in ['enable_pre_adjustment', 'enable_memory_cache', 'enable_disk_cache',
                                    'force_local_refs', 'auto_update_refs']:
                        f.write(f"{flag} = {val}\n")

                # Voice overrides (JSON block)
                f.write("\n# Voice Overrides (JSON block)\n")
                f.write("voice_overrides = {\n")

                for voice, params in self.voice_overrides.items():
                    f.write(f'    "{voice}": {{\n')
                    for k, v in params.items():
                        f.write(f'        "{k}": {v},\n')
                    f.write('    },\n')
                f.write("}\n")

            logger.info(f"Config saved: {config_path} ({len(self.voice_overrides)} voices)")
            self._is_modified = False
            return True, f"Saved | {len(self.voice_overrides)} voices"

        except Exception as e:
            logger.error(f"Save failed: {e}")
            return False, str(e)


    def reload_config(self):
        """Reload (clears cache; re-parses)."""
        global _CONFIG_CACHE
        _CONFIG_CACHE = None

        self.load_config()
        logger.info("Config reloaded")



# Backward Compat: Old Functions as Facades (Non-Breaking; Route to Class) TODO review facades

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