"""
Main Config Module: Orchestrates parsers, creates Config singleton with getters/setters, backward-compat facades.
"""

import threading
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from collections import OrderedDict
import hashlib  # For LRU hashes

from loguru import logger
import torch

# Import helpers from same package
from .defaults import DEFAULTS, _sync_globals, DEVICE, DTYPE, MODEL, MULTILINGUAL, ENABLE_DISK_CACHE, ENABLE_MEMORY_CACHE, FUZZY_CACHE_LIMIT, _CONFIG_CACHE, _CONFIG_FILE, _USE_API_MODE
from .parser_txt import _load_txt_config
from .parser_json import _load_voices_json
from .validation import CAPS, clamp_numeric, parse_bool, parse_list, validate_and_clamp

# No global CONFIG here; lazy instantiation

def sync_cores_to_globals(config_instance):
    """Sync CONFIG cores → globals (bi-directional for device/dtype/etc.). Call after changes."""
    global DEVICE, DTYPE, MODEL, MULTILINGUAL, _USE_API_MODE
    DEVICE = str(config_instance.device)
    DTYPE = config_instance.dtype
    MODEL = config_instance.model
    MULTILINGUAL = config_instance.multilingual
    _USE_API_MODE = config_instance._use_api_mode
    logger.debug("Cores synced to globals")

def sync_globals_to_cores(config_instance):
    """Sync globals → CONFIG cores (fallback during load/init)."""
    global DEVICE, DTYPE, MODEL, MULTILINGUAL, _USE_API_MODE
    if isinstance(DEVICE, str):
        config_instance.device = torch.device(DEVICE if DEVICE.lower() == 'cuda' else 'cpu')
    else:
        config_instance.device = DEVICE
    config_instance.dtype = DTYPE or (torch.bfloat16 if config_instance.device.type == 'cuda' else torch.float32)
    config_instance.model = MODEL
    config_instance.multilingual = bool(MULTILINGUAL)
    config_instance._use_api_mode = bool(_USE_API_MODE)
    logger.debug("Globals synced to cores")


class Config:
    _instance = None
    _lock = threading.Lock()
    _config_file_path = _CONFIG_FILE
    _voices_file_path = Path(__file__).parent.parent / "voices.json"

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
                sync_cores_to_globals(cls._instance)  # Initial sync
            return cls._instance

    def _initialize(self):
        """Init all state (cores direct; no validation)."""
        # Core attrs (direct assignment; detected values)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # str initially
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32  # torch.dtype
        self.model = None
        self.multilingual = False  # Bool direct
        self._use_api_mode = False
        self.sr = 24000  # int fixed

        # Tunable state (for DEFAULTS)
        self._defaults = {}
        self._flags = {}
        self._strings = {}
        self.voice_overrides = {}
        self._is_modified = False
        self._config_cache = None
        self._audio_defaults = None
        self._merged_cache = OrderedDict(maxlen=100)
        self._global_hash = None
        self._voice_hashes = {}
        logger.debug("Config fully initialized (cores + state)")

    def load_config(self):
        """Load: Txt → storage dicts → apply properties/setters → voices → sync once at end."""
        if self._config_cache is not None:
            defaults, modes, global_flags = self._config_cache
            self._defaults.update(defaults)
            # Apply flags directly to storage (no property trigger during reload)
            self._flags['enable_memory_cache'] = global_flags.get('enable_memory_cache', DEFAULTS.get('enable_memory_cache', True))
            self._flags['enable_disk_cache'] = global_flags.get('enable_disk_cache', DEFAULTS.get('enable_disk_cache', True))
            self.voice_overrides = _load_voices_json()
            self._invalidate_merged_cache()
            # Sync at end
            _sync_globals(self)
            logger.debug("Config reloaded from cache")
            return self._config_cache

        # Load txt
        default_config, _, global_flags = _load_txt_config()

        # Update storage dicts with parsed values (no properties yet)
        self._defaults.update({k: v for k, v in default_config.items() if k in DEFAULTS})
        for k in DEFAULTS:
            if k not in self._defaults:
                self._defaults[k] = DEFAULTS[k]
        # Flags from txt/globals (direct to _flags)
        self._flags['enable_memory_cache'] = global_flags.get('enable_memory_cache', DEFAULTS.get('enable_memory_cache', True))
        self._flags['enable_disk_cache'] = global_flags.get('enable_disk_cache', DEFAULTS.get('enable_disk_cache', True))

        # Core overrides (direct, as before)
        dtype_str = default_config.get('dtype', 'bfloat16' if 'cuda' in self.device else 'float32')
        if dtype_str.lower() == 'bfloat16' and torch.cuda.is_available():
            self.dtype = torch.bfloat16
        elif dtype_str.lower() == 'float32':
            self.dtype = torch.float32
        else:
            logger.warning(f"Invalid dtype '{dtype_str}' in config, using detected {self.dtype}")
        self.multilingual = parse_bool(default_config.get('multilingual', self.multilingual))
        self._use_api_mode = parse_bool(default_config.get('use_api_mode', self._use_api_mode))
        if isinstance(self.device, str):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Now apply DEFAULTS to properties (triggers setters/validation; safe now)
        for attr, val in self._defaults.items():
            if hasattr(self.__class__, attr) and isinstance(getattr(type(self), attr, None), property):
                try:
                    setattr(self, attr, val)  # Triggers setter (stores back to _defaults if needed)
                except Exception as e:
                    logger.warning(f"Failed to set {attr}={val} via property: {e} (keeping storage value)")

        # Init remaining _flags bools
        for attr, val in DEFAULTS.items():
            if isinstance(val, bool) and attr not in self._flags:
                self._flags[attr] = parse_bool(val)

        # Load voices (after properties)
        self.voice_overrides = _load_voices_json()

        # Sync model
        self._sync_model_if_loaded()

        # SINGLE sync at end (no recursion, all storage populated)
        _sync_globals(self)
        self._is_modified = False

        # Log
        voices_count = len(self.voice_overrides)
        voices_keys = list(self.voice_overrides.keys()) if voices_count else []
        logging_level = self._get_validated('logging_level') or 'INFO'
        logger.info(
            f"Config loaded: {voices_count} voices (keys: {voices_keys}), {len(self._defaults)} params (logging_level={logging_level}; device={self.device}; dtype={self.dtype}; multilingual={self.multilingual})")

        self._config_cache = (default_config, {}, global_flags)
        return self._config_cache


    def save_config(self, create_backup: bool = True) -> Tuple[bool, str]:
        """Save: Txt (globals/flags) + voices.json."""
        try:
            config_path = Path(self._config_file_path)
            if create_backup and config_path.exists():
                backup_path = config_path.with_suffix('.backup')
                config_path.rename(backup_path)
                logger.debug(f"Backup created: {backup_path}")

            # Save txt (use properties/getters for values—safe post-load)
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
                f.write(f"dtype = {self._defaults.get('dtype', 'bfloat16')}\n")
                f.write("\n# Flags\n")
                for flag, val in self._flags.items():
                    if flag not in ['enable_pre_adjustment', 'enable_memory_cache', 'enable_disk_cache', 'force_local_refs', 'auto_update_refs']:
                        f.write(f"{flag} = {val}\n")

            # Save voices.json
            voices_file = self._voices_file_path
            with open(voices_file, 'w', encoding='utf-8') as f:
                json.dump(self.voice_overrides, f, indent=2)

            logger.info(f"Config saved: {config_path} + voices.json ({len(self.voice_overrides)} voices)")
            self._is_modified = False
            # Sync globals at end
            _sync_globals(self)
            return True, f"Saved | {len(self.voice_overrides)} voices"

        except Exception as e:
            logger.error(f"Save failed: {e}")
            return False, str(e)

    def reload_config(self):
        """Reload: Clears cache; re-calls load_config."""
        self._config_cache = None
        self._merged_cache.clear()
        self._global_hash = None
        self._voice_hashes = {}
        self.load_config()
        logger.info("Config reloaded (txt + voices.json)")

    # Helper Methods (non-recursive; dict lookup only)
    def _get_validated(self, param_name: str) -> Any:
        """Get tunable value from storage (no hasattr/getattr—avoids property recursion)."""
        # Prefer storage dicts first
        if param_name in self._defaults:
            val = self._defaults[param_name]
            if isinstance(val, (int, float)):
                return clamp_numeric(param_name, val)
            return val
        if param_name in self._flags:
            return self._flags[param_name]  # Bool direct
        if param_name in self._strings:
            return self._strings[param_name]
        return DEFAULTS.get(param_name, None)

    def _validate_set(self, param_name: str, value: Any) -> bool:
        """Internal: Validate value (via shared helper); returns True if valid (set it). Warn and return False if invalid."""
        try:
            validated = validate_and_clamp(param_name, value, DEFAULTS)  # Pass DEFAULTS for fallback
            if validated != value:
                logger.warning(f"Invalid value for {param_name}: {value} -> {validated} (clamped/parsed; set anyway)")
            # Always set validated version
            return True
        except Exception as e:
            logger.warning(f"Validation failed for {param_name} = {value}: {e} (value unchanged)")
            return False

    def _invalidate_merged_cache(self):
        """Internal: Purge LRU if global/voice changed (for get_merged_audio_params)."""
        self._global_hash = None
        if self._is_modified:
            self._merged_cache.clear()
        else:
            self._merged_cache.clear()  # Always clear on invalidate for safety

    # Clamp value (shared; original method)
    def clamp_value(self, param_name: str, value: Any) -> Any:
        if isinstance(value, (int, float)):
            return clamp_numeric(param_name, value)
        return value


    # Property getters/setters for all DEFAULTS keys (fixed: flags direct, numerics validated; no sync in setters except direct)
    @property
    def temperature(self):
        return self._get_validated('temperature')

    @temperature.setter
    def temperature(self, value):
        if self._validate_set('temperature', value):
            self._defaults['temperature'] = clamp_numeric('temperature', value)
            self._invalidate_merged_cache()

    @property
    def min_p(self):
        return self._get_validated('min_p')

    @min_p.setter
    def min_p(self, value):
        if self._validate_set('min_p', value):
            self._defaults['min_p'] = clamp_numeric('min_p', value)
            self._invalidate_merged_cache()

    @property
    def top_p(self):
        return self._get_validated('top_p')

    @top_p.setter
    def top_p(self, value):
        if self._validate_set('top_p', value):
            self._defaults['top_p'] = clamp_numeric('top_p', value)
            self._invalidate_merged_cache()

    @property
    def repetition_penalty(self):
        return self._get_validated('repetition_penalty')

    @repetition_penalty.setter
    def repetition_penalty(self, value):
        if self._validate_set('repetition_penalty', value):
            self._defaults['repetition_penalty'] = clamp_numeric('repetition_penalty', value)
            self._invalidate_merged_cache()

    @property
    def cfg_weight(self):
        return self._get_validated('cfg_weight')

    @cfg_weight.setter
    def cfg_weight(self, value):
        if self._validate_set('cfg_weight', value):
            self._defaults['cfg_weight'] = clamp_numeric('cfg_weight', value)
            self._invalidate_merged_cache()

    @property
    def exaggeration(self):
        return self._get_validated('exaggeration')

    @exaggeration.setter
    def exaggeration(self, value):
        if self._validate_set('exaggeration', value):
            self._defaults['exaggeration'] = clamp_numeric('exaggeration', value)
            self._invalidate_merged_cache()

    @property
    def speaking_rate(self):
        return self._get_validated('speaking_rate')

    @speaking_rate.setter
    def speaking_rate(self, value):
        if self._validate_set('speaking_rate', value):
            self._defaults['speaking_rate'] = clamp_numeric('speaking_rate', value)
            self._invalidate_merged_cache()
            self._global_hash = None

    @property
    def eq_gain_db(self):
        return self._get_validated('eq_gain_db')

    @eq_gain_db.setter
    def eq_gain_db(self, value):
        if self._validate_set('eq_gain_db', value):
            self._defaults['eq_gain_db'] = clamp_numeric('eq_gain_db', value)
            self._invalidate_merged_cache()
            self._global_hash = None

    @property
    def eq_cutoff_hz(self):
        return self._get_validated('eq_cutoff_hz')

    @eq_cutoff_hz.setter
    def eq_cutoff_hz(self, value):
        if self._validate_set('eq_cutoff_hz', value):
            self._defaults['eq_cutoff_hz'] = clamp_numeric('eq_cutoff_hz', value)
            self._invalidate_merged_cache()
            self._global_hash = None

    @property
    def notch_gain_db(self):
        return self._get_validated('notch_gain_db')

    @notch_gain_db.setter
    def notch_gain_db(self, value):
        if self._validate_set('notch_gain_db', value):
            self._defaults['notch_gain_db'] = clamp_numeric('notch_gain_db', value)
            self._invalidate_merged_cache()
            self._global_hash = None

    @property
    def notch_low_hz(self):
        return self._get_validated('notch_low_hz')

    @notch_low_hz.setter
    def notch_low_hz(self, value):
        if self._validate_set('notch_low_hz', value):
            self._defaults['notch_low_hz'] = clamp_numeric('notch_low_hz', value)
            self._invalidate_merged_cache()
            self._global_hash = None

    @property
    def notch_high_hz(self):
        return self._get_validated('notch_high_hz')

    @notch_high_hz.setter
    def notch_high_hz(self, value):
        if self._validate_set('notch_high_hz', value):
            self._defaults['notch_high_hz'] = clamp_numeric('notch_high_hz', value)
            self._invalidate_merged_cache()
            self._global_hash = None

    @property
    def gain_max_limit(self):
        return self._get_validated('gain_max_limit')

    @gain_max_limit.setter
    def gain_max_limit(self, value):
        if self._validate_set('gain_max_limit', value):
            self._defaults['gain_max_limit'] = clamp_numeric('gain_max_limit', value)
            self._invalidate_merged_cache()
            self._global_hash = None

    @property
    def trim_threshold_db(self):
        return self._get_validated('trim_threshold_db')

    @trim_threshold_db.setter
    def trim_threshold_db(self, value):
        if self._validate_set('trim_threshold_db', value):
            self._defaults['trim_threshold_db'] = clamp_numeric('trim_threshold_db', value)
            self._invalidate_merged_cache()
            self._global_hash = None

    @property
    def fade_ms(self):
        return self._get_validated('fade_ms')

    @fade_ms.setter
    def fade_ms(self, value):
        if self._validate_set('fade_ms', value):
            self._defaults['fade_ms'] = clamp_numeric('fade_ms', value)
            self._invalidate_merged_cache()
            self._global_hash = None

    @property
    def enable_denoise_normalize(self):
        return self._flags.get('enable_denoise_normalize', DEFAULTS.get('enable_denoise_normalize', False))

    @enable_denoise_normalize.setter
    def enable_denoise_normalize(self, value):
        if self._validate_set('enable_denoise_normalize', value):
            self._flags['enable_denoise_normalize'] = parse_bool(value)
            self._invalidate_merged_cache()
            self._global_hash = None

    @property
    def noise_floor_db(self):
        return self._get_validated('noise_floor_db')

    @noise_floor_db.setter
    def noise_floor_db(self, value):
        if self._validate_set('noise_floor_db', value):
            self._defaults['noise_floor_db'] = clamp_numeric('noise_floor_db', value)
            self._invalidate_merged_cache()
            self._global_hash = None

    @property
    def normalize_method(self):
        return self._strings.get('normalize_method', DEFAULTS.get('normalize_method', 'rms'))

    @normalize_method.setter
    def normalize_method(self, value):
        if self._validate_set('normalize_method', value):
            self._strings['normalize_method'] = str(value)
            self._invalidate_merged_cache()
            self._global_hash = None

    @property
    def n_fft(self):
        return self._get_validated('n_fft')

    @n_fft.setter
    def n_fft(self, value):
        if self._validate_set('n_fft', value):
            self._defaults['n_fft'] = clamp_numeric('n_fft', value)
            self._invalidate_merged_cache()
            self._global_hash = None

    @property
    def hop_length(self):
        return self._get_validated('hop_length')

    @hop_length.setter
    def hop_length(self, value):
        if self._validate_set('hop_length', value):
            self._defaults['hop_length'] = clamp_numeric('hop_length', value)
            self._invalidate_merged_cache()
            self._global_hash = None

    @property
    def max_gain(self):
        return self._get_validated('max_gain')

    @max_gain.setter
    def max_gain(self, value):
        if self._validate_set('max_gain', value):
            self._defaults['max_gain'] = clamp_numeric('max_gain', value)
            self._invalidate_merged_cache()
            self._global_hash = None

    @property
    def target_max(self):
        return self._get_validated('target_max')

    @target_max.setter
    def target_max(self, value):
        if self._validate_set('target_max', value):
            self._defaults['target_max'] = clamp_numeric('target_max', value)
            self._invalidate_merged_cache()
            self._global_hash = None

    @property
    def enable_pre_adjustment(self):
        return self._flags.get('enable_pre_adjustment', DEFAULTS.get('enable_pre_adjustment', True))

    @enable_pre_adjustment.setter
    def enable_pre_adjustment(self, value):
        if self._validate_set('enable_pre_adjustment', value):
            self._flags['enable_pre_adjustment'] = parse_bool(value)
            self._invalidate_merged_cache()
            self._global_hash = None

    @property
    def enable_post_processing(self):
        return self._flags.get('enable_post_processing', DEFAULTS.get('enable_post_processing', True))

    @enable_post_processing.setter
    def enable_post_processing(self, value):
        if self._validate_set('enable_post_processing', value):
            self._flags['enable_post_processing'] = parse_bool(value)
            self._invalidate_merged_cache()
            self._global_hash = None

    @property
    def enable_denoising(self):
        return self._flags.get('enable_denoising', DEFAULTS.get('enable_denoising', True))

    @enable_denoising.setter
    def enable_denoising(self, value):
        if self._validate_set('enable_denoising', value):
            self._flags['enable_denoising'] = parse_bool(value)
            self._invalidate_merged_cache()
            self._global_hash = None


    @property
    def enable_smoothing(self):
        return self._flags.get('enable_smoothing', DEFAULTS.get('enable_smoothing', True))

    @enable_smoothing.setter
    def enable_smoothing(self, value):
        if self._validate_set('enable_smoothing', value):
            self._flags['enable_smoothing'] = parse_bool(value)
            self._invalidate_merged_cache()
            self._global_hash = None

    @property
    def enable_quantization(self):
        return self._flags.get('enable_quantization', DEFAULTS.get('enable_quantization', True))

    @enable_quantization.setter
    def enable_quantization(self, value):
        if self._validate_set('enable_quantization', value):
            self._flags['enable_quantization'] = parse_bool(value)
            self._invalidate_merged_cache()
            self._global_hash = None

    @property
    def enable_resample(self):
        return self._flags.get('enable_resample', DEFAULTS.get('enable_resample', True))

    @enable_resample.setter
    def enable_resample(self, value):
        if self._validate_set('enable_resample', value):
            self._flags['enable_resample'] = parse_bool(value)
            self._invalidate_merged_cache()
            self._global_hash = None

    @property
    def enable_spectral_gating(self):
        return self._flags.get('enable_spectral_gating', DEFAULTS.get('enable_spectral_gating', True))

    @enable_spectral_gating.setter
    def enable_spectral_gating(self, value):
        if self._validate_set('enable_spectral_gating', value):
            self._flags['enable_spectral_gating'] = parse_bool(value)
            self._invalidate_merged_cache()
            self._global_hash = None

    @property
    def notch_enabled(self):
        return self._flags.get('notch_enabled', DEFAULTS.get('notch_enabled', True))

    @notch_enabled.setter
    def notch_enabled(self, value):
        if self._validate_set('notch_enabled', value):
            self._flags['notch_enabled'] = parse_bool(value)
            self._invalidate_merged_cache()
            self._global_hash = None

    @property
    def hp_enabled(self):
        return self._flags.get('hp_enabled', DEFAULTS.get('hp_enabled', True))

    @hp_enabled.setter
    def hp_enabled(self, value):
        if self._validate_set('hp_enabled', value):
            self._flags['hp_enabled'] = parse_bool(value)
            self._invalidate_merged_cache()
            self._global_hash = None

    @property
    def enable_disk_cache(self):
        return self._flags.get('enable_disk_cache', DEFAULTS.get('enable_disk_cache', True))

    @enable_disk_cache.setter
    def enable_disk_cache(self, value):
        if self._validate_set('enable_disk_cache', value):
            self._flags['enable_disk_cache'] = parse_bool(value)
            self._invalidate_merged_cache()
            self._global_hash = None
            globals()['ENABLE_DISK_CACHE'] = self._flags['enable_disk_cache']

    @property
    def enable_memory_cache(self):
        return self._flags.get('enable_memory_cache', DEFAULTS.get('enable_memory_cache', True))

    @enable_memory_cache.setter
    def enable_memory_cache(self, value):
        if self._validate_set('enable_memory_cache', value):
            self._flags['enable_memory_cache'] = parse_bool(value)
            self._invalidate_merged_cache()
            self._global_hash = None
            globals()['ENABLE_MEMORY_CACHE'] = self._flags['enable_memory_cache']

    @property
    def force_local_refs(self):
        return self._flags.get('force_local_refs', DEFAULTS.get('force_local_refs', False))

    @force_local_refs.setter
    def force_local_refs(self, value):
        if self._validate_set('force_local_refs', value):
            self._flags['force_local_refs'] = parse_bool(value)
            self._invalidate_merged_cache()
            self._global_hash = None

    @property
    def auto_update_refs(self):
        return self._flags.get('auto_update_refs', DEFAULTS.get('auto_update_refs', True))

    @auto_update_refs.setter
    def auto_update_refs(self, value):
        if self._validate_set('auto_update_refs', value):
            self._flags['auto_update_refs'] = parse_bool(value)
            self._invalidate_merged_cache()
            self._global_hash = None


    @property
    def fuzzy_cache_limit(self):
        return self._get_validated('fuzzy_cache_limit')

    @fuzzy_cache_limit.setter
    def fuzzy_cache_limit(self, value):
        if self._validate_set('fuzzy_cache_limit', value):
            self._defaults['fuzzy_cache_limit'] = clamp_numeric('fuzzy_cache_limit', value)
            globals()['FUZZY_CACHE_LIMIT'] = self._defaults['fuzzy_cache_limit']

    @property
    def max_new_tokens(self):
        return self._get_validated('max_new_tokens')

    @max_new_tokens.setter
    def max_new_tokens(self, value):
        if self._validate_set('max_new_tokens', value):
            self._defaults['max_new_tokens'] = clamp_numeric('max_new_tokens', value)
            self._invalidate_merged_cache()

    @property
    def min_new_tokens(self):
        return self._get_validated('min_new_tokens')

    @min_new_tokens.setter
    def min_new_tokens(self, value):
        if self._validate_set('min_new_tokens', value):
            self._defaults['min_new_tokens'] = clamp_numeric('min_new_tokens', value)
            self._invalidate_merged_cache()

    @property
    def max_cache_len(self):
        return self._get_validated('max_cache_len')

    @max_cache_len.setter
    def max_cache_len(self, value):
        if self._validate_set('max_cache_len', value):
            self._defaults['max_cache_len'] = clamp_numeric('max_cache_len', value)
            self._invalidate_merged_cache()

    @property
    def stride_length(self):
        return self._get_validated('stride_length')

    @stride_length.setter
    def stride_length(self, value):
        if self._validate_set('stride_length', value):
            self._defaults['stride_length'] = clamp_numeric('stride_length', value)
            self._invalidate_merged_cache()

    @property
    def n_mels(self):
        return self._get_validated('n_mels')

    @n_mels.setter
    def n_mels(self, value):
        if self._validate_set('n_mels', value):
            self._defaults['n_mels'] = clamp_numeric('n_mels', value)
            self._invalidate_merged_cache()

    @property
    def ebu_post_gain_db(self):
        return self._get_validated('ebu_post_gain_db')

    @ebu_post_gain_db.setter
    def ebu_post_gain_db(self, value):
        if self._validate_set('ebu_post_gain_db', value):
            self._defaults['ebu_post_gain_db'] = clamp_numeric('ebu_post_gain_db', value)
            self._invalidate_merged_cache()

    @property
    def ebu_true_peak(self):
        return self._get_validated('ebu_true_peak')

    @ebu_true_peak.setter
    def ebu_true_peak(self, value):
        if self._validate_set('ebu_true_peak', value):
            self._defaults['ebu_true_peak'] = clamp_numeric('ebu_true_peak', value)
            self._invalidate_merged_cache()

    @property
    def base_audio_pad_sec(self):
        return self._get_validated('base_audio_pad_sec')

    @base_audio_pad_sec.setter
    def base_audio_pad_sec(self, value):
        if self._validate_set('base_audio_pad_sec', value):
            self._defaults['base_audio_pad_sec'] = clamp_numeric('base_audio_pad_sec', value)
            self._invalidate_merged_cache()

    @property
    def tiny_audio_pad_multiplier(self):
        return self._get_validated('tiny_audio_pad_multiplier')

    @tiny_audio_pad_multiplier.setter
    def tiny_audio_pad_multiplier(self, value):
        if self._validate_set('tiny_audio_pad_multiplier', value):
            self._defaults['tiny_audio_pad_multiplier'] = clamp_numeric('tiny_audio_pad_multiplier', value)
            self._invalidate_merged_cache()

    @property
    def tiny_threshold_sec(self):
        return self._get_validated('tiny_threshold_sec')

    @tiny_threshold_sec.setter
    def tiny_threshold_sec(self, value):
        if self._validate_set('tiny_threshold_sec', value):
            self._defaults['tiny_threshold_sec'] = clamp_numeric('tiny_threshold_sec', value)
            self._invalidate_merged_cache()

    @property
    def fuzzy_index_size(self):
        return self._get_validated('fuzzy_index_size')

    @fuzzy_index_size.setter
    def fuzzy_index_size(self, value):
        if self._validate_set('fuzzy_index_size', value):
            self._defaults['fuzzy_index_size'] = clamp_numeric('fuzzy_index_size', value)

    @property
    def fuzzy_boost_amount(self):
        return self._get_validated('fuzzy_boost_amount')

    @fuzzy_boost_amount.setter
    def fuzzy_boost_amount(self, value):
        if self._validate_set('fuzzy_boost_amount', value):
            self._defaults['fuzzy_boost_amount'] = clamp_numeric('fuzzy_boost_amount', value)

    @property
    def fuzzy_threshold(self):
        return self._get_validated('fuzzy_threshold')

    @fuzzy_threshold.setter
    def fuzzy_threshold(self, value):
        if self._validate_set('fuzzy_threshold', value):
            self._defaults['fuzzy_threshold'] = clamp_numeric('fuzzy_threshold', value)

    @property
    def fuzzy_enable(self):
        return self._flags.get('fuzzy_enable', DEFAULTS.get('fuzzy_enable', True))

    @fuzzy_enable.setter
    def fuzzy_enable(self, value):
        if self._validate_set('fuzzy_enable', value):
            self._flags['fuzzy_enable'] = parse_bool(value)

    @property
    def compress_pt_saves(self):
        return self._flags.get('compress_pt_saves', DEFAULTS.get('compress_pt_saves', True))

    @compress_pt_saves.setter
    def compress_pt_saves(self, value):
        if self._validate_set('compress_pt_saves', value):
            self._flags['compress_pt_saves'] = parse_bool(value)

    @property
    def compress_level(self):
        return self._get_validated('compress_level')

    @compress_level.setter
    def compress_level(self, value):
        if self._validate_set('compress_level', value):
            self._defaults['compress_level'] = clamp_numeric('compress_level', value)

    @property
    def n_fft_denoise(self):
        return self._get_validated('n_fft_denoise')

    @n_fft_denoise.setter
    def n_fft_denoise(self, value):
        if self._validate_set('n_fft_denoise', value):
            self._defaults['n_fft_denoise'] = clamp_numeric('n_fft_denoise', value)

    @property
    def fade_ms_trail(self):
        return self._get_validated('fade_ms_trail')

    @fade_ms_trail.setter
    def fade_ms_trail(self, value):
        if self._validate_set('fade_ms_trail', value):
            self._defaults['fade_ms_trail'] = clamp_numeric('fade_ms_trail', value)

    @property
    def trailing_silence_db(self):
        return self._get_validated('trailing_silence_db')

    @trailing_silence_db.setter
    def trailing_silence_db(self, value):
        if self._validate_set('trailing_silence_db', value):
            self._defaults['trailing_silence_db'] = clamp_numeric('trailing_silence_db', value)

    @property
    def fuzzy_artifact_threshold_hz(self):
        return self._get_validated('fuzzy_artifact_threshold_hz')

    @fuzzy_artifact_threshold_hz.setter
    def fuzzy_artifact_threshold_hz(self, value):
        if self._validate_set('fuzzy_artifact_threshold_hz', value):
            self._defaults['fuzzy_artifact_threshold_hz'] = clamp_numeric('fuzzy_artifact_threshold_hz', value)

    @property
    def denoise_highpass_hz(self):
        return self._get_validated('denoise_highpass_hz')

    @denoise_highpass_hz.setter
    def denoise_highpass_hz(self, value):
        if self._validate_set('denoise_highpass_hz', value):
            self._defaults['denoise_highpass_hz'] = clamp_numeric('denoise_highpass_hz', value)

    @property
    def denoise_median_ksize(self):
        return self._get_validated('denoise_median_ksize')

    @denoise_median_ksize.setter
    def denoise_median_ksize(self, value):
        if self._validate_set('denoise_median_ksize', value):
            self._defaults['denoise_median_ksize'] = clamp_numeric('denoise_median_ksize', value)

    @property
    def denoise_target_band_low(self):
        return self._get_validated('denoise_target_band_low')

    @denoise_target_band_low.setter
    def denoise_target_band_low(self, value):
        if self._validate_set('denoise_target_band_low', value):
            self._defaults['denoise_target_band_low'] = clamp_numeric('denoise_target_band_low', value)

    @property
    def denoise_target_band_high(self):
        return self._get_validated('denoise_target_band_high')

    @denoise_target_band_high.setter
    def denoise_target_band_high(self, value):
        if self._validate_set('denoise_target_band_high', value):
            self._defaults['denoise_target_band_high'] = clamp_numeric('denoise_target_band_high', value)

    @property
    def enable_deferred_cleanup(self):
        return self._flags.get('enable_deferred_cleanup', DEFAULTS.get('enable_deferred_cleanup', True))

    @enable_deferred_cleanup.setter
    def enable_deferred_cleanup(self, value):
        if self._validate_set('enable_deferred_cleanup', value):
            self._flags['enable_deferred_cleanup'] = parse_bool(value)

    @property
    def enable_audio_padding(self):
        return self._flags.get('enable_audio_padding', DEFAULTS.get('enable_audio_padding', False))

    @enable_audio_padding.setter
    def enable_audio_padding(self, value):
        if self._validate_set('enable_audio_padding', value):
            self._flags['enable_audio_padding'] = parse_bool(value)

    @property
    def logging_level(self):
        return self._strings.get('logging_level', DEFAULTS.get('logging_level', 'INFO'))

    @logging_level.setter
    def logging_level(self, value):
        if self._validate_set('logging_level', value):
            self._strings['logging_level'] = str(value)

    @property
    def tts_name(self):
        return self._strings.get('tts_name', DEFAULTS.get('tts_name', 'default'))

    @tts_name.setter
    def tts_name(self, value):
        if self._validate_set('tts_name', value):
            self._strings['tts_name'] = str(value)

    @property
    def fuzzy_boost_words(self):
        return self._get_validated('fuzzy_boost_words')

    @fuzzy_boost_words.setter
    def fuzzy_boost_words(self, value):
        if self._validate_set('fuzzy_boost_words', value):
            self._defaults['fuzzy_boost_words'] = parse_list(value)
            self._invalidate_merged_cache()


    @property
    def is_modified(self) -> bool:
        return self._is_modified

    @is_modified.setter
    def is_modified(self, value: bool):
        self._is_modified = bool(value)

    def get_value(self, param_name: str, api_value: Any = None, default: Any = None, bypass_config: bool = False) -> Any:
        """Get: Properties first, then direct attrs (cores), then _defaults."""
        if bypass_config or self._use_api_mode:
            fallback = {
                'temperature': 0.7, 'min_p': 0.07, 'top_p': 1.0, 'repetition_penalty': 2.0,
                'cfg_weight': 0.45, 'exaggeration': 0.7, 'hop_length': 256, 'n_fft': 1024,
                'fuzzy_threshold': 0.75, 'fuzzy_min_length': 3, 'fuzzy_boost_amount': 0.1,
                'fuzzy_boost_words': ['ahh', 'mmm', 'ooh', 'gasp'], 'speaking_rate': 1.0,
                'eq_gain_db': 0.0, 'eq_cutoff_hz': 100, 'enable_disk_cache': True, 'enable_memory_cache': True
            }
            val = api_value if api_value is not None else fallback.get(param_name, default or 0.0)
            if param_name == 'fuzzy_boost_words' and isinstance(val, str):
                val = parse_list(val)
            return val

        # Property first
        if hasattr(self.__class__, param_name) and isinstance(getattr(type(self), param_name, None), property):
            return getattr(self, param_name)
        # Direct core attr
        if hasattr(self, param_name) and param_name not in DEFAULTS:  # Skip tunables
            return getattr(self, param_name)
        # Fallback to validated default
        return self._get_validated(param_name) or default

    def set_value(self, param_name: str, value: Any) -> bool:
        """Set: Property for tunables, direct for cores. Direct global sync for flags/cores."""
        if self._validate_set(param_name, value):
            if hasattr(self.__class__, param_name) and isinstance(getattr(type(self), param_name, None), property):
                try:
                    setattr(self, param_name, value)  # Triggers setter
                except Exception as e:
                    logger.warning(f"Property set failed for {param_name}: {e}")
                    return False
            else:
                setattr(self, param_name, value)  # Direct for cores
            self._is_modified = True
            self._invalidate_merged_cache()
            # Direct global sync for specific (no full _sync_globals to avoid recursion)
            if param_name in ['device', 'dtype', 'model', 'multilingual', '_use_api_mode']:
                sync_cores_to_globals(self)
            elif param_name in self._flags:
                globals()[param_name.upper().replace('_', '_')] = self._flags[param_name]  # e.g., ENABLE_DISK_CACHE
            elif param_name == 'fuzzy_cache_limit':
                globals()['FUZZY_CACHE_LIMIT'] = self._defaults[param_name]
            return True
        return False

    @property
    def audio_defaults(self) -> Dict[str, Any]:
        """Audio params subset (merged globals; enforce sr=24000)."""
        if self._audio_defaults is None or self._global_hash is None:
            self._audio_defaults = {
                'enable_post_processing': self.get_value('enable_post_processing', default=True),
                'enable_post_resample': self.get_value('enable_post_resample', default=False),
                'enable_post_jit_gain': self.get_value('enable_post_jit_gain', default=True),
                'enable_post_voice_processing': self.get_value('enable_post_voice_processing', default=True),
                'enable_pre_adjustment': self.get_value('enable_pre_adjustment', default=True),
                'speaking_rate': self.get_value('speaking_rate', default=1.0),
                'eq_gain_db': self.get_value('eq_gain_db', default=0.0),
                'eq_cutoff_hz': self.get_value('eq_cutoff_hz', default=100),
                'gain_max_limit': self.get_value('gain_max_limit', default=10.0),
                'gain_target_max': self.get_value('gain_target_max', default=-10.0),
                'noise_floor_db': self.get_value('noise_floor_db', default=-60.0),
                'trim_threshold_db': self.get_value('trim_threshold_db', default=-50.0),
                'notch_gain_db': self.get_value('notch_gain_db', default=-20.0),
                'notch_low_hz': self.get_value('notch_low_hz', default=100.0),
                'notch_high_hz': self.get_value('notch_high_hz', default=200.0),
                'n_fft': self.get_value('n_fft', default=1024),
                'hop_length': self.get_value('hop_length', default=256),
                'fade_ms': self.get_value('fade_ms', default=10.0),
                'normalize_method': self.get_value('normalize_method', default='rms'),
                'enable_denoise_normalize': self.get_value('enable_denoise_normalize', default=False),
                'enable_denoising': self.get_value('enable_denoising', default=True),
                'enable_audio_padding': self.get_value('enable_audio_padding', default=False),
                'base_audio_pad_sec': self.get_value('base_audio_pad_sec', default=0.5),
                'tiny_audio_pad_multiplier': self.get_value('tiny_audio_pad_multiplier', default=2.0),
                'tiny_threshold_sec': self.get_value('tiny_threshold_sec', default=0.1),
                'sr': self.sr,  # Enforced 24000
            }
        return self._audio_defaults

    def get_merged_audio_params(self, voice_name: Optional[str] = None, api_overrides: Optional[Dict] = None) -> Dict[str, Any]:
        """Merge audio_defaults → voice.json → API; with LRU cache (purge on changes)."""
        if voice_name is None:
            voice_name = 'default'
        params = self.audio_defaults.copy()
        params['voice_name'] = voice_name

        # LRU Key: voice + hashes for invalidation
        global_key = hashlib.md5(str(sorted(self._defaults.items())).encode()).hexdigest()
        voice_params = self.voice_overrides.get(voice_name, {})
        voice_key = hashlib.md5(str(sorted(voice_params.items())).encode()).hexdigest()
        cache_key = (voice_name, global_key, voice_key)

        if self._global_hash != global_key:
            self._merged_cache.clear()
            self._global_hash = global_key

        voice_hash = self._voice_hashes.get(voice_name)
        if voice_hash != voice_key:
            self._merged_cache.pop(cache_key, None)
            self._voice_hashes[voice_name] = voice_key

        if cache_key in self._merged_cache:
            logger.debug(f"LRU cache hit for {voice_name}")
            return self._merged_cache[cache_key]

        # Compute merged
        if voice_params:
            params.update(voice_params)
        if api_overrides:
            params.update(api_overrides)

        # Cache it
        self._merged_cache[cache_key] = params
        logger.debug(f"Merged params for {voice_name} (LRU miss; cached)")
        return params

    def get_voice_parameters(self, voice_name: str) -> Dict[str, Any]:
        return self.voice_overrides.get(voice_name, {})

    def get_all_voices(self) -> List[str]:
        return list(self.voice_overrides.keys())

    def _load_voices_json(self) -> Dict[str, Dict]:
        """Class wrapper for parser_json (instance path)."""
        return _load_voices_json()

    def _sync_model_if_loaded(self):
        """Sync model from ModelManager if loaded."""
        try:
            from src.model import ModelManager
            model_type = 'multilingual' if self.multilingual else 'english'
            manager = ModelManager.get_instance()
            if manager.is_loaded(model_type):
                self.model = manager.get_model(model_type)
                logger.debug(f"Synced model {model_type} to config")
            else:
                self.model = None
        except ImportError:
            self.model = None
        except Exception as e:
            logger.warning(f"Model sync failed: {e}")
            self.model = None

    def load_model(self, model_type: Optional[str] = None, *args, **kwargs) -> bool:
        """Load TTS model via ModelManager."""
        model_type = model_type or ('multilingual' if self.multilingual else 'english')
        try:
            from src.model import ModelManager
            global DEVICE, DTYPE
            DEVICE = str(self.device)
            DTYPE = self.dtype
            success = ModelManager.get_instance().load_model(model_type, *args, **kwargs)
            if success:
                self.model = ModelManager.get_instance().get_model(model_type)
                self._sync_model_if_loaded()
                logger.info(f"Model loaded: {model_type} on {self.device}")
                sync_cores_to_globals(self)
                return True
            return False
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            return False

# Facade Functions (Backward-compat; top-level, delegate to CONFIG; lazy init)
def load_skyrimnet_config():
    """Old API: Load config (lazy init if needed)."""
    global CONFIG
    if CONFIG is None:
        CONFIG = Config()
    if CONFIG._config_cache is None:
        CONFIG.load_config()
    # Return old tuple for compat
    default_config = CONFIG._defaults.copy()
    global_flags = {'enable_memory_cache': CONFIG.enable_memory_cache, 'enable_disk_cache': CONFIG.enable_disk_cache}
    CONFIG._config_cache = (default_config, {'default': 'default'}, global_flags)
    return CONFIG._config_cache

def get_config_value(param_name: str, api_value: Any = None, defaults: Dict = None, modes: Dict = None, bypass_config: bool = False) -> Any:
    """Old API: Get value (lazy load if needed)."""
    load_skyrimnet_config()
    return CONFIG.get_value(param_name, api_value, default=defaults.get(param_name) if defaults else None, bypass_config=bypass_config)

def reload_config():
    """Old API: Reload (lazy load if needed)."""
    load_skyrimnet_config()
    CONFIG.reload_config()

# Lazy global instance (no auto-load on import)
CONFIG = None