"""
Main Config Module: Single-file JSON config with Pydantic validation.
Handles globals + nested voices in config.json. Relies purely on models.py defaults (no hardcodes).
READY: get_merged_audio_params removed; get_voice_params is sole merger (with caching).
"""
import os
import shutil
import threading
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import OrderedDict
import hashlib
from loguru import logger
import torch
from pydantic import BaseModel, ValidationError

from .models import AppConfig, VoiceConfig, CAPS  # Pydantic models with defaults

# Singleton Config
_lock = threading.Lock()
_config_file_path = Path("config.json")  # Single file at root

def get_config():
    """Facade: Get Config instance (lazy init)."""
    with _lock:
        if not hasattr(get_config, '_instance') or get_config._instance is None:
            get_config._instance = Config()
            get_config._instance.load_config()
        return get_config._instance

def load_skyrimnet_config():
    """Facade: Load config from single JSON (lazy init). Call get_config() for instance."""
    return get_config()

def get_config_value(key: str, default: Any = None, api_value: Any = None, bypass_config: bool = False) -> Any:
    """Facade: Get value (lazy load). Supports nested keys like 'globals.tts.temperature'."""
    config = get_config()
    return config.get_value(key, default, api_value, bypass_config)

def reload_config():
    """Facade: Reload from JSON (clears caches)."""
    config = get_config()
    config.reload_config()

class Config:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()  # Sets app_config=None, etc.
            return cls._instance

    def _initialize(self):
        """Init empty (models populated on load)."""
        self.app_config: Optional[AppConfig] = None  # Top-level model
        self._config_cache: Optional[dict] = None  # Cached JSON for reload
        self._merged_cache: OrderedDict = OrderedDict(maxlen=100)
        self._global_hash: Optional[str] = None
        self._voice_hashes: Dict[str, str] = {}
        self._is_modified = False
        self._is_initialized = False
        logger.debug("Config initialized (Pydantic single-JSON)")

    def load_config(self):
        """Load single config.json → validate → populate models (relies on models.py defaults)."""
        if getattr(self, '_loading_in_progress', False):
            logger.warning("load_config called recursively; skipping to avoid loop")
            return True

        self._loading_in_progress = True

        try:
            if self._config_cache is not None:
                json_data = self._config_cache
            else:
                config_path = _config_file_path
                try:
                    with open(config_path, 'r') as f:
                        json_data = json.load(f)
                    logger.info(f"Loaded config.json: {config_path}")
                except FileNotFoundError:
                    logger.warning(f"config.json not found ({config_path}); using models.py defaults")
                    json_data = self._get_default_json()
                    self.save_config(create_backup=False)  # Save pure defaults
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid config.json ({config_path}): {e} – using models.py defaults")
                    json_data = self._get_default_json()
                    self.save_config(create_backup=False, filename=config_path)  # Overwrite invalid

            self.app_config = AppConfig.model_validate(json_data)

            # Coerce globals (device/dtype) post-validate
            globals_ = self.app_config.globals
            globals_.device = str(globals_.device).lower()
            if globals_.device == 'cuda' and not torch.cuda.is_available():
                globals_.device = 'cpu'
                logger.info("CUDA unavailable; forced CPU mode")
            if isinstance(globals_.dtype, str):
                globals_.dtype = torch.bfloat16 if globals_.device == 'cuda' else torch.float32
                logger.debug(f"Dtype coerced to {globals_.dtype}")

            # Cache raw JSON
            self._config_cache = json_data

        except ValidationError as e:
            logger.error(f"Config validation failed: {e} – using models.py defaults")
            # Fallback: Pure model creation (triggers Field defaults from models.py)
            self.app_config = AppConfig()

            # Coerce on fallback
            globals_ = self.app_config.globals
            globals_.device = 'cpu' if not torch.cuda.is_available() else 'cuda'
            globals_.dtype = torch.float32  # Minimal; models.py handles rest

            # Cache the defaulted model dump
            self._config_cache = self.app_config.model_dump()

            # Save once (no recursion)
            self.save_config(create_backup=False)

        finally:
            self._loading_in_progress = False

        # AFTER validation
        if self.app_config:
            try:
                _ = self.app_config.globals.root  # Trigger property
                logger.info(f"Project root: {self.app_config.globals.root}")
                logger.info(f"Cache root: {self.app_config.globals.cache_dir}")
            except Exception as e:
                logger.error(f"Directory initialization failed: {e}")
                if not self.app_config.globals.root:
                    self.app_config.globals.root = Path.cwd()
                if not self.app_config.globals.cache_dir:
                    self.app_config.globals.cache_dir = self.app_config.globals.root / "cache_fallback"

        # Post-load
        self._invalidate_merged_cache()
        self._is_modified = False

        # Log
        globals_config = self.app_config.globals
        voices_config = self.app_config.voices
        voices_count = len(voices_config)
        voices_keys = list(voices_config.keys())
        logging_level = globals_config.logging_level or 'INFO'
        logger.info(
            f"Config loaded: {voices_count} voices (keys: {voices_keys}), "
            f"logging_level={logging_level}; device={globals_config.device}; "
            f"dtype={globals_config.dtype}; multilingual={globals_config.multilingual}")

        self._is_initialized = True
        return True

    def save_config(self, create_backup: bool = True, filename: str = 'config.json') -> bool:
        """Save AppConfig to JSON (exclude runtime model)."""
        try:
            if getattr(self, '_saving_in_progress', False):
                logger.warning("save_config called recursively; skipping to avoid loop")
                return False
            self._saving_in_progress = True

            if create_backup and os.path.exists(filename):
                backup = filename + '.backup'
                shutil.copy2(filename, backup)
                logger.debug(f"Config backup created: {backup}")

            # Exclude non-serializable: globals.model
            dump_kwargs = {
                'indent': 2,
                'default': lambda o: f"{type(o).__name__}({str(o)})" if hasattr(o, '__dict__') else str(o),
            }

            if self.app_config is None:
                logger.error("Cannot save: app_config is None (load first via load_config)")
                return False

            # Dump model to plain dict, excluding non-serializable fields and omitting None values
            # exclude_none=True ensures we don't persist nulls (especially under voices overrides)
            config_dict = self.app_config.model_dump(exclude={'globals': {'model'}}, exclude_none=True)
            if 'model' in config_dict.get('globals', {}):
                del config_dict['globals']['model']

            # Use json.dumps for formatting
            json_data = json.dumps(config_dict, **dump_kwargs, ensure_ascii=False)

            with open(filename, 'w', encoding='utf-8') as f:
                f.write(json_data)

            logger.info(f"Config saved: {filename}")
            self._is_modified = False
            return True

        except Exception as e:
            logger.error(f"Save failed: {e}")
            # Auto-repair: Load pure defaults from models.py
            self.app_config = AppConfig()
            globals_ = self.app_config.globals
            globals_.device = 'cpu' if not torch.cuda.is_available() else 'cuda'
            globals_.dtype = torch.float32
            logger.warning("Auto-repair: Loaded models.py defaults (no emergency save to avoid recursion)")
            return False

        finally:
            self._saving_in_progress = False

    def reload_config(self):
        """Reload: Clears caches; re-loads JSON."""
        self._config_cache = None  # Force re-load
        self._invalidate_merged_cache()  # Purge LRU
        self.load_config()  # Includes validation
        logger.info("Config reloaded from single JSON")

    def get_value(self, key: str, default: Any = None, api_value: Any = None, bypass_config: bool = False) -> Any:
        """Get value: From app_config (supports nested like 'globals.tts.temperature').
        REFACROED: Pure chained getattr on Pydantic models (relies on models.py defaults; no fallbacks)."""
        if self.app_config is None:
            raise ValueError("Config not loaded; call load_config() first")

        # API mode: Minimal, no hardcodes (defaults from caller or None)
        if bypass_config or getattr(self.app_config.globals, 'use_api_mode', False):
            val = api_value if api_value is not None else default
            if key == 'fuzzy_boost_words' and isinstance(val, str):
                val = [w.strip().lower() for w in val.split(',') if w.strip()]
            logger.trace(f"API mode value for '{key}': {val}")
            return val

        # REFACROED: Chained getattr traversal on model instance (fast, direct access)
        val = self.app_config
        parts = key.split('.')
        try:
            for part in parts:
                val = getattr(val, part)
                if val is None:
                    logger.trace(f"None value for '{part}' in {key}; returning default")
                    return default
            logger.trace(f"Retrieved '{key}': {val}")
            return val
        except AttributeError as e:
            logger.trace(f"AttributeError on '{key}' traversal: {e}; returning default")
            return default

    def set_value(self, key: str, value: Any, voice: Optional[str] = None) -> bool:
        """Set value: To globals or specific voice (via model_copy for immutability)."""
        if self.app_config is None:
            raise ValueError("Config not loaded; call load_config() first")

        try:
            if voice:
                # Per-voice (flat overrides; update via model_copy)
                old_voice = self.app_config.voices.get(voice)
                if old_voice:
                    update = {key: value}
                    new_voice = old_voice.model_copy(update=update)
                    self.app_config.voices[voice] = new_voice
                    self._voice_hashes.pop(voice, None)
                else:
                    logger.warning(f"Voice {voice} not found; creating with {key}={value}")
                    self.app_config.voices[voice] = VoiceConfig(**{key: value})
            else:
                # Global: Traverse nested (e.g., 'tts.temperature' → globals.tts.temperature)
                if '.' in key:
                    parts = key.split('.')
                    if parts[0] != 'globals':
                        logger.warning(f"Global keys must start with 'globals.' (got: {key})")
                        return False
                    parts = parts[1:]  # Strip 'globals'
                    target = self.app_config.globals
                    for part in parts[:-1]:
                        if hasattr(target, part):
                            target = getattr(target, part)
                        else:
                            logger.warning(f"Invalid nested path {key}")
                            return False
                    last_part = parts[-1]
                    if hasattr(target, last_part):
                        update = {last_part: value}
                        new_target = target.model_copy(update=update)
                        # Reassign up chain (for sub-models like fuzzy within globals)
                        parent_parts = parts[:-1]
                        parent = self.app_config.globals
                        for p in parent_parts:
                            parent = getattr(parent, p)
                        setattr(parent, last_part, new_target)
                    else:
                        logger.warning(f"Unknown field {last_part} in {key}")
                        return False
                else:
                    # Direct global
                    if hasattr(self.app_config.globals, key):
                        update = {key: value}
                        self.app_config.globals = self.app_config.globals.model_copy(update=update)
                    else:
                        logger.warning(f"Unknown global key {key}")
                        return False

                self._invalidate_merged_cache()

            self._is_modified = True
            return True
        except ValidationError as e:
            logger.warning(f"Set failed for {key}={value} (voice={voice}): {e}")
            return False
        except AttributeError:
            logger.warning(f"Unknown param {key} (voice={voice})")
            return False

    # Comprehensive PARAM_SPECS (based on models.py: tts/audio/fuzzy; expand as needed)
    PARAM_SPECS = {
        # TTS (from TtsConfig)
        'temperature': float, 'exaggeration': float, 'top_p': float, 'min_p': float,
        'repetition_penalty': float, 'cfg_weight': float, 'max_new_tokens': int, 'min_new_tokens': int,
        'max_cache_len': int, 'stride_length': int, 'compile_t3': bool, 'warmup_t3': bool,
        # Audio (from AudioConfig)
        'speaking_rate': float, 'enable_post_processing': bool, 'enable_post_resample': bool,
        'enable_post_jit_gain': bool, 'enable_post_voice_processing': bool, 'enable_pre_adjustment': bool,
        'eq_gain_db': float, 'eq_cutoff_hz': float, 'notch_gain_db': float, 'notch_low_hz': float,
        'notch_high_hz': float, 'fade_ms': float, 'normalize_method': str, 'gain_max_limit': float,
        'noise_floor_db': float, 'trim_threshold_db': float, 'enable_denoise_normalize': bool,
        'enable_denoising': bool, 'enable_audio_padding': bool, 'base_audio_pad_sec': float,
        'tiny_audio_pad_multiplier': float, 'tiny_threshold_sec': float, 'n_fft': int, 'hop_length': int,
        'highpass_cutoff_hz': float, 'n_fft_denoise': int, 'denoise_median_ksize': int,
        'denoise_target_band_low': float, 'denoise_target_band_high': float, 'trailing_silence_db': float,
        'gain_target_max': float, 'max_gain': float, 'n_mels': int, 'ebu_post_gain_db': float,
        'ebu_true_peak': float,
        # Fuzzy (from FuzzyConfig)
        'enable_fuzzy_cache': bool, 'fuzzy_threshold': float, 'fuzzy_boost_amount': float,
        'fuzzy_index_size': int, 'fuzzy_artifact_threshold_hz': float,
        # Core globals
        'sr': int, 'language_id': str, 'voice_name': str,
        # Pre-text padding controls (Audio/Voice)
        'short_padding_threshold': int, 'short_padding_token': str,
        'enable_text_padding': bool, 'text_ellipses_count': int
    }

    # REFACTORED: Delegate to canonical merger in src.voice_params
    def get_voice_params(self, voice_name: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return merged voice parameters using the canonical merger (single source of truth).

        Precedence is handled in `src.voice_params.get_voice_params`:
        models defaults < globals (tts/audio) < voice overrides < explicit overrides

        This method adds a light LRU cache keyed by globals/voice hashes and overrides signature.
        """
        # Safety check
        if not hasattr(self, '_is_initialized') or not self._is_initialized or self.app_config is None:
            logger.warning("Config not initialized – creating pure models.py defaults")
            app_config = AppConfig()  # Triggers Field defaults
            # Use canonical merger directly to obtain sane defaults
            try:
                from src.voice_params import get_voice_params as _merge_voice_params
                return _merge_voice_params(app_config, voice_name or 'default', overrides)
            except Exception:
                return app_config.globals.model_dump()

        voice_name = voice_name or 'default'

        # LRU caching (efficient hash on globals + voice + overrides)
        globals_dump = self.app_config.globals.model_dump()
        global_hash = hashlib.md5(str(sorted(globals_dump.items())).encode()).hexdigest()
        voice = self.app_config.voices.get(voice_name, VoiceConfig())  # Default if missing
        voice_dump = voice.model_dump()
        voice_key = hashlib.md5(str(sorted(voice_dump.items())).encode()).hexdigest()
        # Include overrides signature (order-insensitive) in the cache key
        overrides_sig = None
        if overrides:
            try:
                overrides_sig = hashlib.md5(str(sorted(overrides.items())).encode()).hexdigest()
            except Exception:
                overrides_sig = str(len(overrides))
        # Bumpable schema version to invalidate cache when merger logic changes
        schema_version = 'v3'
        cache_key = (voice_name, global_hash, voice_key, overrides_sig, schema_version)

        if cache_key in self._merged_cache:
            logger.debug(f"LRU cache hit for {voice_name}")
            return self._merged_cache[cache_key]

        # Delegate to canonical merger
        try:
            from src.voice_params import get_voice_params as _merge_voice_params
            params = _merge_voice_params(self.app_config, voice_name, overrides)
        except Exception as e:
            logger.warning(f"Delegation to voice_params merger failed: {e}; falling back to internal resolution")
            # Fallback to internal flat resolution if import fails
            params = {}
            for param, param_type in self.PARAM_SPECS.items():
                params[param] = self._get_nested_value(self.app_config.globals, param, param_type)
            if voice_name != 'default' and voice_name in self.app_config.voices:
                voice = self.app_config.voices[voice_name]
                for param, param_type in self.PARAM_SPECS.items():
                    value = self._get_nested_value(voice, param, param_type)
                    if value is not None:
                        params[param] = value
            params['voice_name'] = voice_name
            if overrides:
                for param, value in overrides.items():
                    if param in self.PARAM_SPECS:
                        param_type = self.PARAM_SPECS[param]
                        try:
                            if param_type == bool:
                                params[param] = bool(value)
                            elif param_type == int:
                                params[param] = int(float(value))
                            elif param_type == float:
                                params[param] = float(value)
                            else:
                                params[param] = str(value)
                        except Exception:
                            pass

        # FIXED: Cache the result
        self._merged_cache[cache_key] = params
        self._global_hash = global_hash
        self._voice_hashes[voice_name] = voice_key
        logger.debug(f"Merged params for {voice_name} (LRU miss; cached)")
        return params

    def _get_nested_value(self, obj: Any, param: str, param_type: type, default: Any = None) -> Any:
        """Helper: Get nested value via chained getattr (for params; type coercion).

        Enhanced: If the immediate attribute is not present on a Globals object,
        look inside known sub-models (tts, audio, fuzzy) for a field with the same name.
        This allows flat PARAM_SPECS keys to resolve nested config fields without
        forcing dotted names.
        """
        val = obj
        # Handle nested params (e.g., 'fuzzy.threshold' – split if needed, but PARAM_SPECS are flat)
        parts = [param]  # Most are flat; if nested, split (e.g., for 'tts.temperature')
        if '.' in param:
            parts = param.split('.')
            # Delegate to sub-model (e.g., globals.tts.temperature → getattr(globals, 'tts').temperature)
            for part in parts[:-1]:
                if hasattr(val, part):
                    val = getattr(val, part)
                else:
                    return default

        param = parts[-1]  # Last part
        if hasattr(val, param):
            val = getattr(val, param)
            if val is None:
                return default
        elif isinstance(val, dict):
            val = val.get(param)
            if val is None:
                return default
        else:
            # Enhanced: If we're at a Globals object, search within tts/audio/fuzzy sub-models
            try:
                # Only attempt if obj looks like a Globals model (has these attributes)
                sub_candidates = []
                for sub_name in ('tts', 'audio', 'fuzzy'):
                    if hasattr(obj, sub_name):
                        sub_candidates.append(getattr(obj, sub_name))
                for sub in sub_candidates:
                    if hasattr(sub, param):
                        sub_val = getattr(sub, param)
                        return default if sub_val is None else sub_val
                    elif isinstance(sub, dict) and param in sub:
                        sub_val = sub.get(param)
                        return default if sub_val is None else sub_val
            except Exception:
                pass
            return default

        try:
            if param_type == bool:
                return bool(val)
            elif param_type == int:
                return int(float(val))
            elif param_type == float:
                return float(val)
            else:
                return str(val)
        except:
            logger.debug(f"Type coercion failed for {param}; using default {default}")
            return default

    def get_voice_parameters(self, voice_name: str) -> Dict[str, Any]:
        """Get full voice config (flat dump)."""
        if self.app_config is None:
            raise ValueError("Config not loaded")
        voice = self.app_config.voices.get(voice_name)
        return voice.model_dump() if voice else {}

    def get_all_voices(self) -> List[str]:
        """Get list of voice names."""
        if self.app_config is None:
            raise ValueError("Config not loaded")
        return list(self.app_config.voices.keys())

    def _get_default_json(self):
        """Default JSON dict (triggers Pydantic models.py defaults)."""
        default_app = AppConfig()  # Uses Field defaults from models.py
        # Minimal overrides (device/dtype from torch)
        default_app.globals.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        default_app.globals.dtype = torch.bfloat16 if default_app.globals.device == 'cuda' else torch.float32
        return default_app.model_dump()

    def _invalidate_merged_cache(self):
        """Purge LRU on changes (shared for get_voice_params)."""
        self._merged_cache.clear()
        self._global_hash = None
        self._voice_hashes.clear()