"""
Main Config Module: Single-file JSON config with Pydantic validation.
Handles globals + nested voices in config.json.
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

from .models import AppConfig, VoiceConfig, CAPS  # CAPS for any fallbacks

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
        self._is_initialized = False  # NEW: Initialize flag
        logger.debug("Config initialized (Pydantic single-JSON)")

    def load_config(self):
        """Load single config.json → validate → populate models."""
        if self._config_cache is not None:
            # Reload if cached (re-validate)
            json_data = self._config_cache
        else:
            # Load JSON
            config_path = _config_file_path
            try:
                with open(config_path, 'r') as f:
                    json_data = json.load(f)
                logger.info(f"Loaded config.json: {config_path}")
            except FileNotFoundError:
                logger.warning(f"config.json not found ({config_path}); using defaults")
                json_data = self._get_default_json()
                # Save defaults as initial
                self.save_config(create_backup=False)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid config.json ({config_path}): {e} – using defaults")
                json_data = self._get_default_json()
                self.save_config(create_backup=False)  # Overwrite invalid

        try:
            # Validate/load to AppConfig (top-level model)
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

            # Cache the validated JSON (for reload)
            self._config_cache = json_data

        except ValidationError as e:
            logger.error(f"Config validation failed: {e}")
            # Fallback: Defaults
            self.app_config = AppConfig()

            # Coerce on fallback
            globals_ = self.app_config.globals
            globals_.device = 'cpu' if not torch.cuda.is_available() else 'cuda'
            globals_.dtype = torch.float32  # Safe fallback

            self._config_cache = self.app_config.model_dump()  # Updated with coercion
            self.save_config(create_backup=False)
            logger.warning("Loaded default config.json (with coercion)")


        # AFTER validation but before caching
        if self.app_config:
            try:
                # Force directory resolution/validation
                _ = self.app_config.globals.root  # Trigger property
                logger.info(f"Project root: {self.app_config.globals.root}")
                logger.info(f"Cache root: {self.app_config.globals.cache_dir}")
            except Exception as e:
                logger.error(f"Directory initialization failed: {e}")
                # Critical failure handling
                if not self.app_config.globals.root:
                    self.app_config.globals.root = Path.cwd()
                if not self.app_config.globals.cache_dir:
                    self.app_config.globals.cache_dir = self.app_config.globals.root / "cache_fallback"

        # Post-load: Setup caches
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

    # In src/config/config.py: save_config (around line 152)
    def save_config(self, create_backup: bool = True, filename: str = 'config.json') -> bool:
        """Save AppConfig to JSON (exclude runtime model)."""
        try:
            if create_backup and os.path.exists(filename):
                backup = filename + '.backup'
                shutil.copy2(filename, backup)
                logger.debug(f"Config backup created: {backup}")

            # Exclude non-serializable: globals.model (runtime object)
            dump_kwargs = {
                'indent': 2,
                'default': lambda o: f"{type(o).__name__}({str(o)})" if hasattr(o, '__dict__') else str(o),
                # Fallback for unknowns
                'exclude': {'app_config': {'globals': {'model'}}}  # Skip model field
            }

            # Guard: Ensure app_config loaded
            if self.app_config is None:
                logger.error("Cannot save: app_config is None (load first via load_config)")
                return False

            json_data = self.app_config.model_dump(**dump_kwargs)
            if 'model' in json_data.get('globals', {}):  # Double-check exclusion
                del json_data['globals']['model']

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Config saved: {filename}")
            return True

        except Exception as e:
            logger.error(f"Save failed: {e}")
            # Auto-repair: Load defaults if corrupt
            self._app_config = AppConfig.model_validate({})  # Empty dict → defaults
            self.save_config(filename=filename + '.emergency')  # Save to alternate
            return False


    def reload_config(self):
        """Reload: Clears caches; re-loads JSON."""
        self._config_cache = None  # Force re-load
        self._merged_cache.clear()
        self._global_hash = None
        self._voice_hashes.clear()
        self.load_config()  # Includes validation and coercion
        logger.info("Config reloaded from single JSON")

    def get_value(self, key: str, default: Any = None, api_value: Any = None, bypass_config: bool = False) -> Any:
        """Get value: From app_config (supports nested like 'globals.tts.temperature')."""
        if self.app_config is None:
            raise ValueError("Config not loaded; call load_config() first")

        if bypass_config or getattr(self.app_config.globals, 'use_api_mode', False):
            # Simple API fallback (hardcoded defaults)
            fallback = {
                'temperature': 0.7, 'min_p': 0.07, 'top_p': 1.0, 'repetition_penalty': 2.0,
                'cfg_weight': 0.45, 'exaggeration': 0.7, 'speaking_rate': 1.0,
                'enable_disk_cache': True, 'enable_memory_cache': True,
                'fuzzy_boost_words': ['ahh', 'mmm', 'ooh', 'gasp']
            }
            val = api_value if api_value is not None else fallback.get(key, default)
            if key == 'fuzzy_boost_words' and isinstance(val, str):
                val = [w.strip().lower() for w in val.split(',') if w.strip()]  # Parse list
            return val

        # Delegate: Flat dump first
        dump = self.app_config.model_dump()
        if '.' not in key:
            return dump.get(key, default)

        # Nested: Traverse (e.g., 'globals.tts.temperature')
        parts = key.split('.')
        val = dump
        for part in parts:
            if isinstance(val, dict):
                val = val.get(part)
            elif hasattr(val, part):
                val = getattr(val, part)
                if isinstance(val, BaseModel):
                    val = val.model_dump()  # Flatten sub-model
            else:
                return default
            if val is None:
                return default
        return val


    def set_value(self, key: str, value: Any, voice: Optional[str] = None) -> bool:
        """Set value: To globals or specific voice (via model_copy for immutability)."""
        if self.app_config is None:
            raise ValueError("Config not loaded; call load_config() first")

        try:
            if voice:
                # Per-voice (update nested; assumes key is flat override)
                old_voice = self.app_config.voices.get(voice)
                if old_voice:
                    update = {key: value}
                    new_voice = old_voice.model_copy(update=update)
                    self.app_config.voices[voice] = new_voice
                    # Invalidate voice cache
                    self._voice_hashes.pop(voice, None)
                else:
                    logger.warning(f"Voice {voice} not found; creating with {key}={value}")
                    self.app_config.voices[voice] = VoiceConfig(**{key: value})
            else:
                # Global: Nested if needed (e.g., key='tts.temperature' → globals.tts.temperature)
                if '.' in key:
                    # Traverse and update
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
                        # Reassign up the chain (simple for direct sub; for deeper, recursive update needed)
                        # For now, assume direct sub like 'globals.tts'; set parent attr
                        if len(parts) == 1:
                            setattr(self.app_config.globals, last_part, new_target)
                        else:
                            # Basic reassign (e.g., for 'audio.filter_q', set globals.audio = new_audio)
                            parent_parts = parts[:-1]
                            parent = self.app_config.globals
                            for p in parent_parts:
                                parent = getattr(parent, p)
                            setattr(parent, last_part, new_target)
                    else:
                        logger.warning(f"Unknown field {last_part} in {key}")
                        return False
                else:
                    # Direct global (error if not in globals)
                    if hasattr(self.app_config.globals, key):
                        update = {key: value}
                        self.app_config.globals = self.app_config.globals.model_copy(update=update)
                    else:
                        logger.warning(f"Unknown global key {key}")
                        return False

                # Invalidate on change
                self._invalidate_merged_cache()

            self._is_modified = True
            return True
        except ValidationError as e:
            logger.warning(f"Set failed for {key}={value} (voice={voice}): {e}")
            return False
        except AttributeError:
            logger.warning(f"Unknown param {key} (voice={voice})")
            return False

    # Merged audio params (use models)
    def get_merged_audio_params(self, voice_name: Optional[str] = None, api_overrides: Optional[Dict] = None) -> Dict[str, Any]:
        if self.app_config is None:
            raise ValueError("Config not loaded")
        voice_name = voice_name or 'default'
        globals_dump = self.app_config.globals.model_dump()
        # Hash for LRU (on globals + voice)
        global_hash = hashlib.md5(str(sorted(globals_dump.items())).encode()).hexdigest()
        voice = self.app_config.voices.get(voice_name, VoiceConfig())  # Default if missing
        voice_dump = voice.model_dump()
        voice_key = hashlib.md5(str(sorted(voice_dump.items())).encode()).hexdigest()
        cache_key = (voice_name, global_hash, voice_key)

        if cache_key in self._merged_cache:
            logger.debug(f"LRU cache hit for {voice_name}")
            return self._merged_cache[cache_key]

        params = globals_dump.copy()
        params.update(voice_dump)
        params['voice_name'] = voice_name
        params['sr'] = self.app_config.globals.sr  # Enforce

        if api_overrides:
            params.update(api_overrides)

        self._merged_cache[cache_key] = params
        self._global_hash = global_hash
        self._voice_hashes[voice_name] = voice_key
        logger.debug(f"Merged params for {voice_name} (LRU miss; cached)")
        return params


    # Merged audio params (use models)
    def get_voice_params(self, voice_name: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[
        str, Any]:
        """Get voice params with proper initialization safety check. Always returns complete dict."""
        # Safety check
        if not hasattr(self, '_is_initialized') or not self._is_initialized or self.app_config is None:
            logger.warning("Config not initialized - using fallback voice params")
            fallback = {
                'temperature': 0.8, 'exaggeration': 0.5, 'top_p': 1.0, 'min_p': 0.05,
                'repetition_penalty': 1.2, 'cfgw': 0.0, 'speaking_rate': 1.0, 'language_id': 'en',
                'enable_pre_adjustment': True, 'sr': 24000, 'min_ref_duration': 3.0,
                'check_artifacts': True, 'hop_length': 256, 'n_fft': 1024, 'normalize_method': 'peak',
                'post_gain': 0.0,  # NEW: For post-processing
                'min_post_duration': 1.0,  # NEW: Prevents None > int
                'gain_max_limit': 1.0, 'noise_floor_db': -60.0,  # NEW: Common post keys
                'voice_name': voice_name or 'default_fallback'
            }
            if overrides:
                # Safe update: Clamp types
                for k, v in overrides.items():
                    if k in fallback:
                        if isinstance(fallback[k], float):
                            fallback[k] = float(v)
                        elif isinstance(fallback[k], int):
                            fallback[k] = int(float(v))
                        elif isinstance(fallback[k], bool):
                            fallback[k] = bool(v)
                fallback.update({k: v for k, v in overrides.items() if k not in fallback})
            return fallback

        # param specs (as yours, expanded)
        PARAM_SPECS = {
            'temperature': float, 'exaggeration': float, 'top_p': float, 'min_p': float,
            'repetition_penalty': float, 'cfgw': float, 'speaking_rate': float, 'language_id': str,
            'enable_pre_adjustment': bool, 'sr': int, 'min_ref_duration': float,
            'check_artifacts': bool, 'hop_length': int, 'n_fft': int, 'normalize_method': str,
            'post_gain': float, 'min_post_duration': float,  # NEW: Essentials for post
            'gain_max_limit': float, 'noise_floor_db': float,
            'voice_name': str
        }

        DEFAULT_VALUES = {  # Full defaults (as yours + new)
            'temperature': 0.8, 'exaggeration': 0.5, 'top_p': 1.0, 'min_p': 0.05,
            'repetition_penalty': 1.2, 'cfgw': 0.0, 'speaking_rate': 1.0, 'language_id': 'en',
            'enable_pre_adjustment': True, 'sr': 24000, 'min_ref_duration': 3.0,
            'check_artifacts': True, 'hop_length': 256, 'n_fft': 1024, 'normalize_method': 'peak',
            'post_gain': 0.0, 'min_post_duration': 1.0, 'gain_max_limit': 1.0, 'noise_floor_db': -60.0,
            'voice_name': voice_name or 'default'
        }

        voice_name = voice_name or 'default'
        params = {}

        def _get_value(config_obj, param, param_type, default):
            """Safely get, ensure no None."""
            if config_obj is None:
                return default
            try:
                value = getattr(config_obj, param, None)
                if value is None:
                    return default
                # Convert (as yours)
                if param_type == bool:
                    return bool(value)
                elif param_type == int:
                    return int(float(value))
                elif param_type == float:
                    return float(value)
                else:
                    return str(value)
            except (TypeError, ValueError, AttributeError):
                logger.debug(f"Failed to get {param}; using default {default}")
                return default

        # Globals
        for param, param_type in PARAM_SPECS.items():
            params[param] = _get_value(self.app_config.globals, param, param_type, DEFAULT_VALUES[param])

        # Voice override (if exists)
        if voice_name != 'default' and voice_name in self.app_config.voices:
            voice = self.app_config.voices[voice_name]
            for param, param_type in PARAM_SPECS.items():
                value = _get_value(voice, param, param_type, None)
                if value is not None:  # Strict: Only if set
                    params[param] = value

        # Overrides (safe, as above)
        if overrides:
            for param, value in overrides.items():
                if param in PARAM_SPECS:
                    param_type = PARAM_SPECS[param]
                    try:
                        if param_type == bool:
                            params[param] = bool(value)
                        elif param_type == int:
                            params[param] = int(float(value))
                        elif param_type == float:
                            params[param] = float(value)
                        else:
                            params[param] = str(value)
                    except:
                        logger.warning(f"Invalid override {param}={value}; keeping {params[param]}")

        # Clamps (prevent extremes)
        params['exaggeration'] = max(0.0, min(2.0, params['exaggeration']))  # For cloning
        params['min_post_duration'] = max(0.5, params['min_post_duration'])  # Safe min
        params['sr'] = int(params['sr'])
        if params['normalize_method'] not in ['peak', 'rms', 'none']:
            params['normalize_method'] = 'peak'
        params['voice_name'] = voice_name

        logger.trace(
            f"Voice params for '{voice_name}': { {k: v for k, v in params.items() if k != 'voice_name'} }")  # Debug (trace to avoid spam)
        return params



    def get_voice_parameters(self, voice_name: str) -> Dict[str, Any]:
        if self.app_config is None:
            raise ValueError("Config not loaded")
        voice = self.app_config.voices.get(voice_name)
        return voice.model_dump() if voice else {}

    def get_all_voices(self) -> List[str]:
        if self.app_config is None:
            raise ValueError("Config not loaded")
        return list(self.app_config.voices.keys())

    def _get_default_json(self):
        """Default JSON dict (use Pydantic defaults)."""
        default_app = AppConfig()  # Uses Field defaults from models
        # Override cores if needed (e.g., device from torch)
        default_app.globals.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        default_app.globals.dtype = torch.bfloat16 if default_app.globals.device == 'cuda' else torch.float32
        return default_app.model_dump()

    def _invalidate_merged_cache(self):
        """Purge LRU on changes."""
        self._merged_cache.clear()
        self._global_hash = None
        self._voice_hashes.clear()