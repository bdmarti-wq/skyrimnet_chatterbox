from typing import Dict, Any, Optional, Tuple
import hashlib
import os

import torch
import torchaudio
from pathlib import Path
import time

from loguru import logger
from src.config import get_config_value, get_config

from .audio_cache import AudioCache
from .conditionals_cache import ConditionalsCache
from .fuzzy_cache import FuzzyAudioCache
from .voice_reference import VoiceReferenceCache, VoiceReferenceEntry
from ...audio_utils import is_artifact_laden
from ...normalize_stem import normalize_stem


class CacheManager:
    """Centralized cache management for audio generation pipeline. FIXED: _validate_path_for_cache uses absolute resolves."""

    def __init__(self, config):
        self.config = config
        self.cache_config = config.app_config.globals

        # Make these directories first
        for d in [
            self.cache_config.audio_cache_dir,
            self.cache_config.conditionals_cache_dir,
            self.cache_config.voices_cache_dir
        ]:
            d.mkdir(parents=True, exist_ok=True)

        # Initialize all cache systems
        self.audio_cache = AudioCache(self.cache_config.audio_cache_dir)
        self.conditionals_cache = ConditionalsCache(self.cache_config.conditionals_cache_dir)
        self.fuzzy_cache = FuzzyAudioCache(
            cache_dir=self.cache_config.audio_cache_dir,
            threshold=self.cache_config.fuzzy.fuzzy_threshold
        )
        self.voice_reference = VoiceReferenceCache(
            cache_dir=self.cache_config.voices_cache_dir,
            content_hash_threshold=self.cache_config.fuzzy.fuzzy_artifact_threshold_hz
        )

        # Initialize validation times (anti-ratelimit)
        self._last_validation = {}
        self.VALIDATION_COOLDOWN = 30  # Prevent excessive remote file validation

        logger.info("Cache manager initialized with all cache systems")

    def generate_audio_cache_key(self,
                               voice_stem: str,
                               text: str,
                               exaggeration: float,
                               cache_uuid: int,
                               stem_only: bool = False) -> str:
        """
        Generate consistent cache key across all cache systems.
        Key format: {voice_stem}_{text_hash}_{exaggeration:.2f}_{uuid_hex}
        """
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8] if text else "empty"
        uuid_hex = hex(cache_uuid)[2:][:8] if isinstance(cache_uuid, int) else str(cache_uuid)[:8]

        if stem_only:
            return voice_stem

        logger.debug(f"The cache_generated_cache_key is: {voice_stem}_{text_hash}_{exaggeration:.2f}_{uuid_hex}")
        return f"{voice_stem}_{text_hash}_{exaggeration:.2f}_{uuid_hex}"

    def get_voice_stem(self, audio_path: Optional[str]) -> str:
        """Get normalized voice stem from path or fallback. Uses centralized utility for consistency."""
        if not audio_path:
            return "default"

        # Try class method first (for encapsulation)
        if hasattr(self.voice_reference, 'normalize_stem'):
            try:
                return self.voice_reference.normalize_stem(audio_path)
            except Exception as e:
                logger.warning(f"VoiceReferenceCache.normalize_stem failed ({e}); falling back to utility")

        # Fallback: Import and use centralized utility (matches fuzzy_cache/inputs_validation)
        try:
            from src.normalize_stem import normalize_stem
            stem = normalize_stem(audio_path)
            logger.debug(f"Derived stem via utility: {stem}")
            return stem
        except ImportError as e:
            logger.warning(f"normalize_stem utility unavailable ({e}); using simple Path fallback")
        except Exception as e:
            logger.warning(f"Utility normalize_stem failed ({e}); using simple Path fallback")

        # Last resort: Direct path extraction
        from pathlib import Path
        path = Path(audio_path)
        stem = path.stem.lower().replace(' ', '_')  # Simple clean as before
        logger.debug(f"Fallback stem from path: {stem}")
        return stem or "default"

    def validate_voice_prompt(self, audio_path: str, voice_stem: str = "default") -> Tuple[bool, str]:
        """Validate a voice prompt against audio requirements. FIXED: Safe globals access (no kwargs to voice_reference)."""
        if not audio_path or not os.path.exists(audio_path):
            return False, "Path does not exist"

        # Rate-limited remote file validation
        if audio_path.startswith("http://") or audio_path.startswith("https://"):
            last_check = self._last_validation.get(audio_path, 0)
            if time.time() - last_check < self.VALIDATION_COOLDOWN:
                return True, "Remote validated recently"
            self._last_validation[audio_path] = time.time()

        # FIXED: Get globals safely (no kwargs to validate_reference_file – assume it uses config internals)
        config = get_config()
        sr = getattr(config.app_config.globals, 'sr', 24000) if hasattr(config, 'app_config') and hasattr(config.app_config, 'globals') else 24000
        is_valid = self.voice_reference.validate_reference_file(audio_path, voice_stem, sr=sr)  # FIXED: No device/dtype kwargs (use internals)
        return is_valid, "Valid" if is_valid else "Invalid voice reference"

    def process_voice_reference(self, audio_path: str, voice_stem: str, force: bool = False) -> Tuple[
        str, str, str, Dict[str, Any], Optional[VoiceReferenceEntry]]:
        """
        Wrapper; SIMPLIFIED: Norm stem on input; call process_new_reference; return tuple + hit_entry.
        FIXED: No device/dtype kwargs in get_voice_params (dict for params only).
        """
        if not audio_path or not os.path.exists(audio_path):
            logger.warning(f"Invalid voice path: {audio_path}")
            voice_params = self.config.get_voice_params('default', {'exaggeration': 1.0})  # FIXED: Dict for params (no device/dtype kwargs)
            return audio_path, '', 'fallback_key', voice_params, None

        # SIMPLIFIED: Always derive norm_stem (stable)
        voice_stem = self.voice_reference.normalize_stem(audio_path) or voice_stem or 'default'
        logger.debug(f"Process voice under stable norm_stem '{voice_stem}'")

        # Basic validation (min dur/artifacts)
        config = get_config()
        min_duration = get_config_value('globals.min_ref_duration', 3.0)
        try:
            info = torchaudio.info(audio_path)
            duration = info.num_frames / info.sample_rate
            if duration < min_duration:
                logger.warning(f"Short {voice_stem}: {duration:.2f}s")
                voice_params = config.get_voice_params(voice_stem, {})  # FIXED: Dict for params (no device/dtype kwargs)
                return audio_path, '', 'short_fallback_key', voice_params, None

            if get_config_value('globals.check_artifacts', True):
                threshold = getattr(config.app_config.globals, 'sr', 24000) // 3 if hasattr(config, 'app_config') and hasattr(config.app_config, 'globals') else 8000
                if is_artifact_laden(audio_path, threshold_hz=threshold):
                    logger.warning(f"Artifacts {voice_stem}; purge if cached")
                    if voice_stem in self.voice_reference.voice_cache:
                        self.voice_reference.voice_cache.pop(voice_stem, None)
        except Exception as v_e:
            logger.warning(f"Validation fail {voice_stem}: {v_e}")
            voice_params = config.get_voice_params(voice_stem, {})  # FIXED: Dict for params (no device/dtype kwargs)
            return audio_path, '', 'invalid_fallback_key', voice_params, None

        # Call with stable norm_stem
        success, processed_path, conds_key, voice_params, hit_entry = self.voice_reference.process_new_reference(
            voice_stem, audio_path, force_update=force
        )

        if not success:
            logger.error(f"Process fail {voice_stem}; purge")
            if voice_stem in self.voice_reference.voice_cache:
                self.voice_reference.voice_cache.pop(voice_stem, None)
                self.voice_reference.save_cache()
            voice_params = config.get_voice_params(voice_stem, {})  # FIXED: Dict for params (no device/dtype kwargs)
            return audio_path, '', f'{voice_stem}_fail_key', voice_params, None

        if hit_entry:
            logger.info(f"Voice HIT/reuse {voice_stem} (stable norm)")
        else:
            logger.info(f"Voice MISS/process {voice_stem} (new stable norm)")

        if not success or not processed_path:
            processed_path = audio_path if os.path.exists(audio_path) else ''

        return str(processed_path), '', conds_key, voice_params, hit_entry

    def _compute_content_hash(self, audio_path: str) -> str:
        """Helper: Compute MD5 hash of file for caching (safe fallback)."""
        if not audio_path or not os.path.exists(audio_path):
            return hashlib.md5(str(audio_path).encode('utf-8')).hexdigest()  # Path-based fallback
        try:
            with open(audio_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.warning(f"File hash failed for '{audio_path}': {e}")
            return hashlib.md5(str(audio_path).encode('utf-8')).hexdigest()

    def get_conditionals(self, conditionals_key: str, model: Any) -> bool:
        """Load conditionals from cache system (memory or disk). FIXED: Safe globals access (no kwargs issues)."""
        if not conditionals_key:
            return False

        config = get_config()
        device = getattr(config.app_config.globals, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu')) if hasattr(config, 'app_config') and hasattr(config.app_config, 'globals') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = getattr(config.app_config.globals, 'dtype', torch.float32) if hasattr(config, 'app_config') and hasattr(config.app_config, 'globals') else torch.float32
        return self.conditionals_cache.get(
            cache_key=conditionals_key,
            model=model,
            device=device,
            dtype=dtype
        ) is not None

    def save_conditionals(self,
                         conditionals_key: str,
                         model: Any) -> bool:
        """Save conditionals to cache system. FIXED: Safe globals access."""
        if not conditionals_key or model is None or not hasattr(model, 'conds'):
            return False

        config = get_config()
        device = getattr(config.app_config.globals, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu')) if hasattr(config, 'app_config') and hasattr(config.app_config, 'globals') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = getattr(config.app_config.globals, 'dtype', torch.float32) if hasattr(config, 'app_config') and hasattr(config.app_config, 'globals') else torch.float32
        return self.conditionals_cache.save(
            cache_key=conditionals_key,
            conditionals=model.conds,
            model=model,
            device=device,
            dtype=dtype
        )

    def get_audio_cache(self, cache_key: str) -> Optional[str]:
        """FIXED: Get with validation (exists? subpath? artifacts?) + purge if invalid."""
        path = self.audio_cache.get(cache_key)
        if path and os.path.exists(path):
            p = Path(path).resolve()
            base_dir = (self.cache_config.cache_dir / "audio" / "output").resolve()
            if p.is_relative_to(base_dir) and not is_artifact_laden(path, 7000):
                return str(p.absolute())
            else:
                # Purge invalid
                self.audio_cache.set(cache_key, None)
                logger.warning(f"Purged invalid audio_cache entry: {path}")
        return None

    def set_audio_cache(self, cache_key: str, audio_path: str) -> None:
        """FIXED: Set only if valid (exists, subpath, no artifacts). Ignore energy arg; use stem fallback 'default'."""
        if audio_path and os.path.exists(audio_path):
            # FIXED: Pass 'default' for stem (no energy_threshold_hz needed)
            if self._validate_path_for_cache(audio_path, 'default'):
                self.audio_cache.set(cache_key, str(Path(audio_path).absolute()))
                logger.info(f"Set audio_cache: {cache_key[:12]} → {Path(audio_path).name}")
            else:
                logger.warning(f"Skipped invalid set for audio_cache: {audio_path}")
        else:
            logger.debug(f"Skipped set (missing): {audio_path}")


    def get_fuzzy_audio_cache(self,
                            audio_path: str = '',
                            text: str = '',
                            stem: str = '',
                            threshold: float = 0.70) -> Optional[str]:
        """FIXED: Pass threshold; return abs validated path."""
        path = self.fuzzy_cache.try_fuzzy_audio_cache(audio_path, text, stem, threshold=threshold)
        return path if path and os.path.exists(path) else None  # Abs from fuzzy

    def index_audio_for_fuzzy(self, text: str, audio_path: str, voice_stem: str) -> None:
        """FIXED: Index only if valid (uses absolute _validate_path_for_cache)."""
        if self._validate_path_for_cache(audio_path, voice_stem):
            self.fuzzy_cache.index_audio(text, audio_path, voice_stem)
            logger.debug(f"Indexed fuzzy: '{text[:20]}...' → {Path(audio_path).name} (stem={voice_stem})")
        else:
            logger.warning(f"Skipped fuzzy index (invalid path): {audio_path}")

    def _validate_path_for_cache(self, path: str, stem: str = 'default') -> bool:
        """Shared validation for caches (exists, subpath output, no artifacts). FIXED: Both absolute resolves."""
        if not path or not os.path.exists(path):
            return False
        p = Path(path).resolve()
        base_dir = (self.cache_config.cache_dir / "audio" / "output").resolve()  # FIXED: Resolve to absolute
        if not p.is_relative_to(base_dir):
            logger.warning(f"Path not in output subdir {base_dir}: {path}")
            return False
        if is_artifact_laden(path, 7000):
            logger.warning(f"Artifact-laden path skipped: {path}")
            return False
        return True

    def clear_caches(self, voice: Optional[str] = None, full: bool = False) -> None:
        """Clear all cache systems with optional voice-specific purge."""
        if voice:
            self.audio_cache.clear(voice)
            self.conditionals_cache.clear(voice)
            self.fuzzy_cache.clear(voice)
        elif full:
            self.audio_cache.clear()
            self.conditionals_cache.clear()
            self.fuzzy_cache.clear()

        logger.info(f"Cleared caches{' for ' + voice if voice else ' entirely'}")

    @property
    def cache_stats(self) -> Dict[str, Any]:
        """Aggregate stats from all cache systems."""
        return {
            "audio_cache": self.audio_cache.get_stats() if hasattr(self.audio_cache, 'get_stats') else {'entries': 0},
            "conditionals_cache": self.conditionals_cache.get_stats(),
            "fuzzy_cache": self.fuzzy_cache.get_stats(),
            "voice_reference": self.voice_reference.get_stats()
        }