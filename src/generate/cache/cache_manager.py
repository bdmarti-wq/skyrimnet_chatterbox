from typing import Dict, Any, Optional, Tuple
import hashlib
import os

import torchaudio
from pathlib import Path
import time

from loguru import logger
from src.config import get_config_value, get_config

from .audio_cache import AudioCache
from .conditionals_cache import ConditionalsCache
from .fuzzy_cache import FuzzyAudioCache
from .voice_reference import VoiceReferenceCache
from ...audio_utils import is_artifact_laden
from ...normalize_stem import normalize_stem


class CacheManager:
    """Centralized cache management for audio generation pipeline."""

    def __init__(self, config: "AppConfig"):
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

        return f"{voice_stem}_{text_hash}_{exaggeration:.2f}_{uuid_hex}"

    # In cache_manager.py, replace the existing get_voice_stem method:

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
        """Validate a voice prompt against audio requirements."""
        if not audio_path or not os.path.exists(audio_path):
            return False, "Path does not exist"

        # Rate-limited remote file validation
        if audio_path.startswith("http://") or audio_path.startswith("https://"):
            last_check = self._last_validation.get(audio_path, 0)
            if time.time() - last_check < self.VALIDATION_COOLDOWN:
                return True, "Remote validated recently"
            self._last_validation[audio_path] = time.time()

        # Use the voice reference cache's validation (which handles resampled versions)
        is_valid = self.voice_reference.validate_reference_file(audio_path, voice_stem)
        return is_valid, "Valid" if is_valid else "Invalid voice reference"

    def process_voice_reference(self, audio_path: str, voice_stem: str, force: bool = False) -> Tuple[
        str, str, str, Dict[str, Any]]:
        """
        Process a voice reference file through the voice cache (handles hashing, reuse, resampling, and conds key).
        Returns: (processed_path, content_hash_str, conds_key, voice_params)
        - processed_path: Resampled/processed path for use in pipeline (reuse if HIT).
        - content_hash_str: Empty '' (handled internally; no need to return for caller).
        - conds_key: Key for conditionals cache.
        - voice_params: Voice-specific parameters from config.

        On HIT: Reuse processed_path and conds_key (fast, no reprocess).
        On MISS: Process (resample/save/update cache) and return new.
        """
        if not audio_path or not os.path.exists(audio_path):
            logger.warning(f"Invalid audio path for voice reference: {audio_path}")
            return audio_path, '', 'fallback_key', {}  # Fallback to raw/empty

        # Normalize stem if not provided (fallback to default)
        if not voice_stem:
            voice_stem = normalize_stem(audio_path) or 'default'
            logger.debug(f"Derived voice stem: '{voice_stem}' from {audio_path}")

        # Basic validation (dur/artifacts; delegate to voice cache for full)
        config = get_config()
        min_duration = get_config_value('globals.min_ref_duration', 3.0)
        try:
            info = torchaudio.info(audio_path)
            duration = info.num_frames / info.sample_rate
            if duration < min_duration:
                logger.warning(f"Voice {voice_stem}: Duration {duration:.2f}s < required {min_duration}s – fallback")
                return audio_path, '', 'short_fallback_key', config.get_voice_params(voice_stem, {})

            # Artifact check (optional)
            if get_config_value('globals.check_artifacts', True) and is_artifact_laden(audio_path,
                                                                                       threshold_hz=config.app_config.globals.sr // 3):
                logger.warning(f"Voice {voice_stem}: Artifacts detected – fallback but flag for purge")
                # Optional: Purge from cache if exists
                if hasattr(self, 'voice_reference') and voice_stem in self.voice_reference.voice_cache:
                    self.voice_reference.voice_cache.pop(voice_stem, None)
                    logger.info(f"Purged artifact-laden entry for {voice_stem}")
        except Exception as v_e:
            logger.warning(f"Validation failed for {voice_stem}: {v_e} – proceed with raw")
            return audio_path, '', 'invalid_fallback_key', config.get_voice_params(voice_stem, {})

        # Call voice reference cache (handles probe/tiers/reuse/resample/conds key)
        if not hasattr(self, 'voice_reference') or not self.voice_reference:
            logger.error("VoiceReferenceCache not initialized – fallback to raw")
            voice_params = config.get_voice_params(voice_stem, {})
            return audio_path, '', f'{voice_stem}_nocache_key', voice_params

        voice_cache = self.voice_reference
        success, processed_path, cond_key, voice_params, entry = voice_cache.process_new_reference(
            voice_stem, audio_path, force_update=force
        )

        if not success:
            logger.error(f"Voice cache processing failed for {voice_stem}; fallback to raw")
            # Purge stem if corrupted/misbehaved
            if voice_stem in voice_cache.voice_cache:
                voice_cache.voice_cache.pop(voice_stem, None)
                voice_cache.save_cache()
                logger.info(f"Purged failed entry for {voice_stem}")
            return audio_path, '', f'{voice_stem}_fail_key', voice_params

        # Success: Use processed (reuse or new resampled)
        if entry:
            logger.debug(
                f"Voice process HIT for {voice_stem}: Reuse entry (processed: {processed_path}, conds: {cond_key[:20]}...)")
        else:
            logger.info(
                f"Voice MISS → processed for {voice_stem}: {processed_path} (new conds key: {cond_key[:20]}...)")

        # Return (no content_hash needed; internal to voice_cache)
        return processed_path, '', cond_key, voice_params



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
        """Load conditionals from cache system (memory or disk)."""
        if not conditionals_key:
            return False

        return self.conditionals_cache.get(
            cache_key=conditionals_key,
            model=model,
            device=self.config.app_config.globals.device,
            dtype=self.config.app_config.globals.dtype
        ) is not None

    def save_conditionals(self,
                         conditionals_key: str,
                         model: Any) -> bool:
        """Save conditionals to cache system."""
        if not conditionals_key or model is None or not hasattr(model, 'conds'):
            return False

        return self.conditionals_cache.save(
            cache_key=conditionals_key,
            conditionals=model.conds,
            model=model,
            device=self.config.app_config.globals.device,
            dtype=self.config.app_config.globals.dtype
        )

    def get_audio_cache(self, cache_key: str) -> Optional[str]:
        """Check exact audio cache for hit."""
        return self.audio_cache.get(cache_key)

    def set_audio_cache(self, cache_key: str, audio_path: str) -> None:
        """Cache an audio result with proper validation."""
        self.audio_cache.set(cache_key, audio_path)

    def get_fuzzy_audio_cache(self,
                            audio_path: str,
                            text: str,
                            stem: str) -> Optional[str]:
        """Check for fuzzy audio cache hit."""
        return self.fuzzy_cache.try_fuzzy_audio_cache(
            audio_path=audio_path,
            text_input=text,
            stem=stem
        )

    def index_audio_for_fuzzy(self, text: str, audio_path: str, voice_stem: str) -> None:
        """Add audio to fuzzy cache for future matching."""
        self.fuzzy_cache.index_audio(
            text=text,
            wav_path=audio_path,
            voice_stem=voice_stem
        )

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
            "audio_cache": self.audio_cache.get_stats(),
            "conditionals_cache": self.conditionals_cache.get_stats(),
            "fuzzy_cache": self.fuzzy_cache.get_stats(),
            "voice_reference": self.voice_reference.get_stats()
        }