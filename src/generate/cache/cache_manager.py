from typing import Dict, Any, Optional, Tuple
import hashlib
import os

import torch
import torchaudio
from pathlib import Path
import time

from loguru import logger
from src.cache_keys import generate_audio_cache_key as _gen_cache_key
from src.config import get_config_value, get_config

from .audio_cache import AudioCache
from .conditionals_cache import ConditionalsCache
from .fuzzy_cache import FuzzyAudioCache
from .voice_reference import VoiceReferenceCache, VoiceReferenceEntry
from ..pipeline import AudioGenerationContext
from src.audio import is_artifact_laden
from src.audio import validate_user_audio
from ...normalize_stem import normalize_stem



_cache_manager_instance = None

def get_cache_manager(config=None):
    """Global singleton for CacheManager."""
    global _cache_manager_instance
    if _cache_manager_instance is None:
        config = config or get_config()
        _cache_manager_instance = CacheManager(config)
        logger.info(f"Global CacheManager singleton #1 created (ID={id(_cache_manager_instance):x})")
    else:
        logger.debug(f"Global CacheManager singleton reuse (ID={id(_cache_manager_instance):x}, entries={_cache_manager_instance.fuzzy_cache.get_stats()['entries']})")
    return _cache_manager_instance

class CacheManager:
    """Centralized cache management for audio generation pipeline. SIMPLIFIED: process_voice_reference takes only context (extracts path/stem internally)."""

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
        self.audio_cache = AudioCache(self.cache_config.audio_cache_dir,  self.config)
        self.conditionals_cache = ConditionalsCache(self.cache_config.conditionals_cache_dir)
        self.fuzzy_cache = FuzzyAudioCache(
            cache_dir=self.cache_config.audio_cache_dir,
            config=self.config
        )
        self.voice_reference = VoiceReferenceCache(
            cache_dir=self.cache_config.voices_cache_dir,
            content_hash_threshold=self.cache_config.fuzzy.fuzzy_artifact_threshold_hz,
            max_entries=getattr(self.cache_config.audio, 'max_voice_entries', 200)
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
        """Delegate to shared cache key generator (single source of truth)."""
        key = _gen_cache_key(voice_stem, text, exaggeration, cache_uuid, stem_only=stem_only)
        if not stem_only:
            logger.debug(f"The cache_generated_cache_key is: {key}")
        return key

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
        if not validate_user_audio(audio_path, min_dur=get_config_value('globals.min_ref_duration', 1.5)):
            return False, "Invalid or too-short path"

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

    def process_voice_reference(self, context: AudioGenerationContext, force: bool = False) -> Tuple[
        bool, str, str, Dict[str, Any], Optional[VoiceReferenceEntry]]:
        """
        Takes context (audio_path/voice_stem); performs multi-stage fallback on invalid path.
        Fallback steps:
        1. Try Configured default voice path.
        2. Search voices directory for any valid .wav.
        3. Fall back to neutral if all fails.
        """
        if context is None:
            logger.warning("process_voice_reference called without context; fallback to defaults")
            voice_params = self.config.get_voice_params('default', {'exaggeration': 1.0})
            return False, "", 'fallback_key', voice_params, None

        audio_path = getattr(context, 'audio_prompt_path', None)
        voice_stem = getattr(context, 'voice_stem', 'default')

        # Multi-stage path fallback if primary is invalid/missing
        if not audio_path or not os.path.exists(audio_path):
            logger.warning(f"Invalid voice path from context: {audio_path}")
            
            # Step 1: Configured default
            default_voice = get_config_value('globals.default_voice_path', 'cache/audio/voices/resampled/malebrute_24000Hz.wav')
            root = get_config_value('globals.root', '')
            if root:
                default_voice_abs = os.path.join(root, default_voice)
                if os.path.exists(default_voice_abs):
                    logger.info(f"Using configured default voice fallback: {default_voice_abs}")
                    audio_path = default_voice_abs
                elif os.path.exists(default_voice):
                    logger.info(f"Using configured default voice fallback (rel): {default_voice}")
                    audio_path = default_voice
            
            # Step 2: Directory search if default missing
            if not audio_path or not os.path.exists(audio_path):
                voices_dir = get_config_value('globals.voices_cache_dir')
                if voices_dir and os.path.exists(voices_dir):
                    logger.debug(f"Searching {voices_dir} for any valid fallback voice")
                    for root_dir, _, files in os.walk(voices_dir):
                        for f in files:
                            if f.lower().endswith('.wav'):
                                candidate = os.path.join(root_dir, f)
                                if os.path.exists(candidate):
                                    logger.info(f"Found directory-search voice fallback: {candidate}")
                                    audio_path = candidate
                                    break
                        if audio_path: break

            if not audio_path or not os.path.exists(audio_path):
                logger.error("All voice fallbacks failed; using neutral conditionals")
                injected = getattr(context, 'voice_params', None)
                voice_params = injected if isinstance(injected, dict) and injected else self.config.get_voice_params(voice_stem, {'exaggeration': 1.0})
                context.audio_prompt_path = ""
                return False, "", 'fallback_key', voice_params, None
            
            # Update context with fallback path
            context.audio_prompt_path = audio_path

        # Always derive norm_stem (stable)
        norm_stem = self.get_voice_stem(audio_path)
        voice_stem = norm_stem or voice_stem or 'default'
        logger.debug(f"Process voice under stable norm_stem '{voice_stem}' from context")

        # Basic validation (min dur/artifacts) using context.sr if available
        min_duration = get_config_value('globals.min_ref_duration', 1.5)
        sr = getattr(context, 'sr', 24000)  # Use context.sr or fallback
        try:
            info = torchaudio.info(audio_path)
            duration = info.num_frames / info.sample_rate
            if duration < min_duration:
                logger.warning(f"Short {voice_stem}: {duration:.2f}s")
                injected = getattr(context, 'voice_params', None)
                voice_params = injected if isinstance(injected, dict) and injected else self.config.get_voice_params(voice_stem, {})
                return False, "", 'short_fallback_key', voice_params, None

            if get_config_value('globals.check_artifacts', True):
                threshold = sr // 3  # Dynamic from context/config SR
                if is_artifact_laden(audio_path, threshold_hz=threshold):
                    logger.warning(f"Artifacts in {voice_stem}; purge if cached")
                    if voice_stem in self.voice_reference.voice_cache:
                        self.voice_reference.voice_cache.pop(voice_stem, None)
        except Exception as v_e:
            logger.warning(f"Validation fail for {voice_stem}: {v_e}")
            injected = getattr(context, 'voice_params', None)
            voice_params = injected if isinstance(injected, dict) and injected else self.config.get_voice_params(voice_stem, {})
            return False, "", 'invalid_fallback_key', voice_params, None

        # FIXED: Pass context to process_new_reference
        success, processed_path, conds_key, voice_params, hit_entry = self.voice_reference.process_new_reference(
            voice_stem, audio_path, force_update=force, context=context
        )

        # Ensure the voice_params we return are consistent with the bridge-injected dict if available
        if isinstance(getattr(context, 'voice_params', None), dict) and context.voice_params:
            # Context overrides are authoritative for this request
            voice_params = {**voice_params, **context.voice_params}
            context.voice_params = voice_params

        if not success:
            logger.error(f"Process fail {voice_stem}; purge")
            if voice_stem in self.voice_reference.voice_cache:
                self.voice_reference.voice_cache.pop(voice_stem, None)
                self.voice_reference.save_cache()
            voice_params = self.config.get_voice_params(voice_stem, {})
            return False, "", f'{voice_stem}_fail_key', voice_params, None

        if hit_entry:
            logger.debug(f"Voice HIT/reuse {voice_stem} (stable norm)")
        else:
            logger.debug(f"Voice MISS/process {voice_stem} (new stable norm)")

        if not success or not processed_path:
            processed_path = audio_path if os.path.exists(audio_path) else ''

        # FIXED: Return bool-first: (success, str(path), conds_key, ...) – aligns unpack; no useless ''
        return success, str(processed_path), conds_key, voice_params, hit_entry


    def _pad_short_audio(self, audio, target_dur=3.0, sr=24000):
        """
        Pad short audio by repeating the clip until it reaches the target duration.
        Assumes audio is a 1D or (1, samples) torch tensor (mono).
        """
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)  # Ensure (1, samples)

        current_samples = audio.shape[1]
        current_dur = current_samples / sr
        if current_dur >= target_dur:
            return audio  # No padding needed

        target_samples = int(target_dur * sr)
        repeats = max(1, int(target_samples / current_samples))  # At least 1 repeat, ceil to reach target

        # Repeat the audio tensor
        padded_audio = audio.repeat(1, repeats)

        # Trim or zero-pad the last bit if over
        if padded_audio.shape[1] > target_samples:
            padded_audio = padded_audio[:, :target_samples]
        else:
            # Add zeros if under (unlikely, but safe)
            extra = torch.zeros((1, target_samples - padded_audio.shape[1]), device=padded_audio.device,
                                dtype=padded_audio.dtype)
            padded_audio = torch.cat([padded_audio, extra], dim=1)

        logger.debug(f"Padded {current_dur:.2f}s audio to {target_dur}s (repeats: {repeats})")
        return padded_audio


    def get_conditionals(self, conditionals_key: str, model: Any) -> bool:
        """Load conditionals from cache system (memory or disk). FIXED: Safe globals access."""
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
        if not path or not os.path.exists(path):
            logger.warning(f"Path not exist: {path}")
            return False
        logger.debug(f": app_config.globals.fuzzy.fuzzy_artifact_threshold_hz is {get_config_value(
            'app_config.globals.fuzzy.fuzzy_artifact_threshold_hz')}")
        logger.debug(f": app_config.globals.fuzzy is {get_config_value(
            'app_config.globals.fuzzy')}")
        p = Path(path).resolve()
        base_dir = (self.cache_config.cache_dir / "audio" / "output").resolve()
        # FIXED: More lenient – startswith for absolute paths, ignore case
        if not str(p).lower().startswith(str(base_dir).lower()):
            logger.warning(f"Path not in output subdir {base_dir}: {path}")
            return False
        if is_artifact_laden(path, get_config_value('app_config.globals.fuzzy.fuzzy_artifact_threshold_hz', 12000)):
            logger.warning(f"Artifact-laden path skipped: {path}")
            return False
        logger.debug(f"Validated path: {p} in {base_dir}")
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