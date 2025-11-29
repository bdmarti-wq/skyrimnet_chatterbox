import os
import json
import time
import hashlib
import threading
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, NamedTuple
from loguru import logger

from src.config import get_config, get_config_value  # For defaults from config
from src.audio_utils import is_artifact_laden  # Assume exists; warn if missing
import torchaudio
import torch

MODEL_SR = 24000  # Global constant for validation

# singleton instance (shared cache)
VOICE_CACHE_INSTANCE = None
RESAMPLED_CACHE = {}  # Dict[hash: str] – key: content_hash, value: resampled_path

class VoiceReferenceEntry(NamedTuple):
    """Represents a specific voice reference in our cache system."""
    stem: str
    reference_path: str  # Raw original path (for metadata)
    resampled_path: str  # Persistent 24kHz path (for fast reuse)
    content_hash: str  # MD5 hash of normalized (24kHz) content
    original_filename: str  # For cheap filename match
    conditionals_key: str
    last_updated: float
    voice_config: Dict[str, Any]  # Additional voice-specific config (from get_voice_params)
    custom_path: Optional[str] = None  # If config specifies a path override
    file_size: Optional[int] = None  # Quick match
    duration: Optional[float] = None  # Quick match
    cleanup_metadata: Optional[Dict[str, Any]] = None  # Future: Trim/noise; None now
    last_processed: Optional[float] = None  # Timestamp; None for legacy

class VoiceReferenceCache:
    """Manages voice reference files and their metadata for cloning."""

    # Cached voice_params per stem (locked, to avoid re-fetch)
    _voice_params_cache: Dict[str, Dict] = {}
    _cache_lock = threading.RLock()

    def __init__(self, cache_dir: Path = None, content_hash_threshold: float = 11000.0):
        """Initialize the voice reference cache system. OPTIMIZED: Cache voice_params per stem."""
        global VOICE_CACHE_INSTANCE
        VOICE_CACHE_INSTANCE = self

        config = get_config()

        if content_hash_threshold is None:
            content_hash_threshold = get_config_value('globals.fuzzy_artifact_threshold_hz', 9000.0)

        self.content_hash_threshold = content_hash_threshold
        logger.info(f"Voice cache initialized with artifact threshold={self.content_hash_threshold}Hz")

        if cache_dir is None:
            cache_dir = get_config().app_config.globals.cache_dir if hasattr(config, 'app_config') else Path('./cache')

        cache_base = cache_dir or Path('./cache')
        self.cache_dir = Path(cache_base).resolve()
        self.resampled_dir = self.cache_dir / "resampled"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.resampled_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Voice cache: {self.cache_dir} (resampled: {self.resampled_dir})")

        self.cache_file = self.cache_dir / "voices_metadata.json"
        self.lock = threading.RLock()  # For cache access during process/save

        self.voice_cache: Dict[str, VoiceReferenceEntry] = {}

        self._voice_params_cache.clear()  # Clear on init

        self.load_cache()

    def apply_voice_params_to_context(self, voice_config: Dict[str, Any], context: Optional[Any] = None):
        """Utility: Apply voice params to context attributes (if context provided). For cache hits."""
        if context is None:
            logger.warning("apply_voice_params_to_context called without context; no attributes set")
            return

        # Apply key TTS params from voice_config (with defaults if None)
        exaggeration = voice_config.get('exaggeration', 0.75)
        temperature = voice_config.get('temperature', 0.75)
        cfg_weight = voice_config.get('cfg_weight', 0.43)
        min_p = voice_config.get('min_p', 0.05)
        top_p = voice_config.get('top_p', 1.0)
        repetition_penalty = voice_config.get('repetition_penalty', 1.2)  # PATCHED: Add default for repetition_penalty

        # Set on context (direct attributes)
        if hasattr(context, 'exaggeration'):
            context.exaggeration = exaggeration
        if hasattr(context, 'temperature'):
            context.temperature = temperature
        if hasattr(context, 'cfg_weight'):
            context.cfg_weight = cfg_weight
        if hasattr(context, 'min_p'):
            context.min_p = min_p
        if hasattr(context, 'top_p'):
            context.top_p = top_p
        if hasattr(context, 'repetition_penalty'):  # PATCHED: Apply repetition_penalty to context
            context.repetition_penalty = repetition_penalty

        logger.trace(f"Applied voice params to context: exagg={exaggeration}, temp={temperature}, cfg={cfg_weight}, rep_pen={repetition_penalty}")

    def _get_voice_params(self, voice_stem: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """CACHED: Get voice params using central get_voice_params (memoized per stem),
        then apply optional overrides (e.g., bridge-injected UI overrides) with final precedence.

        The internal memoization is per-stem for the base values; overrides are applied on top
        per call to avoid polluting the cache with UI-specific ephemeral values.
        """
        with self._cache_lock:
            if voice_stem in self._voice_params_cache:
                logger.trace(f"Cached voice_params for {voice_stem}")
                base = self._voice_params_cache[voice_stem].copy()
            else:
                params = get_config().get_voice_params(voice_stem) or {}

                # FIXED: Ensure key TTS params have defaults if None (avoids validation errors)
                if params.get('exaggeration') is None:
                    params['exaggeration'] = 0.75
                if params.get('temperature') is None:
                    params['temperature'] = 0.75
                if params.get('cfg_weight') is None:
                    params['cfg_weight'] = 0.43
                if params.get('min_p') is None:
                    params['min_p'] = 0.05
                if params.get('top_p') is None:
                    params['top_p'] = 1.0
                # PATCHED: Add default for repetition_penalty (avoids TTS validation error)
                if params.get('repetition_penalty') is None:
                    params['repetition_penalty'] = 1.2
                # Add other defaults as needed based on TTS usage

                self._voice_params_cache[voice_stem] = params  # Cache
                logger.trace(f"Fetched voice_params for {voice_stem} (defaults applied)")
                base = params.copy()
        # Apply overrides last (do not store in cache)
        if isinstance(overrides, dict) and overrides:
            merged = {**base, **overrides}
            return merged
        return base

    def _clear_voice_cache(self, stem: str):
        """Clear cached voice_params for a stem (on update)."""
        with self._cache_lock:
            self._voice_params_cache.pop(stem, None)

    def normalize_stem(self, audio_path: str) -> str:
        """Normalize voice stem from path. OPTIMIZED: Cache if needed."""
        try:
            from src.normalize_stem import normalize_stem
            return normalize_stem(audio_path)
        except ImportError as e:
            logger.warning(f"normalize_stem unavailable ({e}); fallback to Path.stem")
            stem = Path(audio_path).stem or "default"
            return stem.replace('_fixed', '')  # Clean common suffixes

    def _get_audio_info(self, audio_path: str) -> Optional[Tuple[int, int, float]]:
        """Get audio metadata."""
        try:
            info = torchaudio.info(audio_path)
            if info.sample_rate == 0:
                logger.debug(f"Invalid SR (0) for {audio_path}")
                return None
            duration = info.num_frames / info.sample_rate
            return info.num_frames, info.sample_rate, duration
        except Exception as e:
            logger.debug(f"Audio info failed for {audio_path}: {e}")
            return None


    def quick_metadata_match(self, incoming_path: str, cached_entry: VoiceReferenceEntry) -> bool:
        """Cheap match for reuse. OPTIMIZED: Use cached info where possible."""
        try:
            incoming_name = Path(incoming_path).name
            cached_name = cached_entry.original_filename
            if incoming_name != cached_name:
                logger.trace(f"Filename mismatch: '{incoming_name}' != '{cached_name}'")
                return False

            incoming_size = os.path.getsize(incoming_path)
            if cached_entry.file_size is not None and incoming_size != cached_entry.file_size:
                logger.trace(f"Size mismatch: {incoming_size} != {cached_entry.file_size}")
                return False

            incoming_info = self._get_audio_info(incoming_path)
            if incoming_info:
                incoming_frames, _, incoming_dur = incoming_info
                if cached_entry.duration is not None and abs(incoming_dur - cached_entry.duration) > 0.1:
                    logger.trace(f"Dur mismatch: {incoming_dur:.2f}s != {cached_entry.duration:.2f}s")
                    return False
                if cached_entry.file_size is None:
                    cached_entry.file_size = incoming_size  # Update cache
                if cached_entry.duration is None:
                    cached_entry.duration = incoming_dur  # Update cache
            else:
                return False

            logger.trace(f"Quick match: filename + size/dur")
            return True
        except Exception as me:
            logger.warning(f"Metadata match failed: {me}")
            return False

    def calculate_content_hash(self, audio_path: str, full: bool = False) -> str:
        """Calculate content hash. OPTIMIZED: Memoize per path/full combo."""
        cache_key = (audio_path, full)
        if cache_key in RESAMPLED_CACHE:
            logger.trace(f"Cached hash for {audio_path} (full={full})")
            return RESAMPLED_CACHE[cache_key]

        try:
            waveform_orig, sr_orig = torchaudio.load(audio_path)
            if sr_orig != MODEL_SR:
                if waveform_orig.dim() > 1:
                    waveform_orig = waveform_orig.mean(0, keepdim=True)
                resampler = torchaudio.transforms.Resample(sr_orig, MODEL_SR)
                waveform = resampler(waveform_orig)
            else:
                waveform = waveform_orig

            if not full:
                max_samples = min(waveform.shape[1], int(MODEL_SR * 1))  # 1s sample
                waveform = waveform[:, :max_samples]

            TOLERANCE = 1e-6
            chunk_size = 256
            valid_chunks = []
            for i in range(0, waveform.shape[1], chunk_size):
                chunk = waveform[0, i:i + chunk_size]
                is_silent = torch.all(torch.abs(chunk) < TOLERANCE)
                if not is_silent:
                    valid_chunks.append(chunk.cpu().numpy())

            if not valid_chunks:
                hash_val = hashlib.md5(waveform.numpy().tobytes()).hexdigest()
            else:
                concatenated = np.concatenate(valid_chunks)
                hash_val = hashlib.md5(concatenated.tobytes()).hexdigest()

            RESAMPLED_CACHE[cache_key] = hash_val  # Memoize
            return hash_val

        except Exception as e:
            logger.error(f"Hash calculation failed: {str(e)}")
            try:
                info = torchaudio.info(audio_path)
                fallback_str = f"{info.num_frames}_{MODEL_SR}_{info.num_channels}"
                hash_val = hashlib.md5(fallback_str.encode()).hexdigest()
            except:
                return f"fallback_{int(time.time())}"  # Time-based fallback
            RESAMPLED_CACHE[cache_key] = hash_val
            return hash_val

    def load_cache(self) -> None:
        """Load cache; FIXED: Init skipped_count/legacy_resampled outside try (always safe); consistent log."""
        with self.lock:
            skipped_count = 0  # FIXED: Always init (before if; no unbound)
            legacy_resampled = 0  # FIXED: Always init

            if self.cache_file.exists():
                try:
                    with open(self.cache_file, 'r') as f:
                        data = json.load(f)

                    self.voice_cache = {}
                    for stem, entry_data in data.items():
                        try:
                            ref_path = entry_data.get('reference_path', '')
                            if not ref_path or not os.path.exists(ref_path):
                                logger.debug(f"Skipping invalid {stem}: {ref_path}")
                                skipped_count += 1
                                continue

                            # Re-fetch voice_config on load (apply defaults if needed)
                            voice_config = self._get_voice_params(stem)

                            resampled_path = entry_data.get('resampled_path', '')
                            if not resampled_path or not os.path.exists(resampled_path):
                                resampled_path = ''  # Legacy
                                legacy_resampled += 1

                            original_filename = entry_data.get('original_filename',
                                                               Path(ref_path).name if ref_path else 'unknown')

                            content_hash = entry_data.get('content_hash', '')
                            cond_key = entry_data.get('conditionals_key', '')
                            last_updated = entry_data.get('last_updated', time.time())
                            custom_path = entry_data.get('custom_path')
                            file_size = entry_data.get('file_size')
                            duration = entry_data.get('duration')

                            cleanup_meta = entry_data.get('cleanup_metadata', None)
                            last_proc = entry_data.get('last_processed')

                            # SIMPLIFIED: Load under existing stem (no remap); forward uses norm_stem only
                            self.voice_cache[stem] = VoiceReferenceEntry(
                                stem=stem, reference_path=ref_path, resampled_path=resampled_path,
                                content_hash=content_hash, original_filename=original_filename,
                                conditionals_key=cond_key, last_updated=last_updated,
                                voice_config=voice_config, custom_path=custom_path,
                                file_size=file_size, duration=duration,
                                cleanup_metadata=cleanup_meta, last_processed=last_proc
                            )
                        except Exception as e:
                            logger.warning(f"Skipping corrupt {stem}: {e}")
                            skipped_count += 1

                    # FIXED: Log with always-init vars (safe)
                    logger.info(
                        f"Loaded {len(self.voice_cache)} entries (skipped {skipped_count} invalid; legacy resampled: {legacy_resampled})")
                except Exception as e:
                    logger.error(
                        f"Load failed (corrupt JSON): {e}; starting empty. Delete voices_metadata.json to reset.")
                    self.voice_cache = {}
            else:
                logger.info("No cache file; starting empty")
                self.voice_cache = {}

            # FIXED: Runs always (inits before); safe
            if skipped_count > 0 or legacy_resampled > 0:
                logger.warning(
                    "Legacy/invalid entries detected. To reset: Delete voices_metadata.json and resampled/ dir manually for clean slate.")

    def save_cache(self) -> None:
        """Save cache; SIMPLIFIED: Under current stems (forward norm only)."""
        with self.lock:
            try:
                cache_data = {}
                for stem, entry in self.voice_cache.items():
                    try:
                        serializable_config = {}
                        for k, v in entry.voice_config.items():
                            if isinstance(v, (torch.dtype, torch.device)):
                                serializable_config[k] = str(v)
                            else:
                                serializable_config[k] = v

                        cache_data[stem] = {
                            "stem": stem,
                            "reference_path": entry.reference_path,
                            "resampled_path": entry.resampled_path,
                            "content_hash": entry.content_hash,
                            "original_filename": entry.original_filename,
                            "conditionals_key": entry.conditionals_key,
                            "last_updated": entry.last_updated,
                            "voice_config": serializable_config,
                            "custom_path": entry.custom_path,
                            "file_size": entry.file_size,
                            "duration": entry.duration,
                            "cleanup_metadata": entry.cleanup_metadata,
                            "last_processed": entry.last_processed
                        }
                    except Exception as e:
                        logger.warning(f"Serialize failed for {stem}: {e}")
                        continue

                with open(self.cache_file, 'w') as f:
                    json.dump(cache_data, f, indent=2)

                logger.debug(f"Saved {len(cache_data)} voice entries")
            except Exception as e:
                logger.error(f"Save failed: {e}")

    def validate_voice_prompt(self, audio_path: str, stem: str = None) -> Tuple[bool, str]:
        """Validate prompt."""
        if not os.path.exists(audio_path):
            return False, f"Missing: {audio_path}"

        if stem is None:
            from src.normalize_stem import normalize_stem
            stem = normalize_stem(audio_path) or Path(audio_path).stem.replace('_fixed', '') or 'default'

        info_tuple = self._get_audio_info(audio_path)
        if info_tuple is None:
            return False, f"Invalid audio: {stem}"

        num_frames, sample_rate, duration = info_tuple

        config = get_config()
        min_duration = get_config_value('globals.min_ref_duration', 3.0)

        if num_frames == 0 or duration == 0:
            return False, f"Empty for {stem}"

        if duration < min_duration:
            return False, f"Short for {stem}: {duration:.2f}s < {min_duration}s"

        if sample_rate != MODEL_SR:
            logger.warning(f"SR mismatch for {stem}: {sample_rate}Hz != {MODEL_SR}Hz")

        # Artifact check (skip for refs)
        try:
            is_voice_ref = ('voices' in str(audio_path).lower() or
                            any(s in Path(audio_path).stem for s in ['_fixed_new', '_padded', '_resampled', '_24kHz']))
            if not is_voice_ref and get_config_value('globals.check_artifacts', True):
                if hasattr(config, 'app_config') and hasattr(config.app_config, 'globals'):
                    threshold = config.app_config.globals.sr // 3
                else:
                    threshold = 8000
                if is_artifact_laden(audio_path, threshold_hz=self.content_hash_threshold):
                    return False, f"Artifacts in {stem}"
            logger.trace(f"Artifact passed (or skipped) for {audio_path}")
        except ImportError:
            logger.warning("Artifact check skipped (missing func)")
        except Exception as a_e:
            logger.warning(f"Artifact check error for {stem}: {a_e}")

        logger.debug(f"Valid {stem}: {duration:.2f}s @ {sample_rate}Hz")
        return True, f"Valid ({duration:.2f}s)"

    def process_new_reference(self, voice_stem: str, new_path: str, force_update: bool = False, context: Optional[Any] = None) -> Tuple[
        bool, str, str, Dict[str, Any], Optional[VoiceReferenceEntry]]:
        """
        Process voice; SIMPLIFIED: Probe/load by norm_stem (stable); store under norm_stem (overrides legacy if same); no temp _upload; hit_entry on reuse.
        FIXED: Accept context; apply params on hit using utility method. For cache hit, return cached entry for consistency.
        """
        # FIXED: Normalize early (stable for probe/store)
        norm_stem = self.normalize_stem(new_path)
        original_filename = Path(new_path).name
        is_upload = "Temp" in new_path or "gradio" in new_path or "tmp" in new_path

        if is_upload:
            logger.debug(f"Upload: {original_filename} → norm_stem '{norm_stem}'")

        # FIXED: Fetch voice_params once; apply any bridge-injected overrides for this request (do not cache overrides)
        injected_overrides = getattr(context, 'voice_params', None) if context is not None else None
        voice_config = self._get_voice_params(norm_stem, overrides=injected_overrides)

        # Probe by norm_stem for stable HIT (ignores legacy non-norm keys)
        hit_entry = None
        if norm_stem in self.voice_cache and not force_update:
            entry = self.voice_cache[norm_stem]
            # Lightning quick match
            if self.quick_metadata_match(new_path, entry):
                logger.debug(f"LIGHTNING HIT for {norm_stem} (quick match; reuse stable)")
                resampled_path = entry.resampled_path
                if resampled_path and os.path.exists(resampled_path):
                    cond_key = entry.conditionals_key
                    hit_entry = entry
                    # FIXED: Apply cached params to context on hit
                    self.apply_voice_params_to_context(entry.voice_config, context)
                    return True, resampled_path, cond_key, voice_config, hit_entry

            # Tiers if quick miss
            should_update, current_hash, new_hash, entry_from_tiers = self.should_update_reference(norm_stem, new_path)
            if not should_update and entry_from_tiers:
                resampled_path = entry_from_tiers.resampled_path
                if resampled_path and os.path.exists(resampled_path):
                    logger.debug(f"HIT for {norm_stem}: Reuse resampled/conds (stable hash {entry_from_tiers.content_hash[:12]})")
                    cond_key = entry_from_tiers.conditionals_key
                    hit_entry = entry_from_tiers
                    # FIXED: Apply cached params to context on tier hit
                    self.apply_voice_params_to_context(entry_from_tiers.voice_config, context)
                    return True, resampled_path, cond_key, voice_config, hit_entry

        # FIXED: Store under norm_stem always (stable; overrides legacy if norm matches old unique)
        voice_stem = norm_stem
        if is_upload and norm_stem not in self.voice_cache:
            logger.debug(f"New upload under stable '{voice_stem}' (no temp _upload)")

        # Validation
        is_valid, msg = self.validate_voice_prompt(new_path, voice_stem)
        if not is_valid:
            logger.error(f"Validation failed for {voice_stem}: {msg}")
            return False, new_path, "", voice_config, None

        # Tiers for update (under norm_stem)
        should_update, current_hash, new_hash, _ = self.should_update_reference(voice_stem, new_path)
        if not should_update:
            entry = self.voice_cache[voice_stem]
            resampled_path = entry.resampled_path
            if resampled_path and os.path.exists(resampled_path):
                logger.debug(f"HIT for {voice_stem}: Reuse (stable norm)")
                cond_key = entry.conditionals_key
                hit_entry = entry
                # FIXED: Apply cached params to context on update
                self.apply_voice_params_to_context(entry.voice_config, context)
                return True, resampled_path, cond_key, voice_config, hit_entry

        # Force for uploads
        if is_upload:
            should_update = True
            logger.debug(f"Force update for upload {voice_stem}")

        config = get_config()
        if hasattr(config, 'app_config') and hasattr(config.app_config, 'globals'):
            device = config.app_config.globals.device
            dtype = config.app_config.globals.dtype
            sr = config.app_config.globals.sr
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            dtype = torch.float32
            sr = 24000
        voice_params = self._get_voice_params(voice_stem, overrides=injected_overrides)  # Align with unified source

        # Config path?
        config_path = voice_params.get("reference_path")
        if config_path and os.path.exists(config_path) and not is_upload:
            final_path = config_path
            use_config_path = True
            logger.debug(f"Config path for {voice_stem}: {config_path}")
        else:
            final_path = new_path
            use_config_path = False

        # Same path/hash check
        if voice_stem in self.voice_cache and not force_update and not is_upload:
            current_entry = self.voice_cache[voice_stem]
            if current_entry.reference_path == final_path:
                if use_config_path:
                    return True, final_path, current_entry.conditionals_key, voice_params, current_entry

                if current_entry.content_hash == self.calculate_content_hash(final_path, full=True):
                    logger.debug(f"No change for {voice_stem}; reuse")
                    hit_entry = current_entry
                    return True, final_path, current_entry.conditionals_key, voice_params, hit_entry

        # Resample/conds (store under norm)
        resampled_path = self._resample_and_save_persistent(voice_stem, final_path, device=device, dtype=dtype, sr=sr)
        if not resampled_path:
            logger.error(f"Resample failed {voice_stem}; fallback to {final_path}")
            resampled_path = final_path

        new_hash = self.calculate_content_hash(resampled_path, full=True)
        cond_key = self._generate_conditionals_key(voice_stem, resampled_path, voice_params)

        # Cleanup (future, disabled)
        cleanup_meta = None
        last_processed = None
        enable_cleanup = get_config_value('voice.enable_cleanup', False)
        if enable_cleanup:
            # ... (placeholder)
            pass
        else:
            if resampled_path != final_path:
                last_processed = time.time()

        logger.info(f"{voice_stem}: New conds (hash {new_hash[:8]}, resampled {resampled_path})")

        # Update cache under norm_stem (override if legacy)
        with self.lock:
            try:
                info = torchaudio.info(resampled_path)
                file_size = os.path.getsize(resampled_path)
                dur = info.num_frames / info.sample_rate
                entry = VoiceReferenceEntry(
                    stem=voice_stem,
                    reference_path=final_path,
                    resampled_path=resampled_path,
                    content_hash=new_hash,
                    original_filename=original_filename,
                    conditionals_key=cond_key,
                    last_updated=time.time(),
                    voice_config=voice_params,
                    custom_path=config_path if config_path else None,
                    file_size=file_size,
                    duration=dur,
                    cleanup_metadata=cleanup_meta,
                    last_processed=last_processed
                )
                self.voice_cache[voice_stem] = entry  # FIXED: Override to stable norm
            except Exception as me:
                logger.warning(f"Metadata error {voice_stem}: {me}")
                entry = VoiceReferenceEntry(
                    stem=voice_stem,
                    reference_path=final_path,
                    resampled_path=resampled_path,
                    content_hash=new_hash,
                    original_filename=original_filename,
                    conditionals_key=cond_key,
                    last_updated=time.time(),
                    voice_config=voice_params,
                    custom_path=config_path if config_path else None,
                    file_size=None,
                    duration=None,
                    cleanup_metadata=None,
                    last_processed=None
                )
                self.voice_cache[voice_stem] = entry
            self.save_cache()

        logger.debug(f"Voice processed: '{voice_stem}' (norm), final={final_path}, resampled={resampled_path}, conds={cond_key}")

        return True, resampled_path, cond_key, voice_params, None  # MISS/new

    def should_update_reference(self, voice_stem: str, new_path: str) -> Tuple[
        bool, Optional[str], Optional[str], Optional[VoiceReferenceEntry]]:
        """Tiered check."""
        if voice_stem not in self.voice_cache:
            full_new = self.calculate_content_hash(new_path, full=True)
            return True, None, full_new, None

        entry = self.voice_cache[voice_stem]
        current_hash = entry.content_hash

        if not entry.resampled_path or not os.path.exists(entry.resampled_path):
            return True, current_hash, None, None

        if self.quick_metadata_match(new_path, entry):
            logger.debug(f"Quick match {voice_stem}")
            return False, current_hash, None, entry

        partial_new = self.calculate_content_hash(new_path, full=False)
        if partial_new == current_hash[:len(partial_new)]:
            logger.debug(f"Partial match {voice_stem}")
            return False, current_hash, partial_new, entry

        full_new = self.calculate_content_hash(new_path, full=True)
        logger.debug(f"Full check {voice_stem}: {full_new[:8]} vs {current_hash[:8]}")
        if current_hash == full_new:
            logger.info(f"Full match {voice_stem}")
            return False, current_hash, full_new, entry
        else:
            return True, current_hash, full_new, None

    def _resample_and_save_persistent(self, stem: str, raw_path: str, device=torch.device('cpu'), dtype=torch.float32,
                                      sr=24000) -> Optional[str]:
        """Resample/save. FIXED: Full CPU tensor enforcement for torchaudio.save (no GPU tensors); shape handling for mono; robust validation."""
        try:
            # Load waveform (may be on GPU from prior load; force CPU early for all ops)
            waveform, sr_orig = torchaudio.load(raw_path)

            # FIXED: Always to CPU post-load (numpy/ torchaudio.save compatibility)
            if isinstance(waveform, torch.Tensor) and waveform.device.type == 'cuda':
                waveform = waveform.cpu()

            # FIXED: Ensure mono (mean channels if stereo) and float32 dtype for resample
            if waveform.dim() > 1:
                waveform = waveform.mean(0, keepdim=True)  # Mono: [1, frames]
            else:
                waveform = waveform.unsqueeze(0)  # Ensure [1, frames]
            waveform = waveform.float()  # Resample expects float32; not dtype param

            # Resample only if needed (use model SR constant)
            if sr_orig != MODEL_SR:
                resampler = torchaudio.transforms.Resample(orig_freq=sr_orig, new_freq=MODEL_SR)
                waveform = resampler(waveform)  # Already CPU; stays CPU
                logger.debug(f"Resampled {stem} {sr_orig}→{MODEL_SR}Hz (dur post: {waveform.shape[1] / MODEL_SR:.2f}s)")

            # FIXED: Stable filename with norm_stem (your code)
            norm_stem = self.normalize_stem(raw_path)
            resampled_path = self.resampled_dir / f"{norm_stem}_{MODEL_SR}Hz.wav"

            # FIXED: Save on CPU only (torchaudio.save requires CPU tensors; no .to(device) here!)
            # If device needed later (e.g., for conds prep), handle in prepare_conditionals
            torchaudio.save(resampled_path, waveform, MODEL_SR)

            # Validate saved file (non-empty, correct SR)
            if resampled_path.exists():
                file_size = resampled_path.stat().st_size
                if file_size > 0:
                    # Quick SR check
                    verify_info = torchaudio.info(resampled_path)
                    if verify_info.sample_rate == MODEL_SR:
                        logger.debug(f"Resampled success: {stem} → {resampled_path} ({file_size / 1024:.1f}KB)")
                        return str(resampled_path)
                    else:
                        resampled_path.unlink()
                        logger.warning(f"SR validation failed for {stem}; deleted invalid")
                else:
                    resampled_path.unlink()
                    logger.warning(f"Empty resample for {stem}; deleted")
            else:
                logger.warning(f"Resample save path missing for {stem}")

            return None  # Error: Caller falls back to raw_path

        except Exception as re:
            logger.error(f"Resample error {stem}: {re} – fallback to original")
            return None  # Caller (process_new_reference) uses raw_path



    def _generate_conditionals_key(self, voice_stem: str, audio_path: str, voice_config: Dict[str, Any]) -> str:
        """Generate conds key."""
        content_hash = self.calculate_content_hash(audio_path, full=True)
        exag = voice_config.get("exaggeration", 0.5)
        temperature = voice_config.get("temperature", 0.8)
        top_p = voice_config.get("top_p", 1.0)

        return f"v2_{voice_stem}_ref_{content_hash[:12]}_exag{exag:.2f}_temp{temperature:.2f}_topp{top_p:.2f}"

    def get_conditionals_key(self, voice_stem: str) -> Optional[str]:
        """Get conds key."""
        norm_stem = self.normalize_stem(voice_stem) if isinstance(voice_stem, str) else voice_stem
        with self.lock:
            if norm_stem in self.voice_cache:
                return self.voice_cache[norm_stem].conditionals_key
        return None

    def get_reference_path(self, voice_stem: str) -> Optional[str]:
        """Get ref path."""
        norm_stem = self.normalize_stem(voice_stem) if isinstance(voice_stem, str) else voice_stem
        with self.lock:
            if norm_stem in self.voice_cache:
                return self.voice_cache[norm_stem].reference_path
        return None

    def get_entry(self, voice_stem: str) -> Optional[VoiceReferenceEntry]:
        """Get entry."""
        norm_stem = self.normalize_stem(voice_stem) if isinstance(voice_stem, str) else voice_stem
        with self.lock:
            return self.voice_cache.get(norm_stem)

    def get_stats(self) -> Dict[str, Any]:
        """Stats."""
        with self.lock:
            total = len(self.voice_cache)
            valid = 0
            disk_size = 0
            for entry in self.voice_cache.values():
                if os.path.exists(entry.reference_path):
                    valid += 1
                    try:
                        disk_size += os.path.getsize(entry.reference_path)
                    except:
                        pass

            resampled_count = 0
            resampled_size = 0
            for res_path in self.resampled_dir.glob("*.wav"):
                if res_path.exists():
                    resampled_count += 1
                    resampled_size += res_path.stat().st_size

            import sys
            memory_size = sys.getsizeof(self.voice_cache)
            for entry in self.voice_cache.values():
                memory_size += sys.getsizeof(entry)

            return {
                "entries": total,
                "valid_entries": valid,
                "memory_entries": total,
                "memory_size": memory_size,
                "disk_entries": valid,
                "disk_size": disk_size,
                "resampled_entries": resampled_count,
                "resampled_size": resampled_size,
            }


def verify_voice_content_integrity():
    """Verify that voice references and conditionals are properly aligned. FIXED: Safer hash extract; use normalized stems."""
    global VOICE_CACHE_INSTANCE
    if not VOICE_CACHE_INSTANCE or not VOICE_CACHE_INSTANCE.voice_cache:
        logger.warning("⚠ No voice cache entries to verify")
        return False

    all_ok = True
    for stem, entry in VOICE_CACHE_INSTANCE.voice_cache.items():
        if not os.path.exists(entry.reference_path):
            logger.error(f"❌ Voice reference missing: {entry.reference_path} (stem='{stem}')")
            all_ok = False
            continue

        # Calculate actual content hash
        actual_hash = VOICE_CACHE_INSTANCE.calculate_content_hash(entry.reference_path, full=True)

        # Extract expected hash from conditionals key (safer: after '_ref_' , 12 hex chars)
        if '_ref_' in entry.conditionals_key:
            hash_part = entry.conditionals_key.split('_ref_')[1].split('_')[0]
            expected_hash = hash_part[:12] if len(hash_part) >= 12 and all(c in '0123456789abcdef' for c in hash_part[:12]) else None
        else:
            expected_hash = None

        if not expected_hash:
            logger.error(f"❌ Invalid conditionals key format: {entry.conditionals_key} (stem='{stem}')")
            all_ok = False
        elif expected_hash != actual_hash[:12]:
            logger.error(f"❌ HASH MISMATCH for stem '{stem}':\n"
                         f"Expected: {expected_hash}\n"
                         f"Actual:   {actual_hash[:12]}\n"
                         f"Reference: {entry.reference_path}")
            all_ok = False

    if all_ok:
        logger.info("✅ Voice reference and conditionals hashes verified")

    return all_ok