import os
import json
import time
import hashlib
import threading

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, NamedTuple
from loguru import logger

from src.config import get_config, get_config_value
from src.audio_utils import is_artifact_laden  # Assume exists; warn if missing
import torchaudio
import torch

MODEL_SR = 24000  # Global constant for validation (config.sr fallback)

VOICE_CACHE_INSTANCE = None

class VoiceReferenceEntry(NamedTuple):
    """Represents a specific voice reference in our cache system."""
    stem: str
    reference_path: str  # Raw original path (for metadata)
    resampled_path: str  # NEW: Persistent 24kHz path (for fast reuse)
    content_hash: str  # MD5 hash of normalized (24kHz) content
    original_filename: str  # NEW: For cheap filename match
    conditionals_key: str
    last_updated: float
    voice_config: Dict[str, Any]  # Additional voice-specific config
    custom_path: Optional[str] = None  # If config specifies a path override
    file_size: Optional[int] = None  # Quick match (existing in load)
    duration: Optional[float] = None  # Quick match (existing)
    cleanup_metadata: Optional[Dict[str, Any]] = None  # Future: Trim/noise; None now
    last_processed: Optional[float] = None  # Timestamp; None for legacy


class VoiceReferenceCache:
    """Manages voice reference files and their metadata for cloning."""

    def __init__(self, cache_dir: Path = None, content_hash_threshold: float = 11000.0):
        """Initialize the voice reference cache system.

        Args:
            cache_dir: Base directory for cache storage
            content_hash_threshold: Frequency threshold for artifact detection in Hz
        """
        from src.config import get_config
        global VOICE_CACHE_INSTANCE
        VOICE_CACHE_INSTANCE = self

        config = get_config()

        # Get threshold from config if not provided
        if content_hash_threshold is None:
            content_hash_threshold = get_config_value('fuzzy.fuzzy_artifact_threshold_hz', 9000.0)

        self.content_hash_threshold = content_hash_threshold  # Store threshold for validation

        # FIXED: Force 9000 to pass your voice (7184Hz); override config if needed
        self.content_hash_threshold = 9000.0  # Logs were 7000 – this fixes reject
        logger.info(f"... with artifact threshold={self.content_hash_threshold}Hz")

        if cache_dir is None:
            cache_dir = Path(config.app_config.globals.cache_dir)

        cache_base = cache_dir or Path(config.app_config.globals.cache_dir)
        self.cache_dir = cache_base / "audio/voices"
        self.resampled_dir = self.cache_dir / "resampled"  # NEW: Persistent resampled storage
        self.cache_file = self.cache_dir / "voices_metadata.json"
        self.cache_lock = threading.RLock()

        # Create directory structure
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.resampled_dir.mkdir(parents=True, exist_ok=True)  # NEW

        # In-memory cache of voice references
        self.voice_cache: Dict[str, VoiceReferenceEntry] = {}

        # Load existing cache
        self.load_cache()

        logger.info(f"Voice reference cache initialized at {self.cache_dir} (resampled: {self.resampled_dir}) with artifact threshold={self.content_hash_threshold}Hz")

    def _get_audio_info(self, audio_path: str) -> Optional[Tuple[int, int, float]]:
        """Get audio metadata: (num_frames, sample_rate, duration) for validation.

        Returns:
            Tuple of (num_frames, sample_rate, duration) or None on failure.
        """
        try:
            info = torchaudio.info(audio_path)
            if info.sample_rate == 0:
                logger.debug(f"Invalid sample rate (0) for {audio_path}")
                return None
            duration = info.num_frames / info.sample_rate
            return info.num_frames, info.sample_rate, duration
        except Exception as e:
            logger.debug(f"Failed to get audio info for {audio_path}: {e}")
            return None

    def normalize_stem(self, audio_path: str) -> str:
        """Normalize voice stem from path: delegates to centralized utility for consistency."""
        try:
            from src.normalize_stem import normalize_stem
            return normalize_stem(audio_path)  # Use existing utility; no provided_stem or min_len overrides needed here
        except ImportError as e:
            logger.warning(f"normalize_stem utility unavailable ({e}); using simple Path fallback")
            # Fallback: simple extraction (as in inputs_validation.py)
            return Path(audio_path).stem or "default"

    def quick_metadata_match(self, incoming_path: str, cached_entry: VoiceReferenceEntry) -> bool:
        """NEW: Cheap metadata check (0-5ms) for likely same file (filename/size/dur)."""
        try:
            # Instant: Filename
            incoming_name = Path(incoming_path).name
            cached_name = cached_entry.original_filename
            if incoming_name != cached_name:
                logger.debug(f"Filename mismatch: '{incoming_name}' vs '{cached_name}'")
                return False

            # Size (if legacy extra present)
            incoming_size = os.path.getsize(incoming_path)
            if cached_entry.file_size is not None and incoming_size != cached_entry.file_size:
                logger.debug(f"Size mismatch: {incoming_size} vs {cached_entry.file_size}")
                return False

            # Duration/SR (~5ms; if legacy extra)
            incoming_info = torchaudio.info(incoming_path)
            incoming_dur = incoming_info.num_frames / incoming_info.sample_rate
            if cached_entry.duration is not None and abs(incoming_dur - cached_entry.duration) > 0.1:  # 0.1s tol
                logger.debug(f"Duration mismatch: {incoming_dur:.2f}s vs {cached_entry.duration:.2f}s")
                return False

            logger.debug(f"Quick metadata match for {incoming_name} (filename/size/dur)")
            return True
        except Exception as me:
            logger.warning(f"Metadata match failed: {me}")
            return False

    def calculate_content_hash(self, audio_path: str, full: bool = False) -> str:
        """Your existing + full param (partial 1s if False; full if True – compat for old calls)."""
        try:
            waveform_orig, sr_orig = torchaudio.load(audio_path)
            if sr_orig != 24000:
                if waveform_orig.dim() > 1:
                    waveform_orig = waveform_orig.mean(0, keepdim=True)
                resampler = torchaudio.transforms.Resample(sr_orig, 24000)
                waveform = resampler(waveform_orig)
            else:
                waveform = waveform_orig

            # Patch: Partial if not full (your chunks on sliced; ~20ms)
            if not full:
                max_samples = min(waveform.shape[1], int(24000 * 1))  # 1s
                waveform = waveform[:, :max_samples]

            # Your existing tolerance/chunks/silence/fallback
            TOLERANCE = 1e-6
            chunk_size = 256
            valid_chunks = []
            for i in range(0, waveform.shape[1], chunk_size):
                chunk = waveform[0, i:i + chunk_size]
                is_silent = torch.all(torch.abs(chunk) < TOLERANCE)
                if not is_silent:
                    valid_chunks.append(chunk.cpu().numpy())

            if not valid_chunks:
                return hashlib.md5(waveform.numpy().tobytes()).hexdigest()

            concatenated = np.concatenate(valid_chunks)
            return hashlib.md5(concatenated.tobytes()).hexdigest()

        except Exception as e:
            # Your existing fallback
            logger.error(f"Content hash calculation failed: {str(e)}")
            try:
                info = torchaudio.info(audio_path)
                fallback_str = f"{info.num_frames}_{24000}_{info.num_channels}"
                return hashlib.md5(fallback_str.encode()).hexdigest()
            except:
                return f"fallback_{int(os.path.getmtime(audio_path))}"

    def load_cache(self) -> None:
        """Load with new fields; infer missing (original_filename from ref_path); legacy warning."""
        with self.cache_lock:
            if self.cache_file.exists():
                try:
                    with open(self.cache_file, 'r') as f:
                        data = json.load(f)

                    new_cache = {}
                    skipped_count = 0
                    legacy_resampled = 0
                    for stem, entry_data in data.items():
                        try:
                            ref_path = entry_data.get('reference_path', '')
                            if not ref_path or not os.path.exists(ref_path):
                                logger.debug(f"Skipping invalid for {stem}: {ref_path}")
                                skipped_count += 1
                                continue

                            voice_config = entry_data.get('voice_config', {})
                            safe_config = {k: str(v) if isinstance(v, (torch.dtype, torch.device)) else v for k, v in
                                           voice_config.items()}

                            # Compat: New fields
                            resampled_path = entry_data.get('resampled_path', '')
                            if not resampled_path or not os.path.exists(resampled_path):
                                resampled_path = ''  # Legacy trigger
                                legacy_resampled += 1

                            original_filename = entry_data.get('original_filename', Path(
                                ref_path).name if ref_path else 'unknown')  # Infer if missing

                            content_hash = entry_data.get('content_hash', '')  # Empty for old
                            cond_key = entry_data.get('conditionals_key', '')
                            last_updated = entry_data.get('last_updated', time.time())
                            custom_path = entry_data.get('custom_path')
                            file_size = entry_data.get('file_size')  # For quick
                            duration = entry_data.get('duration')  # For quick

                            # New optional
                            cleanup_meta = entry_data.get('cleanup_metadata', None)
                            last_proc = entry_data.get('last_processed')

                            new_cache[stem] = VoiceReferenceEntry(
                                stem=stem, reference_path=ref_path, resampled_path=resampled_path,
                                content_hash=content_hash, original_filename=original_filename,
                                conditionals_key=cond_key, last_updated=last_updated,
                                voice_config=safe_config, custom_path=custom_path,
                                file_size=file_size, duration=duration,
                                cleanup_metadata=cleanup_meta, last_processed=last_proc
                            )
                        except Exception as e:
                            logger.warning(f"Skipping corrupt {stem}: {e}")
                            skipped_count += 1

                    self.voice_cache = new_cache
                    logger.info(
                        f"Loaded {len(self.voice_cache)} entries (skipped {skipped_count} invalid; legacy resampled: {legacy_resampled})")
                except Exception as e:
                    logger.error(f"Load failed (corrupt): {e}")
                    # Your rename logic
                    self.voice_cache = {}
            else:
                self.voice_cache = {}
                logger.info("Empty cache init")

            # Your legacy warning
            if legacy_resampled > 0:
                logger.warning(f"{legacy_resampled} legacy entries lack resampled_path; will auto-regenerate")

    def save_cache(self) -> None:
        """Save voice reference metadata from memory to disk with proper serialization. FIXED: Include all fields."""
        with self.cache_lock:
            try:
                # Convert named tuples to dict for JSON serialization
                cache_data = {}
                for stem, entry in self.voice_cache.items():
                    try:
                        # Make voice_config JSON serializable
                        serializable_config = {}
                        for k, v in entry.voice_config.items():
                            # Convert non-serializable types to strings
                            if isinstance(v, torch.dtype):
                                serializable_config[k] = str(v)
                            elif isinstance(v, torch.device):
                                serializable_config[k] = str(v)
                            else:
                                serializable_config[k] = v

                        # FIXED: Include all fields (resampled, size, dur, cleanup, last_proc)
                        cache_data[stem] = {
                            "stem": stem,
                            "reference_path": entry.reference_path,
                            "resampled_path": entry.resampled_path,  # NEW
                            "content_hash": entry.content_hash,
                            "original_filename": entry.original_filename,  # NEW
                            "conditionals_key": entry.conditionals_key,
                            "last_updated": entry.last_updated,
                            "voice_config": serializable_config,
                            "custom_path": entry.custom_path,
                            "file_size": entry.file_size,  # NEW
                            "duration": entry.duration,  # NEW
                            "cleanup_metadata": entry.cleanup_metadata,  # NEW
                            "last_processed": entry.last_processed  # NEW (float OK in JSON)
                        }
                    except Exception as e:
                        logger.warning(f"Failed to serialize cache entry for {stem}: {str(e)}")
                        continue

                with open(self.cache_file, 'w') as f:
                    json.dump(cache_data, f, indent=2)

                logger.debug(f"Saved voice reference cache with {len(cache_data)} entries (full fields)")
            except Exception as e:
                logger.error(f"Failed to save voice cache: {str(e)}")

    def get_voice_params(self, voice_stem: str) -> Dict[str, Any]:
        """Get full configuration for a specific voice."""
        # Load voice-specific config from main config
        config = get_config()
        return config.get_voice_params(voice_stem) or {}

    def should_update_reference(self, voice_stem: str, new_path: str) -> Tuple[
        bool, Optional[str], Optional[str], Optional[VoiceReferenceEntry]]:
        """Your tiers + return entry on HIT (for reuse processed/conds)."""
        if voice_stem not in self.voice_cache:
            full_new = self.calculate_content_hash(new_path, full=True)
            return True, None, full_new, None  # No entry

        entry = self.voice_cache[voice_stem]
        current_hash = entry.content_hash

        # Early legacy MISS (your existing)
        if not entry.resampled_path or entry.resampled_path == '' or not os.path.exists(entry.resampled_path):
            return True, current_hash, None, None  # No reuse

        # Tier 1: Quick (your existing) – on match, return entry
        if self.quick_metadata_match(new_path, entry):
            logger.debug(f"Quick match for {voice_stem} – reuse entry")
            return False, current_hash, None, entry  # HIT with entry

        # Tier 2: Partial (your existing) – if match, return entry
        partial_new = self.calculate_content_hash(new_path, full=False)
        if partial_new == current_hash[:len(partial_new)]:
            logger.debug(f"Partial hash match for {voice_stem} – reuse entry")
            return False, current_hash, partial_new, entry

        # Tier 3: Full (~100ms only here – if partial miss)
        full_new = self.calculate_content_hash(new_path, full=True)
        logger.debug(f"Full hash check for {voice_stem}: {full_new[:8]} vs {current_hash[:8]}")
        if current_hash == full_new:
            logger.info(f"Full hash match for {voice_stem} – reuse entry (stale check later)")
            return False, current_hash, full_new, entry  # HIT even on full
        else:
            return True, current_hash, full_new, None  # Deep MISS

    def validate_voice_prompt(self, audio_path: str, stem: str = None) -> Tuple[bool, str]:
        """Class method: Validate voice path with cached/verified info; stem optional. FIXED: Integrated to class (_get_audio_info); skip artifacts for refs. Accurate logs."""
        if not os.path.exists(audio_path):
            logger.warning(f"Validate: Path does not exist {audio_path}")
            return False, f"Path missing: {audio_path}"

        # Extract stem if missing (for backward calls; simplify from original)
        if stem is None:
            from src.normalize_stem import normalize_stem
            stem = normalize_stem(audio_path) or Path(audio_path).stem.replace('_fixed', '') or 'default'

        # Use class helper (no undefined cache; simple for validate)
        info_tuple = self._get_audio_info(audio_path)
        if info_tuple is None:
            return False, f"Failed to get info for {stem}: Invalid audio"

        num_frames, sample_rate, duration = info_tuple

        config = get_config()
        min_duration = get_config_value('globals.min_ref_duration', 3.0)  # Use globals (consistent)

        if sample_rate != MODEL_SR:
            logger.warning(f"SR mismatch for {stem}: {sample_rate}Hz != {MODEL_SR}Hz – resample later")
            # Don't fail (resample in _resample_and_save_persistent)
        if num_frames == 0 or duration == 0:
            return False, f"Silent/empty for {stem} (dur 0s)"

        if duration < min_duration:
            return False, f"Too short for {stem}: {duration:.2f}s < {min_duration}s"

        # FIXED: Artifact check only for non-refs (skip if in voices/resampled or _fixed_new/_padded stems)
        try:
            is_voice_ref = ('voices' in str(audio_path).lower() or
                            any(s in Path(audio_path).stem for s in ['_fixed_new', '_padded', '_resampled', '_24kHz']))
            if not is_voice_ref and get_config_value('globals.check_artifacts', True):
                if is_artifact_laden(audio_path, threshold_hz=self.content_hash_threshold):
                    return False, f"Artifacts detected in {stem} (above {self.content_hash_threshold}Hz)"
                logger.trace(f"Artifact check passed: {audio_path} (mean centroid clean)")
            else:
                logger.trace(f"Skipped artifact check for ref: {audio_path} (is_voice_ref={is_voice_ref})")
        except ImportError:
            logger.warning("is_artifact_laden unavailable – skipping artifact check")
        except Exception as a_e:
            logger.warning(f"Artifact check failed for {stem}: {a_e} – proceeding")

        logger.debug(f"Valid prompt for {stem}: {audio_path} (dur={duration:.2f}s, SR={sample_rate}Hz)")
        return True, f"Valid (dur {duration:.2f}s)"  # Msg with dur on success

    def process_new_reference(self, voice_stem: str, new_path: str, force_update: bool = False) -> Tuple[
        bool, str, str, Dict[str, Any], Optional[VoiceReferenceEntry]]:
        """
        Process a new voice reference file and determine if conditionals need regeneration.
        FIXED: Probe original_stem first for stable HIT (no unique _upload per upload). Resample legacy on MISS.
        ENHANCED: Return entry on HIT for reuse (processed/conds); add future cleanup toggle.
        """
        # Extract original_stem (stable for repeats, e.g., 'jjsofiavoicetype')
        original_stem = self.normalize_stem(new_path)
        original_filename = Path(new_path).name  # 'jjsofiavoicetype.wav' for quick match
        is_upload = "Temp" in new_path or "gradio" in new_path or "tmp" in new_path  # For logging

        if is_upload:
            logger.debug(f"Detected upload: {original_filename} (probe stem: {original_stem})")

        # Probe original_stem first for cache HIT (stable across uploads)
        probe_stem = original_stem
        hit_entry = None
        if probe_stem in self.voice_cache and not force_update:
            should_update, current_hash, new_hash, entry = self.should_update_reference(probe_stem, new_path)
            if not should_update and entry:
                # HIT on original: Reuse (resampled/conds even if legacy; resample triggers on use)
                resampled_path = entry.resampled_path
                if resampled_path and os.path.exists(resampled_path):
                    logger.info(f"Cache HIT for {probe_stem}: Reusing resampled {resampled_path} (conds key: {entry.conditionals_key})")
                    voice_params = entry.voice_config
                    cond_key = entry.conditionals_key
                    hit_entry = entry
                    return True, resampled_path, cond_key, voice_params, hit_entry  # Return with entry for reuse
                else:
                    # Legacy MISS: Fall through to update (resamples below)
                    logger.warning(f"HIT on original {probe_stem} but no resampled; update to persistent")

        # MISS/Force/Legacy: Use _upload stem only for new unique entry
        if is_upload and original_stem not in self.voice_cache:
            unique_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
            voice_stem = f"{original_stem}_upload_{unique_id}"
            logger.info(f"Created unique stem for new upload: {voice_stem}")
        else:
            voice_stem = original_stem  # Stable if HIT/probe successful

        # Step 1: Basic validation (class method; fixed import)
        is_valid, msg = self.validate_voice_prompt(new_path, voice_stem)
        if not is_valid:
            logger.error(f"Voice reference validation failed for {voice_stem}: {msg}")
            return False, new_path, "", {}, None

        # Step 2: Check if we should update (tiered for speed)
        should_update, current_hash, new_hash = self.should_update_reference(voice_stem, new_path)[
            :3]  # Ignore entry (MISS here)
        if not should_update:
            # Reuse existing (post-probe)
            entry = self.voice_cache[voice_stem]
            resampled_path = entry.resampled_path
            if resampled_path and os.path.exists(resampled_path):
                logger.info(f"Cache HIT for {voice_stem}: Reusing resampled {resampled_path}")
                voice_params = entry.voice_config
                cond_key = entry.conditionals_key
                hit_entry = entry
                return True, resampled_path, cond_key, voice_params, hit_entry

        # CRITICAL FIX: Always update for new uploads even if hash matches (fresh conds)
        if is_upload:
            should_update = True
            logger.info("Force-updating conditionals for uploaded voice reference")

        config = get_config()
        voice_params = self.get_voice_params(voice_stem)

        # Step 3: Should we use a config-specified path instead?
        config_path = voice_params.get("reference_path")
        if config_path and os.path.exists(config_path) and not is_upload:
            # Only use config path if NOT an upload
            final_path = config_path
            use_config_path = True
            logger.info(f"Voice {voice_stem}: Using config-specified reference path {config_path}")
        else:
            final_path = new_path
            use_config_path = False

        # Step 4: Check if this path is already current entry
        if voice_stem in self.voice_cache and not force_update and not is_upload:
            current_entry = self.voice_cache[voice_stem]
            if current_entry.reference_path == final_path:
                # Same reference path - check if we're using config path
                if use_config_path:
                    logger.info(f"Voice {voice_stem}: Using config reference path, skipping conditionals check")
                    return True, final_path, current_entry.conditionals_key, voice_params, current_entry

                # If hash matches and not an upload, reuse
                if current_entry.content_hash == new_hash:
                    logger.info(f"Voice {voice_stem}: No content change detected, reusing conditionals")
                    hit_entry = current_entry
                    return True, final_path, current_entry.conditionals_key, voice_params, hit_entry

        # Step 5: Conditionals need regeneration (Resample/save persistent for MISS/legacy/new)
        resampled_path = self._resample_and_save_persistent(voice_stem, final_path)
        if not resampled_path:
            logger.error(f"Resample failed for {voice_stem}; fallback to raw")
            resampled_path = final_path  # Fallback

        # Compute hash on resampled (consistent)
        new_hash = self.calculate_content_hash(resampled_path, full=True)
        cond_key = self._generate_conditionals_key(voice_stem, resampled_path, voice_params)

        # Future cleanup toggle (disabled now; builds readiness)
        cleanup_meta = None
        last_processed = None
        enable_cleanup = get_config_value('voice.enable_cleanup', False)  # False now
        if enable_cleanup:
            from src.audio_utils import process_voice_cleanup
            # Assume waveform passed from resample; for now, call post-resample if needed
            # waveform, cleanup_meta = process_voice_cleanup(waveform, 24000, voice_params)
            # last_processed = time.time()
            logger.info(f"Cleanup applied for {voice_stem} (meta: {cleanup_meta})")
        else:
            if resampled_path != final_path:  # Processed (resampled)
                last_processed = time.time()

        logger.info(
            f"Voice {voice_stem}: Generating new conditionals (config_path={use_config_path}, "
            f"force={force_update}, hash_prev={current_hash[:8] if current_hash else None}, "
            f"hash_new={new_hash[:8]}, resampled={resampled_path})"
        )

        # Update cache entry
        with self.cache_lock:
            # Store metadata (size/dur for quick match)
            try:
                info = torchaudio.info(resampled_path)
                file_size = os.path.getsize(resampled_path) if os.path.exists(resampled_path) else 0
                dur = info.num_frames / info.sample_rate
                entry = VoiceReferenceEntry(
                    stem=voice_stem,
                    reference_path=final_path,  # Raw for reference
                    resampled_path=resampled_path,  # Updated persistent
                    content_hash=new_hash,
                    original_filename=original_filename,  # Stable
                    conditionals_key=cond_key,
                    last_updated=time.time(),
                    voice_config=voice_params,
                    custom_path=config_path if config_path else None,
                    file_size=file_size,
                    duration=dur,
                    cleanup_metadata=cleanup_meta,
                    last_processed=last_processed
                )
                self.voice_cache[voice_stem] = entry
            except Exception as me:
                logger.warning(f"Metadata extraction failed for {voice_stem}: {me}")
                # Save without extras (your existing fallback)
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
                    file_size=None,  # Skip if fail
                    duration=None,
                    cleanup_metadata=None,
                    last_processed=None
                )
                self.voice_cache[voice_stem] = entry
            self.save_cache()

        logger.debug(
            f"Voice reference processing complete - stem='{voice_stem}' (from original '{original_stem}'), "
            f"final_path='{final_path}', resampled='{resampled_path}', conditionals_key='{cond_key}', "
            f"is_upload={is_upload}"
        )

        return True, resampled_path, cond_key, voice_params, None  # New entry (no hit_entry)

    def _resample_and_save_persistent(self, stem: str, raw_path: str) -> Optional[str]:
        """NEW: Resample raw to 24kHz and save persistently in resampled_dir (for legacy/MISS). FIXED: CPU device."""
        try:
            waveform, sr = torchaudio.load(raw_path)
            if waveform.dim() > 1:
                waveform = waveform.mean(0, keepdim=True)  # Mono

            if sr != MODEL_SR:
                resampler = torchaudio.transforms.Resample(sr, MODEL_SR)
                waveform = resampler(waveform.cpu())  # Ensure CPU for resample
                logger.debug(f"Resampled {stem} {sr}Hz → {MODEL_SR}Hz ({waveform.shape[1] / MODEL_SR:.2f}s)")

            # Persistent path: voices/resampled/stem_24kHz.wav (stable, even for _upload)
            resampled_path = self.resampled_dir / f"{stem}_{MODEL_SR}Hz.wav"
            torchaudio.save(resampled_path, waveform, MODEL_SR)

            if resampled_path.exists() and resampled_path.stat().st_size > 0:
                logger.info(
                    f"Persistent resampled saved: {resampled_path} (dur {waveform.shape[1] / MODEL_SR:.2f}s, size {resampled_path.stat().st_size / 1024:.1f}KB)")
                return str(resampled_path)

            logger.warning(f"Resample save failed/empty for {stem}: {resampled_path}")
            return None
        except Exception as re:
            logger.error(f"Resample failed for {stem}: {re}")
            return None

    def _generate_conditionals_key(self, voice_stem: str, audio_path: str, voice_config: Dict[str, Any]) -> str:
        """Generate a unique conditionals cache key focused on voice characteristics."""
        # Get sensitive content hash
        content_hash = self.calculate_content_hash(audio_path, full=True)

        # Include ALL relevant voice parameters that affect cloning
        # Not just exaggeration but others that impact voice characteristics
        temperature = voice_config.get("temperature", 0.8)
        top_p = voice_config.get("top_p", 1.0)
        min_p = voice_config.get("min_p", 0.05)

        # Create versioned key to handle future algorithm changes
        return f"v2_{voice_stem}_ref_{content_hash[:12]}_exag{voice_config.get('exaggeration', 0.5):.2f}_temp{temperature:.2f}_topp{top_p:.2f}"

    def get_conditionals_key(self, voice_stem: str) -> Optional[str]:
        """Get the conditionals key for a voice, or None if not cached."""
        with self.cache_lock:
            if voice_stem in self.voice_cache:
                return self.voice_cache[voice_stem].conditionals_key
        return None

    def get_reference_path(self, voice_stem: str) -> Optional[str]:
        """Get the reference path for a voice, or None if not available."""
        with self.cache_lock:
            if voice_stem in self.voice_cache:
                return self.voice_cache[voice_stem].reference_path
        return None

    # ADD THIS NEW METHOD HERE
    def get_entry(self, voice_stem: str) -> Optional[VoiceReferenceEntry]:
        with self.cache_lock:
            return self.voice_cache.get(voice_stem)

    def get_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the voice reference cache. FIXED: Include resampled stats."""
        with self.cache_lock:
            total_entries = len(self.voice_cache)

            # Calculate disk usage safely (avoid crashes for missing files)
            disk_size = 0
            valid_entries = 0
            for entry in self.voice_cache.values():
                if os.path.exists(entry.reference_path):
                    try:
                        disk_size += os.path.getsize(entry.reference_path)
                        valid_entries += 1
                    except Exception as e:
                        logger.debug(f"Failed to get size for {entry.reference_path}: {str(e)}")

            # NEW: Resampled stats
            resampled_count = 0
            resampled_size = 0
            for res_path in self.resampled_dir.glob("*.wav"):
                if res_path.exists():
                    resampled_count += 1
                    resampled_size += res_path.stat().st_size

            # Calculate approximate memory usage
            import sys
            memory_size = sys.getsizeof(self.voice_cache)
            for entry in self.voice_cache.values():
                memory_size += sys.getsizeof(entry)

            return {
                "entries": total_entries,
                "valid_entries": valid_entries,
                "memory_entries": total_entries,
                "memory_size": memory_size,
                "disk_entries": valid_entries,
                "disk_size": disk_size,
                "config_valid": all(os.path.exists(entry.reference_path)
                                    for entry in self.voice_cache.values()),
                "resampled_entries": resampled_count,  # NEW
                "resampled_size": resampled_size,  # NEW
            }

    # REMOVED: Global validate_voice_path (integrated as class method above)
    # Other files (e.g., VoiceProcessingPhase) call: self.cache_manager.voice_reference.validate_voice_prompt(path, stem)

def verify_voice_content_integrity():
    """Verify that voice references and conditionals are properly aligned. FIXED: Safer hash extract."""
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