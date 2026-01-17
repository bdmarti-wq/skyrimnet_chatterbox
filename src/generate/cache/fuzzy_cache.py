import hashlib
import json
import re
import threading
import time
from difflib import SequenceMatcher
from queue import Queue, Empty
from typing import Dict, Any, Optional, List, Tuple
import os
from pathlib import Path

from loguru import logger

# Imports for safe config access (matches audio_cache)
from src.config import get_config, get_config_value

from src.audio import is_artifact_laden
from src.normalize_stem import normalize_stem

class FuzzyAudioCache:
    """Class-based implementation of fuzzy audio cache with proper encapsulation."""
    _instance = None  # Singleton instance
    _lock = threading.Lock()  # Thread-safe init

    def __new__(cls, cache_dir: Path, config=None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-check
                    cls._instance = super(FuzzyAudioCache, cls).__new__(cls)
                    cls._instance._initialized = False  # Flag: init only once
                    logger.info(f"FuzzyAudioCache singleton #1 created (ID={id(cls._instance):x})")
        else:
            logger.debug(f"FuzzyAudioCache singleton reuse (ID={id(cls._instance):x})")
        return cls._instance

    def _get_nested_config(self, section: str, key: str, default=None) -> Any:
        """Helper: Get nested config value from self.config (attr chain); fallback to global get_config_value."""
        if not self.config:
            # No passed config: Direct global fetch
            return get_config_value(f'app_config.globals.{section}.{key}', default=default)

        # Chained attr access on self.config (handles Pydantic/objects)
        try:
            app_config = getattr(self.config, 'app_config')
            globals_obj = getattr(app_config, 'globals')
            section_obj = getattr(globals_obj, section)
            return getattr(section_obj, key)
        except (AttributeError, KeyError):
            # Any missing level: Fallback to global
            return get_config_value(f'app_config.globals.{section}.{key}', default=default)

    def __init__(self, cache_dir: Path, config=None):
        """Initialize fuzzy cache with all settings from config (fallback to globals via helper)."""
        # Singleton check—skip if already init'd
        if hasattr(self, '_initialized') and self._initialized:
            return  # Reuse existing

        # Config for limits/thresholds (concise via helper; prioritizes passed config, fallback to globals)
        self.config = config
        self.threshold = self._get_nested_config('fuzzy', 'fuzzy_threshold', default=0.70)
        self.min_length = self._get_nested_config('fuzzy', 'fuzzy_min_length', default=3)
        self.max_index_size = self._get_nested_config('fuzzy', 'fuzzy_index_size', default=1000)
        self.enable_fuzzy = self._get_nested_config('fuzzy', 'enable_fuzzy_cache', default=True)
        self.max_entries_per_stem = self._get_nested_config('fuzzy', 'fuzzy_max_entries_per_stem', default=1000)
        self.enable_artifact_purge = self._get_nested_config('fuzzy', 'fuzzy_artifact_purge_enable', default=True)
        self.artifact_threshold_hz = self._get_nested_config('fuzzy', 'fuzzy_artifact_threshold_hz', default=12000.0)
        self.boost_words = self._get_nested_config('fuzzy', 'fuzzy_boost_words', default=['ahh', 'mmm', 'ooh', 'gasp', 'oh', 'fuck', 'yes', 'aah', 'gods'])
        self.boost_amount = self._get_nested_config('fuzzy', 'fuzzy_boost_amount', default=0.15)
        self.skip_words = self._get_nested_config('fuzzy', 'fuzzy_force_skip_words', default=[])

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "fuzzy_audio_cache.json"

        # In-memory cache structure...
        self.cache_data: Dict[str, Dict[str, Dict]] = {}
        self.cache_lock = threading.RLock()
        self.save_interval = 5.0
        self.fuzzy_queue = Queue(maxsize=0)  # Non-blocking
        self.last_save_time = 0
        self.last_cleanup_time = 0
        self.save_counter = 0
        self._save_event = threading.Event()  # NEW: Save signal for background worker

        # FIXED: Base dir...
        self.base_dir = self.cache_dir.parent  # "cache" – ensures all paths relative to root

        # Load existing cache (only once)
        self.load_cache()

        # Start background indexer (only once)
        self._start_index_worker()

        # Singleton flag (set after full init)
        self._initialized = True

        logger.info(f"Fuzzy cache singleton initialized at {self.cache_file} with threshold={self.threshold:.2f}, max_index_size={self.max_index_size}")

    def get_stats(self) -> Dict[str, Any]:
        with self.cache_lock:
            total_entries = sum(len(entries) for entries in self.cache_data.values())
        return {
            "entries": total_entries,
            "stems": len(self.cache_data),
            "memory_entries": total_entries,
            "threshold": self.threshold,
            "enabled": self.enable_fuzzy,
            "artifact_purge": self.enable_artifact_purge,
            "artifact_threshold_hz": self.artifact_threshold_hz,
            "max_entries_per_stem": self.max_entries_per_stem,
            "index_size_limit": self.max_index_size,
            "instance_id": hex(id(self))  # Hex ID to track singleton
        }

    def clear(self, voice_stem: Optional[str] = None) -> None:
        """Clear fuzzy cache, optionally for a specific voice stem."""
        with self.cache_lock:
            if voice_stem and voice_stem in self.cache_data:
                del self.cache_data[voice_stem]
                logger.info(f"Cleared fuzzy cache for voice: {voice_stem}")
            else:
                self.cache_data.clear()
                logger.info("Cleared entire fuzzy cache")
            self._save_cache(save_all=True, background_cleanup=True)

    def index_audio(self, text: str, wav_path: str, voice_stem: str) -> None:
        """Add audio to fuzzy cache for future matching. FIXED: Ensure absolute path when queuing (subpath safe)."""
        if not self.enable_fuzzy:
            logger.debug("Fuzzy cache disabled in config - skipping index")
            return

        if not wav_path or not os.path.exists(wav_path):
            logger.debug(f"Skipping index for non-existent path: {wav_path}")
            return

        # FIXED: Ensure wav_path is absolute str (resolve early; prevents relative issues in worker)
        wav_path_abs = str(Path(wav_path).resolve().absolute())
        if not Path(wav_path_abs).is_relative_to(self.base_dir):
            logger.warning(f"Skipping index: Path not in subpath of {self.base_dir}: {wav_path_abs}")
            return

        # Skip very short text
        if len(text.strip()) < self.min_length:
            logger.trace(f"Skipped indexing short text (<{self.min_length}): {text[:10] if text else 'N/A'}")
            return

        # Normalize text for index
        norm_key = self.normalize_text(text)
        clean_text = re.sub(r'[^\w\s]', '', text.lower()).strip()
        words = set(clean_text.split())

        # Skip if any force skip words are detected
        if self.skip_words:
            for skip_word in self.skip_words:
                if skip_word.lower() in words:
                    logger.info(f"Fuzzy index SKIP: force skip word '{skip_word}' detected in text '{text[:30]}...'")
                    return

        # Skip artifacts
        if self.enable_artifact_purge and is_artifact_laden(wav_path_abs, threshold_hz=self.artifact_threshold_hz):
            logger.warning(
                f"Skip fuzzy index: Artifacts in {wav_path_abs} (centroid >{self.artifact_threshold_hz}Hz) for '{text[:20]}' (stem: {voice_stem})"
            )
            return

        # Normalize text for index
        norm_key = self.normalize_text(text)
        clean_text = re.sub(r'[^\w\s]', '', text.lower()).strip()

        # Calculate similarity boost
        sim_boost = 0.0
        if self.boost_amount > 0 and self.boost_words:
            for word in self.boost_words:
                if word in clean_text:
                    sim_boost = self.boost_amount
                    break

        # Queue for background processing (use abs path)
        try:
            import torchaudio
            duration = 0.0
            try:
                info = torchaudio.info(wav_path_abs)
                duration = float(info.num_frames) / float(info.sample_rate)
            except Exception as e:
                logger.debug(f"Failed to get duration for fuzzy index: {e}")

            self.fuzzy_queue.put((text, wav_path_abs, voice_stem, sim_boost, duration), block=False)
            logger.trace(f"Queued fuzzy index: {norm_key[:30]} for {voice_stem}")
        except Exception as e:
            logger.warning(f"Failed to queue fuzzy index: {str(e)}")

    def try_fuzzy_audio_cache(
        self,
        audio_path: str,
        text_input: str,
        stem: str,
        threshold: Optional[float] = None
    ) -> Optional[str]:
        """Check for fuzzy audio cache hit. FIXED: Return absolute path if valid (subpath safe)."""
        if not self.enable_fuzzy:
            logger.debug("Fuzzy cache disabled in config - skipping check")
            return None

        if not audio_path or not stem or not text_input:
            logger.debug(f"Fuzzy skip: Missing required parameters (audio_path={audio_path}, stem={stem}, text_input={text_input})")
            return None

        threshold = threshold or self.threshold

        # Validate text length
        clean_text = re.sub(r'[^\w\s]', '', text_input.lower()).strip()
        words = set(clean_text.split())

        # Immediate rejection for force skip words
        if self.skip_words:
            for skip_word in self.skip_words:
                if skip_word.lower() in words:
                    logger.debug(f"Fuzzy HIT rejected: force skip word '{skip_word}' detected in input")
                    return None

        if len(clean_text.split()) < self.min_length // 2:  # Word-based too (e.g., "aah..." → short)
            logger.trace(f"Fuzzy MISS early: Normalized text too short ('{clean_text}')")
            return None

        # Early exit if no entries for stem
        with self.cache_lock:
            stem_entries = self.cache_data.get(stem, {})
            if not stem_entries:
                logger.trace(f"Fuzzy MISS: No entries for stem '{stem}'")
                return None

            # Optimization: If we have an exact text match, return immediately (fast path)
            norm_key = self.normalize_text(text_input)
            if norm_key in stem_entries:
                entry = stem_entries[norm_key]
                best_match = entry['orig_text']
                best_path = entry['wav_path']
                best_sim = 1.0
                candidates = [] # Skip loop
            else:
                # Convert to list for iteration
                candidates = list(stem_entries.values())

        # Find best match (if not found in fast path)
        if norm_key in stem_entries:
            # We already have best_match, best_path, best_sim from fast path above
            pass
        elif not candidates:
             return None
        else:
            best_match, best_path, best_sim = None, None, 0.0
            for entry in candidates:
                clean_entry = re.sub(r'[^\w\s]', '', entry['orig_text'].lower()).strip()
                # Fast similarity pre-check: if length difference is too large, it can't be a hit
                len_diff = abs(len(clean_text) - len(clean_entry))
                max_len = max(len(clean_text), len(clean_entry))
                if max_len > 0 and (1.0 - (len_diff / max_len)) < (threshold - 0.2):
                    continue

                raw_ratio = SequenceMatcher(None, clean_text, clean_entry).ratio()

                # Apply boost logic
                sim_boost = entry.get('sim_boost', 0.0)
                adjusted_sim = min(1.0, raw_ratio + sim_boost)

                if adjusted_sim > best_sim:
                    best_sim = adjusted_sim
                    best_match = entry['orig_text']
                    best_path = entry['wav_path']
                    if best_sim >= 0.98: # Good enough for early exit
                        break

        # Check if we have a hit
        if best_sim >= threshold and best_path:
            # FIXED: Resolve best_path to absolute and validate subpath (safety)
            best_path_abs = str(Path(best_path).resolve().absolute())
            if not os.path.exists(best_path_abs):
                logger.trace(f"Fuzzy MISS: Best candidate '{best_path_abs}' doesn't exist")
                return None
            
            # Use cached validation if available
            norm_key = self.normalize_text(best_match)
            entry = stem_entries.get(norm_key)
            
            is_artifact_free = entry.get('is_artifact_free') if entry else None
            
            if is_artifact_free is False:
                logger.debug(f"Fuzzy HIT rejected (cached artifact): {best_path_abs}")
                return None

            if not Path(best_path_abs).is_relative_to(self.base_dir):
                logger.warning(f"Fuzzy HIT invalid subpath for {best_path_abs} – purging entry")
                with self.cache_lock:
                    if stem in self.cache_data and norm_key in self.cache_data[stem]:
                        del self.cache_data[stem][norm_key]
                        if not self.cache_data[stem]:
                            del self.cache_data[stem]
                # Start background saving to avoid blocking the pipeline
                self._save_event.set()
                return None

            # Validate for artifacts if enabled and not already known to be free
            if is_artifact_free is None and self.enable_artifact_purge:
                if is_artifact_laden(best_path_abs, threshold_hz=self.artifact_threshold_hz):
                    logger.warning(
                        f"Fuzzy HIT invalid: Artifacts in {best_path_abs} (centroid >{self.artifact_threshold_hz}Hz) – purging entry"
                    )
                    with self.cache_lock:
                        if stem in self.cache_data and norm_key in self.cache_data[stem]:
                            del self.cache_data[stem][norm_key]
                            if not self.cache_data[stem]:
                                del self.cache_data[stem]
                    # Start background saving with cleanup to avoid blocking the pipeline
                    self._save_event.set()
                    return None
                else:
                    # Cache the successful result
                    if entry:
                        entry['is_artifact_free'] = True

            logger.info(
                f"Fuzzy cache HIT: '{text_input[:30]}...' ≈ '{best_match[:30]}...' "
                f"(sim={best_sim:.3f} >= {threshold:.2f}, boost={self.boost_amount if self.boost_amount else 0.0}) "
                f"for stem '{stem}' -> {best_path_abs}"
            )
            return best_path_abs  # FIXED: Return abs str for safety
        else:
            if best_path and not os.path.exists(best_path):
                logger.trace(f"Fuzzy MISS: Best candidate '{best_path}' doesn't exist")
            elif best_sim < threshold:
                logger.trace(f"Fuzzy MISS: Best similarity {best_sim:.3f} < threshold {threshold:.2f}")
            return None

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for fuzzy indexing."""
        cleaned = re.sub(r'[^\w\s]', '', text.lower())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        text_hash = hashlib.md5(cleaned.encode()).hexdigest()[:8]
        return f"{text_hash}_{cleaned}"

    def _start_index_worker(self) -> None:
        """Start background thread for async index processing. FIXED: Path subpath validation in worker (skip invalid)."""
        def worker():
            self.last_cleanup_time = time.time()
            while True:
                try:
                    queue_item = self.fuzzy_queue.get(timeout=1)
                    if queue_item is None: # Shutdown signal
                        break
                    
                    text, wav_path, voice_stem, sim_boost, duration = queue_item
                    orig_text = text

                    # FIXED: Validate wav_path subpath early (skip if not under base_dir – fixes "not in subpath")
                    wav_path_p = Path(wav_path).resolve()
                    if not wav_path_p.is_relative_to(self.base_dir):
                        logger.trace(f"Fuzzy worker skip: Path not in subpath of {self.base_dir}: {wav_path} (stem: {voice_stem})")
                        self.fuzzy_queue.task_done()
                        continue

                    if not wav_path_p.exists():
                        logger.trace(f"Fuzzy worker skip: Path doesn't exist: {wav_path}")
                        self.fuzzy_queue.task_done()
                        continue

                    # Skip very short text
                    min_length = self.min_length
                    if len(orig_text.strip()) < min_length:
                        logger.trace(f"Skipped indexing short text (<{min_length}): {orig_text[:10]}...")
                        self.fuzzy_queue.task_done()
                        continue

                    # Skip if force skip words detected in worker
                    clean_worker_text = re.sub(r'[^\w\s]', '', orig_text.lower()).strip()
                    worker_words = set(clean_worker_text.split())
                    if self.skip_words:
                        skip_detected = False
                        for skip_word in self.skip_words:
                            if skip_word.lower() in worker_words:
                                logger.debug(f"Fuzzy worker skip: force skip word '{skip_word}' in '{orig_text[:20]}...'")
                                skip_detected = True
                                break
                        if skip_detected:
                            self.fuzzy_queue.task_done()
                            continue

                    # Skip artifacts (shouldn't happen after pre-filter but double checking)
                    if self.enable_artifact_purge and is_artifact_laden(str(wav_path_p), threshold_hz=self.artifact_threshold_hz):
                        logger.trace(
                            f"Skip fuzzy index: Artifacts during processing for '{orig_text[:20]}' (stem: {voice_stem})"
                        )
                        self.fuzzy_queue.task_done()
                        continue

                    # Generate normalized key
                    norm_key = self.normalize_text(text)

                    # Store entry
                    with self.cache_lock:
                        # Initialize stem entry if needed
                        if voice_stem not in self.cache_data:
                            self.cache_data[voice_stem] = {}

                        # Skip duplicate entries
                        if norm_key in self.cache_data[voice_stem]:
                            logger.trace(f"Skipped dup fuzzy index: {norm_key[:30]} ({voice_stem})")
                        else:
                            # Enforce per-stem limit
                            if len(self.cache_data[voice_stem]) >= self.max_index_size:
                                # Evict oldest entry (by time_indexed)
                                old_len = len(self.cache_data[voice_stem])
                                oldest_key = min(
                                    self.cache_data[voice_stem].items(),
                                    key=lambda x: x[1].get('time_indexed', 0)
                                )[0]
                                del self.cache_data[voice_stem][oldest_key]
                                new_len = len(self.cache_data[voice_stem])
                                logger.info(f"Per-stem eviction in '{voice_stem}': {old_len - new_len} deleted (limit={self.max_index_size}), now {new_len} entries")

                            # Add new entry (use resolved abs path)
                            self.cache_data[voice_stem][norm_key] = {
                                'wav_path': str(wav_path_p),  # FIXED: Store abs str
                                'orig_text': orig_text,
                                'stem': voice_stem,
                                'sim_boost': sim_boost,
                                'duration': duration,
                                'is_artifact_free': True, # It passed pre-filter if it got here
                                'time_indexed': time.time()
                            }
                        
                        # Apply global size limit check if needed
                        total_entries = sum(len(entries) for entries in self.cache_data.values())
                        if total_entries > self.max_index_size:
                            # Global eviction (LRU across all stems)
                            all_entries = []
                            for s, entries in self.cache_data.items():
                                for k, e in entries.items():
                                    all_entries.append((s, k, e.get('time_indexed', 0)))
                            
                            if all_entries:
                                # Sort by time_indexed
                                all_entries.sort(key=lambda x: x[2])
                                # Evict until under limit
                                to_evict = total_entries - self.max_index_size
                                for i in range(min(len(all_entries), to_evict)):
                                    s, k, _ = all_entries[i]
                                    del self.cache_data[s][k]
                                    if not self.cache_data[s]:
                                        del self.cache_data[s]
                                logger.info(f"Global fuzzy eviction: {to_evict} entries deleted (limit={self.max_index_size})")

                        # Save conditions: Every 25 indexes OR idle >60s
                        time_since_last = time.time() - self.last_save_time
                        time_since_last_cleanup = time.time() - getattr(self, 'last_cleanup_time', 0)
                        if (self.save_counter >= 25 or
                            time_since_last > 60 or
                            self._save_event.is_set()):
                            # Signal worker to save synchronously within its own thread
                            # Perform background cleanup periodically or on explicit save event
                            do_cleanup = (self.save_counter >= 25 or self._save_event.is_set() or time_since_last_cleanup > 300)
                            self._save_cache(save_all=False, background_cleanup=do_cleanup)
                            if do_cleanup:
                                self.last_cleanup_time = time.time()
                            self.save_counter = 0
                            self._save_event.clear()

                    self.fuzzy_queue.task_done()

                except Empty:
                    # Check for idle save condition or signaled save
                    time_since_last = time.time() - self.last_save_time
                    time_since_last_cleanup = time.time() - getattr(self, 'last_cleanup_time', 0)
                    if (time_since_last > 60 and self.save_counter > 0) or self._save_event.is_set():
                        # Idle save: often good to do a cleanup too if it's been a while
                        do_cleanup = self._save_event.is_set() or time_since_last_cleanup > 300
                        self._save_cache(save_all=True, background_cleanup=do_cleanup)
                        if do_cleanup:
                            self.last_cleanup_time = time.time()
                        self.save_counter = 0
                        self._save_event.clear()
                    continue

                except Exception as e:
                    logger.error(f"Fuzzy worker error: {str(e)}")
                    self.fuzzy_queue.task_done()
                    continue

        threading.Thread(
            target=worker,
            daemon=True,
            name="FuzzyIndexer"
        ).start()
        logger.info("Fuzzy cache background indexer started")

    def load_cache(self) -> None:
        """Load fuzzy cache from disk persistence. FIXED: Enhance subpath validation on load (skip invalid relatives)."""
        if not self.cache_file.exists():
            logger.debug("No fuzzy cache file – starting empty")
            return

        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)

            with self.cache_lock:
                self.cache_data = {}
                total_entries = 0
                purged_count = 0

                for stem, stem_entries in data.items():
                    self.cache_data[stem] = {}
                    if not isinstance(stem_entries, dict):
                        logger.trace(f"Load skip invalid stem_entries for {stem}: not dict")
                        purged_count += len(stem_entries) if isinstance(stem_entries, (list, dict)) else 1
                        continue

                    for norm_key, entry in stem_entries.items():
                        if 'wav_path' not in entry:
                            logger.trace(f"Load skip missing wav_path: {stem}:{norm_key}")
                            purged_count += 1
                            continue

                        # FIXED: Convert relative to absolute and validate subpath
                        rel_path = entry['wav_path']
                        full_path = Path(rel_path)
                        if not full_path.is_absolute():
                            full_path = self.base_dir / rel_path  # FIXED: Resolve from base_dir ("cache")

                        full_path_res = full_path.resolve()
                        if not full_path_res.exists():
                            logger.trace(f"Skipped invalid path: {stem}:{norm_key} ({full_path_res})")
                            purged_count += 1
                            continue

                        # FIXED: Subpath check on resolved path
                        if not full_path_res.is_relative_to(self.base_dir):
                            logger.trace(f"Load skip: Path not in subpath of {self.base_dir}: {full_path_res} ({stem}:{norm_key})")
                            purged_count += 1
                            continue

                        # Skip artifact-laden files
                        if self.enable_artifact_purge and is_artifact_laden(str(full_path_res), self.artifact_threshold_hz):
                            logger.trace(
                                f"Load-time purge: Artifacts in {full_path_res} for {stem}:{norm_key} – skipping"
                            )
                            purged_count += 1
                            continue

                        # Store valid entry
                        entry['wav_path'] = str(full_path_res)  # FIXED: Abs str
                        self.cache_data[stem][norm_key] = entry
                        total_entries += 1

                    # Clean up empty stems
                    if not self.cache_data[stem]:
                        del self.cache_data[stem]

                # Log results (trimmed: only INFO if >0 purged; else trace)
                if purged_count > 0:
                    logger.info(f"Loaded fuzzy cache: {total_entries} entries across {len(self.cache_data)} stems (purged {purged_count})")
                else:
                    logger.info(f"Loaded fuzzy cache: {total_entries} entries across {len(self.cache_data)} stems")

        except Exception as e:
            logger.warning(f"Load fuzzy cache failed: {str(e)}")
            self.cache_data = {}

    def _save_cache(self, save_all: bool = False, background_cleanup: bool = False) -> None:
        """Save fuzzy cache to disk with throttling. FIXED: Ensure valid relatives on save (subpath safe)."""
        now = time.time()
        if not save_all and (now - self.last_save_time) < self.save_interval:
            return

        self.last_save_time = now

        with self.cache_lock:
            # Skip if empty
            if not self.cache_data:
                logger.trace("No data to save in fuzzy cache")
                return

            # Pre-save validation and purging
            purged_count = 0
            if background_cleanup:
                if self.enable_artifact_purge:
                    for stem in list(self.cache_data.keys()):
                        for norm_key in list(self.cache_data[stem].keys()):
                            entry = self.cache_data[stem][norm_key]
                            wav_path = entry.get('wav_path', '')

                            # Remove entries with missing files
                            if not wav_path or not os.path.exists(wav_path):
                                del self.cache_data[stem][norm_key]
                                purged_count += 1
                                continue

                            # FIXED: Subpath validation on save (purge if invalid)
                            wav_path_p = Path(wav_path).resolve()
                            if not wav_path_p.is_relative_to(self.base_dir):
                                logger.trace(f"Save-time purge: Path not in subpath of {self.base_dir}: {wav_path} – deleting entry")
                                del self.cache_data[stem][norm_key]
                                purged_count += 1
                                continue

                            # Remove entries with artifacts
                            if is_artifact_laden(wav_path, threshold_hz=self.artifact_threshold_hz):
                                logger.trace(f"Save-time purge: Artifacts in {wav_path} for {stem} – deleting entry")
                                del self.cache_data[stem][norm_key]
                                purged_count += 1

                        # Clean up empty stems
                        if not self.cache_data[stem]:
                            del self.cache_data[stem]

                # Purge force skip words
                if self.skip_words:
                    for stem in list(self.cache_data.keys()):
                        for norm_key in list(self.cache_data[stem].keys()):
                            entry = self.cache_data[stem][norm_key]
                            orig_text = entry.get('orig_text', '')
                            if orig_text:
                                clean_entry_text = re.sub(r'[^\w\s]', '', orig_text.lower()).strip()
                                entry_words = set(clean_entry_text.split())
                                for skip_word in self.skip_words:
                                    if skip_word.lower() in entry_words:
                                        logger.info(f"Purging fuzzy entry due to skip word '{skip_word}': {orig_text[:30]}...")
                                        del self.cache_data[stem][norm_key]
                                        purged_count += 1
                                        break
                        if not self.cache_data[stem]:
                            del self.cache_data[stem]

            # Convert to save format (relative paths)
            save_data = {}
            for stem, stem_entries in self.cache_data.items():
                if not stem_entries:
                    continue

                save_data[stem] = {}
                for norm_key, entry in stem_entries.items():
                    # FIXED: Ensure abs wav_path; compute relative safely
                    abs_path = Path(entry['wav_path']).resolve()
                    if not abs_path.is_relative_to(self.base_dir):
                        logger.trace(f"Save skip: Invalid subpath for {abs_path} – purging")
                        continue  # Skip bad entry
                    rel_path = abs_path.relative_to(self.base_dir)
                    save_entry = entry.copy()
                    save_entry['wav_path'] = str(rel_path)
                    save_data[stem][norm_key] = save_entry

            # Apply global size restriction (if needed beyond per-stem)
            total_entries = sum(len(entries) for entries in save_data.values())
            if total_entries > self.max_index_size * len(save_data):  # Scale with stems
                logger.info(f"Global fuzzy eviction: Total {total_entries} > limit {self.max_index_size * len(save_data)}")

            # Save to disk (trimmed log: only INFO if changed; else trace)
            try:
                with open(self.cache_file, 'w') as f:
                    json.dump(save_data, f, indent=2)

                if purged_count > 0 or save_all:
                    logger.info(f"Fuzzy cache saved: {total_entries} entries across {len(save_data)} stems (purged {purged_count})")
                else:
                    logger.trace(f"Fuzzy cache incremental save: {total_entries} entries")
            except Exception as e:
                logger.error(f"Failed to save fuzzy cache: {str(e)}")

    def __del__(self):
        """Ensure queue is properly drained on destruction."""
        try:
            if hasattr(self, 'fuzzy_queue'):
                # Signal worker to stop
                self.fuzzy_queue.put(None)
                
                # Drain the queue
                while not self.fuzzy_queue.empty():
                    try:
                        self.fuzzy_queue.get_nowait()
                        self.fuzzy_queue.task_done()
                    except:
                        break
                # Force save on shutdown
                self._save_cache(save_all=True, background_cleanup=True)
        except:
            pass