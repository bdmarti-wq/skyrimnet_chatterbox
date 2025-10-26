# src/generate/cache/fuzzy_cache.py
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

from src.config import get_config, get_config_value
from src.audio_utils import is_artifact_laden
from src.normalize_stem import normalize_stem

class FuzzyAudioCache:
    """Class-based implementation of fuzzy audio cache with proper encapsulation."""

    def __init__(self, cache_dir: Path, threshold: float = 0.75):
        """Initialize fuzzy cache with configurable threshold.

        Args:
            cache_dir: Directory for cache persistence
            threshold: Minimum similarity threshold for cache hits
        """
        # Configuration values (from config system)
        self.config = get_config()

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "fuzzy_audio_cache.json"

        self.threshold = threshold
        self.min_length = get_config_value('fuzzy_min_length', default=3)
        self.max_index_size = get_config_value('fuzzy.fuzzy_index_size', default=1000)
        self.enable_fuzzy = get_config_value('fuzzy.enable_fuzzy_cache', default=True)
        self.enable_artifact_purge = get_config_value('fuzzy.fuzzy_artifact_purge_enable', default=True)
        self.artifact_threshold_hz = get_config_value('fuzzy.fuzzy_artifact_threshold_hz', default=8000.0)
        self.boost_words = get_config_value(
            'fuzzy.fuzzy_boost_words',
            default=['ahh', 'mmm', 'ooh', 'throbb', 'moan', 'gasp', 'oh', 'fuck', 'yes', 'aah', 'gods']
        )
        self.boost_amount = get_config_value('fuzzy.fuzzy_boost_amount', default=0.15)

        # In-memory cache structure: {stem: {normalized_key: entry}}
        self.cache_data: Dict[str, Dict[str, Dict]] = {}
        self.cache_lock = threading.RLock()
        self.save_interval = 5.0
        self.fuzzy_queue = Queue(maxsize=0)  # Non-blocking
        self.last_save_time = 0
        self.save_counter = 0

        # FIXED: Base dir for subpath validation (e.g., "cache" root for all audio/output/resampled subpaths)
        self.base_dir = self.cache_dir.parent  # "cache" – ensures all paths relative to root

        # Load existing cache
        self.load_cache()

        # Start background indexer
        self._start_index_worker()

        logger.info(f"Fuzzy cache initialized at {self.cache_file} with threshold={self.threshold:.2f}")

    def get_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the fuzzy cache."""
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
                "index_size_limit": self.max_index_size
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
            self._save_cache(save_all=True)

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
            logger.debug(f"Skipped indexing short text (<{self.min_length}): {text[:10]}")
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
            self.fuzzy_queue.put((text, wav_path_abs, voice_stem, sim_boost), block=False)
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
        min_length = self.min_length
        clean_text = re.sub(r'[^\w\s]', '', text_input.lower()).strip()
        if len(clean_text.split()) < min_length // 2:  # Word-based too (e.g., "aah..." → short)
            logger.debug(f"Fuzzy MISS early: Normalized text too short ('{clean_text}')")
            return None

        # Derive stem from audio path if not provided
        if not stem:
            stem = normalize_stem(audio_path)
        if not stem or len(stem) < 3:
            stem = 'global'  # Fallback to global cache

        # Early exit if no entries for stem
        with self.cache_lock:
            stem_entries = self.cache_data.get(stem, {})
            if not stem_entries:
                logger.debug(f"Fuzzy MISS: No entries for stem '{stem}'")
                return None

            # Convert to list for iteration
            candidates = list(stem_entries.values())

        # Find best match
        best_match, best_path, best_sim = None, None, 0.0

        for entry in candidates:
            clean_entry = re.sub(r'[^\w\s]', '', entry['orig_text'].lower()).strip()
            raw_ratio = SequenceMatcher(None, clean_text, clean_entry).ratio()

            # Apply boost logic
            sim_boost = entry.get('sim_boost', 0.0)
            adjusted_sim = min(1.0, raw_ratio + sim_boost)

            if adjusted_sim > best_sim:
                best_sim = adjusted_sim
                best_match = entry['orig_text']
                best_path = entry['wav_path']

        # Check if we have a hit
        if best_sim >= threshold and best_path:
            # FIXED: Resolve best_path to absolute and validate subpath (safety)
            best_path_abs = str(Path(best_path).resolve().absolute())
            if not os.path.exists(best_path_abs):
                logger.debug(f"Fuzzy MISS: Best candidate '{best_path_abs}' doesn't exist")
                return None
            if not Path(best_path_abs).is_relative_to(self.base_dir):
                logger.warning(f"Fuzzy HIT invalid subpath for {best_path_abs} – purging entry")
                with self.cache_lock:
                    norm_key = self.normalize_text(best_match)
                    if stem in self.cache_data and norm_key in self.cache_data[stem]:
                        del self.cache_data[stem][norm_key]
                        if not self.cache_data[stem]:
                            del self.cache_data[stem]
                self._save_cache(save_all=True)  # Force save after purge
                return None

            # Validate for artifacts if enabled
            if self.enable_artifact_purge and is_artifact_laden(best_path_abs, threshold_hz=self.artifact_threshold_hz):
                logger.warning(
                    f"Fuzzy HIT invalid: Artifacts in {best_path_abs} (centroid >{self.artifact_threshold_hz}Hz) – purging entry"
                )
                with self.cache_lock:
                    norm_key = self.normalize_text(best_match)
                    if stem in self.cache_data and norm_key in self.cache_data[stem]:
                        del self.cache_data[stem][norm_key]
                        if not self.cache_data[stem]:
                            del self.cache_data[stem]
                self._save_cache(save_all=True)  # Force save after purge
                return None

            logger.info(
                f"Fuzzy cache HIT: '{text_input[:30]}...' ≈ '{best_match[:30]}...' "
                f"(sim={best_sim:.3f} >= {threshold:.2f}, boost={self.boost_amount}) "
                f"for stem '{stem}' -> {best_path_abs}"
            )
            return best_path_abs  # FIXED: Return abs str for safety
        else:
            if best_path and not os.path.exists(best_path):
                logger.debug(f"Fuzzy MISS: Best candidate '{best_path}' doesn't exist")
            elif best_sim < threshold:
                logger.debug(f"Fuzzy MISS: Best similarity {best_sim:.3f} < threshold {threshold:.2f}")
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
            while True:
                try:
                    text, wav_path, voice_stem, sim_boost = self.fuzzy_queue.get(timeout=1)
                    orig_text = text

                    # FIXED: Validate wav_path subpath early (skip if not under base_dir – fixes "not in subpath")
                    wav_path_p = Path(wav_path).resolve()
                    if not wav_path_p.is_relative_to(self.base_dir):
                        logger.warning(f"Fuzzy worker skip: Path not in subpath of {self.base_dir}: {wav_path} (stem: {voice_stem})")
                        self.fuzzy_queue.task_done()
                        continue

                    if not wav_path_p.exists():
                        logger.debug(f"Fuzzy worker skip: Path doesn't exist: {wav_path}")
                        self.fuzzy_queue.task_done()
                        continue

                    # Skip very short text
                    min_length = self.min_length
                    if len(orig_text.strip()) < min_length:
                        logger.debug(f"Skipped indexing short text (<{min_length}): {orig_text[:10]}...")
                        self.fuzzy_queue.task_done()
                        continue

                    # Skip artifacts (shouldn't happen after pre-filter but double checking)
                    if self.enable_artifact_purge and is_artifact_laden(str(wav_path_p), threshold_hz=self.artifact_threshold_hz):
                        logger.warning(
                            f"Skip fuzzy index: Artifacts detected during processing for '{orig_text[:20]}' (stem: {voice_stem})"
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
                            logger.debug(f"Skipped dup fuzzy index: {norm_key[:30]} ({voice_stem})")
                        else:
                            # Enforce per-stem limit
                            if len(self.cache_data[voice_stem]) >= self.max_index_size:
                                # Evict oldest entry (by time_indexed)
                                oldest_key = min(
                                    self.cache_data[voice_stem].items(),
                                    key=lambda x: x[1].get('time_indexed', 0)
                                )[0]
                                del self.cache_data[voice_stem][oldest_key]
                                logger.debug(f"Pruned old entry in {voice_stem} - limit reached")

                            # Add new entry (use resolved abs path)
                            self.cache_data[voice_stem][norm_key] = {
                                'wav_path': str(wav_path_p),  # FIXED: Store abs str
                                'orig_text': orig_text,
                                'stem': voice_stem,
                                'sim_boost': sim_boost,
                                'time_indexed': time.time()
                            }

                        # Handle save throttling
                        self.save_counter += 1
                        time_since_last = time.time() - self.last_save_time

                        # Save conditions:
                        # 1. Every 10 index adds
                        # 2. When queue has more than 20 items
                        # 3. When 30+ seconds have passed since last save
                        if (self.save_counter >= 10 or
                            self.fuzzy_queue.qsize() > 20 or
                            time_since_last > 30):
                            self._save_cache(save_all=False)
                            self.save_counter = 0

                    self.fuzzy_queue.task_done()

                except Empty:
                    # Check for idle save condition
                    time_since_last = time.time() - self.last_save_time
                    if time_since_last > 30 and self.cache_data:
                        self._save_cache(save_all=True)
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
                            logger.warning(f"Load skip: Path not in subpath of {self.base_dir}: {full_path_res} ({stem}:{norm_key})")
                            purged_count += 1
                            continue

                        # Skip artifact-laden files
                        if self.enable_artifact_purge and is_artifact_laden(str(full_path_res), self.artifact_threshold_hz):
                            logger.warning(
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

                # Log results
                if purged_count > 0:
                    logger.info(
                        f"Loaded fuzzy cache: {total_entries} valid entries across {len(self.cache_data)} stems "
                        f"(purged {purged_count} bad/invalid)"
                    )
                else:
                    logger.info(
                        f"Loaded fuzzy cache: {total_entries} entries across {len(self.cache_data)} stems"
                    )

        except Exception as e:
            logger.warning(f"Load fuzzy cache failed: {str(e)}")
            self.cache_data = {}

    def _save_cache(self, save_all: bool = False) -> None:
        """Save fuzzy cache to disk with throttling. FIXED: Ensure valid relatives on save (subpath safe)."""
        now = time.time()
        if not save_all and (now - self.last_save_time) < self.save_interval:
            return

        self.last_save_time = now

        with self.cache_lock:
            # Skip if empty
            if not self.cache_data:
                return

            # Pre-save validation and purging
            purged_count = 0
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
                            logger.warning(f"Save-time purge: Path not in subpath of {self.base_dir}: {wav_path} – deleting entry")
                            del self.cache_data[stem][norm_key]
                            purged_count += 1
                            continue

                        # Remove entries with artifacts
                        if is_artifact_laden(wav_path, threshold_hz=self.artifact_threshold_hz):
                            del self.cache_data[stem][norm_key]
                            purged_count += 1

                    # Clean up empty stems
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
                        logger.warning(f"Save skip: Invalid subpath for {abs_path} – purging")
                        continue  # Skip bad entry
                    rel_path = abs_path.relative_to(self.base_dir)
                    save_entry = entry.copy()
                    save_entry['wav_path'] = str(rel_path)
                    save_data[stem][norm_key] = save_entry

            # Apply global size restriction
            total_entries = sum(len(entries) for entries in save_data.values())
            if total_entries > self.max_index_size * 2:  # 2x the per-stem limit
                for stem in list(save_data.keys()):
                    while len(save_data[stem]) > self.max_index_size:
                        oldest_key = min(save_data[stem].keys(), key=lambda k: save_data[stem][k].get('time_indexed', 0))
                        del save_data[stem][oldest_key]

            # Save to disk
            try:
                with open(self.cache_file, 'w') as f:
                    json.dump(save_data, f, indent=2)

                if purged_count > 0 or save_all:
                    logger.info(
                        f"Fuzzy cache saved: {total_entries} entries across {len(save_data)} stems "
                        f"(purged {purged_count} bad/invalid entries)"
                    )
                else:
                    logger.trace(f"Fuzzy cache incremental save: {total_entries} entries")
            except Exception as e:
                logger.error(f"Failed to save fuzzy cache: {str(e)}")

    def __del__(self):
        """Ensure queue is properly drained on destruction."""
        try:
            if hasattr(self, 'fuzzy_queue'):
                while not self.fuzzy_queue.empty():
                    try:
                        self.fuzzy_queue.get_nowait()
                        self.fuzzy_queue.task_done()
                    except:
                        break
                # Force save on shutdown
                self._save_cache(save_all=True)
        except:
            pass