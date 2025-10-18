#audio_cache.py
import os
import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

class AudioCache:
    """Manages caching of generated TTS audio to avoid redundant generations."""

    def __init__(self, cache_dir: Path):
        """Initialize the audio cache system."""
        """Initialize the audio cache system."""
        # CRITICAL: Use proper audio subdirectories
        self.cache_dir = cache_dir / "audio" / "output"
        self.cache_file = self.cache_dir / "audio_cache.json"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache of audio paths
        self.audio_cache: Dict[str, str] = {}
        self.cache_lock = threading.RLock()

        # Load existing cache
        self.load_cache()

        logger.info(f"Audio cache initialized at {self.cache_dir}")

    def load_cache(self) -> None:
        """Load audio cache metadata from disk to memory."""
        with self.cache_lock:
            if self.cache_file.exists():
                try:
                    with open(self.cache_file, 'r') as f:
                        self.audio_cache = json.load(f)
                    logger.info(f"Loaded {len(self.audio_cache)} audio cache entries")
                except Exception as e:
                    logger.error(f"Audio cache load failed: {str(e)} - starting with empty cache")
                    self.audio_cache = {}
            else:
                self.audio_cache = {}

    def save_cache(self) -> None:
        """Save audio cache metadata from memory to disk."""
        with self.cache_lock:
            try:
                with open(self.cache_file, 'w') as f:
                    json.dump(self.audio_cache, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save audio cache: {str(e)}")

    def get_key(self, voice_stem: str, text: str, exaggeration: float) -> str:
        """Generate a cache key for audio generation."""
        from hashlib import md5
        text_hash = md5(text.encode('utf-8')).hexdigest()[:8]
        return f"{voice_stem}_{text_hash}_{exaggeration:.2f}"

    def get(self, key: str) -> Optional[str]:
        """Get the path to cached audio for a given key."""
        with self.cache_lock:
            path = self.audio_cache.get(key)
            if path and os.path.exists(path):
                logger.debug(f"Audio cache hit: {key} → {path}")
                return path
            return None

    def set(self, key: str, path: str) -> None:
        """Cache an audio path for a given key."""
        if not os.path.exists(path):
            logger.warning(f"Cannot cache non-existent path: {path}")
            return

        with self.cache_lock:
            self.audio_cache[key] = path
            self.save_cache()
            logger.debug(f"Audio cached: {key} → {path}")

    def clear(self) -> None:
        """Clear the audio cache."""
        with self.cache_lock:
            self.audio_cache.clear()
            self.save_cache()
            logger.info("Cleared audio cache")


    def get_stats(self) -> Dict[str, Any]:
        """Get audio cache statistics with safety fallbacks."""
        try:
            entries = len(self.audio_cache)
            # Calculate total size safely
            total_size = 0
            for path in self.audio_cache.values():
                try:
                    if os.path.exists(path):
                        total_size += os.path.getsize(path)
                except:
                    pass

            return {
                "entries": entries,
                "total_size": total_size,
                "memory_entries": entries,  # For compatibility with old stats format
                "disk_entries": entries,
                "disk_size": total_size
            }
        except Exception as e:
            logger.warning(f"Audio cache stats failed: {str(e)}")
            return {
                "entries": 0,
                "total_size": 0,
                "memory_entries": 0,
                "disk_entries": 0,
                "disk_size": 0
            }

def save(self, key: str, path: str) -> None:
    """Cache an audio path for a given key with proper validation and eviction."""
    if not os.path.exists(path):
        logger.warning(f"Cannot cache non-existent path: {path}")
        return

    file_size = os.path.getsize(path)

    with self.cache_lock:
        # Apply dual-limit condition (configurable via config)
        max_entries = self.config.audio.max_cache_entries
        max_total_size = self.config.audio.max_cache_size_mb * 1024 * 1024

        # Evict until below limit
        while (len(self.audio_cache) >= max_entries or
               self.total_size + file_size > max_total_size) and self.audio_cache:
            # Find oldest entry
            oldest_key = min(
                self.audio_cache.items(),
                key=lambda x: x[1]["added_time"]
            )[0]
            self._remove_entry(oldest_key)

        # Add new cache entry
        self.audio_cache[key] = {
            "path": path,
            "size": file_size,
            "added_time": time.time(),
            "last_used": time.time()
        }
        self.total_size += file_size

        # Persist immediately if memory cache is full
        if len(self.audio_cache) >= self.config.audio.max_cache_entries * 0.8:
            self.save_cache()

        logger.debug(f"Audio cached: {key} → {path} ({file_size/1024/1024:.1f}MB)")
        logger.debug(f"Cache stats: {len(self.audio_cache)} entries, {self.total_size/(1024*1024):.1f}MB")