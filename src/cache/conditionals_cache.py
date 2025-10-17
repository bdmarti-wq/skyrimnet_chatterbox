import os
import sys
import time
import threading
import logging
from typing import Any, Dict, Optional
from pathlib import Path
import torch
from src.config import get_config
from loguru import logger

# Move this to the very top to ensure proper imports
T3_AVAILABLE = False

try:
    from src.chatterbox.models.t3.modules.cond_enc import T3Cond
    from src.chatterbox.tts import Conditionals
    T3_AVAILABLE = True
except ImportError:
    logger.warning("T3Cond/Conditionals not available – using fallback implementation")

class ConditionalsCache:
    """Manages caching of voice conditionals for the TTS model."""

    def __init__(self, cache_dir: Path):
        """Initialize the conditionals cache system."""
        self.cache_dir = cache_dir / "conditionals"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.memory_cache: Dict[str, Any] = {}
        self.max_memory_entries = 100  # Configurable
        self.cache_lock = threading.RLock()

        logger.info(f"Conditionals cache initialized at {self.cache_dir}")

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{cache_key}.pt"

    def is_cached(self, cache_key: str) -> bool:
        """Check if conditionals are cached for a given key."""
        cache_path = self._get_cache_path(cache_key)
        return cache_key in self.memory_cache or (os.path.exists(cache_path) and os.path.getsize(cache_path) > 0)

    def get(self, cache_key: str, model: Any, device: str, dtype: torch.dtype) -> Optional[Any]:
        """Get conditionals from cache."""
        # 1. Check memory cache first
        with self.cache_lock:
            if cache_key in self.memory_cache:
                conditionals = self.memory_cache[cache_key]
                logger.debug(f"Memory hit for conditionals: {cache_key}")
                return conditionals

        # 2. Check disk cache
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
            try:
                # Load conditionals from disk
                if T3_AVAILABLE:
                    # Load with reconstruction for T3 conditionals
                    loaded = torch.load(cache_path, map_location=device)
                    if isinstance(loaded, dict) and "t3" in loaded:
                        t3_obj = T3Cond(**loaded["t3"])
                        t3_obj = t3_obj.to(device=device, dtype=dtype)
                        conditionals = Conditionals(t3_obj, loaded.get("gen"))
                        conditionals = conditionals.to(device)

                        # Cache in memory
                        with self.cache_lock:
                            self._add_to_memory(cache_key, conditionals)

                        logger.info(f"Disk hit & reconstructed conditionals: {cache_key}")
                        return conditionals
                else:
                    # Simple fallback for non-T3 models
                    conditionals = torch.load(cache_path, map_location=device)
                    with self.cache_lock:
                        self._add_to_memory(cache_key, conditionals)
                    logger.info(f"Disk hit for conditionals (simple): {cache_key}")
                    return conditionals
            except Exception as e:
                logger.warning(f"Failed to load conditionals from disk ({cache_key}): {str(e)}")

        return None

    def _add_to_memory(self, cache_key: str, conditionals: Any) -> None:
        """Add conditionals to memory cache with LRU behavior."""
        if len(self.memory_cache) >= self.max_memory_entries:
            # Remove oldest entry
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]

        self.memory_cache[cache_key] = conditionals

    def save(self, cache_key: str, conditionals: Any, model=None, device: str = "cuda",
             dtype: torch.dtype = torch.float32) -> bool:
        """Save conditionals to both memory and disk caches with T3-specific handling."""
        if cache_key is None or conditionals is None:
            logger.warning("No cache key/conds – skipping save")
            return False

        try:
            # Use Chatterbox TTS's specific serialization method
            if isinstance(conditionals, Conditionals):
                try:
                    arg_dict = {
                        "t3": conditionals.t3.__dict__.copy() if hasattr(conditionals.t3,
                                                                         '__dict__') else conditionals.t3,
                        "gen": conditionals.gen
                    }
                    logger.debug(f"Serialized Conditionals object: {cache_key}")
                except Exception as e:
                    logger.error(f"Failed to serialize Conditionals: {str(e)}")
                    return False
            else:
                arg_dict = conditionals
                logger.warning(f"Using fallback serialization for non-Conditionals object: {cache_key}")

            # FIX #1: Proper memory cache reference (no underscore)
            with self.cache_lock:
                self.memory_cache[cache_key] = arg_dict  # Was _memory_cache (incorrect)
                logger.info(f"Memory saved arg_dict: {cache_key[:8]}... (total: {len(self.memory_cache)})")
                self._current_loaded_cache_key = cache_key

            # Save to disk
            cache_path = self._get_cache_path(cache_key)
            try:
                torch.save(arg_dict, cache_path)
                logger.info(f"Saved arg_dict to disk: {cache_path} (size: {cache_path.stat().st_size / 1e6:.1f}MB)")
                return True
            except Exception as e:
                logger.error(f"Failed to save conditionals to disk ({cache_key}): {str(e)}")
                return False

        except Exception as e:
            logger.error(f"Failed to save conditionals ({cache_key}): {str(e)}")
            return False


    def clear(self, voice_stem: Optional[str] = None) -> None:
        """Clear conditionals cache, optionally for a specific voice."""
        with self.cache_lock:
            if voice_stem:
                # Clear specific voice entries
                keys_to_remove = [k for k in self.memory_cache.keys() if k.startswith(f"{voice_stem}_ref_")]
                for k in keys_to_remove:
                    del self.memory_cache[k]

                # Delete disk files for this voice
                for cache_file in self.cache_dir.glob(f"{voice_stem}_ref_*.pt"):
                    try:
                        cache_file.unlink()
                        logger.debug(f"Deleted conditionals cache: {cache_file.name}")
                    except Exception as e:
                        logger.warning(f"Failed to delete cache file {cache_file}: {str(e)}")

                logger.info(f"Cleared conditionals cache for voice: {voice_stem}")
            else:
                # Clear everything
                self.memory_cache.clear()

                # Delete all disk files
                for cache_file in self.cache_dir.glob("*.pt"):
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete cache file {cache_file}: {str(e)}")

                logger.info("Cleared entire conditionals cache")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.cache_lock:
            disk_size = 0
            for cache_file in self.cache_dir.glob("*.pt"):
                try:
                    disk_size += cache_file.stat().st_size
                except:
                    pass

            return {
                "memory_entries": len(self.memory_cache),
                "memory_size": sum(sys.getsizeof(v) for v in self.memory_cache.values()),
                "disk_entries": len(list(self.cache_dir.glob("*.pt"))),
                "disk_size": disk_size,
                "max_memory": self.max_memory_entries
            }