import os
import sys
import time
import threading
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
                if self._is_empty_conditionals(conditionals):
                    logger.warning(f"Empty conditionals in memory cache for {cache_key}; discarding")
                    del self.memory_cache[cache_key]
                    return None
                logger.debug(f"Memory hit for conditionals: {cache_key}")
                return conditionals

        # 2. Check disk cache
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
            try:
                # Load conditionals from disk
                if T3_AVAILABLE:
                    loaded = torch.load(cache_path, map_location=device)
                    if isinstance(loaded, dict) and "t3" in loaded:
                        t3_obj = T3Cond(**loaded["t3"])
                        t3_obj = t3_obj.to(device=device, dtype=dtype)
                        conditionals = Conditionals(t3_obj, loaded.get("gen"))
                        conditionals = conditionals.to(device)
                    else:
                        conditionals = loaded  # Fallback
                else:
                    conditionals = torch.load(cache_path, map_location=device)

                # NEW: Assign to model state + activate if method (old pattern post-load)
                if model is not None:
                    model.conds = conditionals  # Set state (public self.conds)
                    # Dtype sync on emb (old: .to(dtype) post-load for stability)
                    if T3_AVAILABLE and hasattr(conditionals, 't3') and hasattr(conditionals.t3, 'speaker_emb') and conditionals.t3.speaker_emb is not None:
                        conditionals.t3.speaker_emb = conditionals.t3.speaker_emb.to(dtype=dtype)
                        logger.debug(f"Loaded emb to dtype {dtype} for {cache_key}")
                    if hasattr(model, 'set_conditionals'):
                        model.set_conditionals(conditionals)  # Link emb if exists
                        logger.debug(f"set_conditionals post-cache load for {cache_key} (emb linked)")
                    else:
                        logger.debug(f"Cache load: self.conds = conds (no set; assume direct use)")

                # Validate loaded conditionals
                if self._is_empty_conditionals(conditionals):
                    logger.warning(f"Loaded empty/invalid conditionals for {cache_key} from disk; discarding")
                    try:
                        cache_path.unlink()
                        logger.debug(f"Removed invalid cache file: {cache_path}")
                    except Exception as cleanup_e:
                        logger.debug(f"Could not remove invalid cache: {cleanup_e}")
                    return None

                # Cache in memory
                with self.cache_lock:
                    self._add_to_memory(cache_key, conditionals)

                logger.info(f"Disk hit & reconstructed conditionals: {cache_key}")
                return conditionals
            except Exception as e:
                logger.warning(f"Failed to load conditionals from disk ({cache_key}): {str(e)}")
                try:
                    if os.path.exists(cache_path):
                        cache_path.unlink()
                        logger.debug(f"Removed failed cache file: {cache_path}")
                except Exception as cleanup_e:
                    logger.debug(f"Could not clean up failed cache: {cleanup_e}")
                return None

        return None

    def _is_empty_conditionals(self, conditionals: Any) -> bool:
        """Helper: Check if conditionals are empty/invalid."""
        if conditionals is None:
            return True
        if isinstance(conditionals, torch.Tensor):
            return conditionals.numel() == 0
        if isinstance(conditionals, dict):
            return all(
                (isinstance(v, torch.Tensor) and v.numel() == 0) or
                (hasattr(v, '__len__') and len(v) == 0) or v is None
                for v in conditionals.values()
            )
        if hasattr(conditionals, '__len__') and len(conditionals) == 0:
            return True
        return False

    def _add_to_memory(self, cache_key: str, conditionals: Any) -> None:
        """Add conditionals to memory cache with LRU behavior."""
        if len(self.memory_cache) >= self.max_memory_entries:
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        self.memory_cache[cache_key] = conditionals

    def save(self, cache_key: str, conditionals: Any, model=None, device: str = "cuda",
             dtype: torch.dtype = torch.float32) -> bool:
        """Save conditionals with parameter-aware serialization."""
        if cache_key is None or conditionals is None:
            logger.warning("No cache key/conds – skipping save")
            return False

        try:
            save_data = self._serialize_conditionals(conditionals, device, dtype)

            # Save to memory cache with LRU
            with self.cache_lock:
                self._add_to_memory(cache_key, save_data)
                size_bytes = sys.getsizeof(save_data) if isinstance(save_data, dict) else sum(
                    sys.getsizeof(v) for v in save_data.values()
                )
                logger.info(
                    f"Memory saved: {cache_key[:8]}... (total: {len(self.memory_cache)} entries, "
                    f"{size_bytes / 1e6:.2f}MB)"
                )

                # Purge if memory pressure
                if len(self.memory_cache) > self.max_memory_entries * 0.9:
                    self._purge_memory_cache()

            # Save to disk
            return self._save_to_disk(cache_key, save_data)

        except Exception as e:
            logger.exception(f"Saving conditionals failed: {str(e)}")
            return False

    def delete(self, key: str) -> None:
        """Delete specific key from cache (for purge in manager)."""
        try:
            # Memory: Pop from dict (assume self.conditionals; adjust if self.cache)
            if hasattr(self, 'conditionals') and key in self.conditionals:
                del self.conditionals[key]
            elif hasattr(self, 'cache') and key in self.cache:
                del self.cache[key]
            logger.debug(f"Deleted conds from memory: {key[:12]}...")

            # Disk: Remove file if exists (e.g., key.pkl in cache_dir)
            import pickle  # If using pickle
            file_path = self.cache_dir / f"{key}.pkl"  # Assume pickle format
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Deleted conds file: {file_path}")

            logger.info(f"Purged conds key: {key[:12]}...")
        except Exception as del_e:
            logger.warning(f"Delete failed for {key}: {del_e}; may linger")


    def _serialize_conditionals(self, conditionals: Any, device: str, dtype: torch.dtype) -> Dict:
        """Safely serialize conditionals based on type."""
        if T3_AVAILABLE and isinstance(conditionals, Conditionals):
            t3_dict = {
                k: v for k, v in conditionals.t3.__dict__.items()
                if not isinstance(v, torch.Tensor) or v.numel() < 10000
            }
            return {
                "t3": t3_dict,
                "gen": conditionals.gen,
                "device": str(device),
                "dtype": str(dtype),
                "class": "Conditionals"
            }
        elif isinstance(conditionals, dict):
            return {
                k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                for k, v in conditionals.items()
            }
        else:
            return {
                "data": conditionals,
                "device": str(device),
                "dtype": str(dtype)
            }

    def _save_to_disk(self, cache_key: str, save_data: Any) -> bool:  # MISSING: Add this
        """Save to disk (from your original)."""
        cache_path = self._get_cache_path(cache_key)
        try:
            torch.save(save_data, cache_path)
            logger.debug(f"Disk saved: {cache_path.name}")
            return True
        except Exception as e:
            logger.warning(f"Disk save failed for {cache_key}: {e}")
            return False

    def _purge_memory_cache(self) -> None:  # MISSING: Add this
        """Purge excess memory cache (from your original)."""
        target_size = self.max_memory_entries // 2
        while len(self.memory_cache) > target_size:
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        logger.debug(f"Purged memory cache to {len(self.memory_cache)} entries")

    def clear(self, voice_stem: Optional[str] = None) -> None:
        """Clear conditionals cache, optionally for a specific voice."""
        with self.cache_lock:
            if voice_stem:
                keys_to_remove = [k for k in list(self.memory_cache) if k.startswith(f"{voice_stem}_ref_")]  # list() safe
                for k in keys_to_remove:
                    del self.memory_cache[k]
                for cache_file in self.cache_dir.glob(f"{voice_stem}_ref_*.pt"):
                    try:
                        cache_file.unlink()
                        logger.debug(f"Deleted: {cache_file.name}")
                    except Exception as e:
                        logger.warning(f"Delete failed {cache_file}: {str(e)}")
                logger.info(f"Cleared conds cache for voice: {voice_stem}")
            else:
                self.memory_cache.clear()
                for cache_file in self.cache_dir.glob("*.pt"):
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        logger.warning(f"Delete failed {cache_file}: {str(e)}")
                logger.info("Cleared entire conds cache")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.cache_lock:
            disk_size = sum(cache_file.stat().st_size for cache_file in self.cache_dir.glob("*.pt") if (disk_size := cache_file.stat().st_size or 0))
            return {
                "memory_entries": len(self.memory_cache),
                "memory_size": sum(sys.getsizeof(v) for v in self.memory_cache.values()),
                "disk_entries": len(list(self.cache_dir.glob("*.pt"))),
                "disk_size": disk_size,
                "max_memory": self.max_memory_entries
            }

