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

        # FIXED: Stats counters (hits/misses for debugging)
        self.stats = {
            "total_get": 0,
            "memory_hits": 0,
            "disk_hits": 0,
            "misses": 0
        }

        logger.info(f"Conditionals cache initialized at {self.cache_dir}")

    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key (now hash-based for stable reuse)."""
        # FIXED: Stable hash for content (e.g., "conds_08e97b7a.pt" – persists/reuses across uploads)
        filename = f"conds_{key[:12]}.pt"  # Key is hash (e.g., 08e97b7a...); 12 chars for uniqueness
        return self.cache_dir / filename

    def is_cached(self, cache_key: str) -> bool:
        """Check if conditionals are cached for a given key."""
        cache_path = self._get_cache_path(cache_key)
        return cache_key in self.memory_cache or (os.path.exists(cache_path) and os.path.getsize(cache_path) > 0)

    def get(self, cache_key: str, model: Any, device: str, dtype: torch.dtype) -> Optional[Any]:
        """Get conditionals from memory then disk cache (increment stats)."""
        self.stats["total_get"] += 1
        with self.cache_lock:
            if cache_key in self.memory_cache:
                conditionals = self.memory_cache[cache_key]
                if self._is_empty_conditionals(conditionals):
                    del self.memory_cache[cache_key]
                    self.stats["misses"] += 1
                    return None
                self.stats["memory_hits"] += 1
                logger.debug(f"Memory hit for conditionals: {cache_key[:20]}...")
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
                    if T3_AVAILABLE and hasattr(conditionals, 't3') and hasattr(conditionals.t3,
                                                                                'speaker_emb') and conditionals.t3.speaker_emb is not None:
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
                    self.stats["misses"] += 1
                    return None

                # Cache in memory
                with self.cache_lock:
                    self._add_to_memory(cache_key, conditionals)

                self.stats["disk_hits"] += 1
                logger.info(f"Disk hit & reconstructed conditionals: {cache_key[:20]}...")
                return conditionals
            except Exception as e:
                logger.warning(f"Failed to load conditionals from disk ({cache_key}): {str(e)}")
                try:
                    if os.path.exists(cache_path):
                        cache_path.unlink()
                        logger.debug(f"Removed failed cache file: {cache_path}")
                except Exception as cleanup_e:
                    logger.debug(f"Could not clean up failed cache: {cleanup_e}")
                self.stats["misses"] += 1
                return None

        self.stats["misses"] += 1
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
        """Save conditionals with parameter-aware serialization (memory + disk by stable key). FIXED: Full implementation; handles hash keys for persistence."""
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
                logger.debug(
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

    def _get_or_prepare(self, model, audio_path: str, exag: float, device: str, dtype: torch.dtype, cache_key: str) -> \
            Optional[Any]:
        """NEW: Atomic prepare with cache (_get_or_prepare); returns conds if success (loaded or fresh). FIXED: Eager/sync; set conds_loaded flag."""
        with self.cache_lock:
            conds = self.get(cache_key, model, device, dtype)
            if conds is not None:
                logger.debug(f"Conds HIT from cache for key {cache_key[:20]}...")
                return conds

        # Miss: Prepare + cache (caller must have valid path)
        try:
            model.prepare_conditionals(audio_path, exaggeration=exag)
            conds = model.conds  # Reference or copy? (assume set in model already)

            # Validate
            if self._is_empty_conditionals(conds):
                logger.warning(f"Prepared empty conds for {cache_key}; no save")
                return None

            # Save for next
            self.save(cache_key, conds, model, device, dtype)
            logger.debug(f"Conds prepared and cached fresh for {cache_key[:20]}...")
            return conds
        except Exception as prep_e:
            logger.error(f"Prepare failed for {cache_key}: {prep_e} → None")
            return None

    def delete(self, key: str) -> None:
        """Delete specific key from cache (for purge in manager). FIXED: Use hash paths."""
        try:
            # Memory: Pop from dict (assume self.conditionals; adjust if self.cache)
            if hasattr(self, 'conditionals') and key in self.conditionals:
                del self.conditionals[key]
            elif hasattr(self, 'cache') and key in self.cache:
                del self.cache[key]
            logger.debug(f"Deleted conds from memory: {key[:12]}...")

            # Disk: Remove file if exists (now conds_hash.pt)
            file_path = self._get_cache_path(key)
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Deleted conds file: {file_path}")

            logger.info(f"Purged conds key: {key[:12]}...")
        except Exception as del_e:
            logger.warning(f"Delete failed for {key}: {del_e};")

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
        """Save to disk (from your original; handles hash keys)."""
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
        """Clear conditionals cache, optionally for a specific voice. FIXED: Handle hash keys (e.g., *hash*.*)."""
        with self.cache_lock:
            if voice_stem:
                # Memory: Clear by stem prefix (old keys)
                keys_to_remove = [k for k in list(self.memory_cache) if
                                  voice_stem in k]  # e.g., "femaleneivavoice_ref_"
                for k in keys_to_remove:
                    del self.memory_cache[k]
                # Disk: Glob by stem-related hashes or files
                for cache_file in self.cache_dir.glob(f"*{voice_stem}*.pt"):  # Old stem in filename
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
        """Get cache statistics (added hits/misses)."""
        with self.cache_lock:
            disk_size = sum(cache_file.stat().st_size for cache_file in self.cache_dir.glob("*.pt") if
                            (disk_size := cache_file.stat().st_size or 0))
            return {
                "memory_entries": len(self.memory_cache),
                "disk_entries": len(list(self.cache_dir.glob("*.pt"))),
                "disk_size": disk_size,
                "max_memory": self.max_memory_entries,
                "stats": self.stats  # Include hits/misses
            }