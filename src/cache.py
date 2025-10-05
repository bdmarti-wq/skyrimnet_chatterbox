"""
Merged Cache utilities for SkyrimNet TTS conditionals (memory/disk caching).
Supports serialization (raw torch + T3Cond fallback). Integrates with original globals.
Hybrid: Simple globals + Manager classes (threaded disk saves).
Init loads existing .pt; async saves; audio manager with eviction.
Fallback to no-cache if disabled.
"""

import os
import json
import datetime
import functools
import time
import warnings
import hashlib
import threading
import re
from collections import OrderedDict
from difflib import SequenceMatcher
from threading import Lock, Thread
from queue import Queue
import torch
import torchaudio
from torch.serialization import safe_globals  # For whitelisting in load
from torchaudio.io import StreamReader  # For non-blocking SR probe in async
import numpy as np
from pathlib import Path
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple, Union, List
from loguru import logger  # Assume available; fallback to print if not
import threading  # Ensure imported (likely already is)
from config import CONFIG


# Suppress torchaudio deprecations precisely (exact message/module for backend utils)
warnings.filterwarnings('ignore', message=r'.*torchaudio._backend.utils.info.*', category=UserWarning)
warnings.filterwarnings('ignore', message=r'.*deprecated.*torchaudio.*', category=UserWarning, module='torchaudio')
warnings.filterwarnings('ignore', category=UserWarning, module='torchaudio')  # Broad fallback
warnings.filterwarnings('ignore', category=DeprecationWarning, module='torchaudio')

# Anchor paths to project root (skyrimnet_chatterbox/) for relocatable code
ROOT_DIR = Path(__file__).parent.parent  # From src/ -> skyrimnet_chatterbox/

# Original globals (now rooted)
WAV_OUTPUT_DIR = ROOT_DIR / "output_temp"
CACHE_BASE = ROOT_DIR / "cache"
CACHE_DIR = CACHE_BASE / "conditionals"
CACHE_AUDIO_DIR = CACHE_BASE / "audio"
voices_dir = CACHE_AUDIO_DIR / "voices"  # For check_and_update_ref

# Global model lock (new: serialize access to prevent graph races)
MODEL_LOCK = threading.RLock()

# Tunable constants (hardcoded; override via env vars if needed)
KEY_LEN = 32    # Hash length
MAX_RECURSE = 3 # Reconstruction recursion limit
DEFAULT_DEVICE = "cuda"  # Fallback
DEFAULT_DTYPE = torch.bfloat16  # TODO was float32 test and review
MODEL_SR = 24000  # Assume standard for TTS
MAX_MEMORY_ENTRIES = CONFIG.get_value('max_memory_entries', default=100)
ENABLE_MEMORY_CACHE = CONFIG.get_value('enable_memory_cache', default=True)
ENABLE_DISK_CACHE = CONFIG.get_value('enable_disk_cache', default=True)
ENABLE_THREADED_SAVES = True  # Hardcode or add to CONFIG if needed
MAX_QUEUE = CONFIG.get_value('save_queue_max', default=20)
MAX_INDEX_SIZE = CONFIG.get_value('fuzzy_index_size', default=1000)  # For fuzzy per-stem
ENABLE_FUZZY = CONFIG.get_value('fuzzy_enable', default=True)  # New: Toggle fuzzy
COMPRESS_PT_SAVES = CONFIG.get_value('compress_pt_saves', default=True)
COMPRESS_LEVEL = CONFIG.get_value('compress_level', default=6)

# Global voice cache (stem → dict: fixed_path, file_hash, conds_key)
_voice_cache_lock = threading.RLock()
_voice_cache = OrderedDict()  # LRU, maxlen=200  # Add maxlen=200 (tune via CONFIG if needed)
_voice_info_cache = {}  # {stem: (sr, channels, duration_frames)} – simple, thread-safe with lock
_voice_info_lock = threading.RLock()  # Optional: For concurrent access (shared with _voice_cache_lock)

# Create dirs (rooted)
for d in [WAV_OUTPUT_DIR, CACHE_BASE, CACHE_DIR, CACHE_AUDIO_DIR, voices_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Import fallbacks
T3_AVAILABLE = False
try:
    from src.chatterbox.models.t3.modules.cond_enc import T3Cond
    from src.chatterbox.tts import Conditionals
    T3_AVAILABLE = True
except ImportError:
    logger.warning("T3Cond/Conditionals not available – using raw fallback")

# Dummy conds fallback (simple; assumes model has set_conditionals or conds attr)
def create_dummy_conds(model, device, dtype, reason="fallback"):
    """Create dummy conditionals on failure."""
    if T3_AVAILABLE:
        try:
            dummy_t3 = T3Cond()  # Empty init
            dummy_conds = Conditionals(dummy_t3, None)
            dummy_conds = dummy_conds.to(device, dtype=dtype)
            if hasattr(model, 'set_conditionals'):
                model.set_conditionals(dummy_conds)
            elif hasattr(model, 'conds'):
                model.conds = dummy_conds
            logger.debug(f"Created dummy conds ({reason}): {type(dummy_conds).__name__}")
            return dummy_conds
        except Exception as e:
            logger.error(f"Dummy conds failed ({reason}): {e}")
    return None


class ConditionalsCacheManager:
    """Thread-safe cache manager for conditionals with memory and disk storage (merged)."""

    def __init__(self):
        self._memory_cache = OrderedDict(maxlen=MAX_MEMORY_ENTRIES)  # LRU
        self._cache_lock = threading.RLock()
        self._current_loaded_cache_key = None
        self._disk_save_queue = []

    def get_current_cache_key(self):
        with self._cache_lock:
            return self._current_loaded_cache_key

    def is_cache_key_loaded(self, cache_key):
        with self._cache_lock:
            return self._current_loaded_cache_key == cache_key

    def _save_to_disk_worker(self, cache_key, arg_dict):
        try:
            disk_path = CACHE_DIR / f"{cache_key}.pt"
            torch.save(arg_dict, disk_path, _use_new_zipfile_serialization=False)
            logger.info(f"Saved arg_dict to disk: {disk_path} (size: {disk_path.stat().st_size / 1e6:.1f}MB)")

            with self._cache_lock:
                if cache_key in self._disk_save_queue:
                    self._disk_save_queue.remove(cache_key)

        except Exception as e:
            logger.error(f"Worker save failed for {cache_key}: {e}")
            with self._cache_lock:
                if cache_key in self._disk_save_queue:
                    self._disk_save_queue.remove(cache_key)

    def save(self, cache_key: str, conds: Any, model=None, device: str | torch.device = None,
             dtype: torch.dtype = None, enable_memory_cache: bool = True, enable_disk_cache: bool = True) -> bool:
        if cache_key is None or conds is None:
            logger.warning("No cache key/conds – skipping save")
            return False

        device = device or DEFAULT_DEVICE
        dtype = dtype or DEFAULT_DTYPE

        try:
            # Extract serializable arg_dict
            if T3_AVAILABLE and hasattr(conds, 't3') and hasattr(conds, 'gen'):
                arg_dict = dict(
                    t3=conds.t3.__dict__.copy() if hasattr(conds.t3, '__dict__') else conds.t3,
                    gen=conds.gen
                )
            elif hasattr(conds, '__dict__'):
                arg_dict = conds.__dict__.copy()
            else:
                arg_dict = {'data': conds}

            arg_dict.update({'device': str(device), 'dtype': str(dtype)})

            saved = False
            with self._cache_lock:
                if enable_memory_cache:
                    self._memory_cache[cache_key] = arg_dict
                    self._memory_cache.move_to_end(cache_key)
                    logger.info(f"Memory saved arg_dict: {cache_key[:8]}... (total: {len(self._memory_cache)})")
                    saved = True

                self._current_loaded_cache_key = cache_key

                if enable_disk_cache and cache_key not in self._disk_save_queue:
                    if len(self._disk_save_queue) >= MAX_QUEUE:
                        dropped = self._disk_save_queue.pop(0)
                        logger.debug(f"Dropped queued save: {dropped[:8]}...")
                    self._disk_save_queue.append(cache_key)

                    if ENABLE_THREADED_SAVES:
                        threading.Thread(
                            target=self._save_to_disk_worker,
                            args=(cache_key, arg_dict),
                            daemon=True,
                            name=f"DiskSave-{cache_key[:8]}"
                        ).start()
                        logger.debug(f"Queued threaded save: {cache_key[:8]}...")
                    else:
                        disk_path = CACHE_DIR / f"{cache_key}.pt"
                        torch.save(arg_dict, disk_path, _use_new_zipfile_serialization=False)
                        logger.info(f"Sync saved to disk: {disk_path}")
                    saved = True

            # Set to model if available (locked to prevent race)
            if model:
                with MODEL_LOCK:  # Serialize model access
                    try:
                        if hasattr(model, 'set_conditionals'):
                            model.set_conditionals(conds)
                            logger.debug("Set conds via model.set_conditionals")
                        elif hasattr(model, 'conds'):
                            model.conds = conds
                    except Exception as set_e:
                        logger.error(f"Model set failed: {set_e} – conds saved but not applied")

            return saved

        except Exception as e:
            logger.error(f"Save failed for {cache_key}: {e}")
            with self._cache_lock:
                if cache_key in self._disk_save_queue:
                    self._disk_save_queue.remove(cache_key)
            return False

    def load(self, cache_key: str, model=None, device: str | torch.device = None,
             dtype: torch.dtype = None, enable_memory_cache: bool = True,
             enable_disk_cache: bool = True, quiet: bool = False) -> Optional[Any]:
        if cache_key is None:
            if not quiet:
                logger.debug("No key for load")
            return None

        device = device or DEFAULT_DEVICE
        dtype = dtype or DEFAULT_DTYPE

        if self.is_cache_key_loaded(cache_key):
            if not quiet:
                logger.info(f"Conds already loaded: {cache_key[:8]}...")
            current_conds = getattr(model, 'conds', None) if model else None
            if current_conds:
                return current_conds

        conds = None
        with self._cache_lock:
            cached_item = self._memory_cache.get(cache_key)
            if cached_item is not None:
                self._memory_cache.move_to_end(cache_key)
                if isinstance(cached_item, dict) and ('t3' in cached_item or 'data' in cached_item):
                    conds = self._reconstruct_conds(cached_item, model, device, dtype)
                    if conds:
                        self._memory_cache[cache_key] = conds
                else:
                    conds = cached_item
                if not quiet:
                    logger.info(f"Memory hit: {cache_key[:8]}... ({type(conds).__name__ if conds else 'None'})")

        if conds is None and enable_disk_cache:
            pt_path = CACHE_DIR / f"{cache_key}.pt"
            if pt_path.exists():
                try:
                    with safe_globals([T3Cond, Conditionals]) if T3_AVAILABLE else lambda x: x:
                        loaded = torch.load(pt_path, map_location=device, weights_only=False)
                    if isinstance(loaded, dict) and ('t3' in loaded or 'data' in loaded):
                        conds = self._reconstruct_conds(loaded, model, device, dtype)
                    if conds is not None and enable_memory_cache:
                        with self._cache_lock:
                            self._memory_cache[cache_key] = conds
                            self._memory_cache.move_to_end(cache_key)
                    if not quiet:
                        logger.info(f"Disk hit & reconstructed: {pt_path.name} ({type(conds).__name__ if conds else 'None'})")
                except Exception as e:
                    logger.error(f"Load from {pt_path}: {e}")
                    conds = None
            else:
                if not quiet:
                    logger.debug(f"Disk miss: {pt_path}")

        if conds is None:
            if not quiet:
                logger.warning(f"Cache miss: {cache_key} (recompute)")
            return None

        # Post-load set to model with fallback
        if model:
            try:
                if hasattr(model, 'set_conditionals'):
                    model.set_conditionals(conds)
                elif hasattr(model, 'conds'):
                    model.conds = conds
                if not hasattr(model, 'conds') or model.conds is None:
                    logger.warning(f"Post-load set failed for {cache_key[:8]}... – Dummy fallback")
                    create_dummy_conds(model, device, dtype, 'load')
                    conds = model.conds
                logger.debug(f"Post-load set conds: {type(conds).__name__}")
            except Exception as set_e:
                logger.error(f"Post-load set failed: {set_e} – Dummy fallback")
                create_dummy_conds(model, device, dtype, 'load_error')
                conds = model.conds

        self._current_loaded_cache_key = cache_key
        return conds

    def _reconstruct_conds(self, state_or_item: Any, model=None, device: str | torch.device = None,
                           dtype: torch.dtype = None, depth: int = 0) -> Optional[Any]:
        if depth > MAX_RECURSE:
            logger.warning(f"Reconstruct depth exceeded – raw fallback")
            return state_or_item

        device = device or DEFAULT_DEVICE
        dtype = dtype or DEFAULT_DTYPE

        try:
            if isinstance(state_or_item, dict):
                data = state_or_item.get('data', state_or_item)
                if T3_AVAILABLE and isinstance(data, dict) and 't3' in data:
                    t3_dict = data['t3']
                    gen = data.get('gen')
                    if isinstance(t3_dict, dict):
                        t3_obj = T3Cond(**t3_dict)
                    else:
                        t3_obj = t3_dict
                    t3_obj = t3_obj.to(device=device, dtype=dtype)
                    if hasattr(t3_obj, 'speaker_emb') and t3_obj.speaker_emb is not None:
                        t3_obj.speaker_emb = t3_obj.speaker_emb.to(dtype=dtype)
                    conds = Conditionals(t3_obj, gen)
                    conds = conds.to(device)
                    logger.debug("Reconstructed Conditionals from T3 dict")
                    return conds
                else:
                    # General dict restoration
                    restored = type('RestoredData', (), data)()
                    for attr, val in data.items():
                        if isinstance(val, torch.Tensor):
                            setattr(restored, attr, val.to(device, dtype))
                        else:
                            setattr(restored, attr, self._reconstruct_conds(val, depth=depth+1))
                    return restored
            elif isinstance(state_or_item, torch.Tensor):
                return state_or_item.to(device=device, dtype=dtype)
            return state_or_item
        except Exception as e:
            logger.error(f"_reconstruct failed: {e} – raw fallback")
            return state_or_item.get('data', state_or_item)

    def evict_lru(self, n: int = 50) -> int:
        evicted = 0
        with self._cache_lock:
            for _ in range(n):
                if self._memory_cache:
                    old_key = self._memory_cache.popitem(last=False)[0]
                    pt_file = CACHE_DIR / f"{old_key}.pt"
                    if pt_file.exists():
                        pt_file.unlink(missing_ok=True)
                        logger.debug(f"Evicted disk: {old_key[:8]}...")
                    evicted += 1
                else:
                    break
        logger.info(f"Evicted {evicted} LRU conds")
        return evicted

    def get_cache_stats(self):
        with self._cache_lock:
            disk_count = len(list(CACHE_DIR.glob("*.pt")))
            return {
                'memory_cache_size': len(self._memory_cache),
                'current_loaded_key': self._current_loaded_cache_key,
                'pending_disk_saves': len(self._disk_save_queue),
                'memory_cache_keys': list(self._memory_cache.keys()),
                'disk_files': disk_count
            }


# Global manager
_cache_manager = ConditionalsCacheManager()


class AudioCacheManager:
    """Memory-only LRU cache for generated audio paths."""

    def __init__(self):
        self._audio_cache = {}
        self._lock = threading.RLock()
        self._max_size = 100

    def get(self, key):
        with self._lock:
            return self._audio_cache.get(key)

    def set(self, key, path):
        with self._lock:
            self._audio_cache[key] = path
            if len(self._audio_cache) > self._max_size:
                oldest = next(iter(self._audio_cache))
                del self._audio_cache[oldest]
                logger.debug(f"Evicted audio: {oldest}")

    def clear(self):
        with self._lock:
            self._audio_cache.clear()

    def stats(self):
        return {'audio_cache_size': len(self._audio_cache), 'max_size': self._max_size}


_audio_manager = AudioCacheManager()


# Facades for audio
def get_audio_cache(key):
    return _audio_manager.get(key)


def set_audio_cache(key, path):
    _audio_manager.set(key, path)


# Merged helpers
@functools.lru_cache(maxsize=128)
def get_cache_dir():
    return CACHE_DIR


@functools.lru_cache(maxsize=128)
def get_wavout_dir():
    formatted_start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wavout_dir = WAV_OUTPUT_DIR / formatted_start_time
    wavout_dir.mkdir(parents=True, exist_ok=True)
    return wavout_dir


def get_cache_key(audio_path: str = None, uuid: Any = None, exaggeration: float = None,
                  params: Dict[str, Any] = None) -> str:
    """Robust cache key: Original simple mode + params hashing."""
    if audio_path is None:
        return None

    cache_prefix = Path(audio_path).stem

    # Simple mode (original)
    try:
        uuid_hex = hex(uuid)[2:] if isinstance(uuid, (int, type(None))) else str(uuid)
    except:
        uuid_hex = str(uuid)
    if exaggeration is None:
        cache_key = f"{cache_prefix}_{uuid_hex}"
    else:
        cache_key = f"{cache_prefix}_{uuid_hex}_{exaggeration:.2f}"
    if len(cache_key) <= 100:
        return cache_key

    # Params mode (alternative: hash robustly)
    if params is None:
        params = {}
    param_dict = {'prefix': cache_prefix, 'uuid': uuid_hex, 'exagg': f"{exaggeration:.2f}" if exaggeration else "0.5"}
    param_dict.update(params)

    # Robust-ize (simplified: shapes/str, no slow bytes)
    for k, v in list(param_dict.items()):
        if isinstance(v, (np.ndarray, torch.Tensor)):
            shape_str = f"shape_{v.shape}_{str(v.dtype) if hasattr(v, 'dtype') else 'unknown'}"
            param_dict[k] = f"hash_{hash(shape_str)}"
        elif hasattr(v, '__dict__') and hasattr(v, 't3'):  # Conditionals
            param_dict[k] = f"conds_id_{id(v)}_{getattr(v.t3, 'shape', 'unknown')}"
        else:
            param_dict[k] = str(v)  # Safe: str() on non-tensor

    try:
        serialized = json.dumps(param_dict, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()[:KEY_LEN]
    except:
        return hashlib.md5(cache_key.encode()).hexdigest()[:KEY_LEN]


def save_torchaudio_wav(wav_tensor, sr, audio_path, uuid):
    """Original: Save WAV with timestamp (rooted paths)."""
    formatted_now_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_key = get_cache_key(audio_path, uuid)
    filename = f"{formatted_now_time}_{cache_key}"
    path = get_wavout_dir() / f"{filename}.wav"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torchaudio.save(path, wav_tensor.cpu(), sr, encoding="PCM_S")
    # Cache audio path
    set_audio_cache(cache_key, path.resolve())
    return path.resolve()


def _get_or_cache_audio_info(stem: str = None, audio_path: str = None, force_refresh: bool = False) -> Optional[Tuple[int, int, int]]:
    """Cached wrapper: Get info for stem/path; stem optional (extracts from path if None).
    Backward-compatible: Works if called as _get_or_cache_audio_info(audio_path).
    Always verify cache vs fresh info (invalidate on SR mismatch); log discrepancies."""
    if not audio_path:
        return None

    # Extract stem if missing (for backward calls)
    if stem is None:
        full_stem = Path(audio_path).stem.replace('_fixed', '')  # e.g., 'vayne_csvp_voice' → 'vayne_csvp_voice'
        split_stem = full_stem.split('_')
        stem = split_stem[0] if len(split_stem) > 1 and len(split_stem[0]) >= 2 else full_stem  # Robust: min 2 chars, fallback full

    # Compute fresh always for verify, or if force/invalid
    fresh_info = _get_audio_info_robust(audio_path)
    if not fresh_info:
        logger.warning(f"Fresh info failed for {audio_path} – cannot cache/verify")
        return None

    # If cached, check vs fresh (invalidate on mismatch, esp SR/channels)
    if not force_refresh and stem in _voice_info_cache:
        cached_info = _voice_info_cache[stem]
        if Path(audio_path).exists() and Path(audio_path).stat().st_size > 0:
            if cached_info[:2] != fresh_info[:2]:  # SR + channels mismatch → stale cache
                logger.warning(f"Cache invalid for {stem}: cached SR/ch={cached_info[:2]} vs fresh={fresh_info[:2]} – refresh")
                force_refresh = True  # Proceed to update
            else:
                logger.trace(f"Info: Reused verified cached for '{stem}' (SR={cached_info[0]})")
                return cached_info
        else:
            logger.trace(f"Info: Invalidated cache for '{stem}' (file issue)")

    # Update cache with fresh (if valid)
    with _voice_info_lock:
        _voice_info_cache[stem] = fresh_info
    logger.debug(f"Info: Cached/updated for '{stem}': SR={fresh_info[0]}, ch={fresh_info[1]}, frames={fresh_info[2]}")
    return fresh_info


# Helper: Robust torchaudio info fetch (used everywhere; logs branch)
def _get_audio_info_robust(audio_path: str) -> Optional[Tuple[int, int, int]]:
    """Fetch audio info with new/old API fallback; return (sr, channels, duration_frames) or None."""
    if not os.path.exists(audio_path):
        logger.warning(f"Info: Path does not exist {audio_path}")
        return None

    try:
        # Try new API (2.1+)
        from torchaudio.io import info as torchaudio_info
        info = torchaudio_info(audio_path)
        logger.debug("Audio info: Used new io.info API")
    except (ImportError, AttributeError) as ie:
        try:
            # Fallback to deprecated/old API (2.0.x)
            info = torchaudio.info(audio_path)
            logger.trace(f"Audio info: Fallback to torchaudio.info (suppressed deprecation: {ie})")
        except Exception as e:
            logger.warning(f"Failed to get audio info for {audio_path}: {e}")
            return None

    # Normalize to tuple (handles old tuple vs new object)
    if isinstance(info, tuple) and len(info) >= 2:
        sr = info[0]
        channels = info[1]
        duration_frames = info[2] if len(info) > 2 else 0
    else:
        sr = info.sample_rate
        channels = info.num_channels
        duration_frames = info.num_frames

    return (sr, channels, duration_frames) if sr else None



# Patched validate_voice_path (robust + logging)
# Patched validate_voice_path (robust + logging)
def validate_voice_path(audio_path: str, stem: str = None, force_refresh: bool = False) -> Tuple[bool, Optional[float]]:
    """Validate voice path with cached/verified info; stem optional. Force refresh if provided (e.g., post-load)."""
    if not os.path.exists(audio_path):
        logger.warning(f"Validate: Path does not exist {audio_path}")
        return False, None

    # Use helper with optional force (default False for initial)
    info_tuple = _get_or_cache_audio_info(stem=stem, audio_path=audio_path, force_refresh=force_refresh)
    if info_tuple is None:
        return False, None

    sample_rate, num_channels, duration_frames = info_tuple
    duration = duration_frames / sample_rate if duration_frames > 0 else 0

    if sample_rate != MODEL_SR:  # Use constant (24000)
        logger.warning(f"SR mismatch: {audio_path} ({sample_rate}Hz != {MODEL_SR})")
        return False, None
    if num_channels != 1:
        logger.warning(f"Channels mismatch: {audio_path} ({num_channels} != 1)")
        return False, None

    logger.debug(f"Valid path: {audio_path} (dur={duration:.2f}s)")
    return True, duration



def check_and_update_ref(audio_path: str, exaggeration: float = 0.5, model_sr: int = MODEL_SR,
                         out_path: Optional[Path] = None, enable_pre_adjustment: bool = None, stem: str = None) -> str:
    """Validate/resample + optional pad align for short refs; return validated path.
    enable_pre_adjustment: From CONFIG or kwarg (default False; True for pad).
    Trust load SR over initial info; resample if load != model_sr. Force validate refresh post-load."""
    enable_pre_adjustment = (
        enable_pre_adjustment if enable_pre_adjustment is not None else getattr(CONFIG, 'enable_pre_adjustment', False))

    original_path = audio_path  # Always track for fallback

    # Initial validate (no force)
    validated, duration = validate_voice_path(original_path, stem=stem)
    initial_sr = None
    if info_tuple := _get_audio_info_robust(original_path):  # Fresh for diag
        initial_sr = info_tuple[0]
    if not validated:
        logger.warning(f"Initial validate failed for {original_path} (info SR={initial_sr or 'unknown'}Hz) – load to confirm")

    if validated and not enable_pre_adjustment:
        logger.debug(f"Initial valid {original_path} (no adjustment needed)")
        return original_path

    # Load/resample/mono
    try:
        waveform, load_sr = torchaudio.load(original_path)
        logger.debug(f"Loaded {original_path}: load SR={load_sr}Hz, shape={waveform.shape} (initial info SR={initial_sr})")

        # SR discrepancy check
        if initial_sr and initial_sr != load_sr:
            logger.warning(f"SR discrepancy in {original_path}: info={initial_sr}Hz vs load={load_sr}Hz – use load for decisions")

        # Force mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            logger.debug(f"Converted to mono: {waveform.shape}")

        adjusted_path = original_path
        resampled_path = None

        # Resample if load SR != model_sr
        if load_sr != model_sr:
            logger.info(f"Resampling {original_path} to {model_sr}Hz (load={load_sr}Hz)")
            resampler = torchaudio.transforms.Resample(load_sr, model_sr)
            waveform = resampler(waveform)
            resampled_stem = Path(original_path).stem + '_resampled'
            resampled_path = str(Path(original_path).parent / f"{resampled_stem}.wav")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                torchaudio.save(resampled_path, waveform, model_sr, encoding="PCM_S")
            if Path(resampled_path).exists() and Path(resampled_path).stat().st_size > 0:
                adjusted_path = resampled_path
                logger.info(f"Resampled saved: {adjusted_path} (dur={int(waveform.shape[-1]) / model_sr:.2f}s)")
            else:
                logger.error(f"Resample save failed – fallback original")
                adjusted_path = original_path
        else:
            logger.debug(f"No resample: load SR={load_sr}Hz matches {model_sr}Hz")

        if waveform.numel() == 0:
            logger.error(f"Empty waveform after process for {adjusted_path}")
            return original_path

        # Pad/align if enabled or resampled
        padded = False
        if enable_pre_adjustment or adjusted_path != original_path:
            hop_length = getattr(CONFIG, 'hop_length', 256)
            n_fft = getattr(CONFIG, 'n_fft', 1024)
            audio_samples = int(waveform.shape[-1])
            estimated_tokens = (audio_samples // hop_length) * 1.5  # Conservative
            expected_samples = int(estimated_tokens * hop_length)

            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=model_sr, n_fft=n_fft, hop_length=hop_length, n_mels=80
            )
            mel = mel_transform(waveform)
            actual_mel_len = mel.shape[-1]
            logger.info(f"Mel check for {adjusted_path}: actual={actual_mel_len}, est_tokens={estimated_tokens}")

            if actual_mel_len < estimated_tokens:
                pad_samples = expected_samples - audio_samples
                left_pad = pad_samples // 2
                right_pad = pad_samples - left_pad
                waveform = torch.nn.functional.pad(waveform, (left_pad, right_pad), mode='reflect')
                padded = True
                logger.info(f"Padded: {audio_samples / model_sr:.2f}s → {int(waveform.shape[-1]) / model_sr:.2f}s")

            # Save padded/aligned
            if padded or adjusted_path != original_path:
                orig_stem = Path(original_path).stem
                suffix = '' if orig_stem.endswith(('_resampled', '_padded')) else ('_padded' if padded else '_resampled')
                if suffix:
                    final_stem = orig_stem + suffix if not orig_stem.endswith(('_resampled', '_padded')) else orig_stem.replace(
                        Path(orig_stem).suffix.split('_')[-1], suffix.split('_')[-1])
                else:
                    final_stem = orig_stem
                aligned_out = out_path or voices_dir / f"{final_stem}.wav"
                aligned_out.parent.mkdir(parents=True, exist_ok=True)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    torchaudio.save(str(aligned_out), waveform, model_sr, encoding="PCM_S")
                if aligned_out.exists() and aligned_out.stat().st_size > 0:
                    adjusted_path = str(aligned_out)
                    logger.info(f"Final adjusted saved: {adjusted_path} (mel_len post-pad={mel_transform(waveform).shape[-1]})")
                else:
                    logger.warning(f"Final save failed – use pre-pad {adjusted_path}")

        # Final validate with force_refresh (to clear any stale cache)
        final_valid, final_dur = validate_voice_path(adjusted_path, stem=stem, force_refresh=True)
        if not final_valid:
            post_sr = (_get_audio_info_robust(adjusted_path) or (None,))[0] or 'unknown'
            logger.error(f"Final validate failed for {adjusted_path} (SR={post_sr}Hz) – fallback original")
            adjusted_path = original_path
            # Re-validate fallback
            final_valid, final_dur = validate_voice_path(original_path, stem=stem, force_refresh=True)
        else:
            logger.info(f"Final valid: {adjusted_path} (dur={final_dur:.2f}s)")

        # Dedup
        final_stem = Path(adjusted_path).stem
        if final_stem.endswith(('_fixed', '_padded', '_resampled')):
            logger.debug(f"Deduped: {adjusted_path}")

        if not final_valid:
            logger.warning(f"Final path {adjusted_path} invalid – TTS may fail/artifacts")
        return adjusted_path

    except Exception as e:
        logger.error(f"Ref process failed for {original_path}: {e} – fallback original")
        return original_path



def try_audio_cache(audio_path: str, text: str, exaggeration: float = 0.5, params: Dict = None) -> Optional[str]:
    """Check audio cache for hit; return path if exists."""
    if not audio_path or not text:
        return None
    if params is None:
        params = {'exagg': exaggeration}
    cache_key = get_cache_key(audio_path, uuid="audio_reuse", exaggeration=exaggeration, params=params)
    cached_path = get_audio_cache(cache_key)
    if cached_path and Path(cached_path).exists():
        logger.info(f"Audio cache HIT: {cache_key[:8]}... ({text[:20]}...)")
        return cached_path
    logger.debug(f"Audio cache MISS: {cache_key[:8]}...")
    return None


# Patched _compute_file_hash (robust info)
def _compute_file_hash(file_path: str, method='hybrid', stem: str = None) -> str:
    """Compute hash; use cached info if stem provided and method='quick'."""
    if not Path(file_path).exists():
        return ""
    try:
        if method == 'quick':
            # Use cached if stem known
            if stem:
                cached_info = _get_or_cache_audio_info(stem=stem, audio_path=file_path)
                if cached_info:
                    sr, channels, frames = cached_info
                    size = Path(file_path).stat().st_size
                    return f"{size}_{frames}_{sr}"
            # Fallback to old compute
            info_tuple = _get_audio_info_robust(file_path)
            if info_tuple:
                sr, channels, frames = info_tuple
                size = Path(file_path).stat().st_size
                return f"{size}_{frames}_{sr}"
            return ""  # Fail → empty

        elif method == 'hybrid':
            # Quick first (cached if possible)
            quick_h = None
            if stem:
                cached_info = _get_or_cache_audio_info(stem=stem, audio_path=file_path)
                if cached_info:
                    sr, channels, frames = cached_info
                    size = Path(file_path).stat().st_size
                    quick_h = f"{size}_{frames}_{sr}"
                    logger.debug(f"Hash: Used quick cached metadata for {stem}")
            if not quick_h:
                info_tuple = _get_audio_info_robust(file_path)
                if info_tuple:
                    sr, channels, frames = info_tuple
                    size = Path(file_path).stat().st_size
                    quick_h = f"{size}_{frames}_{sr}"
                    logger.debug("Hash: Used quick metadata")
            if quick_h:
                return quick_h  # Fast path

        # Full MD5 (default/fallback)
        h = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        logger.debug("Hash: Used full MD5")
        return h.hexdigest()
    except Exception as e:
        logger.warning(f"Hash failed {file_path}: {e}")
        return ""





# Patched get_or_queue_voice_process (non-blocking reuse + async verify/update; robust info; dedup)
# Patched get_or_queue_voice_process (non-blocking reuse + async verify/update; robust info; dedup)
# Fixes: Light pre-probe to skip sync refresh if upload SR matches; pass model to async save; consistent globals
def get_or_queue_voice_process(audio_path: str, model, device, dtype, uuid, exaggeration=0.5, quiet=False) -> str:
    stem = Path(audio_path).stem.replace('_fixed', '')  # Normalize early (e.g., 'vayne_csvp_voice')
    with _voice_cache_lock:
        cached = _voice_cache.get(stem, {})
        cached_path = cached.get('fixed_path', '')
        cached_hash = cached.get('file_hash', '')
        cached_conds_key = cached.get('conds_key', '')
        last_bg = cached.get('last_bg_time', 0)

    # Quick validate cached (assume good if exists/valid SR)
    quick_reuse = False
    if cached_path and Path(cached_path).exists():
        cached_info = _get_or_cache_audio_info(stem=stem, audio_path=cached_path)  # Header SR check
        if cached_info and cached_info[0] == MODEL_SR and cached_info[1] == 1:  # 24kHz mono
            quick_reuse = True
            if not quiet:
                logger.debug(f"Cached path valid for {stem} (SR={MODEL_SR}Hz)—immediate reuse")
        else:
            logger.warning(f"Cached path invalid for {stem}, SR={cached_info[0] if cached_info else 'unknown'}Hz – sync fix")

    # Light pre-probe upload SR (non-blocking; skip full refresh if match – reduces gen stall)
    upload_sr_match = False
    if quick_reuse and Path(audio_path).exists():
        try:
            u_info = _get_audio_info_robust(audio_path)
            upload_sr = u_info[0] if u_info else None
            if upload_sr == MODEL_SR:
                upload_sr_match = True
                logger.trace(f"Upload SR match for {stem} ({MODEL_SR}Hz) – full reuse, skip refresh")
            else:
                logger.debug(f"Upload SR {upload_sr}Hz vs cached {MODEL_SR}Hz – async will fix")
        except Exception as probe_e:
            logger.trace(f"Pre-probe failed for {stem}: {probe_e} – use cached (async checks)")

    # Early out if recent BG and cached reuse possible
    if time.time() - last_bg < 30 and quick_reuse:
        if not quiet:
            logger.debug(f"Recent BG + valid cache for {stem}—immediate reuse, skip queue/recheck")
        if cached_conds_key:
            _cache_manager.load(cached_conds_key, model, device, dtype, quiet=quiet)
        return cached_path

    # If no valid cached, sync fix once (rare; first-time or invalid)
    if not quick_reuse:
        # Sync fix as fallback (non-blocking after this)
        fallback_path = check_and_update_ref(audio_path, exaggeration, stem=stem)
        if not fallback_path or not Path(fallback_path).exists():
            logger.error(f"Sync fix failed for {stem}—fallback upload (risky)")
            fallback_path = audio_path
        # Update cache immediately (main thread)
        with _voice_cache_lock:
            _voice_cache[stem] = {
                **cached,
                'fixed_path': fallback_path,
                'last_bg_time': time.time()  # Reset timer
            }
        _save_voice_cache()
        logger.info(f"Sync fixed/updated for {stem}: {fallback_path}")
        # Load conds if available (sync, fast)
        if cached_conds_key:
            _cache_manager.load(cached_conds_key, model, device, dtype, quiet=quiet)
        return fallback_path

    # At this point: Valid cached → immediate reuse, but async verify new upload
    logger.info(f"Cached reuse for {stem}: {cached_path} (async verify new upload)")
    if cached_conds_key:
        conds = _cache_manager.load(cached_conds_key, model, device, dtype, quiet=quiet)
        if conds and not quiet:
            logger.info(f"Reused cached conds for {stem}")

    # Async verify/update thread (non-blocking; always spawn if new upload provided)
    def _async_verify_update():
        if not Path(audio_path).exists() or not quick_reuse:  # Skip if no upload or already fixed
            return
        needs_update = False  # Default
        try:
            # Quick compare: Size + SR probe (no full load/hash initially)
            if Path(audio_path).stat().st_size != Path(cached_path).stat().st_size:
                logger.debug(f"Size differ for {stem} (upload={Path(audio_path).stat().st_size} vs cached={Path(cached_path).stat().st_size}) – async fix")
                needs_update = True
            else:
                # SR probe (light: first 1s, ~10ms via StreamReader)
                sr_u = None
                try:
                    reader_u = StreamReader("file:" + audio_path)
                    probe_u = reader_u._probe_content(return_seconds=1.0)
                    sr_u = probe_u["output"][0][1].sample_rate  # Extract SR from probe output
                    reader_u.close()
                except Exception as probe_e:
                    logger.trace(f"StreamReader probe failed for upload {stem}: {probe_e} – fallback header")
                    u_info = _get_audio_info_robust(audio_path)
                    sr_u = u_info[0] if u_info else None

                sr_c = None
                try:
                    reader_c = StreamReader("file:" + cached_path)
                    probe_c = reader_c._probe_content(return_seconds=1.0)
                    sr_c = probe_c["output"][0][1].sample_rate
                    reader_c.close()
                except Exception as probe_e:
                    logger.trace(f"StreamReader probe failed for cached {stem}: {probe_e} – fallback header")
                    c_info = _get_or_cache_audio_info(stem=stem, audio_path=cached_path)
                    sr_c = c_info[0] if c_info else None

                if sr_u is None or sr_c is None or sr_u != sr_c:
                    logger.debug(f"SR differ for {stem} ({sr_u or 'unknown'}Hz vs {sr_c or 'unknown'}Hz) – async fix")
                    needs_update = True
                else:
                    # Final light hash (partial file, e.g., first 1MB)
                    h_u = hashlib.md5()
                    with open(audio_path, 'rb') as f:
                        chunk = f.read(1024 * 1024)  # 1MB
                        if chunk:
                            h_u.update(chunk)
                    h_c = hashlib.md5()
                    with open(cached_path, 'rb') as f:
                        chunk = f.read(1024 * 1024)
                        if chunk:
                            h_c.update(chunk)
                    needs_update = (h_u.hexdigest() != h_c.hexdigest())
                    if needs_update:
                        logger.debug(f"Partial hash differ for {stem} – async fix")
                    else:
                        logger.trace(f"Quick verify complete for {stem}: Identical upload – no update")

            if needs_update:
                # Async refix (resample/pad/save new fixed)
                new_fixed_path = voices_dir / f"{stem}_fixed_new.wav"  # Temp to avoid overwrite race
                fixed_path = check_and_update_ref(audio_path, exaggeration, out_path=new_fixed_path, stem=stem)
                if fixed_path and Path(fixed_path).exists():
                    # Validate new (force refresh)
                    valid_new, _ = validate_voice_path(fixed_path, stem=stem, force_refresh=True)
                    if valid_new:
                        # Atomic swap in cache
                        with _voice_cache_lock:
                            new_hash = _compute_file_hash(fixed_path, method='hybrid', stem=stem)
                            _voice_cache[stem] = {
                                **cached,
                                'fixed_path': fixed_path,
                                'file_hash': new_hash,
                                'last_bg_time': time.time()
                            }
                        # Cleanup old if different
                        if fixed_path != cached_path and Path(cached_path).exists():
                            Path(cached_path).unlink(missing_ok=True)
                            logger.info(f"Async updated {stem}: {fixed_path} (old: {cached_path})")
                        _save_voice_cache()
                        # Async conds update if possible (defer model access; fix: pass model to save)
                        if hasattr(model, 't3') and hasattr(model.t3, '_bucket_graphs') and len(model.t3._bucket_graphs) > 0:
                            logger.debug(f"Async defer conds for {stem}: Graphs active")
                        else:
                            with MODEL_LOCK:
                                model.prepare_conditionals(fixed_path, exaggeration=exaggeration)
                                new_conds_key = get_cache_key(fixed_path, uuid, exaggeration)
                                _cache_manager.save(new_conds_key, model.conds, model=model, device=device, dtype=dtype)  # Pass model
                                with _voice_cache_lock:
                                    _voice_cache[stem]['conds_key'] = new_conds_key
                                _save_voice_cache()
                                logger.info(f"Async conds updated for {stem}: {new_conds_key[:8]}")
                        if hasattr(model, 'set_conditionals'):
                            model.set_conditionals(None)  # Clear after
                    else:
                        logger.warning(f"Async fix invalid for {stem} – keep old {cached_path}")
                        if Path(fixed_path).exists():
                            Path(fixed_path).unlink()
                else:
                    logger.warning(f"Async fix failed for {stem} – keep old {cached_path}")
            else:
                logger.trace(f"Async verify: No update needed for {stem}")
        except Exception as e:
            logger.error(f"Async verify/update failed for {stem}: {e}")

    # Spawn async if not recent BG (throttle)
    if time.time() - last_bg >= 30 and Path(audio_path).exists():
        Thread(target=_async_verify_update, daemon=True, name=f"Async-Voice-{stem}").start()
        logger.debug(f"Spawned async verify for {stem} (cached reuse, check new upload)")
    elif time.time() - last_bg < 30:
        logger.trace(f"Skipped async for {stem}: Recent BG")

    return cached_path  # Always immediate reuse (non-blocking)

# Facades (unchanged)
def save_conditionals_cache(cache_key: str, cond_cls=None, model=None, device=None, dtype=None,
                            enable_memory_cache: bool = True, enable_disk_cache: bool = True) -> bool:
    return _cache_manager.save(cache_key, cond_cls, model, device, dtype, enable_memory_cache, enable_disk_cache)


def load_conditionals_cache(cache_key: str, model=None, device=None, dtype=None,
                            enable_memory_cache: bool = True, enable_disk_cache: bool = True,
                            quiet: bool = False) -> bool:
    success = _cache_manager.load(cache_key, model, device, dtype, enable_memory_cache, enable_disk_cache, quiet)
    return success is not None


def get_current_cache_key():
    return _cache_manager.get_current_cache_key ()


def is_cache_key_loaded(cache_key):
    return _cache_manager.is_cache_key_loaded(cache_key)


def get_cache_stats() -> Dict[str, Any]:
    cond_stats = _cache_manager.get_cache_stats()
    audio_stats = _audio_manager.stats()
    return {**cond_stats, **audio_stats, 'fuzzy_size': len(_fuzzy_audio_dict)}  # New: Fuzzy stat


# Clears (merged; rooted paths)
def clear_output_directories():
    if not WAV_OUTPUT_DIR.exists():
        logger.info(f"Output dir {WAV_OUTPUT_DIR} does not exist")
        return 0
    removed_count = 0
    try:
        import shutil
        for item in WAV_OUTPUT_DIR.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
                logger.info(f"Removed {item}")
                removed_count += 1
        logger.info(f"Cleared {removed_count} output dirs")
        return removed_count
    except Exception as e:
        logger.error(f"Clear output failed: {e}")
        return 0


def clear_cache_files():
    removed_count = 0
    try:
        for pt_file in CACHE_DIR.glob("*.pt"):
            pt_file.unlink()
            logger.info(f"Removed {pt_file}")
            removed_count += 1
    except Exception as e:
        logger.error(f"Clear cache failed: {e}")

    # Clear memory/audio
    with _cache_manager._cache_lock:
        _cache_manager._memory_cache.clear()
        _cache_manager._current_loaded_cache_key = None
        _cache_manager._disk_save_queue.clear()
    with _voice_info_lock:
        _voice_info_cache.clear()
        logger.debug("Cleared voice info cache")
    with _fuzzy_lock:
        _fuzzy_audio_dict.clear()
        logger.debug("Cleared voice/info/fuzzy caches")
        _fuzzy_save_counter = 0
    _audio_manager.clear()
    logger.info(f"Cleared {removed_count} .pt + memory/audio")
    return removed_count


def clear_cache(voice: Optional[str] = None, full: bool = False):
    if voice:
        keys_to_clear = []
        with _cache_manager._cache_lock:
            for k in list(_cache_manager._memory_cache):
                if voice in k:
                    keys_to_clear.append(k)
        for k in keys_to_clear:
            del _cache_manager._memory_cache[k]
        for pt_file in CACHE_DIR.glob(f"*{voice}*.pt"):
            pt_file.unlink(missing_ok=True)
            logger.debug(f"Cleared disk {pt_file}")
        logger.info(f"Cleared {len(keys_to_clear)} for voice: {voice}")
    elif full:
        return clear_cache_files()




def _load_voice_cache():
    """Load _voice_cache from JSON on init."""
    global _voice_cache
    cache_json = CACHE_AUDIO_DIR / "voice_cache.json"
    if cache_json.exists():
        try:
            with open(cache_json, 'r') as f:
                data = json.load(f)
            _voice_cache = {}
            for stem, info in data.items():
                rel_path = info.get('fixed_path', '')
                if rel_path:
                    full_path = ROOT_DIR / rel_path  # Relative to absolute
                    if full_path.exists():
                        info['fixed_path'] = str(full_path)
                    else:
                        logger.warning(f"Voice cache path invalid: {full_path}—skipping")
                        info['fixed_path'] = ''  # Invalidate
                _voice_cache[stem] = info
            logger.info(f"Loaded voice cache: {len(_voice_cache)} entries (paths rooted)")
        except Exception as e:
            logger.warning(f"Load voice cache failed: {e}—starting fresh")
            _voice_cache = {}


def _save_voice_cache():
    """Save _voice_cache to JSON."""
    with _voice_cache_lock:
        temp_cache = {}
        for stem, info in _voice_cache.items():
            if 'fixed_path' in info:
                abs_path = Path(info['fixed_path'])
                if abs_path.exists():
                    rel_path = abs_path.relative_to(ROOT_DIR)
                    temp_cache[stem] = info.copy()
                    temp_cache[stem]['fixed_path'] = str(rel_path)  # Absolute to relative
                else:
                    logger.warning(f"Save: Invalid path for {stem}—skipping")
            else:
                temp_cache[stem] = info
        if len(temp_cache) > 200:  # Evict oldest if over
            temp_cache.popitem(last=False)
        cache_json = CACHE_AUDIO_DIR / "voice_cache.json"
        try:
            with open(cache_json, 'w') as f:
                json.dump(temp_cache, f, indent=2)  # Indent for readability
            logger.debug(f"Saved voice cache: {len(temp_cache)} entries (paths relative)")
        except Exception as e:
            logger.error(f"Save voice cache failed: {e}")


# Fuzzy globals (updated cap)
_fuzzy_queue = Queue(maxsize=0)  # Non-blocking (unlimited)
_fuzzy_lock = Lock()
_fuzzy_audio_dict = {}  # {normalized_text: {'wav_path': str, 'orig_text': str}}
MAX_INDEX_SIZE = 1000  # Updated to 1000 (reasonable for memory; evict FIFO)


def normalize_text(text: str) -> str:
    """Normalize text for fuzzy indexing: Lowercase, strip punctuation/whitespace, collapse multiples."""
    # Lowercase and remove non-alphanumeric (keep spaces)
    cleaned = re.sub(r'[^\w\s]', '', text.lower())
    # Collapse multiple spaces, strip
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    # Optional: Hash prefix for unique keys (e.g., first 8 chars hash + clean)
    text_hash = hashlib.md5(cleaned.encode()).hexdigest()[:8]  # Need import hashlib
    return f"{text_hash}_{cleaned}"  # e.g., "a1b2c3d4_thane youre killing me"


def string_similarity(s1: str, s2: str, threshold=0.75) -> float:
    """Compute similarity ratio (return float for logging; caller checks >= threshold). Boost for RP."""
    # Use normalized or raw; here using raw for orig_text in sim, but norm_key for index
    s1_clean = re.sub(r'[^\w\s]', '', s1.lower())
    s2_clean = re.sub(r'[^\w\s]', '', s2.lower())
    ratio = SequenceMatcher(None, s1_clean, s2_clean).ratio()
    # Boost for RP patterns (tune as needed)
    if re.search(r'\*moan|\*scream|ahh|mmm|aah|throbb?ing?', s1_clean) and re.search(
            r'\*moan|\*scream|ahh|mmm|aah|throbb?ing?', s2_clean):
        ratio += 0.1
    return ratio  # Caller: if ratio >= threshold


last_fuzzy_save = 0
FUZZY_SAVE_INTERVAL = 5.0  # 5s min; or 10 enqueues

def _save_fuzzy_audio_cache(save_all: bool = False):
    """Save _fuzzy_audio_dict to JSON (throttled: min 5s or save_all=True; evict old if >1000 total)."""
    global last_fuzzy_save
    now = time.time()
    if not save_all and (now - last_fuzzy_save) < FUZZY_SAVE_INTERVAL:
        logger.trace(f"Save skipped: <{FUZZY_SAVE_INTERVAL}s since last (now at {now - last_fuzzy_save:.1f}s)")
        return
    last_fuzzy_save = now
    with _fuzzy_lock:
        if not _fuzzy_audio_dict:
            return
        temp_dict = {}
        total_entries = 0
        for stem, stem_entries in _fuzzy_audio_dict.items():
            if isinstance(stem_entries, dict) and stem_entries:
                temp_dict[stem] = {}
                for norm_key, entry in stem_entries.items():
                    if 'wav_path' in entry and Path(entry['wav_path']).exists():
                        abs_path = Path(entry['wav_path'])
                        rel_path = abs_path.relative_to(ROOT_DIR)
                        save_entry = entry.copy()
                        save_entry['wav_path'] = str(rel_path)
                        temp_dict[stem][norm_key] = save_entry
                        total_entries += 1
                    else:
                        logger.trace(f"Save fuzzy: Skipping invalid for {stem}:{norm_key}")
                if not temp_dict[stem]:
                    del temp_dict[stem]
        # Global eviction if too big (e.g., >1000 total)
        if total_entries > MAX_INDEX_SIZE * 2:  # Over 2000: Prune oldest per-stem
            for stem in list(temp_dict):
                while len(temp_dict[stem]) > MAX_INDEX_SIZE:
                    oldest_key = min(temp_dict[stem], key=lambda k: temp_dict[stem][k].get('time_indexed', 0))
                    del temp_dict[stem][oldest_key]
                    logger.debug(f"Pruned old entry in {stem}: {oldest_key}")
        if total_entries == 0:
            return
        fuzzy_json = CACHE_AUDIO_DIR / "fuzzy_audio_cache.json"
        try:
            with open(fuzzy_json, 'w') as f:
                json.dump(temp_dict, f, indent=2)
            if save_all:
                logger.info(f"Full fuzzy save: {total_entries} entries across {len(temp_dict)} stems")
            else:
                logger.trace(f"Throttled fuzzy save: {total_entries} entries")  # TRACE: Less spam
        except Exception as e:
            logger.error(f"Save fuzzy cache failed: {e}")

# Patched _background_index_worker (add 'sim_boost' meta to entry; optional prune short texts)
# Minor: Track boost words used (debug); evict if < min_length, but index anyway (fuzzy skips short queries, but stores for full texts)
# Patched try_fuzzy_audio_cache (configurable boost words/amount via CONFIG; short text guard tunable)
# Assumes config.py additions: DEFAULTS['fuzzy_boost_words'] = ['ahh', 'mmm', 'ooh', 'throbb', 'moan', 'gasp']
# DEFAULTS['fuzzy_boost_amount'] = 0.1
# DEFAULTS['fuzzy_min_length'] = 3  # For short guard
# get_value: If isinstance(default, list) for words, return list (add if needed: return default if isinstance(default, list) else ...)

def try_fuzzy_audio_cache(audio_path: str = None, text_input: str = None, exaggeration: float = 0.5,
                          stem: str = None, threshold: float = None, quiet: bool = False) -> Optional[str]:
    """Fuzzy audio cache: SequenceMatcher on per-stem DB; returns path or None.
    REQUIRES stem (kwarg or from audio_path); detects/ warns on swap (text short like stem).
    Boosts sim for configurable short/moans words; min length tunable."""
    if not text_input or (
            text_input and len(text_input.strip()) < 3):  # Min guard (hardcode 3 if config fails; tunable below)
        if not quiet:
            logger.debug(f"Fuzzy skip: No/invalid text_input ({text_input[:20] if text_input else 'None'})")
        return None

    # Extract/fix stem (required; fallback if swapped)
    if stem is None:
        if stem is None:
            if audio_path:
                full_stem = Path(audio_path).stem.replace('_fixed', '')  # e.g., 'ba_beatricevoice'
                split_stem = full_stem.split('_')
                stem = split_stem[0] if len(split_stem) > 1 else full_stem  # Fallback to full if short
                if len(stem) < 3:  # Still short → warn but use full
                    stem = full_stem
                    logger.warning(f"Fuzzy stem fallback to full '{stem}' from '{audio_path}' (short prefix)")
        else:
            logger.warning(
                "Fuzzy: No audio_path or stem – cannot filter per-voice; using global fallback (inefficient)")
            stem = 'global'  # Fallback: All entries (less precise)
    if not stem or len(stem) < 3:  # Invalid stem (e.g., swapped long text)
        logger.warning(f"Fuzzy: Invalid stem '{stem[:20]}...' – check caller args (possible swap: text as audio_path?)")
        return None

    threshold = threshold or CONFIG.get_value('fuzzy_threshold', default=0.75)

    # Detect swap: If text_input short/looks like stem (e.g., 'dlc1seranavoice'), warn + auto-swap
    min_length = CONFIG.get_value('fuzzy_min_length', default=3)
    if len(text_input.strip()) < 10 and re.match(r'^[a-z0-9_]+(voice|maid|npc)?$',
                                                 text_input.lower()):  # Heuristic: Looks like stem
        logger.warning(
            f"Fuzzy detect: Possible arg swap (text_input='{text_input}' too stem-like) – auto-fixing (use correct: audio_path, text_input=text, stem=voice)")
        text_input, audio_path = audio_path, text_input  # Swap back
        stem = Path(audio_path).stem.split('_')[0] if audio_path else stem  # Re-extract
        if len(text_input.strip()) < min_length:
            if not quiet:
                logger.debug(f"Fuzzy skip after swap: Text too short (<{min_length} chars)")
            return None

    if not quiet:
        per_stem_size = len(_fuzzy_audio_dict.get(stem, {}))
        global_size = sum(len(entries) for entries in _fuzzy_audio_dict.values()) if _fuzzy_audio_dict else 0
        logger.debug(
            f"Trying fuzzy for stem '{stem}', text: '{text_input[:30]}...' | Global DB: {global_size} (this stem: {per_stem_size}) (threshold: {threshold:.2f})")

    # Early exit if no entries for stem
    if stem not in _fuzzy_audio_dict or not _fuzzy_audio_dict[stem]:
        if not quiet:
            logger.debug(
                f"Fuzzy MISS for '{text_input[:30]}...' (stem '{stem}'): No candidates in DB (global: {global_size})")
        return None

    candidates = list(_fuzzy_audio_dict[stem].values())  # List for iteration
    best_match, best_path, best_sim = None, None, 0.0

    clean_text = re.sub(r'[^\w\s]', '', text_input.lower()).strip()  # Normalize text for sim
    if len(clean_text.split()) < min_length // 2:  # Word-based too (e.g., "aah..." → short)
        if not quiet:
            logger.debug(f"Fuzzy MISS early: Normalized text too short ('{clean_text}')")
        return None

    boost_words = CONFIG.get_value('fuzzy_boost_words',
                                   default=['ahh', 'mmm', 'ooh', 'throbb', 'moan', 'gasp', 'oh', 'fuck', 'yes',
                                            'aah'])  # Extended for your RP logs
    boost_amount = CONFIG.get_value('fuzzy_boost_amount', default=0.1)

    for entry in candidates:
        clean_entry = re.sub(r'[^\w\s]', '', entry['orig_text'].lower()).strip()
        raw_ratio = SequenceMatcher(None, clean_text, clean_entry).ratio()

        # Boost: Query + entry sides (additive; cap)
        sim_boost = entry.get('sim_boost', 0.0)  # Stored meta
        if boost_amount > 0 and boost_words:
            has_boost_query = any(word in clean_text for word in boost_words)
            has_boost_entry = any(word in clean_entry for word in boost_words)
            if has_boost_query or has_boost_entry:
                sim_boost += boost_amount * (1 if has_boost_query else 0.5) * (
                    1 if has_boost_entry else 0.5)  # Weighted: Full if both, half if one
        adjusted_sim = min(1.0, raw_ratio + sim_boost)

        if adjusted_sim > best_sim:
            best_sim = adjusted_sim
            best_match = entry['orig_text']
            best_path = entry['wav_path']
            if not quiet and adjusted_sim >= threshold * 0.9:  # Only log near-hits (e.g., 0.675+ for 0.75 thresh)
                logger.debug(
                    f"  Candidate: '{clean_entry[:30]}...' raw={raw_ratio:.3f} +boost={sim_boost:.3f} = {adjusted_sim:.3f}")

    if best_sim >= threshold:
        if not quiet:
            logger.info(
                f"Fuzzy cache HIT: '{text_input[:30]}...' ≈ '{best_match[:30]}...' (sim={best_sim:.3f} >= {threshold:.2f}, boost={boost_amount}) for stem '{stem}' -> {best_path}")
        return best_path
    else:
        if not quiet:
            logger.debug(
                f"Fuzzy MISS for '{text_input[:30]}...' (stem '{stem}'): best sim={best_sim:.3f} < {threshold:.2f} ({len(candidates)} candidates)")
            if best_match:
                logger.debug(f"  Closest: '{best_match[:30]}...' (sim={best_sim:.3f})")
        return None



# Patched string_similarity (use if refactoring; now with config boost – optional, for consistency)
def string_similarity(s1: str, s2: str, threshold=0.75) -> float:
    """Compute similarity ratio (return float for logging; caller checks >= threshold). Boost for RP/short/moans."""
    s1_clean = re.sub(r'[^\w\s]', '', s1.lower())
    s2_clean = re.sub(r'[^\w\s]', '', s2.lower())
    ratio = SequenceMatcher(None, s1_clean, s2_clean).ratio()
    # Boost for configurable words (same as try_fuzzy)
    boost_words = CONFIG.get_value('fuzzy_boost_words', default=['ahh', 'mmm', 'ooh', 'throbb', 'moan', 'gasp'])
    boost_amount = CONFIG.get_value('fuzzy_boost_amount', default=0.1)
    sim_boost = 0.0
    if boost_amount > 0 and boost_words:
        for word in boost_words:
            if word in s1_clean or word in s2_clean:
                sim_boost = boost_amount
                break
    adjusted_ratio = min(1.0, ratio + sim_boost)
    return adjusted_ratio


# Patched _background_index_worker (add 'sim_boost' meta to entry; optional prune short texts)
# Minor: Track boost words used (debug); evict if < min_length, but index anyway (fuzzy skips short queries, but stores for full texts)
_fuzzy_save_counter = 0  # Global throttle for incremental saves

def _background_index_worker():
    global _fuzzy_save_counter
    while True:
        try:
            text, wav_path, voice_stem = _fuzzy_queue.get(timeout=1)
            orig_text = text
            min_length = CONFIG.get_value('fuzzy_min_length', default=3)
            if len(orig_text.strip()) < min_length:  # Optional: Skip indexing very short (e.g., "a" noise)
                logger.debug(f"Skipped indexing short text (<{min_length}): {orig_text[:10]}...")
                continue
            norm_key = normalize_text(text)
            boost_words = CONFIG.get_value('fuzzy_boost_words', default=['ahh', 'mmm', 'ooh', 'throbb', 'moan', 'gasp'])
            boost_amount = CONFIG.get_value('fuzzy_boost_amount', default=0.1)
            clean_text = re.sub(r'[^\w\s]', '', orig_text.lower())
            sim_boost = 0.0
            if boost_amount > 0 and boost_words:
                for word in boost_words:
                    if word in clean_text:
                        sim_boost = boost_amount
                        break  # Meta: Potential boost for this entry
            with _fuzzy_lock:
                # Ensure per-stem sub-dict
                if voice_stem not in _fuzzy_audio_dict:
                    _fuzzy_audio_dict[voice_stem] = {}
                stem_dict = _fuzzy_audio_dict[voice_stem]
                if norm_key in stem_dict:  # Dedup per-stem/norm_key
                    logger.debug(f"Skipped dup fuzzy index: {norm_key[:30]} ({voice_stem})")
                else:
                    if len(stem_dict) >= MAX_INDEX_SIZE:  # Per-stem cap
                        stem_dict.pop(next(iter(stem_dict)))  # Evict oldest in stem
                        logger.debug(f"Fuzzy per-stem full ({voice_stem}) – evicted oldest")
                    time_indexed = time.time()
                    stem_dict[norm_key] = {
                        'wav_path': wav_path,
                        'orig_text': orig_text,
                        'stem': voice_stem,
                        'sim_boost': sim_boost,  # Meta for future
                        'time_indexed': time_indexed
                    }
                    _fuzzy_save_counter += 1
                    logger.debug(f"Indexed fuzzy audio: {norm_key[:30]} -> {wav_path} (stem: {voice_stem}, boost={sim_boost})")
                # Throttle saves: Every 10 adds, queue >20, or 30s idle
                if (_fuzzy_save_counter >= 10 or _fuzzy_queue.qsize() > 20):
                    _save_fuzzy_audio_cache(save_all=False)  # Incremental
                    _fuzzy_save_counter = 0
            _save_fuzzy_audio_cache()  # Persist (add if not exists; thread-safe)
        except:
            # Idle timeout: Save if dirty (e.g., >30s no queue, but entries > prev)
            if _fuzzy_queue.empty() and _fuzzy_audio_dict:  # Periodic full save
                time_since_last = time.time() - last_fuzzy_save
                if time_since_last > 30:  # >30s idle → full safe save
                    _save_fuzzy_audio_cache(save_all=True)
            pass  # Continue loop

# Start daemon thread at init (fuzzy worker)
threading.Thread(target=_background_index_worker, daemon=True, name="FuzzyIndexer").start()

# Sync pre-extract (threaded for non-blocking; hardcoded top voices)
# Sync pre-extract (threaded for non-blocking; hardcoded top voices)
def pre_extract_fixed_voices(model, device, dtype, top_voices: List[str] = ['nwskatyavoice', 'vp_11_lilia', 'nwsjennavoice', 'ba_ahnivoice']):
    """Pre-extract conds for top voices (threaded; rooted paths). Non-blocking with timeout/error handling."""
    def _extract_worker(voice):
        logger.debug(f"Pre-extract worker start: {voice}")
        if not T3_AVAILABLE:
            logger.debug(f"Pre-extract skip {voice}: T3 unavailable")
            return
        ref_path = voices_dir / f"{voice}.wav"
        if not ref_path.exists():
            logger.warning(f"Skipping pre-extract {voice}: No {ref_path}")
            return
        cache_key = get_cache_key(str(ref_path), uuid=voice, exaggeration=0.5)
        if _cache_manager.load(cache_key, model, device, dtype, quiet=True):
            logger.info(f"Pre-extract HIT: {voice}")
            return
        try:
            # Check graphs before prep (defer if active)
            if hasattr(model, 't3') and hasattr(model.t3, '_bucket_graphs') and len(model.t3._bucket_graphs) > 0:
                logger.debug(f"Pre-extract defer {voice}: Graphs active")
                return
            with MODEL_LOCK:  # Lock to prevent race with main model access
                model.prepare_conditionals(str(ref_path), exaggeration=0.5)
                if model.conds:  # Success check
                    _cache_manager.save(cache_key, model.conds, model, device, dtype)
                    logger.info(f"Pre-extracted: {voice} (key={cache_key[:8]})")
                else:
                    logger.warning(f"Pre-extract conds empty for {voice}")
                model.set_conditionals(None)  # Clear after save
        except Exception as e:
            logger.warning(f"Pre-extract failed {voice}: {e}")
            if hasattr(model, 'set_conditionals') and model.conds:
                model.set_conditionals(None)
        logger.debug(f"Pre-extract worker end: {voice}")

    logger.info(f"Pre-extracting {len(top_voices)} voices")
    threads = []
    for voice in top_voices:
        t = Thread(target=_extract_worker, args=(voice,))
        t.daemon = True
        t.start()
        threads.append(t)

    # Join with timeout (10s per thread; skip hangers)
    for t in threads:
        t.join(timeout=10)  # 10s timeout per worker
        if t.is_alive():
            logger.warning(f"Pre-extract worker timeout (skipped): {t.name} – may need manual WAV/SR check")
        else:
            logger.trace(f"Pre-extract worker complete: {t.name}")
    logger.info("Pre-extract complete")



# Init (merged: preload + optional pre-extract)
# Init (merged: preload + optional pre-extract + pre-validate)
def init_conditional_memory_cache(model=None, device=None, dtype=None, quiet: bool = False,
                                  pre_extract: bool = False, pre_validate_voices: bool = False) -> Tuple[bool, bool]:
    _load_voice_cache()
    _load_fuzzy_cache() # TODO make generic init?
    device = device or DEFAULT_DEVICE
    dtype = dtype or DEFAULT_DTYPE

    # Verify voices dir (log available for debugging)
    available_voices = [f.stem for f in voices_dir.glob("*.wav") if not f.stem.endswith('_fixed')]
    missing_voices = []
    if pre_extract:
        top_voices = ['nwskatyavoice', 'vp_11_lilia', 'nwsjennavoice', 'ba_ahnivoice']
        for v in top_voices:
            if v not in available_voices:
                missing_voices.append(v)
        if missing_voices:
            logger.warning(f"Pre-extract: Missing voices in {voices_dir}: {missing_voices}. Add WAV files for faster hits.")
        if not quiet and available_voices:
            logger.info(f"Found {len(available_voices)} voices in {voices_dir}: {available_voices[:5]}...")  # First 5

    if quiet:
        logger.debug(f"Voices dir {voices_dir} has {len(available_voices)} files")

    # Preload all .pt
    loaded = 0
    for pt_file in CACHE_DIR.glob("*.pt"):
        cache_key = pt_file.stem
        try:
            loaded_raw = torch.load(pt_file, map_location='cpu', weights_only=False)
            state = loaded_raw if isinstance(loaded_raw, dict) else {'data': loaded_raw}
            conds = _cache_manager._reconstruct_conds(state, model, device, dtype)
            if ENABLE_MEMORY_CACHE:
                with _cache_manager._cache_lock:
                    _cache_manager._memory_cache[cache_key] = conds or state
                    _cache_manager._memory_cache.move_to_end(cache_key)
            loaded += 1
            if not quiet:
                logger.debug(f"Preloaded {pt_file.name} → {cache_key[:8]}")
        except Exception as e:
            if not quiet:
                logger.warning(f"Preload failed {pt_file}: {e}")

    # Evict excess
    if len(_cache_manager._memory_cache) > MAX_MEMORY_ENTRIES:
        excess = len(_cache_manager._memory_cache) - MAX_MEMORY_ENTRIES
        _cache_manager.evict_lru(excess)
        if not quiet:
            logger.info(f"Evicted {excess} excess after preload")

    # Optional pre-extract (if enabled; skip if quiet/hang-prone) TODO review
    #if pre_extract and model:
    #    pre_extract_fixed_voices(model, device, dtype)

    # Optional pre-validate all voices (new: gains first-gen speedup; if enabled)
    pre_validate_count = 0
    if pre_validate_voices and model:
        pre_validate_count = pre_validate_all_voices(model, device, dtype)

    # Compute totals/stats *after* all preloads (includes new conds if pre-validated)
    total_pt = len(list(CACHE_DIR.glob('*.pt')))
    stats = _cache_manager.get_cache_stats()

    # Log summary
    if not quiet:
        logger.info(f"Cache init: Memory={ENABLE_MEMORY_CACHE}, Disk={ENABLE_DISK_CACHE}, Loaded {loaded} from {total_pt} (memory: {stats['memory_cache_size']}, pre-valid: {pre_validate_count})")
    return ENABLE_MEMORY_CACHE, ENABLE_DISK_CACHE


# Parallel voice pre-validation (fix SR/pad/cache conds for all; optional, fast)
def pre_validate_all_voices(model, device, dtype, voices_dir: Path = voices_dir, max_workers: int = 8, quiet: bool = False):
    """Pre-validate/fix all voices in dir (resample/pad + conds); threaded for speed."""
    if not model or not T3_AVAILABLE:
        if not quiet:
            logger.debug("Pre-validate skip: No model/T3")
        return 0
    voice_files = [f for f in voices_dir.glob("*.wav") if not f.stem.endswith(('_fixed', '_padded', '_resampled'))]
    if not voice_files:
        if not quiet:
            logger.debug("No voices to pre-validate")
        return 0

    def _validate_worker(file_path: Path):
        stem = file_path.stem.replace('_fixed', '').replace('_padded', '').replace('_resampled', '')  # Normalize
        fixed_path = check_and_update_ref(str(file_path), exaggeration=0.5, stem=stem)  # Auto-resample/pad/save fixed
        if not fixed_path or not Path(fixed_path).exists():
            logger.warning(f"Pre-validate failed {stem}: {fixed_path}")
            return 0
        # Cache conds (locked)
        cache_key = get_cache_key(fixed_path, uuid=stem, exaggeration=0.5)
        if _cache_manager.is_cache_key_loaded(cache_key):
            if not quiet:
                logger.debug(f"Pre-validate hit: {stem}")
            return 1
        try:
            # Defer if model in use (conds not None) or graphs active (safe check)
            if model and (model.conds is not None or
                          (hasattr(model, 't3') and
                           getattr(model.t3, '_bucket_graphs', None) and len(
                                      getattr(model.t3, '_bucket_graphs', [])) > 0)):
                logger.debug(f"Pre-validate defer {stem}: Model busy/graphs active")
                return 0  # Skip conds, but file fixed (partial success)
            with MODEL_LOCK:
                model.prepare_conditionals(fixed_path, exaggeration=0.5)
                if model.conds:
                    _cache_manager.save(cache_key, model.conds, model, device, dtype)
                    model.set_conditionals(None)  # Clear
                    if not quiet:
                        logger.debug(f"Pre-validated: {stem} -> {cache_key[:8]}")
                    return 1
                else:
                    logger.warning(f"Pre-validate conds empty: {stem}")
        except AttributeError as ae:
            logger.trace(f"Pre-validate defer {stem}: Model not ready ({ae}) – file fixed, conds on-demand")
            # Still count file fix as partial success
            return 0.5  # Or 1 if you want; adjust validated_count += result
        except Exception as e:
            logger.warning(f"Pre-validate conds failed {stem}: {e}")
        return 0

    if not quiet:
        logger.info(f"Pre-validating {len(voice_files)} voices (max_workers={max_workers})")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    validated_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_validate_worker, f) for f in voice_files]
        for future in as_completed(futures, timeout=60):  # 30s total timeout
            try:
                validated_count += future.result(timeout=10)  # Per-worker 5s
            except TimeoutError:
                logger.warning("Pre-validate worker timeout – skip")
    if not quiet:
        logger.info(f"Pre-validated {validated_count}/{len(voice_files)} voices")
    return validated_count

# Fuzzy globals (init here if missing)
_fuzzy_audio_dict = {}  # {stem: {norm_key: entry}}
_fuzzy_save_counter = 0  # For throttle

# Load fuzzy on init (add to init_conditional_memory_cache, after _load_voice_cache ~line 1380)
def _load_fuzzy_cache():
    fuzzy_json = CACHE_AUDIO_DIR / "fuzzy_audio_cache.json"
    global _fuzzy_audio_dict
    if fuzzy_json.exists():
        try:
            with open(fuzzy_json, 'r') as f:
                data = json.load(f)
            _fuzzy_audio_dict = {}
            for stem, stem_entries in data.items():
                _fuzzy_audio_dict[stem] = {}
                for norm_key, entry in stem_entries.items():
                    if 'wav_path' in entry:
                        rel_path = entry['wav_path']
                        full_path = ROOT_DIR / rel_path
                        if full_path.exists():
                            entry['wav_path'] = str(full_path)
                        else:
                            continue  # Skip invalid
                    _fuzzy_audio_dict[stem][norm_key] = entry
            logger.info(f"Loaded fuzzy cache: {sum(len(entries) for entries in _fuzzy_audio_dict.values())} entries across {len(_fuzzy_audio_dict)} stems")
        except Exception as e:
            logger.warning(f"Load fuzzy cache failed: {e}")
            _fuzzy_audio_dict = {}



