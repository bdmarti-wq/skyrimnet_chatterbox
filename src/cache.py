# src/cache.py
# Full patched methods: validate_voice_path, _load_and_mono, _resample_if_needed, _pad_if_needed,
# adjust_audio_length_torch, _save_adjusted_and_validate, check_and_update_ref, get_or_queue_voice_process.
# Changes:
# - validate_voice_path: Skip artifact check for refs (voices dir or _fixed_new/_padded stems). Use accurate thresh=7000Hz, ratio>0.5.
# - _pad_if_needed: Full args handling (required out_path, enable_pre_adjustment); reflect pad for quality; save if out_path.
# - adjust_audio_length_torch: Reflect mode for zero artifacts; gated by enable_pre_adjustment.
# - _save_adjusted_and_validate: Dynamic mel after pad; validate force_refresh.
# - check_and_update_ref: Pass out_path=voices_dir/{stem}_padded.wav, enable_pre_adjustment=CONFIG value. Only reprocess if artifacts AND pre_adjust=True.
# - get_or_queue_voice_process: Use patched helpers; spawn async only if needed.
# - is_artifact_laden: Imported/adjusted thresh (7000Hz, ratio>0.5); accurate log (no false ">6000" for low mean).
# No omissions: Full method bodies.

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
import torch.nn.functional as F
import torchaudio
from torch.serialization import safe_globals  # For whitelisting in load
from torchaudio.io import StreamReader  # For non-blocking SR probe in async
import numpy as np
from pathlib import Path
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple, Union, List
from loguru import logger  # Assume available; fallback to print if not
import threading  # Ensure imported (likely already is)
from .config import CONFIG
from src.audio_utils import is_artifact_laden  # Import for artifact check
import hashlib
from threading import Thread
from pathlib import Path
from loguru import logger

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
# Global gen lock (serialize vs async for CUDA graph safety)
GEN_ACTIVE_LOCK = threading.RLock()

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



_local_voice_cache = {}  # Memory: stem → {'content_hash': str, 'sr': int, 'processed_path': str}

def _content_hash(wav_path: str) -> str:
    """SR-agnostic: Hash decoded audio samples (ignores header/SR)."""
    try:
        waveform, _ = torchaudio.load(wav_path)
        if waveform.dim() > 1:
            waveform = waveform.mean(0)  # Mono
        return hashlib.md5(waveform.numpy().tobytes()).hexdigest()
    except Exception as e:
        logger.warning(f"Content hash failed for {wav_path}: {e} – fallback file hash")
        return _compute_file_hash(wav_path, method='hybrid')  # Fallback

def scan_local_voices(voices_dir: Path = voices_dir, rebuild: bool = False, use_content_hash: bool = True) -> int:
    """Pre-scan voices_dir at startup: Hash + SR, cache for API dedup. use_content_hash=True for SR-agnostic."""
    global _local_voice_cache
    if not rebuild and _local_voice_cache:
        logger.info(f"Reusing local voice cache: {len(_local_voice_cache)} entries")
        return len(_local_voice_cache)

    _local_voice_cache.clear()
    scanned = 0
    for wav_file in voices_dir.glob("*.wav"):
        if any(suffix in wav_file.stem for suffix in ['_padded', '_resampled', '_fixed']):  # Skip processed
            continue
        stem = _normalize_stem(str(wav_file))
        if stem in _local_voice_cache:
            continue  # Dedup stems

        # Hash (content or file-level)
        hash_val = _content_hash(str(wav_file)) if use_content_hash else _compute_file_hash(str(wav_file), method='full')
        if not hash_val:
            continue

        info = _get_audio_info_robust(str(wav_file))
        if info:
            sr, ch, frames = info
            processed_file = voices_dir / f"{stem}_padded.wav"
            _local_voice_cache[stem] = {
                'content_hash': hash_val,
                'sr': sr,
                'processed_path': str(processed_file) if processed_file.exists() else None
            }
            scanned += 1
            logger.debug(f"Scanned {stem}: hash={hash_val[:8]}, SR={sr}Hz, processed={processed_file.exists()}")

    logger.info(f"Pre-scanned {scanned} local voices to memory (SR-agnostic: {use_content_hash})")
    return scanned






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
                    # NEW: Proactive evict if pressure (prioritize temps/UUIDs)
                    if len(self._memory_cache) > MAX_MEMORY_ENTRIES * 0.9:
                        # Evict oldest temp-like (UUID/long keys first)
                        to_evict = []
                        for k in list(self._memory_cache):
                            if (len(k) > 20 or any(c.isdigit() for c in k[:20]) and len(k) > 10) and len(
                                    to_evict) < 10:  # Heuristic temp
                                to_evict.append(k)
                        for k in to_evict:
                            self._memory_cache.pop(k, None)
                            pt_file = CACHE_DIR / f"{k}.pt"
                            if pt_file.exists():
                                pt_file.unlink(missing_ok=True)
                            logger.debug(f"Proactive evicted temp: {k[:8]}... (memory={len(self._memory_cache)})")
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
    wavout_dir.mkdir(parents=True, exist_ok=True)  # FIXED: parents=True (not persona; valid kwargs)
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


def validate_voice_path(audio_path: str, stem: str = None, force_refresh: bool = False) -> Tuple[bool, Optional[float]]:
    """Validate voice path with cached/verified info; stem optional. Force refresh if provided (e.g., post-load).
    FIXED: Skip artifact check for refs (voices dir or _fixed_new/_padded stems). Accurate logs."""
    if not os.path.exists(audio_path):
        logger.warning(f"Validate: Path does not exist {audio_path}")
        return False, None

    # Extract stem if missing (for backward calls)
    if stem is None:
        full_stem = Path(audio_path).stem.replace('_fixed', '')  # e.g., 'vayne_csvp_voice' → 'vayne_csvp_voice'
        split_stem = full_stem.split('_')
        stem = split_stem[0] if len(split_stem) > 1 and len(split_stem[0]) >= 2 else full_stem  # Robust: min 2 chars, fallback full

    # Use helper with optional force (default False for initial)
    info_tuple = _get_or_cache_audio_info(stem=stem, audio_path=audio_path, force_refresh=force_refresh)
    if info_tuple is None:
        return False, None

    sample_rate, num_channels, duration_frames = info_tuple
    duration = duration_frames / sample_rate if duration_frames > 0 else 0

    if sample_rate != MODEL_SR:  # Use constant (24000)
        logger.warning(f"SR mismatch: {audio_path} ({sample_rate}Hz != {MODEL_SR}Hz)")
        return False, None
    if num_channels != 1:
        logger.warning(f"Channels mismatch: {audio_path} ({num_channels} != 1)")
        return False, None

    # FIXED: Artifact check only for non-refs/gens (skip if in voices dir or _fixed_new/_padded stems; refs are clean originals)
    is_voice_ref = ('voices' in str(audio_path).lower() or str(Path(audio_path).parent).endswith('voices')) or \
                    any(suffix in Path(audio_path).stem for suffix in ['_fixed_new', '_padded', '_resampled'])
    if not is_voice_ref:  # Only check non-refs (e.g., output_temp gens; refs/Skyrim voices skipped)
        if CONFIG.get_value('fuzzy_artifact_purge_enable', default=True):
            if is_artifact_laden(audio_path):
                logger.warning(f"Artifact detected in {audio_path} (centroid high/ratio high) – invalid for use (purge if gen output)")
                return False, "artifacts_detected"
            logger.trace(f"Artifact check passed: {audio_path} (mean centroid clean)")
        else:
            logger.trace(f"Artifact purge disabled – check skipped for {audio_path}")
    else:
        logger.trace(f"Skipped artifact check for ref: {audio_path} (normal Skyrim voice; is_voice_ref={is_voice_ref})")

    if duration < CONFIG.get_value('min_voice_duration', default=0.5):  # Assume MIN_VOICE_DURATION=0.5s configurable
        logger.warning(f"Too short: {duration:.2f}s < {CONFIG.get_value('min_voice_duration', default=0.5)}s")
        return False, f"Too short ({duration:.2f}s)"

    logger.debug(f"Valid path: {audio_path} (dur={duration:.2f}s)")
    return True, duration



# New Helper: Load and mono (extracted; testable: path → (waveform, sr) or error)
def _load_and_mono(audio_path: str) -> tuple[torch.Tensor, int] | None:
    """Load waveform, force mono; return (waveform, sr) or None on fail."""
    try:
        waveform, load_sr = torchaudio.load(audio_path)
        logger.debug(f"Loaded {audio_path}: load SR={load_sr}Hz, shape={waveform.shape}")
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            logger.debug(f"Converted to mono: {waveform.shape}")
        if waveform.numel() == 0:
            logger.error(f"Empty waveform for {audio_path}")
            return None
        return waveform, load_sr
    except Exception as e:
        logger.error(f"Load/mono failed for {audio_path}: {e}")
        return None

# New Helper: Resample if needed (extracted; testable: waveform/sr → new_waveform/path or orig)
def _resample_if_needed(waveform: torch.Tensor, load_sr: int, model_sr: int, audio_path: str) -> tuple[torch.Tensor, str]:
    """Resample if SR mismatch; save temp if needed, return (new_waveform, path). FIXED: Ensure adjusted_path handles no-resample."""
    adjusted_path = audio_path
    if load_sr != model_sr:
        logger.info(f"Resampling {audio_path} to {model_sr}Hz (load={load_sr}Hz)")
        resampler = torchaudio.transforms.Resample(orig_freq=load_sr, new_freq=model_sr)
        waveform = resampler(waveform)
        resampled_stem = Path(audio_path).stem + '_resampled'
        resampled_path = str(Path(audio_path).parent / f"{resampled_stem}.wav")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torchaudio.save(resampled_path, waveform, model_sr, encoding="PCM_S")
        if Path(resampled_path).exists() and Path(resampled_path).stat().st_size > 0:
            adjusted_path = resampled_path
            logger.info(f"Resampled saved: {adjusted_path} (dur={int(waveform.shape[-1]) / model_sr:.2f}s)")
        else:
            logger.error(f"Resample save failed – fallback original waveform (path unchanged: {audio_path})")
    else:
        logger.debug(f"No resample: load SR={load_sr}Hz matches {model_sr}Hz (path: {adjusted_path})")
    return waveform, adjusted_path


# New Helper: Pad if needed (extracted; testable: waveform/path → padded_waveform/bool)
def _pad_if_needed(waveform: torch.Tensor, audio_path: str, model_sr: int, enable_pre_adjustment: bool = False) -> torch.Tensor:
    """Pad waveform for stability/quality (reflect mode); no out_path dependency – in-mem only. FIXED: Required args handled; return tensor."""
    hop_length = CONFIG.get_value('hop_length', default=256)
    n_fft = CONFIG.get_value('n_fft', default=2048)

    # Step 1: Est min-pad based on stability (est tokens * hop_length)
    est_tokens = len(waveform) / hop_length  # Expected mel frames
    expected_samples = int(est_tokens * hop_length)
    if len(waveform) < expected_samples:
        pad_samples = expected_samples - len(waveform)
        waveform = F.pad(waveform, (0, pad_samples), mode='replicate')  # Replicate initial for safe extension
        logger.info(f"Est min-pad: {len(waveform)} → {expected_samples} samples (est_tokens={est_tokens:.1f}; stability)")
    else:
        logger.debug(f"Est length ok: {len(waveform)} >= {expected_samples} – no min-pad")

    # Step 2: Mel align pad if enabled (exact quality; gated for speed)
    if enable_pre_adjustment:
        from torchaudio.transforms import MelSpectrogram
        mel_transform = MelSpectrogram(sample_rate=model_sr, n_fft=n_fft, hop_length=hop_length, n_mels=80)
        mel = mel_transform(waveform.unsqueeze(0))  # [1, n_mels, frames]
        adjusted = adjust_audio_length_torch(waveform.unsqueeze(0), model_sr, mel.shape, hop_length, enable_pre_adjustment)
        if len(adjusted) != len(waveform):
            logger.info(f"Mel fine-tune pad: {len(waveform)} → {len(adjusted)} samples (exact align)")
            waveform = adjusted.squeeze(0)  # Remove batch dim if added
    else:
        logger.debug(f"Pre-adjust off – est stability only (speed prioritized)")

    logger.debug(f"Pad complete: {len(waveform)} samples (hybrid: stable + {'quality' if enable_pre_adjustment else 'est'})")
    return waveform  # Always return tensor (in-mem; no save here – save in caller)


# Add to src/cache.py (after imports; before _pad_if_needed)
# src/cache.py (replace your adjust_audio_length_torch)



def adjust_audio_length_torch(waveform: torch.Tensor, model_sr: int, mel_shape: torch.Size, hop_length: int,
                             enable_pre_adjustment: bool, stem: str = None) -> torch.Tensor:
    """FIXED: Correct logging (current=len(waveform), final after pad)."""
    if not enable_pre_adjustment:
        logger.debug(f"Pre-adjust disabled ({stem if stem else 'unknown'}) – raw waveform")
        return waveform

    actual_mel_len = mel_shape[-1]
    expected_samples = actual_mel_len * hop_length
    diff_samples = expected_samples - len(waveform)
    if diff_samples == 0:
        logger.debug(f"Mel len exact for {stem}: {actual_mel_len} frames – no adjustment")
        return waveform

    logger.debug(f"Mel align ({stem}): expected={expected_samples}, current={len(waveform)} (diff={diff_samples})")

    if diff_samples > 0:
        waveform = F.pad(waveform, (diff_samples // 2, (diff_samples + 1) // 2), mode='reflect')
        logger.info(f"Pad ({stem}): +{diff_samples} samples, final={len(waveform)} (from {len(waveform)-diff_samples})")
        return waveform
    else:
        excess = -diff_samples
        if excess > hop_length * 10:
            waveform = waveform[..., :expected_samples]
            logger.info(f"Trim ({stem}): -{excess} samples, final={len(waveform)} (from {len(waveform)+excess})")
        else:
            logger.debug(f"Minor excess {excess} ({stem}) – kept")
        return waveform


# New Helper: Save adjusted and validate (extracted; testable: waveform/path/stem → final_path/valid)
import soundfile as sf  # Ensure imported at top of cache.py for info


def _save_adjusted_and_validate(waveform: torch.Tensor, adjusted_path: str, original_path: str, model_sr: int,
                                stem: str = None, out_path: Optional[Path] = None, enable_pre_adjustment: bool = False,
                                exaggeration: float = 0.5) -> tuple[str, bool, Optional[float]]:
    """
    FIXED: Save padded/resampled waveform; validate. Return (final_path, valid: bool, dur: float or None).
    - If no out_path and no adjustment needed (len unchanged), validate original without saving.
    - Logs actual frames/dur using sf.info after save (fixes "len=1" bug).
    - Handles persistent or temp paths (e.g., for voices_dir).
    """
    # Import check (assume soundfile available; skip if not)
    try:
        sf  # Verify imported
    except NameError as e:
        raise ImportError("soundfile required for validation logging – install via pip install soundfile") from e

    # No save if no out_path and no adjustment needed (in-mem only; fast return)
    if not out_path and len(waveform) == _get_audio_info_robust(original_path)[2] if original_path else 0:
        logger.debug(f"No save needed (in-mem only): {adjusted_path} (dur={len(waveform) / model_sr:.2f}s)")
        valid, dur = validate_voice_path(original_path, stem=stem, force_refresh=True)
        return adjusted_path if valid else original_path, valid, dur

    # Determine out_path (persistent if voices_dir, else temp padded)
    if out_path is None:
        out_path = Path(adjusted_path).parent / f"{Path(original_path).stem}_padded.wav"
        # Prefer persistent if stem known (e.g., voices_dir)
        if stem and voices_dir.exists():
            persistent_out = voices_dir / f"{stem}_padded.wav"
            if persistent_out.parent == voices_dir:  # Valid stem
                out_path = persistent_out
                logger.debug(f"Using persistent out_path for {stem}: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save adjusted waveform (ensure 2D for save if needed)
    if waveform.dim() == 1:
        save_wav = waveform.unsqueeze(0)  # [1, samples]
    else:
        save_wav = waveform  # Assume already [1, samples]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torchaudio.save(str(out_path), save_wav, model_sr, encoding="PCM_S")

    # Verify save (log actual frames/dur; fix "len=1" bug)
    if not out_path.exists() or out_path.stat().st_size == 0:
        logger.warning(f"Save failed for {out_path} (empty) – fallback to original: {adjusted_path}")
        valid, dur = validate_voice_path(original_path, stem=stem, force_refresh=True)
        return original_path, valid, dur

    # Log actual info (sf.info ~0ms)
    try:
        info = sf.info(str(out_path))
        actual_frames = info.frames
        actual_dur = actual_frames / model_sr if model_sr > 0 else 0.0
        logger.info(f"Adjusted saved: {out_path} (final len={actual_frames}, dur={actual_dur:.2f}s)")
        logger.debug(f"Save verified: Expected len ~{len(waveform)}, actual {actual_frames} (SR={model_sr}Hz)")
    except Exception as info_e:
        logger.warning(f"Could not get info for {out_path}: {info_e} – assuming valid (dur unavailable)")
        actual_frames = len(waveform)
        actual_dur = actual_frames / model_sr if model_sr > 0 else 0.0

    # Final validate (force_refresh=True to check post-save SR=24000/ch=1)
    valid, dur = validate_voice_path(str(out_path), stem=stem, force_refresh=True)
    if not valid:
        logger.error(
            f"Post-save validate failed for {out_path} ({dur or 'unknown'}) – fallback to original: {original_path}")
        # Clean up failed save
        if out_path.exists():
            out_path.unlink(missing_ok=True)
        valid, dur = validate_voice_path(original_path, stem=stem, force_refresh=True)
        return original_path, valid, dur
    else:
        logger.info(f"Post-save valid: {out_path} (dur={dur:.2f}s; matches info {actual_dur:.2f}s)")

    # Update adjustment flag if enabled (for logging)
    if enable_pre_adjustment:
        logger.debug(f"Saved with pre-adjustment: {out_path} (enable_pre=True)")

    return str(out_path), valid, actual_dur  # Return str path, valid bool, actual dur from sf.info


# Refactored Main Method (now ~60 lines; pipeline calls)
def check_and_update_ref(audio_path: str, exaggeration: float = 0.5, model_sr: int = MODEL_SR,
                         out_path: Optional[Path] = None, enable_pre_adjustment: bool = None,
                         stem: str = None, force_reprocess: bool = False) -> str:
    """
    UPDATED: Full process only if needed; update cache with fixed_path. No redundant validation.
    - NEW: Early check vs local memory cache; if API hash matches, return existing processed (skip all).
    - Initial quick validate on original (SR check only – hash already checked in caller).
    - If pre-existing processed (resampled/padded) in temp dir is valid (SR=24000), use it without reprocess.
    - Scans local voices if enabled and matches hash (SR-agnostic via raw samples).
    - Always process if force_reprocess or no valid candidate.
    - After save, update _local_voice_cache for future dedup.
    """
    if enable_pre_adjustment is None:
        enable_pre_adjustment = CONFIG.get_value('enable_pre_adjustment', default=True)  # Default True as per logs

    stem = _normalize_stem(audio_path, stem)

    # NEW: Quick match vs pre-scanned local voices (SR-agnostic content hash)
    if stem in _local_voice_cache:
        upload_hash = _content_hash(audio_path)  # Content hash of API temp (ignores SR header)
        cached_hash = _local_voice_cache[stem]['content_hash']
        if upload_hash == cached_hash:
            processed_path = _local_voice_cache[stem].get('processed_path')
            if processed_path and Path(processed_path).exists():
                # Validate processed SR (quick)
                info = _get_audio_info_robust(processed_path)
                if info and info[0] == model_sr and info[1] == 1:
                    logger.info(f"Local hit for {stem}: Identical content (hash match) – use cached {processed_path}")
                    return processed_path  # Hit! Skip all processing
                else:
                    logger.warning(f"Cached processed invalid for {stem} (SR/ch mismatch) – fallback process")
            else:
                logger.debug(f"Local hash match but no processed path for {stem} – fallback")

    # Quick validate original (SR check only – hash already checked in caller)
    valid, msg = validate_voice_path(audio_path, stem=stem, force_refresh=False)

    logger.debug(f"Processing {stem}: Load/mono → Resample? → Pad? → Save")

    # FIXED: Check for existing resampled or padded in temp dir (reuse if valid 24000Hz)
    dir_path = Path(audio_path).parent
    resampled_path = dir_path / f"{Path(audio_path).stem}_resampled.wav"
    padded_path = dir_path / f"{Path(audio_path).stem}_padded.wav"

    # Prefer padded (final), then resampled
    candidate_path = None
    if padded_path.exists():
        candidate_path = padded_path
        logger.debug(f"Prioritize existing padded for {stem}: {candidate_path}")
    elif resampled_path.exists():
        candidate_path = resampled_path
        logger.debug(f"Prioritize existing resampled for {stem}: {candidate_path}")

    # Validate candidate (processed = 24000Hz)
    if candidate_path and candidate_path.exists():
        valid_candidate, _ = validate_voice_path(str(candidate_path), stem=stem)
        if valid_candidate:
            logger.info(f"Temp processed valid for {stem}: reusing {candidate_path} – skip resample/pad")
            return str(candidate_path)

    # No valid temp candidate – load original and process
    load_result = _load_and_mono(audio_path)
    if not load_result:
        logger.error(f"Load failed for {audio_path} – fallback original")
        return audio_path
    waveform, load_sr = load_result

    # Resample if needed (SR mismatch)
    waveform, adjusted_path = _resample_if_needed(waveform, load_sr, model_sr, audio_path)

    # Pad if enabled (gated for quality)
    if enable_pre_adjustment:
        from torchaudio.transforms import MelSpectrogram
        mel_transform = MelSpectrogram(sample_rate=model_sr, n_fft=CONFIG.get_value('n_fft', default=2048),
                                       hop_length=CONFIG.get_value('hop_length', default=256), n_mels=80)
        mel = mel_transform(waveform.unsqueeze(0))
        waveform = adjust_audio_length_torch(waveform, model_sr, mel.shape, CONFIG.get_value('hop_length', default=256),
                                             enable_pre_adjustment, stem)
        logger.info(f"Pad applied for {stem}: {len(waveform)} samples (enable_pre={enable_pre_adjustment})")
    else:
        logger.debug(f"Pad skipped for {stem} (enable_pre={enable_pre_adjustment})")

    # Save adjusted (use out_path if provided, else temp padded)
    out_path = out_path or dir_path / f"{Path(audio_path).stem}_padded.wav"

    # Optional: Prefer persistent if stem known (e.g., voices_dir)
    if stem and voices_dir.exists():
        persistent_out = voices_dir / f"{stem}_padded.wav"
        if persistent_out.parent == voices_dir:  # Valid stem
            out_path = persistent_out
            logger.debug(f"Using persistent out_path for {stem}: {out_path}")

    final_path, final_valid, _ = _save_adjusted_and_validate(
        waveform, adjusted_path, audio_path, model_sr, stem, out_path, enable_pre_adjustment, exaggeration
    )

    # NEW: After successful save, update _local_voice_cache for future API dedup
    if final_valid:
        new_hash = _content_hash(final_path)
        if stem not in _local_voice_cache:
            _local_voice_cache[stem] = {
                'content_hash': new_hash,
                'sr': _probe_upload_sr(audio_path, quiet=True) or model_sr,
                'processed_path': final_path
            }
            logger.debug(f"Added new voice to local cache: {stem} (path={final_path})")
        else:
            _local_voice_cache[stem]['content_hash'] = new_hash
            _local_voice_cache[stem]['processed_path'] = final_path
            logger.debug(f"Updated local cache for {stem}: {final_path} (new hash {new_hash[:8]})")

    if final_valid:
        logger.debug(f"Processed {stem}: {final_path} (SR={model_sr}Hz)")
        return final_path
    else:
        logger.warning(f"Process invalid for {stem} – fallback {audio_path}")
        return audio_path



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


# New Helper: Shared stem normalization (extracted from multiple places; testable: input path → expected stem)
def _normalize_stem(audio_path: str, provided_stem: str | None = None, min_len: int = 3) -> str:
    """Derive/clean stem from path or provided; handles temps/UUIDs. Returns str >= min_len or fallback."""
    if provided_stem is not None and isinstance(provided_stem, (int, float)):
        provided_stem = str(provided_stem)  # Handle old int calls

    if provided_stem and len(str(provided_stem)) >= min_len:
        # Clean if provided (remove suffixes)
        stem = str(provided_stem).replace('_fixed', '').replace('_padded', '').replace('_resampled', '').replace(
            '_ui_resampled', '')
        if len(stem) >= min_len:
            return stem
        logger.debug(f"Provided stem '{provided_stem}' too short/invalid → derive from path")

    if not audio_path:
        raise ValueError("No audio_path for stem derivation")

    basename = Path(audio_path).stem
    # Regex extract voice before UUID/temp (e.g., 'vp_11_lilia_123hex' → 'vp_11_lilia')
    match = re.match(r'([a-zA-Z0-9_]+[voice]?)(_?[0-9a-f]{15,})?$', basename)
    stem = match.group(1) if match else basename.replace('_fixed', '').replace('_padded', '').replace('_resampled',
                                                                                                      '').replace(
        '_ui_resampled', '').replace('_temp', '')
    if len(stem) < min_len:
        stem = basename  # Fallback to full
    logger.trace(f"Normalized stem for '{basename}': '{stem}'")
    return stem


# New Helper: Quick cache path validation (extracted; testable: stem/path → quick_reuse bool + info)
# Updated _quick_cache_validate (use fixed_path if available, hash check if provided)
def _quick_cache_validate(stem: str, cached_path: str = None, original_path: str = None, quiet: bool = False) -> tuple[
    bool, Optional[tuple]]:
    """FIXED: Validate cached fixed_path (processed SR=24000); add hash check vs original for dedup."""
    # Prefer cached fixed_path (processed)
    if cached_path and Path(cached_path).exists():
        info = _get_audio_info_robust(cached_path)
        if info and info[0] == MODEL_SR and info[1] == 1:
            if not quiet:
                logger.debug(f"Cached path valid for {stem} (SR={MODEL_SR}Hz)—immediate reuse")
            return True, info
        else:
            logger.warning(f"Cached path invalid for {stem}, SR={info[0] if info else 'unknown'}Hz – sync fix")

    # If no fixed or invalid, but original provided, fallback to original SR check (rare)
    if original_path and Path(original_path).exists():
        info = _get_audio_info_robust(original_path)
        if info and info[0] != MODEL_SR:  # Original mismatch only if no fixed
            if not quiet:
                logger.debug(f"Fallback to original for {stem}: SR={info[0]}Hz (no fixed cached)")
            return (info[0] == MODEL_SR and info[1] == 1), info

    return False, None


# New Helper: Light SR probe for upload (extracted/shared; testable: path → SR or None)
def _probe_upload_sr(audio_path: str, quiet: bool = False) -> Optional[int]:
    """Quick SR probe (header/info; ~1ms). Logs if mismatch."""
    try:
        info = _get_audio_info_robust(audio_path)
        sr = info[0] if info else None
        if sr == MODEL_SR:
            if not quiet:
                logger.trace(f"Upload SR match for {Path(audio_path).stem} ({MODEL_SR}Hz)")
        else:
            logger.debug(f"Upload SR {sr}Hz vs expected {MODEL_SR}Hz")
        return sr
    except Exception as e:
        logger.trace(f"SR probe failed for {audio_path}: {e}")
        return None


# New Helper: Orig metadata compute (for dedup same server files; testable: path → (hash, sr))
def _compute_orig_metadata(audio_path: str) -> tuple[str, Optional[int]]:
    """Full MD5 hash + SR of raw upload (ignores local mods like resample). ~0.01s."""
    if not Path(audio_path).exists():
        return "", None
    orig_hash = _compute_file_hash(audio_path, method='full')  # Full MD5
    orig_sr = _probe_upload_sr(audio_path, quiet=True)
    return orig_hash, orig_sr


# New Helper: Orig match check (for async/same source skip; testable: upload + cached → bool + msg)
def _is_same_server_file(upload_path: str, cached: dict) -> tuple[bool, str]:
    """Check if upload is same source (raw hash/SR match cached orig). Return (match, msg)."""
    cached_orig_hash = cached.get('orig_hash', '')
    cached_orig_sr = cached.get('orig_sr')
    upload_hash, upload_sr = _compute_orig_metadata(upload_path)

    if upload_hash == cached_orig_hash:
        if upload_sr == cached_orig_sr:
            return True, "Identical server file (hash/SR match) – skip"
        else:
            return True, f"SR var for same server file (hash match, SR {upload_sr} vs {cached_orig_sr}) – keep cached 24kHz"
    return False, f"Different file (hash {upload_hash[:8]} vs {cached_orig_hash[:8]}) – update"


# New Helper: Sync fix handler (extracted; testable: path/stem → updated path + cache update)
def _handle_sync_fix(audio_path: str, stem: str, exaggeration: float, model, device, dtype, cached: dict,
                     cached_conds_key: str, quiet: bool = False) -> str:
    """Perform sync refix + cache update (first-time/invalid). Returns fixed path."""
    fallback_path = check_and_update_ref(audio_path, exaggeration, stem=stem)
    if not fallback_path or not Path(fallback_path).exists():
        logger.error(f"Sync fix failed for {stem}—fallback upload (risky)")
        fallback_path = audio_path
    # Update cache (main thread)
    with _voice_cache_lock:
        _voice_cache[stem] = {**cached, 'fixed_path': fallback_path, 'last_bg_time': time.time()}
        # Add orig metadata for future dedup
        orig_hash, orig_sr = _compute_orig_metadata(audio_path)
        _voice_cache[stem].update({'orig_hash': orig_hash, 'orig_sr': orig_sr})
    # Save only if persistent
    if ROOT_DIR in Path(fallback_path).parents or voices_dir in Path(fallback_path).parents:
        _save_voice_cache()
        logger.debug(f"Updated voice cache post-sync: {stem}")
    else:
        logger.debug(f"Skipped voice cache save for temp: {fallback_path} (stem: {stem})")
    logger.info(f"Sync fixed/updated for {stem}: {fallback_path}")
    # Load conds if available
    if cached_conds_key:
        _cache_manager.load(cached_conds_key, model, device, dtype, quiet=quiet)
    # Update info cache post-fix (reduces future invalidations)
    with _voice_info_lock:
        new_info = _get_audio_info_robust(fallback_path)
        if new_info:
            _voice_info_cache[stem] = new_info
    return fallback_path


# New Helper: Reuse + conds load (extracted; testable: cached_key/path → conds loaded)
def _handle_reuse_and_load_conds(cached_conds_key: str, cached_path: str, model, device, dtype, stem: str,
                                 quiet: bool = False) -> None:
    """Load conds from key; log reuse."""
    if cached_conds_key:
        conds = _cache_manager.load(cached_conds_key, model, device, dtype, quiet=quiet)
        if conds and not quiet:
            logger.info(f"Reused cached conds for {stem}")
    if not quiet:
        logger.info(f"Cached reuse for {stem}: {cached_path}")


# Extracted Async Function (now outer; testable via mock Thread)
def _async_verify_update(audio_path: str, stem: str, cached: dict, cached_path: str, exaggeration: float, model, device,
                         dtype, quick_reuse: bool) -> None:
    from src.generate_audio import GEN_ACTIVE_LOCK
    """Async verify/update logic (extracted for testing; logs time)."""
    start_time = time.time()
    if not Path(audio_path).exists() or not quick_reuse:
        return
    needs_update = False
    try:
        # Orig check first (new dedup)
        is_same, msg = _is_same_server_file(audio_path, cached)
        if is_same:
            logger.debug(f"Async skip for {stem}: {msg}")
            return  # Skip all processing

        # Fallback: Current quick compares
        if Path(audio_path).stat().st_size != Path(cached_path).stat().st_size:
            logger.debug(
                f"Size differ for {stem} (upload={Path(audio_path).stat().st_size} vs cached={Path(cached_path).stat().st_size}) – async fix")
            needs_update = True
        else:
            sr_u = _probe_upload_sr(audio_path, quiet=True)
            sr_c = _get_or_cache_audio_info(stem=stem, audio_path=cached_path)[0]
            if sr_u != sr_c:
                logger.debug(f"SR differ for {stem} ({sr_u or 'unknown'}Hz vs {sr_c or 'unknown'}Hz) – async fix")
                needs_update = True
            else:
                # Partial hash (1MB)
                h_u = hashlib.md5(open(audio_path, 'rb').read(1024 * 1024) or b'')
                h_c = hashlib.md5(open(cached_path, 'rb').read(1024 * 1024) or b'')
                needs_update = (h_u.hexdigest() != h_c.hexdigest())
                if needs_update:
                    logger.debug(f"Partial hash differ for {stem} – async fix")
                else:
                    logger.trace(f"Quick verify complete for {stem}: Identical upload – no update")
                    return

        if needs_update:
            new_fixed_path = voices_dir / f"{stem}_fixed_new.wav"
            fixed_path = check_and_update_ref(audio_path, exaggeration, out_path=new_fixed_path, stem=stem)
            if fixed_path and Path(fixed_path).exists():
                valid_new, _ = validate_voice_path(fixed_path, stem=stem, force_refresh=True)
                if valid_new:
                    # Atomic swap
                    with _voice_cache_lock:
                        new_hash = _compute_file_hash(fixed_path, method='hybrid', stem=stem)
                        _voice_cache[stem] = {**cached, 'fixed_path': fixed_path, 'file_hash': new_hash,
                                              'last_bg_time': time.time()}
                        # Update orig (from current upload)
                        orig_hash, orig_sr = _compute_orig_metadata(audio_path)
                        _voice_cache[stem].update({'orig_hash': orig_hash, 'orig_sr': orig_sr})
                        # Save if persistent
                        if voices_dir in Path(fixed_path).parents:
                            _save_voice_cache()
                    if fixed_path != cached_path and Path(cached_path).exists():
                        Path(cached_path).unlink(missing_ok=True)
                        logger.info(f"Async updated {stem}: {fixed_path} (old: {cached_path})")
                    # Conds update (defer if graphs)
                    # Async conds update if possible (defer if graphs or gen active)
                    if hasattr(model, 't3') and len(model.t3._bucket_graphs) > 0:
                        logger.debug(f"Async defer conds for {stem}: Graphs active")
                    elif GEN_ACTIVE_LOCK.acquire(blocking=False):  # NEW: Try-acquire (non-block); defer if gen ongoing
                        logger.debug(f"Async defer conds for {stem}: Gen active")
                        GEN_ACTIVE_LOCK.release()  # Release immediately
                    else:
                        # Safe to prepare (no gen/graphs)
                        orig_backend = getattr(model.t3, 'generate_token_backend', None) if hasattr(model,
                                                                                                    't3') else None
                        if hasattr(model.t3, 'generate_token_backend'):
                            model.t3.generate_token_backend = 'eager'  # NEW: Force eager in async (no graph capture)
                        try:
                            with MODEL_LOCK:
                                model.prepare_conditionals(fixed_path, exaggeration=exaggeration)
                                new_conds_key = get_cache_key(fixed_path, uuid=stem, exaggeration=exaggeration)
                                _cache_manager.save(new_conds_key, model.conds, model=model, device=device, dtype=dtype)
                                with _voice_cache_lock:
                                    _voice_cache[stem]['conds_key'] = new_conds_key
                                logger.info(f"Async conds updated for {stem}: {new_conds_key[:8]}")
                        except Exception as conds_e:
                            logger.warning(f"Async conds failed for {stem}: {conds_e} – path updated only")
                        finally:
                            if orig_backend:
                                model.t3.generate_token_backend = orig_backend
                            torch.cuda.empty_cache()
                        if hasattr(model, 'set_conditionals'):
                            model.set_conditionals(None)  # Clear after
                    # Update info cache
                    with _voice_info_lock:
                        _voice_info_cache[stem] = _get_audio_info_robust(fixed_path)
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
    async_time = time.time() - start_time
    logger.debug(f"Async verify for {stem} took {async_time:.3f}s")



def get_or_queue_voice_process(audio_path: str, model, device, dtype, stem: str = None,
                               exaggeration: float = 0.5, quiet: bool = False) -> str:
    """FIXED: Hash check original vs cached orig; if match, use cached fixed_path (no process). Update cache after sync fix. Background if diff."""
    if not audio_path:
        return audio_path  # No-op

    stem = _normalize_stem(audio_path, stem)

    with _voice_cache_lock:
        cached = _voice_cache.get(stem, {})
        cached_fixed_path = cached.get('fixed_path', '')
        cached_orig_hash = cached.get('orig_hash', '')

    # FIXED: Hash check for dedup (fast skip if same server file)
    current_orig_hash = _compute_orig_metadata(audio_path)[0]  # Fast hash of current original
    if cached_orig_hash and cached_orig_hash == current_orig_hash:
        # Same file – use cached fixed if valid
        if Path(cached_fixed_path).exists():
            quick_reuse, info = _quick_cache_validate(stem, cached_fixed_path, quiet=quiet)
            if quick_reuse:
                _handle_reuse_and_load_conds(cached.get('conds_key', ''), cached_fixed_path, model, device, dtype, stem,
                                             quiet)
                if not quiet:
                    logger.info(
                        f"Ref hit for {stem}: Identical content (hash match) – reuse cached {cached_fixed_path}")
                return cached_fixed_path  # Skip process
            else:
                logger.warning(f"Ref cached invalid (SR/ch mismatch) for {stem} – sync fix after hash match")
        else:
            logger.warning(f"Ref cached path missing for {stem} (hash match) – sync fix")

    # No match or invalid – sync process
    fallback_path = check_and_update_ref(audio_path, exaggeration, stem=stem)

    # FIXED: Update cache with new fixed_path and current orig_hash
    with _voice_cache_lock:
        _voice_cache[stem] = {
            'fixed_path': fallback_path,
            'orig_hash': current_orig_hash,  # Cache current hash for next dedup
            'last_bg_time': time.time(),
            # Preserve other fields
            **{k: v for k, v in cached.items() if k not in ['fixed_path', 'orig_hash', 'last_bg_time']}
        }
        # Save persistent if fixed_path in voices_dir
        if voices_dir in Path(fallback_path).parents:
            _save_voice_cache()
        if not quiet:
            logger.info(f"Cache updated for {stem}: New fixed {fallback_path} (orig hash {current_orig_hash[:8]})")

    # If different content, queue background (async update for future, e.g., if exaggeration changed)
    if cached_orig_hash and cached_orig_hash != current_orig_hash:
        from threading import Thread
        Thread(target=_async_verify_update,
               args=(audio_path, stem, _voice_cache[stem], fallback_path, exaggeration, model, device, dtype, False),
               daemon=True).start()
        logger.debug(f"Queued BG update for {stem}: Content changed (hash diff)")

    if not quiet:
        logger.info(f"Sync processed/updated for {stem}: {fallback_path}")
    return fallback_path




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
    from src.fuzzy_cache import FUZZY_AUDIO_DICT
    cond_stats = _cache_manager.get_cache_stats()
    audio_stats = _audio_manager.stats()
    fuzzy_total = sum(len(entries) for entries in FUZZY_AUDIO_DICT.values()) if FUZZY_AUDIO_DICT else 0  # NEW: Total entries (not stems)
    return {**cond_stats, **audio_stats, 'fuzzy_size': fuzzy_total, 'fuzzy_stems': len(FUZZY_AUDIO_DICT)}  # Accurate


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
    from src.fuzzy_cache import FUZZY_AUDIO_DICT, FUZZY_LOCK
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
    with FUZZY_LOCK:
        FUZZY_AUDIO_DICT.clear()
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
    """Save _voice_cache to JSON; skip temp UUIDs quietly."""
    with _voice_cache_lock:
        temp_cache = {}
        for stem, info in _voice_cache.items():
            if 'fixed_path' in info:
                abs_path = Path(info['fixed_path'])
                # NEW: Skip temp UUIDs (long numeric; non-persistent)
                if len(str(stem)) > 15 and str(stem).isdigit():
                    logger.trace(f"Save skip temp UUID stem: {stem} (non-persistent)")
                    continue
                if abs_path.exists():
                    try:
                        rel_path = abs_path.relative_to(ROOT_DIR)
                        temp_cache[stem] = info.copy()
                        temp_cache[stem]['fixed_path'] = str(rel_path)  # Absolute to relative
                    except ValueError:
                        logger.debug(f"Save: Temp path non-relative for {stem}—skipping")
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
def init_conditional_memory_cache(model=None, device=None, dtype=None, quiet: bool = False,
                                  pre_extract: bool = False, pre_validate_voices: bool = False) -> Tuple[bool, bool]:
    _load_voice_cache()  # Existing: Load persistent JSON
    scan_local_voices(rebuild=False, use_content_hash=True)  # NEW: Scan locals to memory (SR-agnostic)

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
            logger.warning(
                f"Pre-extract: Missing voices in {voices_dir}: {missing_voices}. Add WAV files for faster hits.")
        if not quiet and available_voices:
            logger.info(f"Found {len(available_voices)} voices in {voices_dir}: {available_voices[:5]}...")

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
    # if pre_extract and model:
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
        logger.info(
            f"Cache init: Memory={ENABLE_MEMORY_CACHE}, Disk={ENABLE_DISK_CACHE}, Loaded {loaded} from {total_pt} (memory: {stats['memory_cache_size']}, pre-valid: {pre_validate_count})")
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
        # Cache conds (locked) – removed busy check; always try in init
        cache_key = get_cache_key(fixed_path, uuid=stem, exaggeration=0.5)
        if _cache_manager.is_cache_key_loaded(cache_key):
            logger.debug(f"Pre-validate hit: {stem}")
            return 1
        try:
            # No defer check in init (attempt all conds; lock safe)
            with MODEL_LOCK:
                model.prepare_conditionals(fixed_path, exaggeration=0.5)
                if model.conds:
                    _cache_manager.save(cache_key, model.conds, model, device, dtype)
                    model.set_conditionals(None)  # Clear
                    logger.info(f"Pre-validated conds: {stem} -> {cache_key[:8]}")
                    return 2  # File + conds
                else:
                    logger.warning(f"Pre-validate conds empty for {stem}")
                    return 1  # File only
        except AttributeError as ae:
            logger.debug(f"Pre-validate defer {stem}: Model not ready ({ae}) – file fixed")
            return 1
        except Exception as e:
            logger.warning(f"Pre-validate conds failed {stem}: {e}")
            if hasattr(model, 'set_conditionals') and model.conds:
                model.set_conditionals(None)
            return 1  # File success

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


# In init_conditional_memory_cache (add after _load_voice_cache())
scan_local_voices()  # Call on startup







