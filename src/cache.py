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
from difflib import SequenceMatcher
from threading import Lock, Thread
from queue import Queue
import torch
import torchaudio
from torch.serialization import safe_globals  # For whitelisting in load
import numpy as np
from pathlib import Path
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple, Union, List
from loguru import logger  # Assume available; fallback to print if not
import threading  # Ensure imported (likely already is)

# Suppress torchaudio deprecations precisely (exact message/module for backend utils)
warnings.filterwarnings('ignore', message=r'.*torchaudio._backend.utils.info.*', category=UserWarning)
warnings.filterwarnings('ignore', message=r'.*deprecated.*torchaudio.*', category=UserWarning, module='torchaudio')
warnings.filterwarnings('ignore', category=UserWarning, module='torchaudio')  # Broad fallback

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
MAX_MEMORY_ENTRIES = int(os.getenv('COND_CACHE_MAX_ENTRIES', 50))
ENABLE_MEMORY = bool(int(os.getenv('ENABLE_MEMORY_CACHE', 1)))
ENABLE_DISK = bool(int(os.getenv('ENABLE_DISK_CACHE', 1)))
ENABLE_THREADED_SAVES = bool(int(os.getenv('ENABLE_THREADED_SAVES', 1)))
MAX_QUEUE = 10  # Disk save queue limit
KEY_LEN = 32    # Hash length
MAX_RECURSE = 3 # Reconstruction recursion limit
DEFAULT_DEVICE = "cuda"  # Fallback
DEFAULT_DTYPE = torch.float32
MODEL_SR = 24000  # Assume standard for TTS

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
            shape_str = f"shape_{v.shape}_{v.dtype}" if hasattr(v, 'dtype') else f"shape_{v.shape}"
            param_dict[k] = f"hash_{hash(shape_str)}"
        elif hasattr(v, '__dict__') and hasattr(v, 't3'):  # Conditionals
            param_dict[k] = f"conds_id_{id(v)}_{getattr(v.t3, 'shape', 'unknown')}"
        else:
            param_dict[k] = str(v)

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
            logger.debug(f"Audio info: Fallback to torchaudio.info (suppressed deprecation: {ie})")
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
def validate_voice_path(audio_path: str) -> Tuple[bool, Optional[float]]:
    """Validate voice path with robust torchaudio handling; return (valid, duration)."""
    if not os.path.exists(audio_path):
        logger.warning(f"Validate: Path does not exist {audio_path}")
        return False, None

    info_tuple = _get_audio_info_robust(audio_path)
    if info_tuple is None:
        return False, None

    sample_rate, num_channels, duration_frames = info_tuple
    duration = duration_frames / sample_rate if duration_frames > 0 else 0

    if sample_rate != 24000:
        logger.warning(f"SR mismatch: {audio_path} ({sample_rate}Hz != 24000)")
        return False, None
    if num_channels != 1:
        logger.warning(f"Channels mismatch: {audio_path} ({num_channels} != 1)")
        return False, None

    logger.debug(f"Valid path: {audio_path} (dur={duration:.2f}s)")
    return True, duration


def check_and_update_ref(audio_path: str, exaggeration: float = 0.5, model_sr: int = MODEL_SR, out_path: Optional[Path] = None) -> str:
    """Validate/resample audio to model SR/mono if needed; return validated path (rooted). Dedup '_fixed' to prevent loops."""
    validated, _ = validate_voice_path(audio_path)
    if validated:
        stem = Path(audio_path).stem
        if stem.endswith('_fixed') or stem.endswith('_fixed_fixed'):  # Dedup loop
            logger.debug(f"Validate passed, but deduped path {audio_path}")
        return audio_path  # No fix needed

    # Resample/fix (torchaudio primary)
    try:
        waveform, orig_sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:  # Mono
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if orig_sr != model_sr:
            resampler = torchaudio.transforms.Resample(orig_sr, model_sr)
            waveform = resampler(waveform)
            logger.info(f"Resampled {audio_path} to {model_sr}Hz mono")
        else:
            logger.debug(f"No resample needed for {audio_path}")

        # Validate waveform (non-empty)
        if waveform.numel() == 0:
            logger.error(f"Empty waveform after load/resample for {audio_path}")
            return audio_path

        # Save fixed (direct to final for BG; use abs str path; dedup '_fixed')
        orig_stem = Path(audio_path).stem
        if orig_stem.endswith('_fixed_fixed'):
            final_stem = orig_stem[:-13] + '_fixed'  # Strip dupe suffix
        elif orig_stem.endswith('_fixed'):
            final_stem = orig_stem  # No extra
        else:
            final_stem = orig_stem + '_fixed'
        final_out = out_path or voices_dir / f"{final_stem}.wav"
        final_out.parent.mkdir(parents=True, exist_ok=True)
        try:
            torchaudio.save(
                str(final_out.absolute()),  # Abs path for Windows
                waveform,
                model_sr,
                format="wav",
                encoding="PCM_S",
                bits_per_sample=16
            )
        except Exception as save_e:
            logger.warning(f"Save failed: {save_e}—retrying raw")
            torchaudio.save(str(final_out.absolute()), waveform, model_sr)  # No extras
        if not final_out.exists() or final_out.stat().st_size == 0:
            logger.error(f"Save produced empty file: {final_out}")
            return audio_path
        logger.info(f"Fixed audio ref: {final_out}")
        return str(final_out)
    except Exception as e:
        logger.error(f"Audio ref fix failed: {e}")
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
def _compute_file_hash(file_path: str, method='hybrid') -> str:  # New 'hybrid'
    if not Path(file_path).exists():
        return ""
    try:
        if method == 'quick':
            info_tuple = _get_audio_info_robust(file_path)
            if info_tuple:
                sr, channels, frames = info_tuple
                size = Path(file_path).stat().st_size
                return f"{size}_{frames}_{sr}"
            return ""  # Fail → empty hash
        elif method == 'hybrid':  # Quick first; full if needed (for debug)
            quick_h = None
            info_tuple = _get_audio_info_robust(file_path)
            if info_tuple:
                sr, channels, frames = info_tuple
                size = Path(file_path).stat().st_size
                quick_h = f"{size}_{frames}_{sr}"
                logger.debug("Hash: Used quick metadata")
            if quick_h:
                return quick_h  # Fast; fallback full only on metadata fail
        # Full MD5 (default/always for 'full' or hybrid fallback)
        h = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        logger.debug("Hash: Used full MD5 fallback")
        return h.hexdigest()
    except Exception as e:
        logger.warning(f"Hash failed {file_path}: {e}")
        return ""


# Patched get_or_queue_voice_process (skip validate on quick match; robust info; dedup)
def get_or_queue_voice_process(audio_path: str, model, device, dtype, uuid, exaggeration=0.5, quiet=False) -> str:
    stem = Path(audio_path).stem
    with _voice_cache_lock:
        cached = _voice_cache.get(stem, {})
        cached_path = cached.get('fixed_path', '')
        cached_hash = cached.get('file_hash', '')
        cached_conds_key = cached.get('conds_key', '')
        last_bg = cached.get('last_bg_time', 0)

    if time.time() - last_bg < 30:
        if not quiet:
            logger.debug(f"Recent BG for {stem}—skipping queue")
        # Ensure fixed even on skip
        return cached_path if (cached_path and Path(cached_path).exists()) else check_and_update_ref(audio_path, exaggeration)

    # Quick pre-check: If cached_path exists, quick metadata match (size + SR/channels via robust info) → skip full hash/validate
    quick_match = False
    if cached_path and Path(audio_path).exists() and Path(cached_path).exists():
        if Path(audio_path).stat().st_size == Path(cached_path).stat().st_size:  # Size first
            upload_info = _get_audio_info_robust(audio_path)
            cached_info = _get_audio_info_robust(cached_path)
            if upload_info and cached_info and upload_info[:2] == cached_info[:2]:  # SR + channels match
                quick_match = True
                logger.debug(f"Quick metadata match for {stem}—reusing without full validate/hash")
                upload_hash = cached_hash  # Assume same
            else:
                logger.debug(f"Quick metadata mismatch for {stem}—full hash")

    upload_hash = _compute_file_hash(audio_path, method='hybrid' if not quick_match else 'quick')
    is_different = (not quick_match) and (upload_hash != cached_hash or not cached_path)

    if not is_different and Path(cached_path).exists():
        logger.info(f"Server WAV same as cached for {stem}—reusing {cached_path}")
        if cached_conds_key:
            conds = _cache_manager.load(cached_conds_key, model, device, dtype, quiet=quiet)
            if conds and not quiet:
                logger.info(f"Reused cached conds for {stem}")
        return cached_path  # Valid; skipped validate

    # Always fix/return valid for current (main thread; covers new/old invalid)
    # Skip full validate/check if quick_match (already done)
    candidate_path = cached_path if cached_path else audio_path
    if quick_match:
        fallback_path = candidate_path  # No need to refix
    else:
        fallback_path = check_and_update_ref(candidate_path, exaggeration)
    if not fallback_path or not Path(fallback_path).exists():
        logger.warning(f"Fix failed for {stem}—using original (may fail SR)")
        fallback_path = candidate_path
    logger.info(f"Fixed/used for {stem} in main: {fallback_path}")
    if cached_conds_key and not is_different:
        _cache_manager.load(cached_conds_key, model, device, dtype, quiet=quiet)

    def _bg_process_new():
        if not is_different or Path(fallback_path).exists():  # Skip if already fixed/same
            return
        try:
            final_fixed = voices_dir / f"{stem}_fixed.wav"
            fixed_path = check_and_update_ref(str(audio_path), exaggeration, out_path=final_fixed)  # Direct
            if not fixed_path or not Path(fixed_path).exists():
                if not quiet:
                    logger.warning(f"BG full fail for {stem}")
                return

            # Prep/conds (locked)
            conds_key = None
            with MODEL_LOCK:
                if (hasattr(model, 't3') and hasattr(model.t3, '_bucket_graphs') and len(model.t3._bucket_graphs) > 0):
                    if not quiet:
                        logger.debug(f"BG defer for {stem}: Graphs active")
                else:
                    model.prepare_conditionals(fixed_path, exaggeration=exaggeration)
                    temp_conds = model.conds
                    if temp_conds:
                        model.set_conditionals(None)
                        conds_key = get_cache_key(fixed_path, uuid, exaggeration)
                        _cache_manager.save(conds_key, temp_conds, model=None, device=device, dtype=dtype)

            with _voice_cache_lock:
                update_entry = {'fixed_path': fixed_path, 'file_hash': upload_hash, 'last_bg_time': time.time()}
                if conds_key:
                    update_entry['conds_key'] = conds_key
                _voice_cache[stem] = {**cached, **update_entry}
            _save_voice_cache()
            if not quiet:
                logger.info(f"BG processed {stem}: {fixed_path}" + (f", conds {conds_key[:8]}" if conds_key else ""))
        except Exception as e:
            if not quiet:
                logger.error(f"BG failed for {stem}: {e}")

    if ENABLE_THREADED_SAVES and (Path(fallback_path).stem.endswith('_fixed') or is_different):  # Queue only if needed
        threading.Thread(target=_bg_process_new, daemon=True, name=f"BG-Voice-{stem}").start()
        logger.debug(f"Queued bg for {stem}")
    return fallback_path  # Always fixed/valid


# Sync pre-extract (threaded for non-blocking; hardcoded top voices)
def pre_extract_fixed_voices(model, device, dtype, top_voices: List[str] = ['nwskatyavoice', 'vp_11_lilia', 'nwsjennavoice', 'ba_ahnivoice']):
    """Pre-extract conds for top voices (sync with threading; rooted paths)."""
    def _extract_worker(voice):
        if not T3_AVAILABLE:
            return
        ref_path = voices_dir / f"{voice}.wav"  # Rooted to cache/audio/voices/
        if not ref_path.exists():
            logger.warning(f"Skipping pre-extract {voice}: No {ref_path}")
            return
        cache_key = get_cache_key(str(ref_path), uuid=voice, exaggeration=0.5)
        if _cache_manager.load(cache_key, model, device, dtype, quiet=True):
            logger.info(f"Pre-extract HIT: {voice}")
            return
        try:
            model.prepare_conditionals(str(ref_path), exaggeration=0.5)
            _cache_manager.save(cache_key, model.conds, model, device, dtype)
            logger.info(f"Pre-extracted: {voice} (key={cache_key[:8]})")
        except Exception as e:
            logger.warning(f"Pre-extract failed {voice}: {e}")

    logger.info(f"Pre-extracting {len(top_voices)} voices")
    threads = []
    for voice in top_voices:
        t = threading.Thread(target=_extract_worker, args=(voice,))
        t.daemon = True
        t.start()
        threads.append(t)
    for t in threads:
        t.join()  # Wait for completion


# Init (merged: preload + optional pre-extract)
def init_conditional_memory_cache(model=None, device=None, dtype=None, quiet: bool = False,
                                  pre_extract: bool = True) -> Tuple[bool, bool]:
    _load_voice_cache()
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
            if ENABLE_MEMORY:
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

    # Pre-extract if enabled
    if pre_extract and model:
        pre_extract_fixed_voices(model, device, dtype)

    total_pt = len(list(CACHE_DIR.glob('*.pt')))
    stats = _cache_manager.get_cache_stats()
    if not quiet:
        logger.info(f"Cache init: Memory={ENABLE_MEMORY}, Disk={ENABLE_DISK}, Loaded {loaded} from {total_pt} (memory: {stats['memory_cache_size']})")
    return ENABLE_MEMORY, ENABLE_DISK


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
    return _cache_manager.get_current_cache_key()


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


# Global voice cache (stem → dict: fixed_path, file_hash, conds_key)
_voice_cache = {}  # In-memory; persist below
_voice_cache_lock = threading.RLock()

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


# Background indexing worker (unchanged; FIFO evict)
def _background_index_worker():
    while True:
        try:
            text, wav_path, voice_stem = _fuzzy_queue.get(timeout=1)  # Unpack as tuple
            orig_text = text  # For sim
            norm_key = normalize_text(text)
            with _fuzzy_lock:
                if norm_key in _fuzzy_audio_dict:  # Simple dedup (skip enqueue dups)
                    logger.debug(f"Skipped dup fuzzy index: {norm_key[:30]}")
                else:
                    if len(_fuzzy_audio_dict) >= MAX_INDEX_SIZE:
                        _fuzzy_audio_dict.pop(next(iter(_fuzzy_audio_dict)))  # Oldest key (FIFO)
                        logger.debug(f"Fuzzy cache full ({MAX_INDEX_SIZE}) – evicted oldest")
                    _fuzzy_audio_dict[norm_key] = {'wav_path': wav_path, 'orig_text': orig_text}
                    logger.debug(f"Indexed fuzzy audio: {norm_key} -> {wav_path} (stem: {voice_stem})")
        except:
            pass


# Patched try_fuzzy_audio_cache (detailed logs for testing; uses float from string_similarity)
def try_fuzzy_audio_cache(input_text: str, voice_stem: str, threshold: float = 0.75) -> Optional[str]:
    """
    Fuzzy matching for audio cache: Checks string similarity in _fuzzy_audio_dict for voice-specific entries.
    Logs attempts, matches/MISSES with ratios, and DB stats for debugging.
    Assumes _fuzzy_audio_dict is dynamic (filled via enqueue from generations; capped elsewhere).
    """
    if not input_text or not voice_stem:
        logger.debug("Fuzzy query skipped: Empty text or stem")
        return None

    if not _fuzzy_audio_dict:
        logger.debug(f"Fuzzy DB empty – no matches possible ({len(_fuzzy_audio_dict)} entries)")
        return None

    # Normalize for lookup/sim (clean query)
    norm_query_key = normalize_text(input_text)
    norm_query_text = input_text.lower().strip()  # Semi-clean for sim (preserve words)

    logger.debug(
        f"Trying fuzzy for stem '{voice_stem}', text: '{norm_query_text[:50]}...' "
        f"DB size: {len(_fuzzy_audio_dict)} (threshold: {threshold})"
    )

    candidate_count = 0
    max_sim = 0.0
    best_entry = None
    with _fuzzy_lock:
        for stored_key, data in _fuzzy_audio_dict.items():
            if voice_stem in stored_key:  # Voice filter (stem in key ensures match)
                candidate_count += 1
                sim_ratio = string_similarity(norm_query_text, data['orig_text'], threshold)  # Get float
                if sim_ratio > max_sim:
                    max_sim = sim_ratio
                    best_entry = data
                if sim_ratio >= threshold:
                    logger.info(
                        f"Fuzzy cache HIT: '{norm_query_text[:20]}...' ≈ '{data['orig_text'][:20]}...' "
                        f"(sim={sim_ratio:.3f} >= {threshold}) for stem '{voice_stem}' -> {data['wav_path']}"
                    )
                    return data['wav_path']

    # Log MISS with stats (best sim, candidates) for tuning
    logger.debug(
        f"Fuzzy MISS for '{norm_query_text[:20]}...' (stem '{voice_stem}'): "
        f"best sim={max_sim:.3f} < {threshold} ({candidate_count} candidates in DB)"
    )
    if best_entry:
        logger.debug(
            f"  Closest: '{best_entry['orig_text'][:20]}...' (sim={max_sim:.3f})"
        )

    return None

# Start daemon thread at init (fuzzy worker)
threading.Thread(target=_background_index_worker, daemon=True, name="FuzzyIndexer").start()