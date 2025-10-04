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
import torch
import torchaudio
from torch.serialization import safe_globals  # For whitelisting in load
import numpy as np
from pathlib import Path
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple, Union, List
from loguru import logger  # Assume available; fallback to print if not

# Suppress torchaudio deprecation warnings (clean logs)
warnings.filterwarnings('ignore', category=UserWarning, module='torchaudio')

# Anchor paths to project root (skyrimnet_chatterbox/) for relocatable code
ROOT_DIR = Path(__file__).parent.parent  # From src/ -> skyrimnet_chatterbox/

# Original globals (now rooted)
WAV_OUTPUT_DIR = ROOT_DIR / "output_temp"
CACHE_BASE = ROOT_DIR / "cache"
CACHE_DIR = CACHE_BASE / "conditionals"
CACHE_AUDIO_DIR = CACHE_BASE / "audio"
voices_dir = CACHE_AUDIO_DIR / "voices"  # For check_and_update_ref

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

            # Set to model if available
            if model and hasattr(model, 'set_conditionals'):
                model.set_conditionals(conds)
                logger.debug("Set conds via model.set_conditionals")

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


def validate_voice_path(audio_path: str) -> Tuple[Optional[str], Optional[Path]]:
    """Validate audio path (exists, dur>0, SR match)."""
    if not audio_path or not Path(audio_path).exists():
        return None, None
    try:
        info = torchaudio.info(audio_path)
        if info.num_frames == 0 or info.sample_rate != MODEL_SR:
            logger.warning(f"Invalid path: {audio_path} (dur=0 or SR={info.sample_rate} != {MODEL_SR})")
            return None, None
        logger.debug(f"Valid path: {audio_path} (dur={info.num_frames/info.sample_rate:.2f}s)")
        return audio_path, Path(audio_path)
    except Exception as e:
        logger.warning(f"Validate failed {audio_path}: {e}")
        return None, None


def check_and_update_ref(audio_path: str, exaggeration: float = 0.5, model_sr: int = MODEL_SR) -> str:
    """Validate/resample audio to model SR/mono if needed; return validated path (rooted)."""
    validated_path, p = validate_voice_path(audio_path)
    if validated_path:
        return validated_path

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

        # Save fixed to voices_dir
        voices_dir.mkdir(parents=True, exist_ok=True)
        fixed_path = voices_dir / f"{Path(audio_path).stem}_fixed.wav"
        torchaudio.save(fixed_path, waveform, model_sr)
        logger.info(f"Fixed audio ref: {fixed_path}")
        return str(fixed_path)
    except ImportError:
        logger.error("torchaudio unavailable for fix – return original (may fail)")
        return audio_path
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
    device = device or DEFAULT_DEVICE
    dtype = dtype or DEFAULT_DTYPE

    # NEW: Verify voices dir (log available for debugging)
    available_voices = [f.stem for f in voices_dir.glob("*.wav") if not f.stem.endswith('_fixed')]
    missing_voices = []
    if pre_extract:
        top_voices = ['nwskatyavoice', 'vp_11_lilia', 'nwsjennavoice', 'ba_ahnivoice']
        for v in top_voices:
            if v not in available_voices:
                missing_voices.append(v)
        if missing_voices:
            logger.warning(f"Pre-extract: Missing voices in {voices_dir}: {missing_voices}. Add WAV files for faster hits.")
        if available_voices:
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
    return {**cond_stats, **audio_stats}


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