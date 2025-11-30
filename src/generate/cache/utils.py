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
import tempfile
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


from src.config import get_config, get_config_value, find_project_root
from src.audio import is_artifact_laden  # Import for artifact check
import hashlib
from threading import Thread
from pathlib import Path
from loguru import logger

from src.normalize_stem import normalize_stem

# Suppress torchaudio deprecations precisely (exact message/module for backend utils)
warnings.filterwarnings('ignore', message=r'.*torchaudio._backend.utils.info.*', category=UserWarning)
warnings.filterwarnings('ignore', message=r'.*deprecated.*torchaudio.*', category=UserWarning, module='torchaudio')
warnings.filterwarnings('ignore', category=UserWarning, module='torchaudio')  # Broad fallback
warnings.filterwarnings('ignore', category=DeprecationWarning, module='torchaudio')

CONFIG = get_config()

def get_project_root():
    """Always return the properly anchored project root"""
    return CACHE_ROOT.parent

def get_cache_root():
    """Return the verified cache directory within project root"""
    return CACHE_ROOT



def get_valid_cache_root():
    """Ensure cache is stored in project root, never in src/"""
    project_root = find_project_root()
    if not project_root:
        raise RuntimeError("Cannot determine project root - required for cache management")

    cache_dir = project_root / "cache"
    # Verify cache is not in src/ directory
    if "src" in str(cache_dir).lower():
        bad_path = str(cache_dir)
        cache_dir = project_root / "cache"
        logger.critical(f"SECURITY: Redirected cache from src/ ({bad_path}) to {cache_dir}")

    # Create proper subdirectories
    for d in ["conditionals", "audio/output", "audio/voices", "fallbacks"]:
        (cache_dir / d).mkdir(parents=True, exist_ok=True)

    return cache_dir

# Use throughout code instead of direct path references
CACHE_ROOT = get_valid_cache_root()
WAV_OUTPUT_DIR = CACHE_ROOT / "audio" / "output"
voices_dir = CACHE_ROOT / "audio" / "voices"


# Global model lock (new: serialize access to prevent graph races)
MODEL_LOCK = threading.RLock()
# Global gen lock (serialize vs async for CUDA graph safety)
GEN_ACTIVE_LOCK = threading.RLock() #TODO in generate?

# TODO cleanup these constants and use config
# Tunable constants (hardcoded; override via env vars if needed)
KEY_LEN = 32    # Hash length
MAX_RECURSE = 3 # Reconstruction recursion limit
DEFAULT_DEVICE = "cuda"  # Fallback
DEFAULT_DTYPE = torch.bfloat16  # TODO was float32 test and review
MODEL_SR = 24000  # Assume standard for TTS
MAX_MEMORY_ENTRIES = get_config_value('max_memory_entries', default=100)
ENABLE_MEMORY_CACHE = get_config_value('enable_audio_cache', default=True)
ENABLE_DISK_CACHE = get_config_value('enable_fuzzy_cache', default=True)
ENABLE_THREADED_SAVES = True  # Hardcode or add to CONFIG if needed
MAX_QUEUE = get_config_value('save_queue_max', default=20)

COMPRESS_PT_SAVES = get_config_value('compress_pt_saves', default=True) #TODO
COMPRESS_LEVEL = get_config_value('compress_level', default=6) #TODO

# Global voice cache (stem → dict: fixed_path, file_hash, conds_key)
_voice_cache_lock = threading.RLock()
_voice_cache = OrderedDict()  # LRU, maxlen=200  # Add maxlen=200 (tune via CONFIG if needed)
_voice_info_cache = {}  # {stem: (sr, channels, duration_frames)} – simple, thread-safe with lock
_voice_info_lock = threading.RLock()  # Optional: For concurrent access (shared with _voice_cache_lock)

# Import fallbacks
T3_AVAILABLE = False
try:
    from src.chatterbox.models.t3.modules.cond_enc import T3Cond
    from src.chatterbox.tts import Conditionals
    T3_AVAILABLE = True
except ImportError:
    logger.warning("T3Cond/Conditionals not available – using raw fallback")


def _content_hash(wav_path: str) -> str:
    if wav_path is None or not os.path.exists(wav_path):
        logger.warning(f"Content hash: Invalid path {wav_path}")
        return ""  # Empty hash fallback
    try:
        waveform, _ = torchaudio.load(wav_path)
        if waveform.dim() > 1:
            waveform = waveform.mean(0)  # Mono
        return hashlib.md5(waveform.numpy().tobytes()).hexdigest()
    except Exception as e:
        logger.warning(f"Content hash failed for {wav_path}: {e} – fallback file hash")
        return _compute_file_hash(wav_path, method='hybrid') or ""


# Dummy conds fallback (simple; assumes model has set_conditionals or conds attr)
def create_dummy_conds(model, device, dtype, reason="fallback"):
    """Create dummy conditionals with dimension-validated speaker embedding."""
    if T3_AVAILABLE:
        try:
            # FIX: Validate dtype is actually a torch dtype
            if not isinstance(dtype, torch.dtype):
                logger.warning(f"Invalid dtype {dtype} - using torch.float32")
                dtype = torch.float32

            # FIX: Ensure proper speaker embedding dimensions (1, 1280 is common)
            dummy_speaker_emb = torch.zeros(1, 1280, device=device, dtype=dtype)

            # Create T3Cond with required speaker_emb parameter
            dummy_t3 = T3Cond(speaker_emb=dummy_speaker_emb)

            # Create conditionals
            dummy_conds = Conditionals(dummy_t3, None)

            # Proper transfer to target device
            if str(device) != "cpu":
                dummy_conds = dummy_conds.to(device=device)

            # Set to model
            if hasattr(model, 'set_conditionals'):
                model.set_conditionals(dummy_conds)
            elif hasattr(model, 'conds'):
                model.conds = dummy_conds

            logger.debug(f"Created valid dummy conds ({reason})")
            return dummy_conds
        except Exception as e:
            logger.error(f"Full dummy conds creation failed ({reason}): {str(e)}")

    logger.warning(f"Fallback to minimal dummy conds ({reason})")
    return None


@functools.lru_cache(maxsize=128)
def get_wavout_dir(cache: bool = True) -> Path:
    """Get output directory with proper audio directory structure"""
    if cache:
        # CRITICAL FIX: Ensure we're using the proper audio output subdirectory
        wavout_dir = CACHE_ROOT / "audio" / "output"
    else:
        formatted_start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        wavout_dir = CACHE_ROOT / "temp" / formatted_start_time

    # Ensure directory exists
    wavout_dir.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Using wavout dir: {wavout_dir}")
    return wavout_dir



def get_audio_cache_key(conditionals_key: str, text: str, exaggeration: float) -> str:
    """Generate safe cache key with conditionals validation"""
    if not conditionals_key or len(conditionals_key) < 5:
        raise ValueError(f"Invalid conditionals_key: {conditionals_key}")

    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:10]
    return f"{conditionals_key}_{text_hash}_{exaggeration:.2f}".replace("/", "_")



def save_torchaudio_wav(
        wav_tensor=None,
        sr: int = 24000,
        audio_path: str = None,
        uuid: Any = None,
        cache_key: str = None,
        text: str = None,
        cache: bool = True
):
    """Save WAV tensor to file with proper directory structure."""
    from cache_manager import CacheManager
    """
    Save WAV tensor to file and return path (fallback to temp if issues).
    FIXED: Proper directory structure with cache/audio/output
    :param wav_tensor: Audio tensor (or None → silence).
    :param sr: Sample rate.
    :param audio_path: For filename prefix (or None → "default_voice").
    :param uuid: For filename (or None → "default").
    :param cache_key: Optional cache key (cache if not None).
    :param text: Text for per-gen filename uniqueness (hash included).
    :param cache: Use cache dir (True) or temp (False).
    :return: str path (always; temp fallback on errors).
    """
    # Upfront guards (minimal fallbacks)
    if wav_tensor is None:
        wav_tensor = torch.zeros(1, sr * 2, dtype=torch.float32)  # 2s silence
    if sr <= 0:
        sr = 24000
    if audio_path is None:
        audio_path = "default_voice"
    if uuid is None:
        uuid = "default"
    if cache is None:
        cache = True

    try:
        # Compute path/filename (coerce audio_path)
        audio_path = str(audio_path)
        # Use normalize_stem for consistent unpadded prefix
        cache_prefix = normalize_stem(audio_path)
        uuid_hex = hex(uuid)[2:][:8] if isinstance(uuid, int) else str(uuid)[:8] or "default"

        # Compute text_hash if text provided (per-gen unique)
        text_hash = "default"  # Fallback
        if text and isinstance(text, str):
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
            logger.debug(f"Filename text_hash computed: {text_hash} for text='{text[:20]}...'")

        # Include text_hash in filename (prevents overwrite); uses normalized prefix
        filename = f"{cache_prefix}_{text_hash}_{uuid_hex}.wav"

        # CRITICAL FIX: Use the proper audio output directory
        out_dir = get_wavout_dir(cache)
        path = out_dir / filename

        # Verify directory exists
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Saving to directory: {out_dir}")

        # Save tensor (to CPU/FP32)
        safe_wav = wav_tensor.cpu().to(torch.float32) if hasattr(wav_tensor, 'cpu') else torch.zeros_like(wav_tensor)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torchaudio.save(str(path), safe_wav, sr, encoding="PCM_S")

        # Verify and cache if key provided
        if not path.exists() or path.stat().st_size == 0:
            raise OSError(f"Empty save for {path}")

        if cache_key:
            set_audio_cache(cache_key, str(path.resolve()))
            logger.debug(
                f"Saved & cached: {path.name} (key={cache_key[:8]}, text_hash={text_hash}, prefix={cache_prefix})")
        else:
            logger.debug(f"Saved: {path.name} (no cache key, text_hash={text_hash}, prefix={cache_prefix})")

        return str(path)

    except Exception as e:
        logger.error(
            f"save_torchaudio_wav failed (path={path or 'N/A'}, key={cache_key[:8] or 'N/A'}): {type(e).__name__}: {e} – temp fallback")
        try:
            temp_path = Path(tempfile.mktemp(suffix=".wav"))
            torch.save(torch.zeros(1, sr * 2, dtype=torch.float32), temp_path)  # Simple silence fallback
            if cache_key:
                set_audio_cache(cache_key, str(temp_path))  # Cache fallback if key
            logger.debug(f"Temp fallback: {temp_path}")
            return str(temp_path)
        except:
            return str(Path(tempfile.gettempdir()) / "error.wav")  # Last-resort empty str path


def get_content_hash(audio_path: str) -> str:
    from voice_reference import VoiceReferenceCache
    """Get content hash via cache system if available, otherwise calculate."""
    # Try to get from voice cache first if initialized
    if hasattr(VoiceReferenceCache, 'instance') and VoiceReferenceCache.instance:
        return VoiceReferenceCache.instance.calculate_content_hash(audio_path)

    # Fallback to direct calculation
    try:
        import torchaudio
        waveform, _ = torchaudio.load(audio_path)
        if waveform.dim() > 1:
            waveform = waveform.mean(0)
        return hashlib.md5(waveform.numpy().tobytes()).hexdigest()
    except Exception:
        return "fallback_hash"


def _get_or_cache_audio_info(stem: str = None, audio_path: str = None, force_refresh: bool = False) -> Optional[
    Tuple[int, int, int]]:
    """Cached wrapper: Get info for stem/path. IMPROVED: Disk JSON persistence (TTL 1hr); verifies vs fresh."""
    if not audio_path:
        return None

    # Extract stem if missing (for backward calls)
    if stem is None:
        full_stem = Path(audio_path).stem.replace('_fixed', '')  # e.g., 'vayne_csvp_voice' → 'vayne_csvp_voice'
        split_stem = full_stem.split('_')
        stem = split_stem[0] if len(split_stem) > 1 and len(
            split_stem[0]) >= 2 else full_stem  # Robust: min 2 chars, fallback full

    # IMPROVED: Disk JSON path (persistent meta)
    meta_path = Path(audio_path).with_suffix('.meta.json')
    fresh_needed = force_refresh or not meta_path.exists()

    cached_info = None
    if not fresh_needed:
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            if meta.get('stem') == stem and meta.get('valid_until', 0) > time.time():  # TTL 1hr
                cached_info = (meta['sr'], meta['channels'], meta['frames'])
                logger.trace(f"Meta loaded from JSON: {audio_path} (cached for {stem})")
        except (json.JSONDecodeError, KeyError, OSError) as e:
            fresh_needed = True
            logger.debug(f"Meta JSON invalid/missing for {audio_path} ({e}) – fresh probe")

    # If no valid cache, compute fresh always for verify, or if force/invalid
    if fresh_needed or cached_info is None:
        fresh_info = _get_audio_info_robust(audio_path)
        if not fresh_info:
            logger.warning(f"Fresh info failed for {audio_path} – cannot cache/verify")
            return None

        # Verify vs cached if available (invalidate on mismatch, esp SR/channels)
        if cached_info and cached_info[:2] != fresh_info[:2]:  # SR + channels mismatch → stale
            logger.warning(
                f"Cache invalid for {stem}: cached SR/ch={cached_info[:2]} vs fresh={fresh_info[:2]} – refresh")
            cached_info = None  # Force update

        # Update in-mem cache with fresh
        # Update in-mem cache with fresh
        with _voice_info_lock:
            _voice_info_cache[stem] = fresh_info

        # IMPROVED: Save to JSON (persistent lazy)
        meta = {
            'stem': stem,
            'sr': fresh_info[0],
            'channels': fresh_info[1],
            'frames': fresh_info[2],
            'valid_until': time.time() + 3600,  # 1hr TTL
            'probed_at': datetime.datetime.now().isoformat()
        }
        try:
            with open(meta_path, 'w') as f:
                json.dump(meta, f)
            logger.debug(f"Saved meta JSON: {audio_path} (info={fresh_info})")
        except OSError as e:
            logger.warning(f"Meta save failed for {audio_path}: {e}")

        return fresh_info

    # Use cached if valid
    logger.trace(f"Info: Reused verified cached for '{stem}' (SR={cached_info[0]})")
    return cached_info



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
        if get_config_value('fuzzy_artifact_purge_enable', default=True):
            if is_artifact_laden(audio_path):
                logger.warning(f"Artifact detected in {audio_path} (centroid high/ratio high) – invalid for use (purge if gen output)")
                return False, "artifacts_detected"
            logger.trace(f"Artifact check passed: {audio_path} (mean centroid clean)")
        else:
            logger.trace(f"Artifact purge disabled – check skipped for {audio_path}")
    else:
        logger.trace(f"Skipped artifact check for ref: {audio_path} (normal Skyrim voice; is_voice_ref={is_voice_ref})")

    if duration < get_config_value('min_voice_duration', default=0.5):  # Assume MIN_VOICE_DURATION=0.5s configurable
        logger.warning(f"Too short: {duration:.2f}s < {get_config_value('min_voice_duration', default=0.5)}s")
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
    hop_length = get_config_value('hop_length', default=256)
    n_fft = get_config_value('n_fft', default=2048)

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










