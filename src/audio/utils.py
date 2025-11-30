"""
Centralized audio utility functions.

This module contains general-purpose audio helpers
directly from `src.audio.utils`.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from loguru import logger
import numba
import numpy as np
import librosa
import torch
import torchaudio
import soundfile as sf
import tempfile
from scipy.signal import sosfilt, butter, iirnotch
from src.config import CONFIG, get_config_value, get_config

# Quiet torchaudio backend warnings
import warnings
warnings.filterwarnings("ignore", message="torchaudio._backend.*")


# UTILITY: Cleanup function for test files
def cleanup_old_test_files(max_age_hours: int = 1):
    """Clean up old test output files to prevent conflicts."""
    output_dir = Path("test_suite/test_outputs")
    main_output_dir = Path(".")

    if not output_dir.exists():
        return 0

    now = time.time()
    deleted_count = 0

    for file_path in output_dir.glob("*.wav"):
        try:
            if "dynamic_" in file_path.name or "test_run_" in file_path.name:
                file_age = now - file_path.stat().st_mtime
                if file_age > (max_age_hours * 3600):
                    file_path.unlink()
                    deleted_count += 1
        except Exception:
            pass

    for file_path in main_output_dir.glob("output_audio_dynamic_*.wav"):
        try:
            file_age = now - file_path.stat().st_mtime
            if file_age > (max_age_hours * 3600):
                file_path.unlink()
                deleted_count += 1
        except Exception:
            pass

    if deleted_count > 0:
        logger.info(f"🧹 Auto-cleaned {deleted_count} old test files")

    return deleted_count


# UTILITY: Get WAV duration
def get_wav_duration(path: str) -> float | None:
    """Get WAV duration using librosa or fallback."""
    try:
        return librosa.get_duration(path=path)
    except Exception:
        return None


def set_torchaudio_backend():
    """Configure torchaudio backend with SOX preference (Windows-friendly; debug deps)."""
    preferred_backend = 'sox_io'
    fallback_backend = 'soundfile'

    # Try pre-set env for SOX (helps detection)
    if 'TORCHAUDIO_BACKEND' not in os.environ:
        os.environ['TORCHAUDIO_BACKEND'] = preferred_backend

    try:
        available_backends = torchaudio.list_audio_backends()
        logger.debug(f"Available torchaudio backends: {available_backends}")

        if preferred_backend in available_backends:
            old_backend = torchaudio.get_audio_backend()
            if old_backend != preferred_backend:
                torchaudio.set_audio_backend(preferred_backend)
                new_backend = torchaudio.get_audio_backend()
                logger.info(f"✓ Switched to {preferred_backend} (from {old_backend})")
            else:
                logger.info(f"✓ Already on {preferred_backend}")
            return preferred_backend
        else:
            logger.warning(f"⚠ {preferred_backend} not in backends ({available_backends}). Check SOX DLLs in PATH.")
            logger.warning("Install full SOX: choco install sox, or conda install -c conda-forge torchaudio.")
            torchaudio.set_audio_backend(fallback_backend)
            return fallback_backend

    except RuntimeError as load_e:  # SOX load fail (DLL missing)
        logger.error(f"❌ Failed to load {preferred_backend}: {load_e} – Using {fallback_backend}")
        try:
            torchaudio.set_audio_backend(fallback_backend)
        except Exception:
            logger.error("Torchaudio backend switch failed – audio ops may break")
        return fallback_backend
    except ImportError:
        logger.error("❌ Torchaudio not installed – install via pip/conda")
        return None
    except Exception as e:
        logger.error(f"❌ Backend config failed: {e} – Fallback to {fallback_backend}")
        return fallback_backend


# Simple async denoising and normalization (unused but kept)
async def denoise_and_normalize_in_memory(wav: np.ndarray, sr: int) -> np.ndarray:
    try:
        # Placeholder: return same audio (hook for future use)
        return wav
    except Exception as e:
        logger.debug(f"denoise_and_normalize_in_memory failed: {e}")
        return wav


def get_silence(duration: float = 1.0, sr: int = 24000, dtype: torch.dtype = torch.float32, device: torch.device | str = 'cpu') -> torch.Tensor:
    """Create a silence tensor [1, samples]."""
    try:
        dev = torch.device(device) if not isinstance(device, torch.device) else device
    except Exception:
        dev = torch.device('cpu')
    samples = int(sr * max(0.0, float(duration)))
    return torch.zeros((1, samples), dtype=dtype, device=dev)


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2) + 1e-12))


def is_artifact_laden_array(y: np.ndarray, sr: int, threshold_hz: float = 12000.0, ratio_threshold: float = 0.5,
                            n_fft: int = 1024, hop_length: int = 256) -> bool:
    """Heuristic detector on in-memory array. Returns True if high-band energy ratio is large.

    Uses a smaller FFT by default for performance. Caller provides y (mono float) and sr.
    """
    try:
        if y is None or y.size == 0 or sr <= 0:
            return False
        y = y.astype(np.float32, copy=False)
        S = np.abs(librosa.stft(y, n_fft=int(n_fft), hop_length=int(hop_length)))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=int(n_fft))
        high_mask = freqs >= float(threshold_hz)
        if not np.any(high_mask):
            return False
        # Sum power in high band vs total
        high_energy = float(np.sum(S[high_mask] ** 2))
        total_energy = float(np.sum(S ** 2) + 1e-12)
        ratio = high_energy / total_energy
        flagged = ratio >= float(ratio_threshold)
        # Log sparingly
        if flagged:
            logger.info(f"artifact ratio={ratio:.3f} (thr={ratio_threshold:.2f}) in-memory")
        else:
            logger.trace(f"artifact ratio={ratio:.3f} (thr={ratio_threshold:.2f}) in-memory")
        return flagged
    except Exception as e:
        logger.debug(f"Artifact detect (array) failed: {e}")
        return False


def is_artifact_laden(path: str, threshold_hz: float = 12000.0, ratio_threshold: float = 0.5) -> bool:
    """File-path wrapper for artifact detection, delegates to array-based detector."""
    try:
        y, sr = librosa.load(path, sr=None, mono=True)
        return is_artifact_laden_array(y, sr, threshold_hz=threshold_hz, ratio_threshold=ratio_threshold)
    except Exception as e:
        logger.debug(f"Artifact detect failed for {path}: {e}")
        return False


def pad_short_text(text: str, params: Dict[str, Any]) -> str:
    """Pad very short texts with ellipses/soft tokens to stabilize prosody.

    params:
      - enable_text_padding: bool
      - text_ellipses_count: int (blocks of '...')
      - max_short_word_len: int
      - vocalise_patterns: list[str]
    """
    try:
        if not isinstance(text, str):
            return str(text)
        enable = bool(params.get('enable_text_padding', True))
        if not enable:
            return text
        ellipses_blocks = int(params.get('text_ellipses_count', 2))
        max_short = int(params.get('max_short_word_len', 3))
        vocalise = set([str(v).lower() for v in params.get('vocalise_patterns', ['ah', 'oh', 'aah', 'mmm', 'uh', 'mmh'])])

        base = text.strip()
        if not base:
            return text
        # If it's a vocalise token (e.g., "ah"), don't pad with ellipses
        if base.lower() in vocalise and len(base) <= max_short:
            return base
        # If it's very short, wrap with ellipses blocks
        if len(base) <= max_short:
            return ("... " * ellipses_blocks) + base + (" ..." * ellipses_blocks)
        return base
    except Exception:
        return text


def notch_filter(audio: np.ndarray, sr: int, notch_freq: float, Q: float = 30.0) -> np.ndarray:
    """Apply a notch filter around `notch_freq` with quality factor Q."""
    try:
        b, a = iirnotch(w0=notch_freq / (sr / 2.0), Q=Q)
        return sosfilt(np.array([[b[0], b[1], b[2], 1.0, a[1], a[2]]]), audio)
    except Exception:
        return audio


def apply_highpass(audio: np.ndarray, sr: int, cutoff_hz: float = 50.0) -> np.ndarray:
    try:
        w = float(cutoff_hz) / (sr / 2.0)
        sos = butter(2, w, btype='high', output='sos')
        return sosfilt(sos, audio)
    except Exception:
        return audio


def apply_lowpass(audio: np.ndarray, sr: int, cutoff_hz: float = 8000.0) -> np.ndarray:
    try:
        w = float(cutoff_hz) / (sr / 2.0)
        sos = butter(2, w, btype='low', output='sos')
        return sosfilt(sos, audio)
    except Exception:
        return audio


def resample_to(audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    try:
        if orig_sr == target_sr:
            return audio
        from torchaudio.transforms import Resample
        resampler = Resample(orig_sr, target_sr)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        out = resampler(audio)
        return out.squeeze(0)
    except Exception as e:
        logger.debug(f"Resample failed: {e}")
        return audio


def save_wav_numpy(path: str | Path, audio_np: np.ndarray, sr: int) -> bool:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(p), audio_np, sr)
        return True
    except Exception as e:
        logger.debug(f"NumPy save failed: {e}")
        return False


def load_wav_numpy(path: str | Path) -> Tuple[np.ndarray, int] | Tuple[None, None]:
    try:
        data, sr = sf.read(str(path), dtype='float32')
        if data.ndim == 2:
            data = data.mean(axis=1)
        return data, int(sr)
    except Exception as e:
        logger.debug(f"NumPy load failed: {e}")
        return None, None


def ensure_mono(wav: torch.Tensor) -> torch.Tensor:
    if wav.dim() == 2 and wav.size(0) > 1:
        return wav.mean(dim=0, keepdim=True)
    if wav.dim() == 1:
        return wav.unsqueeze(0)
    return wav


def normalize_peak(wav: torch.Tensor) -> torch.Tensor:
    peak = torch.max(torch.abs(wav)) if wav.numel() > 0 else torch.tensor(1.0, dtype=wav.dtype, device=wav.device)
    if peak > 0:
        return wav / peak
    return wav


__all__ = [
    'cleanup_old_test_files',
    'get_wav_duration',
    'set_torchaudio_backend',
    'denoise_and_normalize_in_memory',
    'get_silence',
    'is_artifact_laden',
    'pad_short_text',
    'notch_filter',
    'apply_highpass',
    'apply_lowpass',
    'resample_to',
    'save_wav_numpy',
    'load_wav_numpy',
    'ensure_mono',
    'normalize_peak',
]
