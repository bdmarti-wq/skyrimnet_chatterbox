import asyncio
import os
import time
from typing import Optional

import numba
import numpy as np
import librosa
import torch
from scipy.signal import sosfilt, butter
import soundfile as sf
import tempfile
import asyncio
import os
import time
import numpy as np
import librosa
from scipy.signal import sosfilt, butter
from pathlib import Path
from loguru import logger

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


# UTILITY: Simple async denoising and normalization
async def denoise_and_normalize_in_memory(
        audio: np.ndarray, sr: int, normalize_method: str = "peak",
        noise_floor_db: float | None = -60.0  # PATCH: None to skip denoising
) -> np.ndarray:
    """Simple in-memory audio denoising and normalization."""
    if noise_floor_db is None:
        logger.debug("Denoise/normalize skipped (noise_floor_db=None)")
        return audio  # No-op

    loop = asyncio.get_running_loop()

    # Simple denoising (only if noise_floor_db provided)
    def simple_denoise(audio, sr, noise_floor_db):
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        noise_floor = max(10 ** (noise_floor_db / 20), np.mean(np.abs(audio)) * 0.01)
        stft = librosa.stft(audio)
        magnitude, phase = librosa.magphase(stft)
        magnitude[magnitude < noise_floor] *= 0.1
        denoised_stft = magnitude * phase
        return librosa.istft(denoised_stft)

    audio = await loop.run_in_executor(None, simple_denoise, audio, sr, noise_floor_db)
    audio = np.clip(audio, -1.0, 1.0)

    # Simple normalization (skip if method=None)
    if normalize_method is None:
        return audio

    def simple_normalize(audio, method):
        if method == "peak":
            max_amp = np.max(np.abs(audio))
            if max_amp > 0:
                return audio / max_amp * 0.707
        elif method == "rms":
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 0:
                return audio / rms * 0.1  # -20dB RMS
        return audio

    audio = await loop.run_in_executor(None, simple_normalize, audio, normalize_method)
    return np.clip(audio, -1.0, 1.0)


# Modular Post-Processing Functions (extracted from generate) - Updated for no-op values
def trim_silence(audio: np.ndarray, threshold_db: float | None = -25.0) -> np.ndarray:
    """Trim leading/trailing silence based on threshold."""
    if threshold_db is None:
        logger.debug("Trim skipped (threshold_db=None)")
        return audio  # No-op
    threshold = 10 ** (threshold_db / 20)
    abs_audio = np.abs(audio)
    start_idx = np.argmax(abs_audio > threshold)
    end_idx = len(audio) - np.argmax(abs_audio[::-1] > threshold)
    if start_idx < end_idx:
        return audio[start_idx:end_idx]
    return audio

# FIXED: Add notch from gen_utils (modular; no-op on gain_db=None)
def apply_notch(audio: np.ndarray, sr: int, low_hz: float | None = 8000.0, high_hz: float | None = 11000.0,
                gain_db: float | None = -12.0) -> np.ndarray:
    """Apply bandstop notch filter (e.g., 8-11kHz -12dB)."""
    if gain_db is None or gain_db >= 0 or low_hz is None or high_hz is None or low_hz >= high_hz:
        logger.debug("Notch skipped (gain_db=None/>=0 or invalid range)")
        return audio  # No-op
    nyquist = sr / 2.0
    low_norm = max(0.01, min(0.99, low_hz / nyquist))
    high_norm = min(0.99, high_hz / nyquist)
    if low_norm >= high_norm:
        logger.warning(f"Invalid notch range {low_hz}-{high_hz}; skipping")
        return audio
    try:
        sos_notch = butter(2, [low_norm, high_norm], btype='bandstop', output='sos')
        gain_factor = 10 ** (gain_db / 20.0)
        filtered = sosfilt(sos_notch, audio) * gain_factor + audio * (1 - gain_factor)
        return np.clip(filtered, -1.0, 1.0)
    except (ValueError, NameError) as e:
        logger.warning(f"Notch failed: {e}")
        return audio

def apply_eq(audio: np.ndarray, sr: int, gain_db: float | None = 0.0, cutoff_hz: float | None = 3000.0) -> np.ndarray:
    """Apply simple EQ (low/high-pass with gain)."""
    if gain_db is None or gain_db == 0.0:
        logger.debug("EQ skipped (gain_db=None or 0)")
        return audio  # No-op
    if cutoff_hz is None:
        logger.debug("EQ skipped (cutoff_hz=None)")
        return audio
    nyquist = sr / 2
    cutoff = cutoff_hz / nyquist
    if not (0 < cutoff < 1):
        logger.warning(f"Invalid cutoff {cutoff_hz}; skipping EQ")
        return audio
    sos = butter(2, cutoff, btype='lowpass' if gain_db < 0 else 'highpass', output='sos')
    gain_factor = 10 ** (gain_db / 20)
    filtered = sosfilt(sos, audio)
    if gain_db < 0:
        audio = filtered * gain_factor + audio * (1 - gain_factor)
    else:
        audio = filtered * (1 + gain_factor) + audio
    return np.clip(audio, -1.0, 1.0)


# ... (other imports/funcs unchanged: trim_silence, denoise..., eq, notch, etc.)

def apply_gain_normalization(audio: np.ndarray, target_max: Optional[float] = None,
                             max_gain_limit: float = 1.0) -> np.ndarray:
    """Apply gain to reach target_max (clamps <= max_gain_limit). Jit-safe branch."""
    if target_max is None or target_max <= 0:
        return audio  # No-op

    current_max = np.max(np.abs(audio))  # Sync numpy (fast ~0.01ms)
    if current_max <= 0 or current_max >= target_max:
        return audio  # Already good/no need

    max_gain = min(target_max / current_max, max_gain_limit)  # Compute once (scalar; no jit issue)
    if max_gain <= 1.0:  # No boost needed
        return audio

    # FIXED: Jit loop with ifs (avoids max/min overload on scalar*float)
    @numba.jit(nopython=True)
    def apply_gain_jit(data, max_gain):
        # Numba-friendly: Explicit if for clamp (no built-in max/min with float lit)
        for i in range(len(data)):
            amplified = data[i] * max_gain
            if amplified > 1.0:
                data[i] = 1.0
            elif amplified < -1.0:
                data[i] = -1.0
            else:
                data[i] = amplified
        return data

    # Call jit (audio.copy() to avoid mutating original; astype(np.float64) for Numba)
    data_copy = audio.copy().astype(np.float64)  # Numba prefers float64 for precision
    wav_np = apply_gain_jit(data_copy, max_gain)

    logger.debug(f"Gain applied: {max_gain:.2f}x (target={target_max}, current_max={current_max:.3f})")
    return wav_np  # Returns clipped [-1,1]




def adjust_speaking_rate(audio: np.ndarray, rate: float = 1.0) -> np.ndarray:
    """Adjust speaking rate via time stretching."""
    if abs(rate - 1.0) <= 0.1:
        logger.debug(f"Speaking rate adjustment skipped (rate={rate} ≈1.0)")
        return audio  # No-op (as original)
    stretch_rate = 1.0 / rate
    stretched = librosa.effects.time_stretch(audio, rate=stretch_rate)
    target_length = int(len(audio) * rate)
    if len(stretched) > target_length:
        stretched = stretched[:target_length]
    else:
        pad_length = target_length - len(stretched)
        stretched = np.pad(stretched, (0, pad_length), mode='constant')
    return np.clip(stretched, -1.0, 1.0)


def apply_fade(audio: np.ndarray, sr: int, fade_ms: float | None = 20.0) -> np.ndarray:
    """Apply fade-in/out."""
    if fade_ms is None or fade_ms <= 0:
        logger.debug("Fade skipped (fade_ms=None or <=0)")
        return audio  # No-op
    fade_samples = int(sr * (fade_ms / 1000.0))
    audio_len = len(audio)
    if audio_len <= fade_samples * 2:
        logger.debug("Audio too short for fade; skipped")
        return audio
    adaptive_fade = min(fade_samples, int(audio_len * 0.05))
    if adaptive_fade <= 0:
        return audio
    fade_in = np.linspace(0.0, 1.0, adaptive_fade)
    audio[:adaptive_fade] *= fade_in
    fade_out = np.linspace(1.0, 0.0, adaptive_fade)
    audio[-adaptive_fade:] *= fade_out
    return audio


# High-Level Wrapper for Voice-Specific Processing - Revised: Dict-based, no booleans
async def apply_voice_specific_processing(
        audio: np.ndarray, sr: int, voice_params: dict | None = None,
        overrides: dict | None = None  # Bundled (None skips)
) -> np.ndarray:
    """Async chain: Denoise → Trim → EQ → Notch → Gain → Rate → Fade (offloads heavy; modular)."""
    if overrides is None:
        overrides = {}

    # Legacy override (unchanged; add notch keys)
    if voice_params:
        overrides['trim_threshold_db'] = voice_params.get('trim_threshold_db', overrides.get('trim_threshold_db'))
        overrides['eq_gain_db'] = voice_params.get('eq_gain_db', overrides.get('eq_gain_db'))
        overrides['eq_cutoff_hz'] = voice_params.get('eq_cutoff_hz', overrides.get('eq_cutoff_hz'))
        overrides['notch_low_hz'] = voice_params.get('notch_low', overrides.get('notch_low_hz', 8000))
        overrides['notch_high_hz'] = voice_params.get('notch_high', overrides.get('notch_high_hz', 11000))
        overrides['notch_gain_db'] = voice_params.get('notch_gain_db', overrides.get('notch_gain_db', -12))
        overrides['gain_target_max'] = voice_params.get('target_max', overrides.get('gain_target_max', 0.5))
        overrides['gain_max_limit'] = voice_params.get('max_gain', overrides.get('gain_max_limit', 2.0))
        overrides['speaking_rate'] = voice_params.get('speaking_rate', overrides.get('speaking_rate', 1.0))
        overrides['fade_ms'] = voice_params.get('fade_ms', overrides.get('fade_ms'))
        overrides['noise_floor_db'] = voice_params.get('noise_floor_db', overrides.get('noise_floor_db'))
        overrides['normalize_method'] = voice_params.get('normalize_method', overrides.get('normalize_method', 'peak'))
        overrides['enable_denoise_normalize'] = voice_params.get('enable_denoise_normalize', False)

    # Unpack (no-op defaults)
    trim_threshold_db = overrides.get('trim_threshold_db', None)
    eq_gain_db = overrides.get('eq_gain_db', None)
    eq_cutoff_hz = overrides.get('eq_cutoff_hz', None)
    notch_low_hz = overrides.get('notch_low_hz')
    notch_high_hz = overrides.get('notch_high_hz')
    notch_gain_db = overrides.get('notch_gain_db')
    gain_target_max = overrides.get('gain_target_max', None)
    gain_max_limit = overrides.get('gain_max_limit', 2.0)
    speaking_rate = overrides.get('speaking_rate', 1.0)
    fade_ms = overrides.get('fade_ms', None)
    enable_denoise_normalize = overrides.get('enable_denoise_normalize', False)
    normalize_method = overrides.get('normalize_method', 'peak') if enable_denoise_normalize else None
    noise_floor_db = overrides.get('noise_floor_db', -60.0) if enable_denoise_normalize else None

    logger.debug(f"Voice processing: { {k: v for k, v in locals().items() if k in overrides and v is not None} }")

    loop = asyncio.get_running_loop()

    # Chain: Offload heavy (denoise/stretch); light inline/async-possible
    if enable_denoise_normalize:
        audio = await denoise_and_normalize_in_memory(audio, sr, normalize_method, noise_floor_db)  # Already async offload

    # Light sync (inline; <10ms each)
    audio = trim_silence(audio, trim_threshold_db)
    audio = apply_eq(audio, sr, eq_gain_db, eq_cutoff_hz)
    audio = apply_notch(audio, sr, notch_low_hz, notch_high_hz, notch_gain_db)  # FIXED: New modular
    audio = apply_gain_normalization(audio, gain_target_max, gain_max_limit)

    # Offload stretch if rate !=1 (librosa ~50ms medium; heavy)
    if abs(speaking_rate - 1.0) > 0.01:
        def sync_stretch(audio, sr, rate):
            return adjust_speaking_rate(audio, rate)  # Internal call
        audio = await loop.run_in_executor(None, sync_stretch, audio, sr, speaking_rate)

    audio = apply_fade(audio, sr, fade_ms)  # Light; inline

    return np.clip(audio, -1.0, 1.0)


async def apply_post_processing(wav: torch.Tensor, model_sr: int, audio_params: dict | None) -> np.ndarray:
    """Post-processing: Applies merged params directly (no re-checks; Trusts merge for enables/defaults)."""
    if not audio_params:
        logger.debug("Post skipped: No params")
        return wav.cpu().squeeze(0).numpy()  # Raw

    """Sync/Async post-chain: Ensure 2D input → 1D mono output."""
    if not audio_params.get('enable_post_processing', False):
        # Passthru: Squeeze to 1D np
        return wav.cpu().squeeze().numpy() if wav.dim() > 1 else wav.cpu().numpy()

    # FIXED: Ensure 2D mono input (Gradio/TTS may give 1D/3D)
    if wav.dim() == 1:
        wav_np = wav.cpu().squeeze().numpy()
        wav = torch.from_numpy(wav_np).unsqueeze(0).unsqueeze(-1) if wav_np.ndim == 1 else torch.from_numpy(np.expand_dims(wav_np, -1))
    elif wav.dim() > 2:
        wav = torch.mean(wav, dim=-1)  # Avg channels → 2D
    logger.debug(f"Post input ensured: {wav.shape} @ {model_sr}Hz")

    wav_np = wav.cpu().squeeze(-1).squeeze(0).numpy()  # Final 1D mono np

    voice_name_log = audio_params.get('voice_name', 'Unknown')  # FIXED: From merge (no voice_name param needed)
    enable_post = audio_params.get('enable_post_processing', True)
    if not enable_post:
        logger.info(f"Post disabled for {voice_name_log} – raw")
        return wav_np
    logger.debug(f"Post enabled for {voice_name_log}")  # FIXED: Consistent log (no None)

    # FIXED: Direct apply from audio_params (merged: If key present/non-default, process; Else skip)
    # Rate (resample if !=1.0 or present)
    speaking_rate = audio_params.get('speaking_rate')
    if speaking_rate is not None and speaking_rate != 1.0:
        wav_np = adjust_speaking_rate(wav_np, speaking_rate)
        logger.debug(f"Rate applied: {speaking_rate}x for {voice_name_log}")

    # EQ (if gain_db !=0.0 or present)
    eq_gain_db = audio_params.get('eq_gain_db')
    eq_cutoff_hz = audio_params.get('eq_cutoff_hz', 3000)
    if eq_gain_db is not None and eq_gain_db != 0.0:
        wav_np = apply_eq(wav_np, model_sr, eq_gain_db, eq_cutoff_hz)
        logger.debug(f"EQ applied: {eq_gain_db}dB @ {eq_cutoff_hz}Hz for {voice_name_log}")

    # Gain Norm (if target_max not None or limit !=1.0)
    gain_target_max = audio_params.get('gain_target_max')
    gain_max_limit = audio_params.get('gain_max_limit', 1.0)
    if gain_target_max is not None or gain_max_limit != 1.0:
        wav_np = apply_gain_normalization(wav_np, gain_target_max, gain_max_limit)
        logger.debug(f"Gain norm: target={gain_target_max}, limit={gain_max_limit} for {voice_name_log}")

    # JIT Clamp (always if enabled; From merged flag)
    if audio_params.get('enable_post_jit_gain', True):
        @numba.jit(nopython=True)
        def apply_gain_jit(data, max_gain):
            for i in range(len(data)):
                data[i] = min(max(data[i] * max_gain, -1.0), 1.0)
            return data

        max_gain_limit = audio_params.get('gain_max_limit', 1.0)  # From merged
        wav_np = apply_gain_jit(wav_np.copy().astype(np.float32), max_gain_limit)
        logger.debug(f"JIT clamp {max_gain_limit} for {voice_name_log}")

    # Voice-Specific (if enable flag; Builds from merged keys)
    if audio_params.get('enable_post_voice_processing', True):
        voice_overrides = {
            'trim_threshold_db': audio_params.get('trim_threshold_db', -28),
            'eq_gain_db': 0.0,  # Already applied; Reset
            'eq_cutoff_hz': eq_cutoff_hz,  # From above
            'notch_low_hz': audio_params.get('notch_low_hz', 8000),
            'notch_high_hz': audio_params.get('notch_high_hz', 11000),
            'notch_gain_db': audio_params.get('notch_gain_db'),  # e.g., -12 for dlc1 (applies if not None)
            'gain_max_limit': gain_max_limit,
            'speaking_rate': 1.0,  # Already applied
            'fade_ms': audio_params.get('fade_ms', 20),
            'enable_denoise_normalize': audio_params.get('enable_denoise_normalize', False),
            'normalize_method': audio_params.get('normalize_method', 'rms') if audio_params.get(
                'enable_denoise_normalize', False) else None,
            'noise_floor_db': audio_params.get('noise_floor_db', -60.0) if audio_params.get('enable_denoise_normalize',
                                                                                            False) else None
        }
        # FIXED: Apply/log only if voice-specific (e.g., notch_gain_db not None)
        applied_overrides = {k: v for k, v in voice_overrides.items() if
                             v is not None and k in ['notch_gain_db', 'trim_threshold_db']}
        if applied_overrides:
            logger.debug(f"Voice overrides applied for {voice_name_log}: {applied_overrides}")
        else:
            logger.debug(f"Default processing for {voice_name_log}")

        wav_np = await apply_voice_specific_processing(wav_np, model_sr, None,
                                                                   overrides=voice_overrides)  # FIXED: No voice_params (merged in overrides)

    return np.clip(wav_np, -1.0, 1.0)  # 1D output


# Set torchaudio backend: Prefer 'sox' for speed/reliability, fallback to 'soundfile'
import os
import logging
from loguru import logger as loguru_logger  # If using loguru


def set_torchaudio_backend():
    """Configure torchaudio backend with SOX preference (Windows-friendly; debug deps)."""
    preferred_backend = 'sox_io'
    fallback_backend = 'soundfile'
    sox_exe = 'sox'

    # Try pre-set env for SOX (helps detection)
    if 'TORCHAUDIO_BACKEND' not in os.environ:
        os.environ['TORCHAUDIO_BACKEND'] = preferred_backend

    try:
        import torchaudio
        available_backends = torchaudio.list_audio_backends()
        logger.debug(f"Available torchaudio backends: {available_backends}")  # Or print

        if preferred_backend in available_backends:
            # Force load/test
            old_backend = torchaudio.get_audio_backend()
            if old_backend != preferred_backend:
                torchaudio.set_audio_backend(preferred_backend)  # Explicit set
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
            import torchaudio
            torchaudio.set_audio_backend(fallback_backend)
        except:
            logger.error("Torchaudio backend switch failed – audio ops may break")
        return fallback_backend
    except ImportError:
        logger.error("❌ Torchaudio not installed – install via pip/conda")
        return None
    except Exception as e:
        logger.error(f"❌ Backend config failed: {e} – Fallback to {fallback_backend}")
        return fallback_backend

