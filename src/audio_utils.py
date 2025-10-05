import asyncio
import os
import time
import numpy as np
import librosa
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


def apply_gain_normalization(audio: np.ndarray, target_max: float | None = 0.5,
                             max_gain_limit: float = 2.0) -> np.ndarray:
    """Apply gain to reach target max amplitude, limited by max_gain."""
    if target_max is None:
        logger.debug("Gain normalization skipped (target_max=None)")
        return audio  # No-op
    current_max = np.max(np.abs(audio))
    if current_max >= target_max or current_max <= 0:
        return audio
    gain_factor = min(target_max / current_max, max_gain_limit)
    return np.clip(audio * gain_factor, -1.0, 1.0)


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


# Set torchaudio backend: Prefer 'sox' for speed/reliability, fallback to 'soundfile'
def set_torchaudio_backend():
    """Configure torchaudio backend with SOX preference (Windows-friendly)."""
    # Set env var BEFORE importing torchaudio
    preferred_backend = 'sox'
    fallback_backend = 'soundfile'

    # Early set to env (torchaudio reads on import)
    os.environ['TORCHAUDIO_BACKEND'] = fallback_backend  # Default fallback

    try:
        import torchaudio  # Temp import to check backends
        available_backends = torchaudio.list_audio_backends()
        if preferred_backend in available_backends:
            os.environ['TORCHAUDIO_BACKEND'] = preferred_backend
            print(f"✓ Using torchaudio backend: {preferred_backend} (faster resample/metadata)")
            return preferred_backend
        else:
            print(f"⚠ SOX not available (install via 'choco install sox'). Using: {fallback_backend}")
            return fallback_backend
    except ImportError:
        print(f"❌ torchaudio not installed—audio ops will fail.")
        return None
    except Exception as e:
        print(f"❌ Backend check failed: {e}. Using fallback: {fallback_backend}")
        return fallback_backend

