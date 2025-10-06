import asyncio
import os
import time
from typing import Optional
import os
import numba
import numpy as np
import librosa
import torch
import torchaudio
import librosa
import soundfile as sf
import tempfile
import asyncio
import os
import time
import numpy as np
import librosa
from scipy.signal import sosfilt, butter, iirnotch
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


def is_artifact_laden(wav_path: str, threshold_hz: float = 7000.0, sr: int = 24000) -> bool:
    """Detect gen artifacts (only; mean >7000Hz OR >50% high frames). Skip refs."""
    try:
        y, actual_sr = torchaudio.load(wav_path)
        y = y.mean(dim=0).numpy()
        if len(y) < sr * 0.5:  # <0.5s skip
            return False

        centroid = librosa.feature.spectral_centroid(y=y, sr=actual_sr, n_fft=2048, hop_length=512)[0]
        mean_centroid = np.mean(centroid)
        half_thresh = threshold_hz / 2  # 3500Hz
        high_freq_ratio = np.mean(centroid > half_thresh)

        # FIXED: Higher thresh/ratio (voice <6000Hz; purge only severe chirps)
        is_bad = (mean_centroid > threshold_hz) or (high_freq_ratio > 0.5)  # 50% (sibilants ok <0.5)

        reason = "high_mean" if mean_centroid > threshold_hz else "high_ratio"
        logger.debug(
            f"Artifact check {wav_path}: mean={mean_centroid:.0f}{' >' if mean_centroid > threshold_hz else ' <= '}{threshold_hz}Hz (ratio={high_freq_ratio:.2f}{' >0.5' if high_freq_ratio > 0.5 else ' <=0.5'}, len={len(y) / actual_sr:.2f}s) – {'bad (' + reason + ')' if is_bad else 'clean'}")

        return is_bad
    except Exception as e:
        logger.trace(f"Check failed {wav_path}: {e} – clean")
        return False



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



def apply_post_processing(wav: torch.Tensor, sr: int, params: dict | None = None) -> np.ndarray:
    """Merged post-processing: Trim → Denoise (if enabled) → EQ (if gain!=0) → Notch (if gain<0) → Normalize (if enabled) → Rate → Fade → Clamp.
    Uses full params dict from merge (e.g., 'enable_denoising': True, 'eq_gain_db': -3.0). Logs each step. FIXED: No sub-call/kwargs (single pass, no redundancy)."""
    if params is None:
        params = {}
        logger.debug("Post skipped: No params – raw output")
        return wav.cpu().squeeze().numpy() if wav.dim() > 1 else wav.cpu().numpy()

    if not params.get('enable_post_processing', False):
        logger.debug("Post disabled – raw output")
        return wav.cpu().squeeze().numpy() if wav.dim() > 1 else wav.cpu().numpy()

    voice_name = params.get('voice_name', 'unknown')
    logger.debug(f"Post params for {voice_name}: {params}")  # Full dict (confirms merge)

    # Ensure 1D np (mono)
    if wav.dim() > 1:
        wav_np = wav.mean(dim=0).cpu().numpy()  # Avg channels if multi
    else:
        wav_np = wav.cpu().numpy()
    logger.debug(f"Post input: {len(wav_np)} samples @ {sr}Hz ({len(wav_np)/sr:.2f}s)")

    # Trim (if threshold)
    trim_db = params.get('trim_threshold_db', None)
    if trim_db is not None:
        wav_np, _ = librosa.effects.trim(wav_np, top_db=trim_db)
        logger.debug(f"Trim applied ({trim_db}dB): {len(wav_np)/sr:.2f}s")

    # Denoise (spectral if enabled)
    enable_denoise = params.get('enable_denoising', False)
    if enable_denoise:
        noise_floor_db = params.get('noise_floor_db', -60.0)
        n_fft = params.get('n_fft', 2048)
        hop_length = params.get('hop_length', 512)
        stft = librosa.stft(wav_np, n_fft=n_fft, hop_length=hop_length)
        mag, phase = np.abs(stft), np.angle(stft)
        noise_floor = 10 ** (noise_floor_db / 20.0)
        clean_mag = np.maximum(mag - noise_floor, 0.0)
        clean_stft = clean_mag * np.exp(1j * phase)
        wav_np = librosa.istft(clean_stft, hop_length=hop_length, length=len(wav_np))
        logger.debug(f"Denoise applied (spectral subtract, {noise_floor_db}dB, n_fft={n_fft}, hop={hop_length})")

    # EQ (if gain !=0)
    eq_gain_db = params.get('eq_gain_db', None)
    eq_cutoff_hz = params.get('eq_cutoff_hz', 4000)
    if eq_gain_db is not None and eq_gain_db != 0.0:
        nyquist = sr / 2.0
        cutoff_norm = min(1.0, max(0.01, eq_cutoff_hz / nyquist))
        sos = butter(4, cutoff_norm, btype='lowpass' if eq_gain_db < 0 else 'highpass', output='sos')
        wav_np = sosfilt(sos, wav_np)
        logger.debug(f"EQ applied ({eq_gain_db}dB {'low' if eq_gain_db < 0 else 'high'}-pass @ {eq_cutoff_hz}Hz)")

    # Notch (if gain <0)
    notch_gain_db = params.get('notch_gain_db', None)
    notch_low_hz = params.get('notch_low_hz', 8000)
    notch_high_hz = params.get('notch_high_hz', 11000)
    if notch_gain_db is not None and notch_gain_db < 0:
        # Use bandstop butter (sosfilt; iirnotch b,a needs lfilter)
        # from scipy.signal import butter
        nyquist = sr / 2.0
        low_norm = max(0.01, min(0.99, notch_low_hz / nyquist))
        high_norm = min(0.99, max(0.01, notch_high_hz / nyquist))
        if low_norm < high_norm:
            sos_notch = butter(4, [low_norm, high_norm], btype='bandstop', output='sos')
            gain_factor = 10 ** (notch_gain_db / 20.0)
            notched = sosfilt(sos_notch, wav_np)
            wav_np = notched * gain_factor + wav_np * (1 - gain_factor)  # Blend attenuated
            logger.debug(f"Notch applied ({notch_gain_db}dB @ {notch_low_hz}-{notch_high_hz}Hz, 4th-order bandstop)")

    # Normalize (if enabled, post-denoise/EQ)
    enable_normalize = params.get('enable_denoise_normalize', False)
    norm_method = params.get('normalize_method', 'peak')
    if enable_normalize:
        if norm_method == 'peak':
            peak = np.max(np.abs(wav_np))
            if peak > 0:
                wav_np /= peak
                wav_np *= 0.95  # Headroom
                logger.debug(f"Normalize applied ({norm_method}: peak -1dB)")
        elif norm_method == 'rms':
            rms = np.sqrt(np.mean(wav_np ** 2))
            if rms > 0:
                target_rms = 10 ** (-18 / 20)  # -18dB
                wav_np *= target_rms / rms
                logger.debug(f"Normalize applied ({norm_method}: RMS -18dB)")

    # Rate (time stretch if !=1.0)
    rate = params.get('speaking_rate', 1.0)
    if abs(rate - 1.0) > 0.05:
        stretch_rate = 1.0 / rate
        wav_np = librosa.effects.time_stretch(wav_np, rate=stretch_rate)
        # Resample/trim to original length if stretch altered
        target_len = int(len(wav_np) * rate)
        if len(wav_np) > target_len:
            wav_np = wav_np[:target_len]
        else:
            pad_len = target_len - len(wav_np)
            wav_np = np.pad(wav_np, (0, pad_len), mode='constant')
        logger.debug(f"Rate applied ({rate}x time-stretch)")

    # Fade (if ms >0)
    fade_ms = params.get('fade_ms', None)
    if fade_ms is not None and fade_ms > 0:
        fade_samples = int(sr * (fade_ms / 1000.0))
        if len(wav_np) > 2 * fade_samples:
            fade_in = np.linspace(0.0, 1.0, fade_samples)
            fade_out = np.linspace(1.0, 0.0, fade_samples)
            wav_np[:fade_samples] *= fade_in
            wav_np[-fade_samples:] *= fade_out
            logger.debug(f"Fade applied ({fade_ms}ms in/out)")

    # Clamp gain (limit max)
    gain_limit = params.get('gain_max_limit', None)
    if gain_limit is not None:
        wav_np = np.clip(wav_np, -gain_limit, gain_limit)
        logger.debug(f"Gain clamped to {gain_limit}")

    # Final clip/return 1D np
    wav_np = np.clip(wav_np, -1.0, 1.0)
    logger.debug(f"Post complete for {voice_name}: {len(wav_np)/sr:.2f}s (applied: denoise={enable_denoise}, eq={eq_gain_db or 'no'}, notch={notch_gain_db or 'no'}, rate={rate}, etc.)")
    return wav_np




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

