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
from config import CONFIG
import asyncio

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


def is_artifact_laden(wav_path: str, threshold_hz: float = None, ratio_threshold: float = 0.5, sr: int = 24000) -> bool:
    """Detect artifacts. FIXED: Default 7000Hz (config); accurate log/comp (no '8000')."""
    threshold_hz = threshold_hz or CONFIG.get_value('fuzzy_artifact_threshold_hz', 7000.0)
    threshold_hz = CONFIG.clamp_value('fuzzy_artifact_threshold_hz', threshold_hz)  # Assume CAP added
    ratio_threshold = 0.5  # Fixed; add CAP 'FUZZY_RATIO_THRESHOLD' if tune

    try:
        y, actual_sr = torchaudio.load(wav_path)
        y = y.mean(dim=0).numpy()
        min_dur_sec = CONFIG.get_value('min_post_duration_sec', 0.5)
        if len(y) < actual_sr * min_dur_sec:
            logger.trace(f"Short {Path(wav_path).name} (<{min_dur_sec}s) – clean")
            return False

        n_fft = CONFIG.get_value('n_fft', 2048)
        hop_length = CONFIG.get_value('hop_length', 512)
        centroid = librosa.feature.spectral_centroid(y=y, sr=actual_sr, n_fft=n_fft, hop_length=hop_length)[0]
        mean_centroid = np.mean(centroid)
        high_frames = np.mean(centroid > (threshold_hz / 2))
        dur = len(y) / actual_sr

        is_bad = (mean_centroid > threshold_hz) or (high_frames > ratio_threshold)
        reason = "high_mean" if mean_centroid > threshold_hz else "high_ratio" if high_frames > ratio_threshold else "clean"
        comp_mean = " > " if mean_centroid > threshold_hz else " <= "
        comp_ratio = " > " if high_frames > ratio_threshold else " <= "
        logger.debug(f"Artifact {Path(wav_path).name}: mean={mean_centroid:.0f}{comp_mean}{threshold_hz}Hz (ratio={high_frames:.2f}{comp_ratio}{ratio_threshold}, dur={dur:.2f}s) – {reason}")

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





import asyncio  # Ensure at top
from config import CONFIG

async def apply_post_processing(wav: torch.Tensor, sr: int, params: dict | None = None) -> np.ndarray:
    """Async post: Trim → Denoise → EQ → Notch → Normalize → Rate → Trail Cut → Fade → Clamp. FIXED: Async (executor for heavy librosa); speed (n_fft=1024, skip stretch=1.0); trail cut (-45dB post-rate); config 'enable_post_resample'=False no-op. FIXED: All vars (wav_np, enable_denoise, eq_gain_db, etc.) defined early in outer/inner (no reference/scope errors on skip/exception). Applied flags via nonlocal (track if ran)."""
    if params is None:
        params = {}
        logger.debug("Post skipped (no params) – raw")
        return wav.cpu().squeeze().numpy()

    enable_post = params.get('enable_post_processing', CONFIG.get_value('enable_post_processing', True))
    if not enable_post:
        logger.debug("Post disabled – raw")
        return wav.cpu().squeeze().numpy()

    voice_name = params.get('voice_name', 'unknown')
    non_none_params = {k: v for k, v in params.items() if v is not None}
    logger.debug(f"Post params for {voice_name}: {non_none_params}")

    # Ensure 1D np (sync)
    if wav.dim() > 1:
        wav_np = wav.mean(dim=0).cpu().numpy()
    else:
        wav_np = wav.cpu().numpy()
    orig_len = len(wav_np)
    orig_dur = orig_len / sr
    logger.debug(f"Post input: {orig_len} samples @ {sr}Hz ({orig_dur:.2f}s)")

    if orig_len == 0:
        logger.warning("Post input empty – raw fallback")
        return wav_np

    loop = asyncio.get_running_loop()

    # Config min guard
    min_dur_sec = CONFIG.clamp_value('min_post_duration_sec', params.get('min_post_duration_sec', CONFIG.get_value('min_post_duration_sec', 0.05)))
    min_samples = int(sr * min_dur_sec)
    light_mode = orig_len < min_samples
    if light_mode:
        logger.debug(f"Short input < {min_dur_sec}s – light post (skip heavy)")

    # Trim (sync; fast)
    trim_db = CONFIG.clamp_value('trim_threshold_db', params.get('trim_threshold_db', CONFIG.get_value('trim_threshold_db', -30.0)))
    did_trim = False
    if trim_db is not None and abs(trim_db) > 5 and not light_mode:
        hop = params.get('hop_length', CONFIG.get_value('hop_length', 256))
        frame_factor = CONFIG.clamp_value('trim_frame_length_factor', params.get('trim_frame_length_factor', CONFIG.get_value('trim_frame_length_factor', 4)))
        frame_length = hop * frame_factor
        n_fft_trim = CONFIG.clamp_value('max_n_fft_for_trim', min(params.get('max_n_fft_for_trim', CONFIG.get_value('max_n_fft_for_trim', 2048)), orig_len))

        try:
            wav_np_trim, _ = librosa.effects.trim(wav_np, top_db=abs(trim_db), frame_length=frame_length, hop_length=hop)
            if len(wav_np_trim) == 0:
                logger.warning(f"Trim {trim_db}dB zeroed – fallback full (tune > -35)")
                wav_np_trim = wav_np
            else:
                logger.debug(f"Trim applied ({trim_db}dB, frame={frame_length}, hop={hop}, n_fft={n_fft_trim}): {len(wav_np_trim)/sr:.2f}s")
                did_trim = True
            wav_np = wav_np_trim
        except Exception as trim_e:
            logger.warning(f"Trim failed: {trim_e} – skip")
    else:
        logger.debug(f"Trim skipped (db={trim_db}; light={light_mode})")

    new_len = len(wav_np)
    new_dur = new_len / sr
    heavy_skip = new_len < min_samples
    if heavy_skip:
        logger.debug("Post-trim short – skip heavy")

    # FIXED: Define applied flags in outer (track if each step ran; nonlocal to heavy)
    did_denoise = False
    did_eq = False
    did_notch = False
    did_normalize = False
    # All config vars early (for applied checks; no reliance on heavy)
    enable_denoise = params.get('enable_denoising', CONFIG.get_value('enable_denoising', False))
    eq_gain_db = CONFIG.clamp_value('eq_gain_db', params.get('eq_gain_db', CONFIG.get_value('eq_gain_db', 0.0)))
    notch_gain_db = params.get('notch_gain_db', CONFIG.get_value('notch_gain_db', None))
    if notch_gain_db is not None:
        notch_gain_db = CONFIG.clamp_value('notch_gain_db', notch_gain_db)
    enable_norm = params.get('enable_denoise_normalize', CONFIG.get_value('enable_denoise_normalize', False))

    # Heavy steps in executor (async speed; offloads CPU)
    def heavy_processing():
        nonlocal wav_np, did_denoise, did_eq, did_notch, did_normalize  # FIXED: Nonlocal for wav_np + flags (mod in inner, access outer)
        # Config denoise params early (always define; no scope error)
        noise_floor_db = CONFIG.clamp_value('noise_floor_db', params.get('noise_floor_db', CONFIG.get_value('noise_floor_db', -60.0)))
        min_denoise_samples = CONFIG.clamp_value('min_samples_for_denoise', params.get('min_samples_for_denoise', CONFIG.get_value('min_samples_for_denoise', 100)))
        n_fft_denoise = CONFIG.clamp_value('n_fft_denoise', params.get('n_fft_denoise', CONFIG.get_value('n_fft_denoise', 1024)))  # Always defined (faster default)
        n_fft_denoise = min(n_fft_denoise, new_len * 2)  # Scale to len
        hop_length = params.get('hop_length', CONFIG.get_value('hop_length', 256))
        eq_cutoff_hz = CONFIG.clamp_value('eq_cutoff_hz', params.get('eq_cutoff_hz', CONFIG.get_value('eq_cutoff_hz', 3000)))
        notch_low_hz = CONFIG.clamp_value('notch_low_hz', params.get('notch_low_hz', CONFIG.get_value('notch_low_hz', 8000)))
        notch_high_hz = CONFIG.clamp_value('notch_high_hz', params.get('notch_high_hz', CONFIG.get_value('notch_high_hz', 11000)))
        norm_method = params.get('normalize_method', CONFIG.get_value('normalize_method', 'peak'))

        # Denoise (if enabled and not skipped)
        if enable_denoise and not heavy_skip and new_len >= min_denoise_samples:
            try:
                stft = librosa.stft(wav_np, n_fft=n_fft_denoise, hop_length=hop_length)  # Now always defined
                mag, phase = np.abs(stft), np.angle(stft)
                noise_floor = 10 ** (noise_floor_db / 20.0)
                clean_mag = np.maximum(mag - noise_floor, 0.0)
                clean_stft = clean_mag * np.exp(1j * phase)
                wav_np = librosa.istft(clean_stft, hop_length=hop_length, length=new_len)  # Modify nonlocal
                did_denoise = True  # FIXED: Set flag if success
                logger.debug(f"Denoise applied ({noise_floor_db}dB, n_fft={n_fft_denoise}, hop={hop_length}, min_samples={min_denoise_samples})")
            except Exception as denoise_e:
                logger.warning(f"Denoise failed ({new_len} samples, n_fft={n_fft_denoise}): {denoise_e} – skip")
        else:
            logger.debug(f"Denoise skipped (enable={enable_denoise}, heavy_skip={heavy_skip}, len={new_len} < {min_denoise_samples})")

        # EQ (similar: define params early)
        if eq_gain_db != 0.0 and not heavy_skip and len(wav_np) > 0:
            nyquist = sr / 2.0
            cutoff_norm = min(1.0, max(0.01, eq_cutoff_hz / nyquist))
            sos = butter(4, cutoff_norm, btype='lowpass' if eq_gain_db < 0 else 'highpass', output='sos')
            wav_np = sosfilt(sos, wav_np)  # Modify nonlocal
            did_eq = True  # FIXED: Set flag if success
            logger.debug(f"EQ applied ({eq_gain_db}dB {'low' if eq_gain_db < 0 else 'high'}-pass @ {eq_cutoff_hz}Hz)")
        else:
            logger.debug(f"EQ skipped (gain={eq_gain_db}, heavy_skip={heavy_skip})")

        # Notch (define early)
        if notch_gain_db is not None and notch_gain_db < 0 and not heavy_skip and len(wav_np) > 0:
            nyquist = sr / 2.0
            low_norm = max(0.01, min(0.99, notch_low_hz / nyquist))
            high_norm = min(0.99, max(low_norm + 0.01, notch_high_hz / nyquist))
            if low_norm < high_norm:
                sos_notch = butter(4, [low_norm, high_norm], btype='bandstop', output='sos')
                gain_factor = 10 ** (notch_gain_db / 20.0)
                notched = sosfilt(sos_notch, wav_np)
                wav_np = notched * gain_factor + wav_np * (1 - gain_factor)  # Modify nonlocal
                did_notch = True  # FIXED: Set flag if success
                logger.debug(f"Notch applied ({notch_gain_db}dB @ {notch_low_hz}-{notch_high_hz}Hz)")
            else:
                logger.debug(f"Notch skipped (invalid range {notch_low_hz}-{notch_high_hz})")
        else:
            logger.debug(f"Notch skipped (gain={notch_gain_db}, heavy_skip={heavy_skip})")

        # Normalize (early params)
        if enable_norm and len(wav_np) > 0:
            if norm_method == 'peak':
                peak = np.max(np.abs(wav_np))
                if peak > 0:
                    wav_np = (wav_np / peak) * 0.95  # Modify nonlocal
                    did_normalize = True  # FIXED: Set flag if success
                    logger.debug(f"Normalize ({norm_method}: peak -1dB)")
            elif norm_method == 'rms':
                rms = np.sqrt(np.mean(wav_np ** 2))
                if rms > 0:
                    target_rms_db = params.get('target_rms_db', CONFIG.get_value('ebu_post_gain_db', -18))
                    target_rms = 10 ** (target_rms_db / 20)
                    wav_np *= target_rms / rms  # Modify nonlocal
                    did_normalize = True  # FIXED: Set flag if success
                    logger.debug(f"Normalize ({norm_method}: RMS {target_rms_db}dB)")

        return wav_np  # Returns processed

    # Async heavy (offloads to thread; ~50% faster than sync librosa)
    wav_np_heavy = await loop.run_in_executor(None, heavy_processing)
    new_len_heavy = len(wav_np_heavy)
    logger.debug(f"Post-heavy: {new_len_heavy/sr:.2f}s (from {new_dur:.2f}s)")

    # Rate (if enabled; executor if stretch heavy)
    enable_resample = params.get('enable_post_resample', CONFIG.get_value('enable_post_resample', False))
    rate = CONFIG.clamp_value('speaking_rate', params.get('speaking_rate', CONFIG.get_value('speaking_rate', 1.0)))
    did_rate = False
    if enable_resample and abs(rate - 1.0) > 0.05 and new_len_heavy > 0:
        def rate_stretch():
            target_len = int(new_len_heavy * rate)
            if target_len > 0:
                stretch_rate = 1.0 / rate
                wav_stretch = librosa.effects.time_stretch(wav_np_heavy, rate=stretch_rate)
                if len(wav_stretch) > target_len:
                    return wav_stretch[:target_len]
                else:
                    pad_len = target_len - len(wav_stretch)
                    return np.pad(wav_stretch, (0, pad_len), 'constant')
            return wav_np_heavy
        wav_np = await loop.run_in_executor(None, rate_stretch)
        did_rate = True
        logger.debug(f"Rate applied ({rate}x; {len(wav_np)/sr:.2f}s)")
    else:
        logger.debug(f"Rate skipped (enable={enable_resample}; rate={rate})")
        wav_np = wav_np_heavy

    # Trail cut (new: Silence after rate; for breathy trails)
    trail_db = params.get('trailing_silence_db', CONFIG.get_value('trailing_silence_db', -45.0))
    trail_db = CONFIG.clamp_value('trailing_silence_db', trail_db)
    did_trail_cut = False
    if abs(trail_db) > 30 and len(wav_np) > sr * 0.1:  # Apply if meaningful
        def cut_trails():
            # Trim end only (keep start breaths)
            threshold = 10 ** (trail_db / 20)
            abs_audio = np.abs(wav_np)
            end_idx = len(wav_np) - np.argmax(abs_audio[::-1] > threshold)
            if end_idx < len(wav_np):
                trimmed = wav_np[:end_idx]
                logger.debug(f"Trail cut ({trail_db}dB): {len(trimmed)/sr:.2f}s (cut { (len(wav_np)-end_idx)/sr :.2f}s trail)")
                return trimmed
            return wav_np
        wav_np = await loop.run_in_executor(None, cut_trails)
        did_trail_cut = len(wav_np) < new_len_heavy  # If shortened
    else:
        logger.debug(f"Trail cut skipped (db={trail_db})")

    # Fade (sync; fast)
    fade_ms = params.get('fade_ms', CONFIG.get_value('fade_ms', None))
    if fade_ms is not None:
        fade_ms = CONFIG.clamp_value('fade_ms', fade_ms)
    did_fade = False
    if fade_ms is not None and fade_ms > 0 and len(wav_np) > sr * 0.05:
        fade_samples = int(sr * (fade_ms / 1000.0))
        if len(wav_np) > 2 * fade_samples:
            fade_in = np.linspace(0.0, 1.0, fade_samples)
            fade_out = np.linspace(1.0, 0.0, fade_samples)
            wav_np[:fade_samples] *= fade_in
            wav_np[-fade_samples:] *= fade_out
            did_fade = True
            logger.debug(f"Fade applied ({fade_ms}ms)")
        else:
            logger.debug(f"Fade skipped (too short)")
    else:
        logger.debug(f"Fade skipped (ms={fade_ms})")

    # Clamp (sync)
    gain_limit = CONFIG.clamp_value('gain_max_limit', params.get('gain_max_limit', CONFIG.get_value('gain_max_limit', 1.0)))
    wav_np = np.clip(wav_np, -gain_limit, gain_limit)
    if gain_limit != 1.0:
        logger.debug(f"Gain clamped to {gain_limit}")

    # Final clip
    wav_np = np.clip(wav_np, -1.0, 1.0)
    final_len = len(wav_np)
    final_dur = final_len / sr
    # FIXED: Use outer-defined vars/flags (all available; no reference error from inner)
    applied = []
    if did_trim: applied.append('trim')
    if did_denoise: applied.append('denoise')  # Flag from heavy
    if did_eq: applied.append('eq')  # Flag from heavy
    if did_notch: applied.append('notch')  # Flag from heavy
    if did_normalize: applied.append('normalize')  # Flag from heavy
    if did_rate: applied.append('rate')
    if did_trail_cut: applied.append('trail_cut')
    if did_fade: applied.append('fade')
    applied_str = ', '.join(applied) if applied else 'none (no-op)'
    logger.debug(f"Post complete for {voice_name}: {final_dur:.2f}s (from {orig_dur:.2f}s; applied: {applied_str}; light={light_mode})")

    # Final empty guard
    min_final_sec = min_dur_sec / 2
    if final_dur < min_final_sec:
        fallback_sec = params.get('fallback_silence_sec', CONFIG.get_value('fallback_silence_sec', 2.0))
        fallback_len = int(sr * fallback_sec)
        if final_len < sr * 0.1:
            wav_np = np.zeros(fallback_len)
            logger.debug(f"Fallback silence {fallback_sec}s")

    return wav_np  # Awaitable now (async func)




