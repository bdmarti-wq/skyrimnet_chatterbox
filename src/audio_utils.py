import asyncio
import os
import time
from typing import Optional
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
from src.config import CONFIG, get_config_value  # Adjusted import (from .config if package)

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
        except:
            logger.error("Torchaudio backend switch failed – audio ops may break")
        return fallback_backend
    except ImportError:
        logger.error("❌ Torchaudio not installed – install via pip/conda")
        return None
    except Exception as e:
        logger.error(f"❌ Backend config failed: {e} – Fallback to {fallback_backend}")
        return fallback_backend


# UTILITY: Simple async denoising and normalization (unused but kept)
async def denoise_and_normalize_in_memory(
        audio: np.ndarray, sr: int, normalize_method: str = "peak",
        noise_floor_db: float | None = -60.0
) -> np.ndarray:
    """Simple in-memory audio denoising and normalization."""
    if noise_floor_db is None:
        logger.debug("Denoise/normalize skipped (noise_floor_db=None)")
        return audio  # No-op

    loop = asyncio.get_running_loop()

    # Simple denoising
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

    # Simple normalization
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
    """Detect artifacts. FIXED: Default from config; accurate log/comp (no '8000'). Keep one version."""
    threshold_hz = threshold_hz or CONFIG.get_value('fuzzy_artifact_threshold_hz', 7000.0)
    # threshold_hz = CONFIG.clamp_value('fuzzy_artifact_threshold_hz', threshold_hz)  # Assume CAP added
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


# Modular Post-Processing Functions
def trim_silence(audio: np.ndarray, threshold_db: float | None = -25.0) -> np.ndarray:
    """Trim leading/trailing silence based on threshold."""
    if threshold_db is None or threshold_db > -99:  # No-op guard (high = trim nothing)
        logger.debug("Trim skipped (threshold_db=None or high)")
        return audio
    threshold = 10 ** (threshold_db / 20)
    abs_audio = np.abs(audio)
    start_idx = np.argmax(abs_audio > threshold)
    end_idx = len(audio) - np.argmax(abs_audio[::-1] > threshold)
    if start_idx < end_idx:
        return audio[start_idx:end_idx]
    return audio

def apply_notch(audio: np.ndarray, sr: int, low_hz: float | None = 8000.0, high_hz: float | None = 11000.0,
                gain_db: float | None = -12.0) -> np.ndarray:
    """Apply bandstop notch filter (e.g., 8-11kHz -12dB). FIXED: None guards before / *."""
    if gain_db is None or gain_db >= 0 or low_hz is None or high_hz is None or low_hz >= high_hz:
        logger.debug("Notch skipped (gain_db=None/>=0 or invalid range)")
        return audio  # No-op
    nyquist = sr / 2.0
    low_norm = max(0.01, min(0.99, low_hz / nyquist))
    high_norm = min(0.99, max(low_norm + 0.01, high_hz / nyquist))
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
    """Apply simple EQ (low/high-pass with gain). FIXED: None guards before / *."""
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

def apply_gain_normalization(audio: np.ndarray, target_max: Optional[float] = None,
                             max_gain_limit: float = 1.0) -> np.ndarray:
    """Apply gain to reach target_max (clamps <= max_gain_limit). FIXED: Guards target_max/None."""
    if target_max is None or target_max <= 0:
        logger.debug("Gain norm skipped (target_max=None or <=0)")
        return audio  # No-op

    current_max = np.max(np.abs(audio))
    if current_max <= 0 or current_max >= target_max:
        return audio  # Already good/no need

    max_gain = min(target_max / current_max, max_gain_limit)
    if max_gain <= 1.0:  # No boost needed
        return audio

    @numba.jit(nopython=True)
    def apply_gain_jit(data, max_gain):
        for i in range(len(data)):
            amplified = data[i] * max_gain
            if amplified > 1.0:
                data[i] = 1.0
            elif amplified < -1.0:
                data[i] = -1.0
            else:
                data[i] = amplified
        return data

    data_copy = audio.copy().astype(np.float64)
    wav_np = apply_gain_jit(data_copy, max_gain)

    logger.debug(f"Gain applied: {max_gain:.2f}x (target={target_max}, current_max={current_max:.3f})")
    return wav_np

def adjust_speaking_rate(audio: np.ndarray, rate: float = 1.0) -> np.ndarray:
    """Adjust speaking rate via time stretching. FIXED: Guard rate/None."""
    if rate is None:
        rate = 1.0  # Fallback
        logger.debug("Rate fallback to 1.0 (was None)")
    if abs(rate - 1.0) <= 0.1:
        logger.debug(f"Speaking rate adjustment skipped (rate={rate} ≈1.0)")
        return audio  # No-op
    stretch_rate = 1.0 / rate
    stretched = librosa.effects.time_stretch(audio, rate=stretch_rate)
    target_length = int(len(audio) * rate)  # int() safe (rate float)
    if len(stretched) > target_length:
        stretched = stretched[:target_length]
    else:
        pad_length = target_length - len(stretched)
        stretched = np.pad(stretched, (0, pad_length), mode='constant')
    return np.clip(stretched, -1.0, 1.0)

def apply_fade(audio: np.ndarray, sr: int, fade_ms: float | None = 20.0) -> np.ndarray:
    """Apply fade-in/out. FIXED: Guard fade_ms/None before * / int()."""
    if fade_ms is None or fade_ms <= 0:
        logger.debug("Fade skipped (fade_ms=None or <=0)")
        return audio  # No-op
    fade_samples = int(sr * (fade_ms / 1000.0))  # fade_ms float/guard above → no None
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

# Helper: Prioritize params → config → default; coerce type with guards
def _get_audio_param(key: str, param_dict: dict, default=None, target_type=float, required=False):
    """Get value: params first, then config (validated), then default. Coerce to target_type."""
    val = param_dict.get(key)
    if val is not None:
        try:
            return target_type(val)  # Coerce (e.g., float('30') → 30.0)
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Invalid audio param '{key}'={val} (type error: {e}); fallback to config/default")
            val = None  # Treat as missing

    # Fallback to config (handles nested like 'audio.trim_threshold_db')
    config_val = get_config_value(key, default=default)  # Your facade; assumes validated/non-None

    if config_val is None and required:
        raise ValueError(f"Required param '{key}' missing from params/config")

    try:
        return target_type(config_val) if config_val is not None else target_type(default)
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid config/default '{key}' (type error: {e}); using raw default {default}")
        return target_type(default)


async def apply_post_processing(wav: torch.Tensor, sr: int, params: dict | None = None) -> np.ndarray:
    """Async post: Trim → Pad → Heavy (denoise/EQ/notch/norm) → Rate → Trail Cut → Fade → Clamp. FIXED: Guards all * ops; frame_factor/None fallback; outer vars safe."""
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

    # FIXED: All params/vars defined early (outer; before heavy; None guards)
    min_dur_sec =  _get_audio_param('min_post_duration_sec', params, 0.05)
    min_samples = int(sr * min_dur_sec)
    light_mode = orig_len < min_samples
    if light_mode:
        logger.debug(f"Short input < {min_dur_sec}s – light post (skip heavy)")

    # Trim params (FIXED: Guard frame_factor/None before *)
    trim_db = _get_audio_param('trim_threshold_db', params, -30.0)
    hop = _get_audio_param('hop_length', params, 256 ) # Guard None
    hop = int(hop)  # Ensure int
    frame_factor = _get_audio_param('trim_frame_length_factor', params, 4)
    frame_factor = int(frame_factor) if frame_factor is not None else 4  # FIXED: Fallback int on None
    frame_length = hop * frame_factor  # Now safe: both int
    n_fft_trim = _get_audio_param('max_n_fft_for_trim', params, 2048)
    n_fft_trim = min(int(n_fft_trim), orig_len)  # Ensure int/guard
    did_trim = False

    # Pad params (FIXED: Guard base_pad_sec/None before *)
    enable_audio_pad = params.get('enable_audio_padding', CONFIG.get_value('enable_audio_padding', True))
    base_pad_sec = params.get('base_audio_pad_sec', params.get('post_pad_sec', CONFIG.get_value('base_audio_pad_sec', 0.15)) or 0.15)
    base_pad_sec = float(base_pad_sec) if base_pad_sec is not None else 0.15  # Guard
    multiplier = _get_audio_param('tiny_audio_pad_multiplier', params, 2.0)
    multiplier = float(multiplier) if multiplier is not None else 2.0
    tiny_threshold = _get_audio_param('tiny_threshold_sec', params, 0.5)
    tiny_threshold = float(tiny_threshold) if tiny_threshold is not None else 0.5
    did_pad = False
    is_tiny = False

    # Heavy params (FIXED: Guards on None)
    noise_floor_db = _get_audio_param('noise_floor_db', params, -60.0)
    min_denoise_samples = _get_audio_param('min_samples_for_denoise', params, 100)
    n_fft_denoise = _get_audio_param('n_fft_denoise', params, 1024)
    n_fft_denoise = min(int(n_fft_denoise), orig_len * 2)  # Safe * (orig_len int)
    eq_gain_db = _get_audio_param('eq_gain_db', params, 0.0)
    eq_cutoff_hz = _get_audio_param('eq_cutoff_hz', params, 3000)
    notch_gain_db = _get_audio_param('notch_gain_db', params, 0)

    notch_low_hz = _get_audio_param('notch_low_hz', params, 8000)
    notch_high_hz = _get_audio_param('notch_high_hz', params, 11000)
    norm_method = _get_audio_param('normalize_method', params,'peak')
    enable_norm = _get_audio_param('enable_denoise_normalize', params, False)
    enable_denoise = _get_audio_param('enable_denoising', params, False)
    enable_resample = _get_audio_param('enable_post_resample', params, False)
    rate = _get_audio_param('speaking_rate', params, 1.0)
    trail_db = _get_audio_param('trailing_silence_db', params, -45.0)
    fade_ms = _get_audio_param('fade_ms', params, 0)
    gain_limit = _get_audio_param('gain_max_limit', params, 1.0)


    # Trim (sync; fast) – FIXED: Now frame_length safe
    if trim_db is not None and abs(trim_db) > 5 and not light_mode:
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

    # Gated pad (after trim; FIXED: int() guards on *)
    if not enable_audio_pad:
        logger.debug("Audio padding skipped (enable=False)")
    else:
        is_tiny = orig_dur < tiny_threshold
        if is_tiny:
            base_pad_sec *= multiplier  # float * float safe
            logger.debug(f"Tiny audio ({orig_dur:.2f}s < {tiny_threshold}s) – pad x{multiplier}: {base_pad_sec}s/side")
        if base_pad_sec > 0 and len(wav_np) > 0:
            # FIXED: Guard before int(*)
            pad_sec_total = base_pad_sec * 2  # Side pad *2
            pad_len = int(sr * pad_sec_total) if pad_sec_total is not None else 0
            silence_side_len = int(sr * base_pad_sec) if base_pad_sec is not None else 0
            if pad_len > 0:
                silence = np.zeros(silence_side_len, dtype=wav_np.dtype)
                wav_np = np.concatenate([silence, wav_np, silence])
                did_pad = True
                logger.debug(f"Pad applied: {base_pad_sec}s silence each side (total {pad_len} samples; tiny_extra={is_tiny})")
        else:
            logger.debug(f"Pad skipped (sec={base_pad_sec}; len={len(wav_np)})")

    # new_len early (post-trim/pad)
    new_len = len(wav_np)
    new_dur = new_len / sr
    heavy_skip = new_len < min_samples
    if heavy_skip:
        logger.debug("Post-trim/pad short – skip heavy")
    elif did_pad:
        logger.debug(f"Post-pad length: {new_len} samples ({new_dur:.2f}s)")

    # Flags
    did_denoise = did_eq = did_notch = did_normalize = did_rate = did_trail_cut = did_fade = False

    # Heavy steps in executor (FIXED: All outer, no new Nones)
    def heavy_processing(params: dict):
        from .config import get_config_value
        nonlocal wav_np, did_denoise, did_eq, did_notch, did_normalize

        # Denoise (gated)
        if enable_denoise and not heavy_skip and new_len >= min_denoise_samples:
            try:
                # Pre: High-pass (FIXED: Guard hz/None before /)
                highpass_hz = _get_audio_param('denoise_highpass_hz', params, 80)
                if highpass_hz > 0:
                    nyquist = sr / 2.0
                    highpass_norm = highpass_hz / nyquist
                    sos_hp = butter(4, highpass_norm, btype='high', output='sos')
                    wav_np = sosfilt(sos_hp, wav_np)
                    logger.debug(f"Denoise pre-highpass: {highpass_hz}Hz")

                # Spectral gating
                stft = librosa.stft(wav_np, n_fft=n_fft_denoise, hop_length=hop)  # hop int safe
                mag, phase = np.abs(stft), np.angle(stft)

                freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft_denoise)
                target_low = _get_audio_param('denoise_target_band_low', params,5000)
                target_high = _get_audio_param('denoise_target_band_high', params, 12000)
                band_mask = (freqs >= target_low) & (freqs <= target_high)
                ksize = _get_audio_param('denoise_median_ksize', params,3)
                ksize = int(ksize)  # Ensure int
                for frame in range(mag.shape[1]):
                    high_mag = mag[band_mask, frame]
                    if len(high_mag) > ksize:
                        median_high = np.median(high_mag)
                        mag[band_mask, frame] = np.clip(high_mag, 0, median_high * 0.5)  # * 0.5 float safe

                clean_stft = mag * np.exp(1j * phase)
                wav_np = librosa.istft(clean_stft, hop_length=hop, length=new_len)
                did_denoise = True
                logger.debug(f"Denoise gating applied (band {target_low}-{target_high}Hz, k={ksize}, n_fft={n_fft_denoise})")
            except Exception as denoise_e:
                logger.warning(f"Denoise failed ({new_len} samples): {denoise_e} – skip")
        else:
            logger.debug(f"Denoise skipped (enable={enable_denoise}; skip={heavy_skip})")

        # EQ (safe, no new *)
        if eq_gain_db != 0.0 and not heavy_skip and len(wav_np) > 0:
            nyquist = sr / 2.0
            cutoff_norm = min(1.0, max(0.01, eq_cutoff_hz / nyquist))
            sos = butter(4, cutoff_norm, btype='lowpass' if eq_gain_db < 0 else 'highpass', output='sos')
            wav_np = sosfilt(sos, wav_np)
            did_eq = True
            logger.debug(f"EQ applied ({eq_gain_db}dB {'low' if eq_gain_db < 0 else 'high'}-pass @ {eq_cutoff_hz}Hz)")
        else:
            logger.debug(f"EQ skipped (gain={eq_gain_db}, heavy_skip={heavy_skip})")

        # Notch (safe, guards in function)
        if notch_gain_db < 0 and not heavy_skip and len(wav_np) > 0:
            wav_np = apply_notch(wav_np, sr, notch_low_hz, notch_high_hz, notch_gain_db)
            did_notch = True  # Assume applied if no except in func
        else:
            logger.debug(f"Notch skipped (gain={notch_gain_db}, heavy_skip={heavy_skip})")

        # Normalize
        if enable_norm and len(wav_np) > 0:
            if norm_method == 'peak':
                peak = np.max(np.abs(wav_np))
                if peak > 0:
                    wav_np = (wav_np / peak) * 0.95
                    did_normalize = True
                    logger.debug(f"Normalize ({norm_method}: peak -1dB)")
            elif norm_method == 'rms':
                rms = np.sqrt(np.mean(wav_np ** 2))
                if rms > 0:
                    target_rms_db = params.get('ebu_post_gain_db', CONFIG.get_value('ebu_post_gain_db', -18)) or -18
                    target_rms = 10 ** (target_rms_db / 20)
                    wav_np *= target_rms / rms
                    did_normalize = True
                    logger.debug(f"Normalize ({norm_method}: RMS {target_rms_db}dB)")

        return wav_np

    # Async heavy
    wav_np_heavy = await loop.run_in_executor(None, heavy_processing)
    new_len_heavy = len(wav_np_heavy)
    logger.debug(f"Post-heavy: {new_len_heavy/sr:.2f}s (from {new_dur:.2f}s)")

    # Rate (FIXED: Guard rate before *)
    speaking_rate = _get_audio_param('speaking_rate', params, 1.0)
    if enable_resample and abs(speaking_rate - 1.0) > 0.05 and new_len_heavy > 0:
        def rate_stretch(rate: float):
            nonlocal wav_np
            speaking_rate = float(rate) if rate is not None else 1.0  # Guard
            target_len = int(new_len_heavy * speaking_rate)
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

    # Trail cut (gated)
    if abs(trail_db) > 30 and len(wav_np) > sr * 0.1:
        def cut_trails():
            nonlocal wav_np
            threshold = 10 ** (trail_db / 20)
            abs_audio = np.abs(wav_np)
            end_idx = len(wav_np) - np.argmax(abs_audio[::-1] > threshold)
            if end_idx < len(wav_np):
                trimmed = wav_np[:end_idx]
                logger.debug(f"Trail cut ({trail_db}dB): {len(trimmed)/sr:.2f}s (cut {(len(wav_np)-end_idx)/sr :.2f}s trail)")
                return trimmed
            return wav_np
        if is_tiny:
            logger.debug("Trail cut skipped for tiny audio (preserve sustain)")
        else:
            wav_np = await loop.run_in_executor(None, cut_trails)
            did_trail_cut = len(wav_np) < new_len_heavy
    else:
        logger.debug(f"Trail cut skipped (db={trail_db})")

    # Fade (safe from guards)
    wav_np = apply_fade(wav_np, sr, fade_ms)
    if fade_ms > 0:
        did_fade = True

    # Clamp (FIXED: gain_limit guard)
    gain_limit = float(gain_limit) if gain_limit is not None else 1.0
    wav_np = np.clip(wav_np, -gain_limit, gain_limit)
    if gain_limit != 1.0:
        logger.debug(f"Gain clamped to {gain_limit}")

    # Final clip
    wav_np = np.clip(wav_np, -1.0, 1.0)
    final_len = len(wav_np)
    final_dur = final_len / sr

    # Flags summary
    applied = []
    if did_trim: applied.append('trim')
    if did_denoise: applied.append('denoise')
    if did_eq: applied.append('eq')
    if did_notch: applied.append('notch')
    if did_normalize: applied.append('normalize')
    if did_rate: applied.append('rate')
    if did_trail_cut: applied.append('trail_cut')
    if did_fade: applied.append('fade')
    applied_str = ', '.join(applied) if applied else 'none (no-op)'
    logger.debug(f"Post complete for {voice_name}: {final_dur:.2f}s (from {orig_dur:.2f}s; applied: {applied_str}; light={light_mode})")

    # Final empty guard
    min_final_sec = min_dur_sec / 2
    if final_dur < min_final_sec:
        fallback_sec = params.get('fallback_silence_sec', CONFIG.get_value('fallback_silence_sec', 2.0)) or 2.0
        fallback_len = int(sr * fallback_sec)
        if final_len < sr * 0.1:
            wav_np = np.zeros(fallback_len)
            logger.debug(f"Fallback silence {fallback_sec}s")

    return wav_np