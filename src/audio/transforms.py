"""
Stateless numpy-based audio post-processing transforms.

These were extracted from PostProcessingPhase to improve testability and
maintainability. Functions here operate on numpy arrays and simple params.

Moved from src/audio_post/transforms.py to src/audio/transforms.py
"""
from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
from scipy.signal import sosfilt, butter, sosfiltfilt
from loguru import logger
import librosa
import torch
import torchaudio

from .post_config import PostParams, normalize_post_params


def short_padding_trim_head(y: np.ndarray, sr: int, params: Dict[str, Any]) -> np.ndarray:
    """Remove leading short-padding up to the first voiced island, keeping a small pre-roll and tail pad."""
    if y is None or not isinstance(y, np.ndarray) or y.size == 0:
        return y

    try:
        head_ms = float(params.get('short_padding_head_sil_ms', 60.0) or 60.0)
        tail_ms = float(params.get('short_padding_tail_sil_ms', 60.0) or 60.0)
        top_db = float(params.get('short_padding_split_db', 40.0) or 40.0)
        hop = int(params.get('short_padding_hop_length', 256) or 256)
        frame = int(params.get('short_padding_frame_length', 1024) or 1024)
    except Exception:
        head_ms, tail_ms, top_db, hop, frame = 60.0, 60.0, 40.0, 256, 1024

    try:
        intervals = librosa.effects.split(y, top_db=top_db, frame_length=frame, hop_length=hop)
        if intervals is None or len(intervals) == 0:
            return y

        head_pad = int(sr * (head_ms / 1000.0))
        tail_pad = int(sr * (tail_ms / 1000.0))

        near_db_offset = float(params.get('short_trim_near_db_offset', 6.0) or 6.0)
        base_sil_val = params.get('short_trim_silence_db', params.get('short_repeat_silence_db', -45.0))
        base_sil_db = float(base_sil_val if base_sil_val is not None else -45.0)
        near_db = base_sil_db + near_db_offset
        near_thr = 10 ** (near_db / 20.0)
        win = max(64, int(sr * 0.010))
        hop_near = max(32, int(sr * 0.005))
        if len(y) > win:
            num_frames = 1 + (len(y) - win) // hop_near
            starts = np.arange(num_frames, dtype=np.int64) * hop_near
            rms = np.empty(num_frames, dtype=np.float32)
            for i in range(num_frames):
                seg = y[starts[i]:starts[i] + win]
                rms[i] = float(np.sqrt(np.mean(seg * seg)) + 1e-12)
            near_silent = rms < near_thr

            if len(intervals) >= 1:
                first_end = int(intervals[0][1])
                first_end_frame = min(num_frames - 1, max(0, first_end // hop_near))
                min_near_ms = float(params.get('short_trim_min_near_silence_ms', 40.0) or 40.0)
                min_near_frames = max(1, int((min_near_ms / 1000.0) * sr / hop_near))

                j = first_end_frame
                cut_frame_after_near = None
                while j < num_frames:
                    if near_silent[j]:
                        k = j
                        while k < num_frames and near_silent[k]:
                            k += 1
                        run_len = k - j
                        if run_len >= min_near_frames:
                            cut_frame_after_near = k
                            break
                        j = k
                    else:
                        j += 1

                if cut_frame_after_near is not None:
                    start = max(0, int(starts[min(cut_frame_after_near, len(starts) - 1)]) - head_pad)
                    end = min(len(y), int(intervals[-1][1]) + tail_pad)
                    logger.debug(
                        f"short_padding_trim_head: near-silence cut start={start/sr:.3f}s end={end/sr:.3f}s")
                    return y[start:end]

        if len(intervals) >= 2:
            best_gap = -1
            best_after_idx = None
            for i in range(len(intervals) - 1):
                g = int(intervals[i + 1][0]) - int(intervals[i][1])
                if g > best_gap:
                    best_gap = g
                    best_after_idx = i + 1
            if best_after_idx is not None:
                start = max(0, int(intervals[best_after_idx][0]) - head_pad)
                end = min(len(y), int(intervals[-1][1]) + tail_pad)
                return y[start:end]
    except Exception as e:
        logger.debug(f"short_padding_trim_head failed: {e}")
    return y


def short_trim_padding(y: np.ndarray, sr: int, params: Dict[str, Any]) -> np.ndarray:
    """A higher-level helper that first trims head padding using short_padding_trim_head."""
    try:
        return short_padding_trim_head(y, sr, params)
    except Exception:
        return y


def trim_trailing_artifacts(audio: np.ndarray, sr: int,
                            tail_threshold_db: Optional[float] = None,
                            tail_fraction: Optional[float] = None) -> np.ndarray:
    """Trim low-level tail based on global RMS and dB threshold.

    None parameters mean skip (no-op).
    """
    if audio.size == 0 or tail_threshold_db is None or tail_fraction is None:
        return audio
    try:
        rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
        thr = 10 ** (float(tail_threshold_db) / 20.0)
        tf = max(0.0, min(1.0, float(tail_fraction)))
        min_len = max(1, int(len(audio) * (1.0 - tf)))
        last_idx = len(audio) - 1
        # Vectorized search from the end: find last index above threshold
        segment = audio[min_len:]
        idx = np.where(np.abs(segment) > thr * rms)[0]
        if idx.size > 0:
            last_idx = min_len + int(idx[-1])
        else:
            last_idx = min_len
        return audio[:last_idx + 1]
    except Exception:
        return audio


def gate_trailing_phantoms(audio: np.ndarray, sr: int,
                           gate_threshold: Optional[float] = None,
                           tail_fraction: Optional[float] = None,
                           smooth_ms: float = 10.0) -> np.ndarray:
    """Soft-gate the tail below threshold with a short smoothing envelope.

    None parameters mean skip. Uses a raised-cosine fade to reduce clicks.
    """
    if audio.size == 0 or gate_threshold is None or tail_fraction is None:
        return audio
    try:
        gate_thr = max(0.0, float(gate_threshold))
        tf = max(0.0, min(1.0, float(tail_fraction)))
        end = len(audio)
        start = int(end * (1.0 - tf))
        tail = audio[start:end]
        mask = (np.abs(tail) < gate_thr).astype(np.float32)
        if mask.size == 0:
            return audio
        # Smooth mask with a short fade on both sides
        smooth = max(1, int(sr * (smooth_ms / 1000.0)))
        if 2 * smooth < mask.size:
            window = np.ones_like(mask)
            ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, smooth))
            window[:smooth] = ramp
            window[-smooth:] = ramp[::-1]
            mask = mask * window
        # Apply gating by blending towards zero
        audio[start:end] = tail * (1.0 - mask)  # where mask==1 → zero
        return audio
    except Exception:
        return audio


def suppress_tail_artifacts(audio: np.ndarray, sr: int,
                            tail_fraction: Optional[float] = None,
                            low_hz: Optional[float] = None,
                            high_hz: Optional[float] = None,
                            strength: Optional[float] = None) -> np.ndarray:
    if audio.size == 0 or tail_fraction is None or low_hz is None or high_hz is None or strength is None:
        return audio
    try:
        low = float(low_hz) / (sr / 2.0)
        high = float(high_hz) / (sr / 2.0)
        sos = butter(4, [low, high], btype='band', output='sos')
        tail_len = int(len(audio) * float(max(0.0, min(1.0, tail_fraction))))
        if tail_len <= 0:
            return audio
        tail = audio[-tail_len:]
        # zero-phase filtering to avoid phase lag
        try:
            filtered = sosfiltfilt(sos, tail)
        except Exception:
            filtered = sosfilt(sos, tail)
        alpha = max(0.0, min(1.0, float(strength)))
        audio[-tail_len:] = (1.0 - alpha) * tail + alpha * (tail - filtered)
        return audio
    except Exception as e:
        logger.debug(f"suppress_tail_artifacts failed: {e}")
        return audio


def reverse_tail_suppress(audio: np.ndarray, sr: int,
                          tail_sec: Optional[float] = None,
                          onset_threshold: Optional[float] = None,
                          fade_ms: float = 10.0) -> np.ndarray:
    if audio.size == 0 or tail_sec is None or onset_threshold is None:
        return audio
    try:
        n = int(sr * float(max(0.0, tail_sec)))
        n = min(n, len(audio))
        if n <= 0:
            return audio
        tail = audio[-n:]
        rev = tail[::-1].copy()
        thr = float(onset_threshold)
        over = np.where(np.abs(rev) > thr)[0]
        if over.size > 0:
            idx = int(over[0])
            cut = len(audio) - idx
            out = audio[:cut].copy()
            # apply short fade-out at the end to avoid click
            fm = int(sr * (fade_ms / 1000.0))
            fm = min(fm, out.size)
            if fm > 1:
                fade = np.linspace(1.0, 0.0, fm)
                out[-fm:] *= fade
            return out
        return audio
    except Exception as e:
        logger.debug(f"reverse_tail_suppress failed: {e}")
        return audio


def apply_notch(audio: np.ndarray, sr: int,
                low_hz: Optional[float] = None,
                high_hz: Optional[float] = None,
                gain_db: Optional[float] = None) -> np.ndarray:
    if audio.size == 0 or gain_db is None or gain_db >= 0 or low_hz is None or high_hz is None:
        return audio
    try:
        w1 = float(low_hz) / (sr / 2.0)
        w2 = float(high_hz) / (sr / 2.0)
        sos = butter(4, [w1, w2], btype='band', output='sos')
        try:
            out = sosfiltfilt(sos, audio)
        except Exception:
            out = sosfilt(sos, audio)
        g = 10 ** (float(abs(gain_db)) / 20.0)
        return audio - g * out
    except Exception as e:
        logger.debug(f"apply_notch failed: {e}")
        return audio


def apply_eq(audio: np.ndarray, sr: int,
             gain_db: Optional[float] = None,
             cutoff_hz: Optional[float] = None) -> np.ndarray:
    if audio.size == 0 or gain_db is None or gain_db == 0 or cutoff_hz is None:
        return audio
    try:
        w = float(cutoff_hz) / (sr / 2.0)
        sos = butter(4, w, btype='low', output='sos') if gain_db > 0 else butter(4, w, btype='high', output='sos')
        try:
            out = sosfiltfilt(sos, audio)
        except Exception:
            out = sosfilt(sos, audio)
        g = 10 ** (abs(float(gain_db)) / 20.0)
        if gain_db > 0:
            return audio + g * out
        else:
            return audio - g * out
    except Exception as e:
        logger.debug(f"apply_eq failed: {e}")
        return audio


def adjust_speaking_rate(audio: np.ndarray, rate: Optional[float] = None) -> np.ndarray:
    """Time-stretch using torchaudio TimeStretch (phase vocoder). None/≈1.0 → no-op.

    Fallback to librosa for robustness if torchaudio path fails.
    """
    if audio.size == 0 or rate is None or abs(float(rate) - 1.0) < 1e-3:
        return audio
    try:
        rate_f = float(rate)
        rate_f = max(0.5, min(1.5, rate_f))
        # Torch STFT
        n_fft = 1024
        hop = 256
        win = torch.hann_window(n_fft)
        x = torch.from_numpy(audio.astype(np.float32))
        X = torch.stft(x, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
        # torchaudio TimeStretch expects complex with shape (..., freq, time)
        ts = torchaudio.transforms.TimeStretch(hop_length=hop, n_freq=X.size(0))
        Y = ts(X.unsqueeze(0), rate_f).squeeze(0)
        y = torch.istft(Y, n_fft=n_fft, hop_length=hop, window=win, length=audio.size)
        return y.numpy().astype(np.float32)
    except Exception as e:
        logger.debug(f"adjust_speaking_rate (torchaudio) failed: {e}; falling back to librosa")
        try:
            out = librosa.effects.time_stretch(audio.astype(np.float32), rate=float(rate))
            return out.astype(np.float32)
        except Exception as ee:
            logger.debug(f"adjust_speaking_rate (librosa) failed: {ee}")
            return audio


def apply_fade(audio: np.ndarray, sr: int, fade_ms: float | None = 20.0) -> np.ndarray:
    if audio.size == 0 or fade_ms is None or float(fade_ms) <= 0:
        return audio
    try:
        fade_samples = int(sr * (float(fade_ms) / 1000.0))
    except Exception:
        fade_samples = int(sr * 0.02)
    adaptive_fade = min(fade_samples, int(audio.size * 0.05))
    if adaptive_fade <= 0:
        return audio
    fade_in = np.linspace(0.0, 1.0, adaptive_fade)
    audio[:adaptive_fade] *= fade_in
    fade_out = np.linspace(1.0, 0.0, adaptive_fade)
    audio[-adaptive_fade:] *= fade_out
    logger.debug("Fade applied")
    return audio


def apply_post_processing(wav_np: np.ndarray, sr: int, params: dict | PostParams | None = None, text: str = '') -> np.ndarray:
    """Compose transforms using None-as-noop semantics via PostParams.

    - If params is None or enable flag is False → passthrough.
    - Only transforms with non-None controller params are applied.
    """
    pp = normalize_post_params(params)
    if not pp.enable_post_processing:
        logger.debug("Post disabled – no-op passthrough")
        return wav_np

    voice_name = pp.voice_name or 'unknown'
    logger.debug(f"Post params for {voice_name} (None=skip)")

    if wav_np.size == 0:
        min_sec = pp.min_post_duration_sec if pp.min_post_duration_sec is not None else 0.5
        min_samples = int(sr * float(min_sec))
        return np.zeros(min_samples, dtype=np.float32)

    # Light tail fixes for short vocalizes
    text = text or ''
    is_vocalize = len(text.strip()) <= 3 and text.lower() in ['ah', 'oh', 'aah', 'mmm', 'uh', 'mmh', 'eh']
    light_mode = is_vocalize or len(text) < 10
    if light_mode and pp.trailing_silence_db is not None:
        # derive tail_fraction preference if provided via pp.tail_fraction
        tf = pp.tail_fraction
        if tf is None and pp.tail_suppress_sec is not None:
            dur = max(1, wav_np.size)
            tf = min(0.9, max(0.0, (pp.tail_suppress_sec * sr) / dur))
        wav_np = trim_trailing_artifacts(wav_np, sr, pp.trailing_silence_db, tf if tf is not None else 0.2)

    # Heavy transforms (apply only when their params are provided)
    if pp.gate_threshold is not None:
        tf = pp.tail_fraction
        if tf is None and pp.tail_suppress_sec is not None:
            dur = max(1, wav_np.size)
            tf = min(0.9, max(0.0, (pp.tail_suppress_sec * sr) / dur))
        wav_np = gate_trailing_phantoms(wav_np, sr, pp.gate_threshold, tf if tf is not None else 0.3)

    wav_np = suppress_tail_artifacts(wav_np, sr,
                                     tail_fraction=pp.tail_fraction if pp.tail_fraction is not None else (
                                         min(0.9, max(0.0, (pp.tail_suppress_sec * sr) / max(1, wav_np.size)))
                                         if pp.tail_suppress_sec is not None else None
                                     ),
                                     low_hz=pp.tail_suppress_low_hz,
                                     high_hz=pp.tail_suppress_high_hz,
                                     strength=pp.tail_suppress_strength)

    wav_np = reverse_tail_suppress(wav_np, sr, tail_sec=pp.tail_suppress_sec, onset_threshold=pp.tail_onset_threshold)

    wav_np = apply_notch(wav_np, sr, pp.notch_low_hz, pp.notch_high_hz, pp.notch_gain_db)
    wav_np = apply_eq(wav_np, sr, pp.eq_gain_db, pp.eq_cutoff_hz)

    wav_np = adjust_speaking_rate(wav_np, pp.speaking_rate)

    try:
        wav_np = apply_fade(wav_np, sr, pp.fade_ms)
    except Exception as e:
        logger.debug(f"Fade failed/skipped: {e}")

    if pp.gain_max_limit is not None and pp.gain_max_limit > 0:
        peak = float(np.max(np.abs(wav_np))) if wav_np.size > 0 else 0.0
        if peak > 0 and peak > pp.gain_max_limit:
            scale = pp.gain_max_limit / peak
            wav_np = wav_np * scale
            logger.debug(f"Limiter scale {scale:.3f} to enforce peak≤{pp.gain_max_limit:.3f}")
        ceiling = min(1.0, float(pp.gain_max_limit))
        wav_np = np.clip(wav_np, -ceiling, ceiling)
    else:
        wav_np = np.clip(wav_np, -1.0, 1.0)

    return wav_np
