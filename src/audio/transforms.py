"""
Stateless numpy-based audio post-processing transforms.

These were extracted from PostProcessingPhase to improve testability and
maintainability. Functions here operate on numpy arrays and simple params.

Moved from src/audio_post/transforms.py to src/audio/transforms.py
"""
from __future__ import annotations

from typing import Dict, Any
import numpy as np
from scipy.signal import sosfilt, butter
from loguru import logger
import librosa


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


def trim_trailing_artifacts(audio: np.ndarray, sr: int, tail_threshold_db: float = -45.0, tail_fraction: float = 0.2) -> np.ndarray:
    if audio.size == 0:
        return audio
    try:
        rms = np.sqrt(np.mean(audio ** 2) + 1e-12)
        thr = 10 ** (tail_threshold_db / 20.0)
        min_len = max(1, int(len(audio) * (1.0 - tail_fraction)))
        last_idx = len(audio) - 1
        for i in range(len(audio) - 1, min_len, -1):
            if abs(audio[i]) > thr * rms:
                last_idx = i
                break
        return audio[:last_idx + 1]
    except Exception:
        return audio


def gate_trailing_phantoms(audio: np.ndarray, sr: int, gate_threshold: float = 0.05, tail_fraction: float = 0.3) -> np.ndarray:
    if audio.size == 0:
        return audio
    try:
        gate_thr = max(0.0, float(gate_threshold))
    except Exception:
        gate_thr = 0.05
    end = len(audio)
    start = int(end * (1.0 - tail_fraction))
    audio[start:end] = np.where(np.abs(audio[start:end]) < gate_thr, 0.0, audio[start:end])
    return audio


def suppress_tail_artifacts(audio: np.ndarray, sr: int, tail_fraction: float = 0.25, low_hz: float = 2000.0, high_hz: float = 4000.0, strength: float = 0.6) -> np.ndarray:
    if audio.size == 0:
        return audio
    try:
        low = float(low_hz) / (sr / 2.0)
        high = float(high_hz) / (sr / 2.0)
        sos = butter(2, [low, high], btype='band', output='sos')
        tail_len = int(len(audio) * float(tail_fraction))
        if tail_len <= 0:
            return audio
        tail = audio[-tail_len:]
        filtered = sosfilt(sos, tail)
        alpha = max(0.0, min(1.0, float(strength)))
        audio[-tail_len:] = (1.0 - alpha) * tail + alpha * (tail - filtered)
        return audio
    except Exception as e:
        logger.debug(f"suppress_tail_artifacts failed: {e}")
        return audio


def reverse_tail_suppress(audio: np.ndarray, sr: int, tail_sec: float = 0.2, onset_threshold: float = 0.3) -> np.ndarray:
    if audio.size == 0:
        return audio
    try:
        n = int(sr * float(tail_sec))
        n = min(n, len(audio))
        tail = audio[-n:]
        rev = tail[::-1].copy()
        thr = float(onset_threshold)
        idx = np.argmax(np.abs(rev) > thr)
        if idx > 0:
            cut = len(audio) - idx
            return audio[:cut]
        return audio
    except Exception as e:
        logger.debug(f"reverse_tail_suppress failed: {e}")
        return audio


def apply_notch(audio: np.ndarray, sr: int, low_hz: float = 8000.0, high_hz: float = 11000.0, gain_db: float = -12.0) -> np.ndarray:
    if audio.size == 0:
        return audio
    try:
        w1 = float(low_hz) / (sr / 2.0)
        w2 = float(high_hz) / (sr / 2.0)
        sos = butter(2, [w1, w2], btype='band', output='sos')
        out = sosfilt(sos, audio)
        g = 10 ** (float(gain_db) / 20.0)
        return audio - g * out
    except Exception as e:
        logger.debug(f"apply_notch failed: {e}")
        return audio


def apply_eq(audio: np.ndarray, sr: int, gain_db: float = 0.0, cutoff_hz: float = 3000.0) -> np.ndarray:
    if audio.size == 0 or gain_db == 0.0:
        return audio
    try:
        w = float(cutoff_hz) / (sr / 2.0)
        sos = butter(2, w, btype='low', output='sos') if gain_db > 0 else butter(2, w, btype='high', output='sos')
        out = sosfilt(sos, audio)
        g = 10 ** (abs(float(gain_db)) / 20.0)
        if gain_db > 0:
            return audio + g * out
        else:
            return audio - g * out
    except Exception as e:
        logger.debug(f"apply_eq failed: {e}")
        return audio


def adjust_speaking_rate(audio: np.ndarray, rate: float = 1.0) -> np.ndarray:
    if audio.size == 0 or rate == 1.0:
        return audio
    try:
        rate = max(0.5, min(1.5, float(rate)))
        out = librosa.effects.time_stretch(audio.astype(np.float32), rate=rate)
        return out.astype(np.float32)
    except Exception as e:
        logger.debug(f"adjust_speaking_rate failed: {e}")
        return audio


def apply_fade(audio: np.ndarray, sr: int, fade_ms: float | None = 20.0) -> np.ndarray:
    if audio.size == 0 or fade_ms is None:
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


def apply_post_processing(wav_np: np.ndarray, sr: int, params: dict | None = None, text: str = '') -> np.ndarray:
    """Top-level helper that composes the above transforms using params dict."""
    if params is None:
        logger.debug("Post params missing/empty – no-op")
        return wav_np
    enable_post = params.get('enable_post_processing', False)
    if not enable_post:
        logger.debug("Post disabled or not enabled – no-op")
        return wav_np

    voice_name = params.get('voice_name', 'unknown')
    non_none_params = {k: v for k, v in params.items() if v is not None}
    logger.debug(f"Post params for {voice_name}: {non_none_params}")

    text = text or ''
    is_vocalize = len(text.strip()) <= 3 and text.lower() in ['ah', 'oh', 'aah', 'mmm', 'uh', 'mmh', 'eh']
    light_mode = is_vocalize or len(text) < 10
    if light_mode:
        logger.debug(f"Vocalize/short '{text}' – applying light tail fixes PLUS full heavy post")

    if wav_np.size == 0:
        min_dur_sec = params.get('min_post_duration_sec', 0.5)
        min_samples = int(sr * min_dur_sec)
        silence = np.zeros(min_samples, dtype=np.float32)
        logger.warning("Empty input – raw fallback")
        return silence

    # Light tail-focused fixes
    tail_threshold_db = params.get('trailing_silence_db', -45.0)
    if light_mode:
        wav_np = trim_trailing_artifacts(wav_np, sr, tail_threshold_db)
        logger.debug("Light post: Tail fixes for phantoms")

    # Heavy transforms
    gate_threshold = params.get('gate_threshold', 0.05)
    wav_np = gate_trailing_phantoms(wav_np, sr, gate_threshold)
    tail_fraction_val = params.get('tail_suppress_sec', 0.2)
    try:
        tail_fractions = 0.25 if tail_fraction_val is None else float(tail_fraction_val)
    except Exception:
        tail_fractions = 0.25
    low_hz = params.get('tail_suppress_low_hz', 2000)
    high_hz = params.get('tail_suppress_high_hz', 4000)
    strength = params.get('tail_suppress_strength', 0.6)
    wav_np = suppress_tail_artifacts(wav_np, sr, tail_fraction=tail_fractions, low_hz=low_hz, high_hz=high_hz, strength=strength)
    onset_thresh = params.get('tail_onset_threshold', 0.3)
    wav_np = reverse_tail_suppress(wav_np, sr, tail_sec=0.2, onset_threshold=onset_thresh)

    notch_gain = params.get('notch_gain_db', 0)
    notch_low = params.get('notch_low_hz', 8000)
    notch_high = params.get('notch_high_hz', 11000)
    if notch_gain < 0:
        wav_np = apply_notch(wav_np, sr, notch_low, notch_high, notch_gain)

    eq_gain_db = params.get('eq_gain_db', 0.0)
    eq_cutoff = params.get('eq_cutoff_hz', 3000)
    if eq_gain_db != 0.0:
        wav_np = apply_eq(wav_np, sr, eq_gain_db, eq_cutoff)

    rate = params.get('speaking_rate', 1.0)
    if abs(rate - 1.0) > 0.05:
        wav_np = adjust_speaking_rate(wav_np, rate)

    fade_ms = params.get('fade_ms', None)
    try:
        wav_np = apply_fade(wav_np, sr, fade_ms)
    except Exception as e:
        logger.debug(f"Fade failed/skipped: {e}")

    gain_max_limit = params.get('gain_max_limit', None)
    if isinstance(gain_max_limit, (int, float)) and gain_max_limit is not None and gain_max_limit > 0:
        peak = float(np.max(np.abs(wav_np))) if wav_np.size > 0 else 0.0
        if peak > 0 and peak > gain_max_limit:
            scale = gain_max_limit / peak
            wav_np = wav_np * scale
            logger.debug(f"Applied limiter scale {scale:.3f} to enforce peak≤{gain_max_limit:.3f}")
        ceiling = min(1.0, float(gain_max_limit))
        wav_np = np.clip(wav_np, -ceiling, ceiling)
    else:
        wav_np = np.clip(wav_np, -1.0, 1.0)

    return wav_np
