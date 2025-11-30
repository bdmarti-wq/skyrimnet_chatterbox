# src/generate/pipeline/phases/post_processing.py
import os
import tempfile
from typing import Dict, Any

import torch
import numpy as np
import librosa
import torchaudio
from scipy.signal import sosfilt, butter
from loguru import logger

from src.audio import is_artifact_laden, is_artifact_laden_array
from src.generate.pipeline.phases.base import BaseGenerationPhase
from src.generate.pipeline.context import AudioGenerationContext
from src.audio import (
    apply_post_processing as _np_apply_post,
    short_padding_trim_head as _ap_short_head,
    short_trim_padding as _ap_short_trim,
)
from src.audio import (
    PostParams,
    normalize_post_params,
    build_params_from_preset,
)


def _short_padding_trim_head(y: np.ndarray, sr: int, params: Dict[str, Any]) -> np.ndarray:
    """When short-padding is active, remove the leading padding up to the first voiced island.
    Uses silence-based segmentation; keeps a small configurable pre-roll before the first speech
    and an optional tail pad. Safe and optional.
    """
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

        # Prefer cutting AFTER the first token when a near-silence gap occurs, keeping the rest.
        head_pad = int(sr * (head_ms / 1000.0))
        tail_pad = int(sr * (tail_ms / 1000.0))

        # Compute a near-silence mask using frame-wise RMS with a slightly higher threshold than strict silence
        near_db_offset = 6.0
        try:
            near_db_offset = float(params.get('short_trim_near_db_offset', 6.0) or 6.0)
        except Exception:
            pass
        base_sil_val = params.get('short_trim_silence_db', params.get('short_repeat_silence_db', -45.0))
        try:
            base_sil_db = float(base_sil_val if base_sil_val is not None else -45.0)
        except Exception:
            base_sil_db = -45.0
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

            # Determine region after the first voiced island
            if len(intervals) >= 1:
                first_end = int(intervals[0][1])
                # Convert first_end to frame index
                first_end_frame = min(num_frames - 1, max(0, first_end // hop_near))
                # Find first extended near-silence run after first_end_frame
                try:
                    min_near_ms = float(params.get('short_trim_min_near_silence_ms', 40.0) or 40.0)
                except Exception:
                    min_near_ms = 40.0
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
                            cut_frame_after_near = k  # first non-near-silent frame after the run
                            break
                        j = k
                    else:
                        j += 1

                if cut_frame_after_near is not None:
                    start = max(0, int(starts[min(cut_frame_after_near, len(starts) - 1)]) - head_pad)
                    end = min(len(y), int(intervals[-1][1]) + tail_pad)
                    logger.debug(
                        f"short_padding_trim_head: near-silence cut start={start/sr:.3f}s (near_db={near_db}dB, min_run={min_near_ms}ms) end={end/sr:.3f}s"
                    )
                    return y[start:end]

        # Fallback: work backwards – cut after the largest internal silence gap
        if len(intervals) >= 2:
            best_gap = -1
            best_after_idx = None
            for i in range(len(intervals) - 1):
                g = int(intervals[i + 1][0]) - int(intervals[i][1])
                if g > best_gap:
                    best_gap = g
                    best_after_idx = i + 1
            if best_after_idx is not None and best_gap >= int(sr * 0.02):  # ~20ms minimum
                start = max(0, int(intervals[best_after_idx][0]) - head_pad)
                end = min(len(y), int(intervals[-1][1]) + tail_pad)
                logger.debug(
                    f"short_padding_trim_head: largest-gap cut start={start/sr:.3f}s (gap={best_gap/sr:.3f}s) end={end/sr:.3f}s"
                )
                return y[start:end]

        # Last resort: keep from first non-silent region with head preroll (original behavior)
        first_start, last_end = int(intervals[0][0]), int(intervals[-1][1])
        start = max(0, first_start - head_pad)
        end = min(len(y), last_end + tail_pad)
        logger.debug(
            f"short_padding_trim_head: fallback first-region start={start/sr:.3f}s end={end/sr:.3f}s"
        )
        return y[start:end]
    except Exception:
        return y


def _short_trim_padding(y: np.ndarray, sr: int, params: Dict[str, Any]) -> np.ndarray:
    """Aggressively trim off short-padding by working backwards from the end.

    Strategy (requested):
    - Ignore any trailing silence at the very end (do not cut there).
    - From the end, once a SOUND is encountered, search BACKWARDS for the first SILENCE
      run and cut at the end of that silence (i.e., keep only the final phrase).
    - Be aggressive: even ~20 ms of silence should trigger a cut (configurable).

    Tunables (all optional, with safe defaults):
    - short_repeat_head_sil_ms / short_repeat_tail_sil_ms: preroll/keep pads around kept audio.
    - short_repeat_silence_db or short_trim_silence_db: silence threshold in dBFS (default -45).
    - short_repeat_min_gap_ms or short_trim_min_silence_ms: minimum silence to qualify (default 20 ms).

    Returns original on failure.
    """
    if y is None or not isinstance(y, np.ndarray) or y.size == 0:
        return y

    try:
        # Pads (retain a tiny preroll, optional tail pad kept as-is)
        try:
            head_ms = float(params.get('short_repeat_head_sil_ms', 120.0) or 120.0)
            tail_ms = float(params.get('short_repeat_tail_sil_ms', 120.0) or 120.0)
        except Exception:
            head_ms, tail_ms = 120.0, 120.0
        head_pad = int(sr * (head_ms / 1000.0))
        tail_pad = int(sr * (tail_ms / 1000.0))

        # Silence definition (aggressive)
        sil_db = None
        for k in ('short_trim_silence_db', 'short_repeat_silence_db'):
            if isinstance(params.get(k, None), (int, float)):
                sil_db = float(params.get(k))
                break
        if sil_db is None:
            sil_db = -60.0
        sil_thr = 10 ** (sil_db / 20.0)

        # Near-silence threshold: slightly higher than strict silence to tolerate tiny sounds
        try:
            near_db_offset = float(params.get('short_trim_near_db_offset', 18.0) or 5.0)
        except Exception:
            near_db_offset = 6.0
        near_thr = 10 ** ((sil_db + near_db_offset) / 20.0)

        # Minimum silence chunk to qualify (default ~20ms)
        min_sil_ms = None
        for k in ('short_trim_min_silence_ms', 'short_repeat_min_gap_ms'):
            if isinstance(params.get(k, None), (int, float)):
                min_sil_ms = float(params.get(k))
                break
        if min_sil_ms is None:
            min_sil_ms = 5.0
        # Minimum near-silence run to qualify (default: max(40ms, min_sil_ms))
        try:
            min_near_ms = float(params.get('short_trim_min_near_silence_ms', max(5.0, min_sil_ms)))
        except Exception:
            min_near_ms = max(30.0, min_sil_ms)

        # Frame analysis (~10ms window, 5ms hop)
        win = max(64, int(sr * 0.010))
        hop = max(32, int(sr * 0.005))
        if hop <= 0:
            hop = 32
        n = len(y)
        if n < win:
            return y
        # Compute RMS per frame
        num_frames = 1 + (n - win) // hop
        rms = np.empty(num_frames, dtype=np.float32)
        starts = np.arange(num_frames, dtype=np.int64) * hop
        for i in range(num_frames):
            seg = y[starts[i]:starts[i] + win]
            rms[i] = float(np.sqrt(np.mean(seg * seg)) + 1e-12)
        silent = rms < sil_thr
        near_silent = rms < near_thr

        # 1) Skip trailing silence entirely
        last_idx = num_frames - 1
        while last_idx >= 0 and silent[last_idx]:
            last_idx -= 1
        if last_idx < 0:
            logger.debug("short_trim_padding: all-silent; returning original")
            return y

        # 2) From the first SOUND encountered (from end), search backwards for the
        #    first NEAR-SILENCE run of sufficient length, and cut at the end of that run.
        min_sil_frames = max(1, int((min_sil_ms / 1000.0) * sr / hop))
        min_near_frames = max(1, int((min_near_ms / 1000.0) * sr / hop))
        j = last_idx
        cut_frame = None
        while j >= 0:
            if near_silent[j]:
                # Count backward run
                k = j
                while k >= 0 and near_silent[k]:
                    k -= 1
                run_len = j - k
                if run_len >= min_near_frames:
                    # Cut at the first non-silent frame after this silence when moving forward
                    cut_frame = j + 1
                    break
                j = k
            else:
                j -= 1

        if cut_frame is None:
            # Fallback: find the largest near-silence valley before the last_idx
            # Identify all near-silent runs and pick the longest before last_idx
            best_len = 0
            best_after = None
            j = min(last_idx, num_frames - 1)
            while j >= 0:
                if near_silent[j]:
                    k = j
                    while k >= 0 and near_silent[k]:
                        k -= 1
                    run_len = j - k
                    if run_len > best_len:
                        best_len = run_len
                        best_after = j + 1
                    j = k
                else:
                    j -= 1
            cut_frame = best_after if best_after is not None else 0

        # Convert frame index to sample index
        start_idx = int(starts[min(cut_frame, len(starts) - 1)])
        # Apply preroll head pad (keep a bit before start)
        start_idx = max(0, start_idx - head_pad)
        end_idx = len(y)  # Keep full tail; optional tail_pad keeps extra naturally
        kept = y[start_idx:end_idx]
        logger.debug(
            f"short_trim_padding: last_idx_frame={last_idx}, cut_frame={cut_frame}, start={start_idx/sr:.3f}s, kept={len(kept)/sr:.3f}s (sil_db={sil_db}dB, near_off={near_db_offset}dB, min_sil={min_sil_ms}ms, min_near={min_near_ms}ms)"
        )
        return kept
    except Exception as e:
        logger.debug(f"short_trim_padding: failed with {e}; returning original")
        return y


def trim_trailing_artifacts(audio: np.ndarray, sr: int, tail_threshold_db: float = -45.0, tail_fraction: float = 0.2) -> np.ndarray:
    """Trim trailing low-energy (aggressive for phantoms)."""
    if tail_threshold_db is None or tail_threshold_db > -20:
        logger.debug("Tail trim skipped")
        return audio
    threshold = 10 ** (tail_threshold_db / 20)
    tail_len = int(len(audio) * tail_fraction)
    if tail_len == 0:
        return audio
    tail_audio = audio[-tail_len:] if len(audio) > tail_len else audio
    abs_tail = np.abs(tail_audio[::-1])
    end_idx = len(tail_audio) - np.argmax(abs_tail > threshold)
    if end_idx < len(tail_audio):
        trimmed_tail = tail_audio[:len(tail_audio) - end_idx]
        if len(trimmed_tail) < len(tail_audio):
            trimmed_tail = np.pad(trimmed_tail, (0, end_idx), 'constant')
        audio[-len(trimmed_tail):] = trimmed_tail
        logger.debug(f"Tail trimmed: {end_idx / sr:.2f}s")
    return audio


def gate_trailing_phantoms(audio: np.ndarray, sr: int, gate_threshold: float = 0.05, tail_fraction: float = 0.3) -> np.ndarray:
    """Light gate on tail: Ramp to zero if below relative threshold (for weak 'ee')."""
    if gate_threshold <= 0:
        logger.debug("Tail gate skipped")
        return audio
    tail_len = int(len(audio) * tail_fraction)
    if tail_len == 0:
        return audio
    tail_audio = audio[-tail_len:]
    peak = np.max(np.abs(tail_audio))
    if peak <= 0:
        return audio
    rel_threshold = gate_threshold * peak
    gate_mask = np.abs(tail_audio) < rel_threshold
    ramp_start = np.argmax(gate_mask[::-1])  # From end
    if ramp_start > 0:
        ramp_len = len(tail_audio) - ramp_start
        ramp = np.linspace(1.0, 0.0, ramp_len)
        tail_audio[ramp_start:] *= ramp
        audio[-len(tail_audio):] = tail_audio
        logger.debug(f"Tail gated: {ramp_len / sr:.2f}s @ rel_threshold={gate_threshold}")
    return audio


def suppress_tail_artifacts(audio: np.ndarray, sr: int, tail_fraction: float = 0.25, low_hz: float = 2000.0, high_hz: float = 4000.0, strength: float = 0.6) -> np.ndarray:
    """Spectral gating on tail: Suppress high-freq phantoms (e.g., 'ee')."""
    tail_len = int(len(audio) * tail_fraction)
    if tail_len < sr * 0.1:
        return audio
    tail_audio = audio[-tail_len:]
    hop_length = min(512, len(tail_audio) // 4)
    stft = librosa.stft(tail_audio, n_fft=1024, hop_length=hop_length)
    mag, phase = np.abs(stft), np.angle(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    mag[mask] *= strength
    suppressed_tail = librosa.istft(mag * np.exp(1j * phase), hop_length=hop_length, length=len(tail_audio))
    audio[-len(suppressed_tail):] = suppressed_tail
    logger.debug(f"Tail spectral suppressed: {low_hz}-{high_hz}Hz, strength={strength}")
    return audio


def reverse_tail_suppress(audio: np.ndarray, sr: int, tail_sec: float = 0.2, onset_threshold: float = 0.3) -> np.ndarray:
    """Reverse for tail onset detection + suppress low-confidence ends."""
    tail_samples = int(sr * tail_sec)
    if len(audio) < tail_samples * 2:
        return audio
    tail_reversed = audio[-tail_samples:][::-1]
    # Use 'delta' kw for librosa onset detection for broader version compatibility
    onsets = librosa.onset.onset_detect(y=tail_reversed, sr=sr, units='samples', hop_length=512, delta=onset_threshold)
    if len(onsets) == 0:  # No strong onsets – suppress tail
        suppress_len = int(tail_samples * 0.7)
        taper = np.linspace(1.0, 0.0, suppress_len)
        tail_audio = audio[-tail_samples:]
        tail_audio[-suppress_len:] *= taper
        audio[-tail_samples:] = tail_audio
        logger.debug(f"Tail suppressed (no onsets): {tail_sec}s taper")
    return audio


def apply_notch(audio: np.ndarray, sr: int, low_hz: float = 8000.0, high_hz: float = 11000.0, gain_db: float = -12.0) -> np.ndarray:
    """Bandstop notch (e.g., 8-11kHz -12dB)."""
    if gain_db >= 0 or low_hz >= high_hz:
        logger.debug("Notch skipped")
        return audio
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
    except Exception as e:
        logger.warning(f"Notch failed: {e}")
        return audio


def apply_eq(audio: np.ndarray, sr: int, gain_db: float = 0.0, cutoff_hz: float = 3000.0) -> np.ndarray:
    """Simple EQ (low/high-pass with gain). FIXED: None guards, better logging."""
    if gain_db == 0.0:
        logger.debug("EQ skipped (gain_db=0)")
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
    logger.debug(f"EQ applied ({gain_db}dB {'low' if gain_db < 0 else 'high'}-pass @ {cutoff_hz}Hz)")
    return np.clip(audio, -1.0, 1.0)


def adjust_speaking_rate(audio: np.ndarray, rate: float = 1.0) -> np.ndarray:
    """Adjust speaking rate via time stretching. FIXED: Guard rate/None."""
    if rate is None:
        rate = 1.0
    if abs(rate - 1.0) <= 0.1:
        logger.debug("Rate adjustment skipped (rate≈1.0)")
        return audio
    stretch_rate = 1.0 / rate
    try:
        stretched = librosa.effects.time_stretch(audio, rate=stretch_rate)
        target_length = int(len(audio) * rate)  # int() safe
        if len(stretched) > target_length:
            stretched = stretched[:target_length]
        else:
            pad_length = target_length - len(stretched)
            stretched = np.pad(stretched, (0, pad_length), mode='constant')
        logger.debug(f"Rate adjusted: {rate}x")
        return np.clip(stretched, -1.0, 1.0)
    except Exception as e:
        logger.warning(f"Rate adjustment failed: {e}")
        return audio


def apply_fade(audio: np.ndarray, sr: int, fade_ms: float | None = 20.0) -> np.ndarray:
    """Apply fade-in/out. FIXED: Guard fade_ms/None before * / int()."""
    if fade_ms is None or fade_ms <= 0:
        logger.debug("Fade skipped (fade_ms=None or <=0)")
        return audio
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
    logger.debug("Fade applied")
    return audio


def apply_post_processing(wav_np: np.ndarray, sr: int, params: dict | None = None, text: str = '') -> np.ndarray:
    """Apply post-processing. FIXED: Optional text for vocalize, light/heavy conditional; no-op defaults."""
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

    # Detect if short vocalize (previously used to skip heavy; now only for logging/tuning)
    text = text or ''  # Empty text fallback
    is_vocalize = len(text.strip()) <= 3 and text.lower() in ['ah', 'oh', 'aah', 'mmm', 'uh', 'mmh', 'eh']
    light_mode = is_vocalize or len(text) < 10  # Keep a notion of shortness, but don't skip heavy anymore
    if light_mode:
        logger.debug(f"Vocalize/short '{text}' – applying light tail fixes PLUS full heavy post")

    orig_dur = len(wav_np) / sr
    orig_len = len(wav_np)
    logger.debug(f"Post input: {orig_len} samples @ {sr}Hz ({orig_dur:.2f}s)")

    if orig_len == 0:
        min_dur_sec = params.get('min_post_duration_sec', 0.5)
        min_samples = int(sr * min_dur_sec)
        silence = np.zeros(min_samples, dtype=np.float32)
        logger.warning("Empty input – raw fallback")
        return silence

    # Light: Tail-focused fixes (always for shorts)
    tail_threshold_db = params.get('trailing_silence_db', -45.0)  # Note: Use trailing_silence_db or your default
    light_mode = light_mode  # From above
    if light_mode:
        wav_np = trim_trailing_artifacts(wav_np, sr, tail_threshold_db)
        logger.debug(f"Light post: Tail fixes for phantoms")

    # Heavy: Always apply when post-processing is enabled (even for short strings)
    gate_threshold = params.get('gate_threshold', 0.05)
    wav_np = gate_trailing_phantoms(wav_np, sr, gate_threshold)
    # Respect explicit 0.0 (do not coerce to default using 'or')
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

    logger.debug("Heavy post: Spectral/reverse/notch/EQ applied")

    # Rate adjustment
    rate = params.get('speaking_rate', 1.0)
    if abs(rate - 1.0) > 0.05:
        wav_np = adjust_speaking_rate(wav_np, rate)

    # Optional fade from overrides
    fade_ms = params.get('fade_ms', None)
    try:
        wav_np = apply_fade(wav_np, sr, fade_ms)
    except Exception as e:
        logger.debug(f"Fade failed/skipped: {e}")

    # Limiter/clipping based on overrides (peak-safe for short/garbled samples)
    gain_max_limit = params.get('gain_max_limit', None)
    if isinstance(gain_max_limit, (int, float)) and gain_max_limit is not None and gain_max_limit > 0:
        peak = float(np.max(np.abs(wav_np))) if wav_np.size > 0 else 0.0
        if peak > 0 and peak > gain_max_limit:
            scale = gain_max_limit / peak
            wav_np = wav_np * scale
            logger.debug(f"Applied limiter scale {scale:.3f} to enforce peak≤{gain_max_limit:.3f}")
        # Final clip to configured ceiling (and still within [-1,1])
        ceiling = min(1.0, float(gain_max_limit))
        wav_np = np.clip(wav_np, -ceiling, ceiling)
    else:
        # Standard safety clip
        wav_np = np.clip(wav_np, -1.0, 1.0)
    final_len = len(wav_np)
    final_dur = final_len / sr
    # Cosmetic: simplify string (last branch unreachable previously)
    light_str = "light" if light_mode else "heavy"
    logger.debug(f"Post complete for {voice_name}: {final_dur:.2f}s from {orig_dur:.2f}s ({light_str}; vocal={is_vocalize})")

    # Fallback if too short (but relaxed for all)
    min_dur_sec = params.get('min_post_duration_sec', 0.5)
    if final_dur < min_dur_sec * 0.5:
        min_samples = int(sr * min_dur_sec * 0.5)
        if final_len < min_samples:
            wav_np = np.pad(wav_np, (0, min_samples - final_len), 'constant')
            logger.debug(f"Fallback pad to {min_dur_sec * 0.5}s")

    return wav_np


class PostProcessingPhase(BaseGenerationPhase):
    """Post-processes generated audio: Trim, pad, denoise, EQ, notch, rate adjust, fade, clamp.
    Delegates to apply_post_processing (config-driven; light/heavy modes)."""

    def _execute_core(self, context: AudioGenerationContext) -> AudioGenerationContext:
        """REFACTORED: Validate wav, extract params from context, apply post-processing, set processed_wav."""
        if not hasattr(context, 'generated_wav') or context.generated_wav is None:
            logger.warning("No or empty WAV for post-processing – skip")
            return context

        # FIXED: Ensure voice_params dict (defensive)
        voice_params = getattr(context, 'voice_params', {})
        if not isinstance(voice_params, dict):
            logger.warning(f"voice_params not dict ({type(voice_params)}), defaulting to empty")
            voice_params = {}

        try:
            sr = context.sr
            wav = context.generated_wav

            if wav.numel() == 0:
                raise ValueError("Empty WAV tensor")

            if wav.dim() == 2 and wav.size(0) == 1:
                wav = wav.squeeze(0)

            peak = torch.max(torch.abs(wav))
            logger.debug(f"Norm: peak={peak:.3f}")

            # FIXED: Use guarded dict
            post_gain = voice_params.get('post_gain', 0.0)
            if post_gain != 0:
                wav = wav * (1 + post_gain)
                if torch.max(torch.abs(wav)) > 1.0:
                    wav = wav / torch.max(torch.abs(wav))

            target_sr = 24000
            if sr != target_sr:
                from torchaudio.transforms import Resample
                resampler = Resample(sr, target_sr)
                context.sr = target_sr
                input_wav = wav.unsqueeze(0) if wav.dim() == 1 else wav
                resampled = resampler(input_wav)
                wav = resampled.squeeze(0)

            # Optional pre-check: detect artifacts on current waveform to auto-enable post-processing
            device = wav.device
            dtype = wav.dtype
            wav_np = wav.detach().cpu().float().numpy()
            text = getattr(context, 'text', '') or ''
            voice_params = voice_params or {}

            # If short-padding active: trim off the head padding up to first voiced island
            meta = getattr(context, 'meta', {})
            if isinstance(meta, dict) and meta.get('short_padding_active'):
                before_len = len(wav_np)
                wav_np = _ap_short_head(wav_np, context.sr, voice_params)
                after_len = len(wav_np)
                if after_len != before_len:
                    logger.debug(f"Short-padding head trim applied: {before_len/context.sr:.2f}s → {after_len/context.sr:.2f}s")

            # Optional short-repeat trimming: if earlier phase marked it active, cut to last instance now
            try:
                threshold_cfg = int(voice_params.get('short_repeat_threshold', 0) or 0)
            except Exception:
                threshold_cfg = 0
            if isinstance(meta, dict) and meta.get('short_repeat_active') and threshold_cfg > 0:
                before = len(wav_np)
                wav_np = _ap_short_trim(wav_np, context.sr, voice_params)
                after = len(wav_np)
                if after != before:
                    logger.debug(f"Short-repeat trim_to_last applied: {before/context.sr:.2f}s → {after/context.sr:.2f}s")

            # Build preset-driven params with None-as-noop semantics
            preset_name = None
            if isinstance(voice_params, dict):
                preset_name = voice_params.get('post_preset') or voice_params.get('postprocessing_preset')
            effective_params = build_params_from_preset(preset_name, voice_params)
            auto_enabled = False
            try:
                if is_artifact_laden_array(wav_np, context.sr):
                    auto_enabled = True
                    # Choose a safe preset and merge with overrides
                    effective_params = build_params_from_preset('light_tail_cleanup', voice_params)
                    # Ensure enabled
                    effective_params.enable_post_processing = True
            except Exception as det_e:
                logger.debug(f"Artifact detection skipped/failed: {det_e}")

            if auto_enabled:
                logger.info("Artifact-laden audio detected – auto-enabling post-processing with safe defaults")

            # Apply numpy-based post processing (None-as-noop params supported)
            processed_np = _np_apply_post(wav_np, context.sr, effective_params, text)
            if processed_np is not None and isinstance(processed_np, np.ndarray) and processed_np.size > 0:
                wav = torch.from_numpy(processed_np).to(device=device, dtype=dtype)
            else:
                logger.debug("Post-processing returned empty/invalid – using original wav")

            context.processed_wav = wav.unsqueeze(0) if wav.dim() == 1 else wav
            # Update duration based on processed wav
            try:
                context.audio_duration = len(context.processed_wav.squeeze(0)) / context.sr
            except Exception:
                pass
            logger.info(f"Post-processing: {context.audio_duration:.2f}s @ {context.sr}Hz")
        except Exception as e:
            logger.error(f"Post-processing error: {e} – raw fallback")
            raw_wav = context.generated_wav
            # FIXED: Guard again in fallback (though attrs/ensure should prevent)
            fallback_params = getattr(context, 'voice_params', {})
            if not isinstance(fallback_params, dict):
                fallback_params = {}
            if raw_wav is not None and raw_wav.numel() > 0:
                context.processed_wav = self._inline_simple_norm(raw_wav, fallback_params)
            else:
                context.processed_wav = self.create_silence_tensor(context.sr, 2.0)

        return context

    @staticmethod
    def create_silence_tensor(sr: int, duration_s: float = 2.0, device: str = 'cpu') -> torch.Tensor:
        """Create silence as a 2D tensor [1, samples] to satisfy downstream expectations."""
        if device == 'cuda' and torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'  # Safe for save
        dev = torch.device(device)
        samples = int(sr * duration_s)
        return torch.zeros((1, samples), dtype=torch.float32, device=dev)

    # UTILITY: Inline simple norm (fallback)
    def _inline_simple_norm(self, wav: torch.Tensor, voice_params: Dict[str, Any]) -> torch.Tensor:
        """FIXED: Guard voice_params (input param); same Tensor."""
        # FIXED: Ensure dict (caller should, but defensive)
        if not isinstance(voice_params, dict):
            logger.warning("Inline voice_params not dict, defaulting")
            voice_params = {}

        if wav is None or wav.numel() == 0:
            return self.create_silence_tensor(24000, 1.0)

        if wav.dim() == 2 and wav.size(0) == 1:
            wav = wav.squeeze(0)

        max_abs = torch.max(torch.abs(wav))
        if max_abs > 0:
            wav = wav / max_abs

        post_gain = voice_params.get('post_gain', 0.0)
        if post_gain != 0:
            wav = wav * (1 + post_gain)
            if torch.max(torch.abs(wav)) > 1.0:
                wav = wav / torch.max(torch.abs(wav))

        min_dur = voice_params.get('min_post_duration', 1.0)
        current_dur = wav.numel() / 24000
        if current_dur < min_dur:
            pad_samples = int((min_dur - current_dur) * 24000)
            wav = torch.nn.functional.pad(wav, (0, pad_samples), mode='constant')

        logger.debug(f"Inline norm: peak=1.0, gain={post_gain:+.2f}")
        return wav.unsqueeze(0) if wav.dim() == 1 else wav

    def handle_error(self, context: AudioGenerationContext, error: Exception) -> AudioGenerationContext:
        """FIXED: Guard voice_params in fallback."""
        logger.error(f"Post-processing failed: {error} – raw with inline norm")
        raw_wav = getattr(context, 'generated_wav', None)
        fallback_params = getattr(context, 'voice_params', {})
        if not isinstance(fallback_params, dict):
            fallback_params = {}
        if raw_wav is not None and raw_wav.numel() > 0:
            context.processed_wav = self._inline_simple_norm(raw_wav, fallback_params)
        return super().handle_error(context, error)