# src/generate/pipeline/phases/post_processing.py
from typing import Dict, Any

import torch
import numpy as np
import librosa
from scipy.signal import sosfilt, butter
from loguru import logger

from src.generate.pipeline.phases.base import BaseGenerationPhase
from src.generate.pipeline.context import AudioGenerationContext


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
    onsets = librosa.onset.onset_detect(y=tail_reversed, sr=sr, units='samples', hop_length=512, threshold=onset_threshold)
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
        params = {}
        logger.debug("Post params skipped – raw")
        return wav_np
    enable_post = params.get('enable_post_processing', True)
    if not enable_post:
        logger.debug("Post disabled – raw")
        return wav_np

    voice_name = params.get('voice_name', 'unknown')
    non_none_params = {k: v for k, v in params.items() if v is not None}
    logger.debug(f"Post params for {voice_name}: {non_none_params}")

    # Detect if short vocalize (skip heavy for "ah", "mmm", etc.)
    text = text or ''  # Empty text fallback
    is_vocalize = len(text.strip()) <= 3 and text.lower() in ['ah', 'oh', 'aah', 'mmm', 'uh', 'mmh', 'eh']
    light_mode = is_vocalize or len(text) < 10  # Light for short/vocal
    if light_mode:
        logger.debug(f"Vocalize/short '{text}' – light post (skip heavy)")

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

    # Heavy: Only if enabled and not light
    if not light_mode:
        gate_threshold = params.get('gate_threshold', 0.05)
        wav_np = gate_trailing_phantoms(wav_np, sr, gate_threshold)
        tail_fractions = params.get('tail_suppress_sec', 0.2) or 0.25  # From params
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

    # Final clip
    wav_np = np.clip(wav_np, -1.0, 1.0)
    final_len = len(wav_np)
    final_dur = final_len / sr
    light_str = "light" if light_mode else "heavy" if not light_mode else "none"
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

            context.processed_wav = wav.unsqueeze(0) if wav.dim() == 1 else wav
            context.audio_duration = context.audio_duration
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
                context.processed_wav = self.create_silence(context.sr, 2.0)

        return context

    def create_silence_tensor(sr: int, duration_s: float = 2.0, device: str = 'cpu') -> torch.Tensor:
        """Sample silence (1D fp32 mono; [samples])."""
        if device == 'cuda' and torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'  # Safe for save
        dev = torch.device(device)
        samples = int(sr * duration_s)
        return torch.zeros(samples, dtype=torch.float32, device=dev)

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

        logger.debug("Inline norm: peak=1.0, gain={post_gain:+.2f}")
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