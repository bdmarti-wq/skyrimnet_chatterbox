import time
import os
from typing import Optional, Dict, Any
import torch
import torchaudio  # Optional: For any SR checks (unused now)

from src.audio_utils import create_silence_tensor, apply_post_processing  # From prior patches; fallback if missing
from src.config import get_config, get_config_value
from .base import GenerationPhase
from ...pipeline.context import AudioGenerationContext
from loguru import logger


class PostProcessingPhase(GenerationPhase):
    """Applies audio post-processing to enhance quality of generated audio."""

    process_after_cache = True  # Should run even if we have cache hits (for quality control)

    def execute(self, context: AudioGenerationContext) -> AudioGenerationContext:
        if context.generated_wav is None or len(context.generated_wav) == 0:
            logger.warning("No WAV for post-processing – skip")
            return context

        try:
            sr = context.sr  # FIXED: Use context.sr (int) instead of context.config.sr
            logger.debug(f"Post-processing: WAV {context.generated_wav.shape}, sr={sr}Hz")

            # FIXED: Norm/gain calc (prevent iterable error on config)
            wav = context.generated_wav.squeeze(0)  # [T] or [C,T]
            if wav.dim() == 1:
                peak = torch.max(torch.abs(wav))  # Scalar
            else:
                peak = torch.max(torch.abs(wav))  # Still scalar

            if peak > 0:
                gain = 0.95 / (peak + 1e-8)  # Avoid div0
                wav = wav * gain
                logger.debug(f"Applied norm + gain: peak={peak:.3f}, gain={20 * torch.log10(gain):+.1f}dB")
            else:
                logger.debug("WAV is zero – no norm needed")

            # FIXED: Resample if needed (use context.sr, not config)
            if context.sr != 24000:  # Target SR
                from torchaudio.transforms import Resample
                resampler = Resample(context.sr, 24000)
                target_sr = 24000
                context.sr = target_sr  # Update
                context.processed_wav = resampler(wav.unsqueeze(0)).squeeze(0)  # [1, T] → [T]
            else:
                context.processed_wav = wav.unsqueeze(0) if wav.dim() == 1 else wav  # Ensure [1,T]

            context.audio_duration = len(context.processed_wav.squeeze(0)) / context.sr
            logger.info(f"Post-processing success: {context.audio_duration:.2f}s @ {context.sr}Hz")

        except Exception as e:
            logger.error(f"Post-processing error: {e} – raw pass-through")
            context.processed_wav = context.generated_wav  # Fallback
            # FIXED: In handle_error, use context.sr too
            # e.g., sr = int(context.sr) if hasattr(context, 'sr') else 24000

        return context



    def _inline_simple_norm(self, wav: torch.Tensor, voice_params: Dict[str, Any]) -> torch.Tensor:
        """Simple torch-based peak normalization + voice gain (fallback; no deps)."""
        # Peak norm (safe clamp)
        max_abs = wav.abs().max()
        if max_abs > 0:
            post_wav = wav / max_abs  # Normalize to [-1,1]
        else:
            post_wav = wav  # Already zero/silent

        # Voice-specific: Post-gain from params (e.g., boost for quiet clones)
        post_gain = voice_params.get('post_gain', 0.0)
        if post_gain != 0:
            post_wav = post_wav * (1 + post_gain)
            # Re-clamp if exploded
            if post_wav.abs().max() > 1.0:
                post_wav = post_wav / post_wav.abs().max()

        # Optional: Min duration pad (if short artifact)
        min_post_duration = voice_params.get('min_post_duration', 1.0)
        sr = get_config_value('sr', 24000)
        current_dur = post_wav.shape[-1] / sr
        if current_dur < min_post_duration:
            pad_samples = int((min_post_duration - current_dur) * sr)
            post_wav = torch.nn.functional.pad(post_wav, (0, pad_samples), mode='constant', value=0)

        logger.debug("Applied inline simple norm + gain (peak=1.0, gain={:+.2f}dB)".format(
            20 * torch.log10(torch.tensor(1 + post_gain)) if post_gain > 0 else 0))
        return post_wav

    # Advanced methods (commented: Optional expansions; enable via config + deps)
    # def _apply_voice_eq(self, audio_data: np.ndarray, sr: int, gain_db: float, cutoff_hz: float) -> np.ndarray:
    #     """High-shelf EQ (requires scipy)."""
    #     if np.isclose(gain_db, 0.0):
    #         return audio_data
    #     # ... (your existing; convert wav to np first if used)
    #     return audio_data

    # Future: Integrate cleaning toggle
    # if get_config_value('voice.enable_cleanup', False):
    #     from src.audio_utils import process_voice_cleanup
    #     post_wav, _ = process_voice_cleanup(post_wav, sr, voice_params)

    def handle_error(self, context: AudioGenerationContext, error: Exception) -> AudioGenerationContext:
        """Handle post-processing errors by falling back to raw audio."""
        logger.error(f"Post-processing phase failed: {str(error)} – falling back to raw generation")
        if hasattr(context, 'generated_wav') and context.generated_wav is not None:
            # Inline simple norm as last resort
            voice_params = getattr(context, 'voice_params', {})
            context.generated_wav = self._inline_simple_norm(context.generated_wav, voice_params)
        return context