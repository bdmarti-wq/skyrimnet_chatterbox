import torch
import torchaudio
from typing import Dict, Any  # Add for voice_params
from .base import GenerationPhase
from ...pipeline.context import AudioGenerationContext
from loguru import logger

class PostProcessingPhase(GenerationPhase):
    def _execute_core(self, context: AudioGenerationContext) -> AudioGenerationContext:
        """FIXED: Local guard for voice_params dict (if str from mis-set); same Tensor fixes."""
        if not hasattr(context, 'generated_wav') or context.generated_wav is None or context.generated_wav.numel() == 0:
            logger.warning("No/empty WAV for post-processing – skip")
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
            if peak > 0:
                gain = 0.95 / (peak + 1e-8)
                wav = wav * gain
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

    def _inline_simple_norm(self, wav: torch.Tensor, voice_params: Dict[str, Any]) -> torch.Tensor:
        """FIXED: Guard voice_params (input param); same Tensor."""
        # FIXED: Ensure dict (caller should, but defensive)
        if not isinstance(voice_params, dict):
            logger.warning(f"Inline voice_params not dict, defaulting")
            voice_params = {}

        if wav is None or wav.numel() == 0:
            return self.create_silence(24000, 1.0)

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
            wav = torch.nn.functional.pad(wav, (0, pad_samples), mode='constant', value=0)

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