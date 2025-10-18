from pathlib import Path
from typing import Optional

import torch

from .base import GenerationPhase
from ...pipeline.context import AudioGenerationContext
from loguru import logger

class InputsValidationPhase(GenerationPhase):
    """Validates essential input parameters and prepares them for pipeline."""

    def execute(self, context: AudioGenerationContext) -> AudioGenerationContext:
        start_time = self._start_timer(context)

        # Validate text input
        if not context.text.strip():
            logger.warning("No text provided - using fallback silence")
            context.text = "Silence"
            context.original_text = context.text

        # Prepare voice stem
        context.voice_stem = context.voice_stem or (
            "default" if not context.audio_prompt_path
            else self._normalize_stem(context.audio_prompt_path)
        )

        # Set default seed if needed
        if context.seed_num is None or context.seed_num <= 0:
            context.seed_num = self._get_default_seed(context.cache_uuid)
            logger.debug(f"Using default seed: {context.seed_num}")

        # Record timing
        self._record_timer(context, start_time, "validation")
        return context

    def _normalize_stem(self, path: str) -> str:
        """Extract normalized voice stem from path."""
        try:
            from src.normalize_stem import normalize_stem
            return normalize_stem(path)
        except ImportError:
            logger.error("normalize_stem not available - using fallback stem")
            return Path(path).stem

    def _get_default_seed(self, cache_uuid: int) -> int:
        """Generate seed from UUID with proper bounds."""
        max_seed = 2**32 - 1
        return abs(hash(cache_uuid)) % max_seed

    def handle_error(self, context: AudioGenerationContext, error: Exception) -> AudioGenerationContext:
        """Handle validation failures with meaningful fallbacks."""
        logger.error(f"Input validation failed: {str(error)}")
        context.text = "Error Generating Audio"
        context.generated_wav = self._create_silence_fallback()
        return context

    def _create_silence_fallback(self, duration: int = 1) -> torch.Tensor:
        """Create silence tensor as generation fallback."""
        return torch.zeros(1, 24000 * duration, dtype=torch.float32, device='cpu')