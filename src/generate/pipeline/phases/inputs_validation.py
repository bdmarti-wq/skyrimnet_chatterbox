from pathlib import Path
from typing import Optional
import torch
from .base import GenerationPhase
from ...pipeline.context import AudioGenerationContext
from loguru import logger
from src.normalize_stem import normalize_stem

class InputsValidationPhase(GenerationPhase):
    """REFACTORED: Master validator – shared helpers for text, stem, seed. Other phases import/call static methods."""

    def _execute_core(self, context: AudioGenerationContext) -> AudioGenerationContext:
        """REFACTORED: Core validation using shared helpers."""
        # Text
        self.validate_text(context)

        # Voice stem
        self.derive_stem(context)

        # Seed
        self.default_seed(context)

        # Delegate to base for conds/paths (already in _validate)
        return context

    @staticmethod
    def validate_text(context: AudioGenerationContext):
        """REFACTORED: Shared text validator (importable)."""
        if not context.text or not context.text.strip():
            context.text = "Silence"  # Fallback
            logger.warning("Empty text – fallback to silence")
        context.original_text = context.text

    @staticmethod
    def derive_stem(context: AudioGenerationContext):
        """REFACTORED: Shared stem derive (importable; avoids dupes in cache_check)."""
        if not context.voice_stem or context.voice_stem == "default":
            stem = "default" if not context.audio_prompt_path else normalize_stem(context.audio_prompt_path) or "default"
            object.__setattr__(context, 'voice_stem', stem)
            logger.debug(f"Derived stem: {stem}")

    @staticmethod
    def default_seed(context: AudioGenerationContext):
        """REFACTORED: Shared seed default (importable)."""
        seed = context.seed or context.seed_num or 42
        if seed <= 0:
            from src.seeding import cpp_uuid_to_seed
            seed = cpp_uuid_to_seed(context.cache_uuid) or 42
        object.__setattr__(context, 'seed', seed)
        logger.debug(f"Seed set: {seed}")

    def handle_error(self, context: AudioGenerationContext, error: Exception) -> AudioGenerationContext:
        """REFACTORED: Delegate to base (silence)."""
        return super().handle_error(context, error)