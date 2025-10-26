from pathlib import Path
from typing import Optional
import torch
from .base import GenerationPhase
from ...pipeline.context import AudioGenerationContext
from loguru import logger
from src.normalize_stem import normalize_stem
from src.config import get_config  # For defaults from config

class InputsValidationPhase(GenerationPhase):
    """REFACTORED: Master validator – shared helpers for text, stem, seed, and audio params (exagg, temp, etc.). Validates early to prevent None errors.
    OPTIMIZED: Single call to get_voice_params, then use the full dict for all params (avoids repetition)."""

    def _execute_core(self, context: AudioGenerationContext) -> AudioGenerationContext:
        """REFACTORED: Core validation using shared helpers; ensure all audio params have sane defaults."""
        # Text
        self.validate_text(context)

        # Voice stem
        self.derive_stem(context)

        # Seed
        self.default_seed(context)

        # FIXED: Validate/default all audio params with single voice_params fetch
        self.validate_audio_params(context)

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

    @staticmethod
    def validate_audio_params(context: AudioGenerationContext):
        """OPTIMIZED: Ensure all audio params have sane non-None defaults (single get_voice_params call).
        Fetches voice_params once, then uses the dict for all params (no repetition). Adds t3_params validation."""
        config = get_config() if context.config is None else context.config

        # FIXED: Fetch voice_params once if voice_stem is set (for the entire pipeline)
        voice_params = {}
        if hasattr(config, 'get_voice_params') and context.voice_stem != "default":
            voice_params = config.get_voice_params(context.voice_stem) or {}  # Single call!

        # Helper: Get from voice_params, fall back to context, then hardcoded
        def sane_default(key: str, fallback_default: float) -> float:
            context_val = getattr(context, key, None)
            if context_val is not None and isinstance(context_val, (int, float)) and 0 <= context_val <= 2.0:
                return context_val  # Use context if sane
            voice_val = voice_params.get(key, None)
            if voice_val is not None and isinstance(voice_val, (int, float)) and 0 <= voice_val <= 2.0:
                return voice_val  # Use voice override
            return fallback_default  # Hardcoded fallback

        # Set each param with sane_default
        context.exaggeration = sane_default('exaggeration', 0.5)
        context.temperature = sane_default('temperature', 0.7)
        context.cfg_weight = sane_default('cfg_weight', 0.45)
        context.min_p = sane_default('min_p', 0.05)
        context.top_p = sane_default('top_p', 1.0)
        context.repetition_penalty = sane_default('repetition_penalty', 1.2)

        # Also ensure t3_params is present (fallback dict if None)
        if context.t3_params is None:
            context.t3_params = {
                "generate_token_backend": "cudagraphs-manual",
                "stride_length": 4,
                "skip_when_1": True
            }
            logger.debug("Set default t3_params (was None)")

        logger.trace(f"Audio params validated: exagg={context.exaggeration}, temp={context.temperature}, cfg={context.cfg_weight} (from single voice_params fetch)")

    def handle_error(self, context: AudioGenerationContext, error: Exception) -> AudioGenerationContext:
        """REFACTORED: Delegate to base (silence)."""
        logger.error(f"Validation error: {error} – fallback to defaults")
        self.validate_audio_params(context)  # Re-ensure defaults on error
        return super().handle_error(context, error)