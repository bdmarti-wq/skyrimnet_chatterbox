from pathlib import Path
from typing import Optional
import torch

from src.audio import pad_short_text
from .base import BaseGenerationPhase
from ...pipeline.context import AudioGenerationContext
from loguru import logger
from src.normalize_stem import normalize_stem
from src.config import get_config  # For defaults from config

class InputsValidationPhase(BaseGenerationPhase):
    """REFACTORED: Master validator – shared helpers for text, stem, seed, and audio params (exagg, temp, etc.). Validates early to prevent None errors.
    OPTIMIZED: Single call to get_voice_params, then use the full dict for all params (avoids repetition)."""

    def _execute_core(self, context: AudioGenerationContext) -> AudioGenerationContext:
        """REFACTORED: Core validation using shared helpers; ensure all audio params have sane defaults."""
        # Text
        self.validate_text(context)
        # After self.validate_text(context):
        text = context.text  # Get current validated text (unmodified yet)
        # Use context.voice_params so UI in-memory overrides (e1) take effect immediately
        voice_params = getattr(context, 'voice_params', {}) or {}

        # NEW: Optional short-padding – prepend a configurable token followed by a single ellipsis
        try:
            sp_threshold = int(voice_params.get('short_padding_threshold', None)
                               if 'short_padding_threshold' in voice_params else None)
        except Exception:
            sp_threshold = None

        logger.info(
            f"voice '{context.voice_stem}' voice params'{context.voice_params}'" )

        short_padding_applied = False
        if sp_threshold and sp_threshold > 0:
            base_stripped = (text or '').strip()
            if 0 < len(base_stripped) < sp_threshold:
                # Get padding token (can be any string). Fallbacks: explicit short_padding_token,
                # else legacy short_repeat_separator, else empty string.
                token = voice_params.get('short_padding_token', None)
                if not isinstance(token, str):
                    token = None

                # Build: <token> + '...' + <original text>
                ellipsis = '...'
                new_text = f"{token} {base_stripped}"

                # Persist metadata for post-generation trimming
                setattr(context, 'meta', getattr(context, 'meta', {}))
                context.meta['short_padding_active'] = True
                context.meta['short_padding_token'] = token
                context.meta['short_padding_ellipsis'] = ellipsis

                # Replace context.text BEFORE cache checks (affects cache keys)
                logger.info(
                    f"Short-padding active for voice '{context.voice_stem}': base_len={len(base_stripped)} < {sp_threshold}; token_len={len(token or '')} text= {new_text}"
                )
                context.text = new_text
                short_padding_applied = True

        # If short-padding did not apply, fall back to universal padding (legacy behavior)
        if not short_padding_applied:
            padding_params = {
                'enable_text_padding': voice_params.get('enable_text_padding', True),  # Configurable per voice
                'text_ellipses_count': voice_params.get('text_ellipses_count', 2),
                'max_short_word_len': voice_params.get('max_short_word_len', 3),
                'vocalise_patterns': voice_params.get('vocalise_patterns', ['ah', 'oh', 'aah', 'mmm', 'uh', 'mmh'])
                # Extend as needed
            }
            padded_text = pad_short_text(text, padding_params)
            context.text = padded_text  # Set padded text for downstream (cache keys, TTS)
            logger.debug(f"Applied text padding: '{text}' → '{padded_text}' (params={padding_params})")

        # Voice stem
        self.derive_stem(context)

        # Seed
        self.default_seed(context)

        # Validate/default all audio params using the already-injected context.voice_params
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
        """Resolve seed only if the bridge did not set it.

        Bridge is authoritative; only compute a deterministic fallback when missing.
        """
        try:
            seed = getattr(context, 'seed', None)
            if isinstance(seed, int) and seed > 0:
                logger.debug(f"Seed preserved from bridge: {seed}")
                return
            # Fallback: derive deterministically from cache_uuid
            from src.seeding import resolve_seed
            final_seed = resolve_seed(getattr(context, 'cache_uuid', 0), provided_seed=None, randomize=False)
            object.__setattr__(context, 'seed', final_seed)
            logger.debug(f"Seed set (fallback): {final_seed}")
        except Exception:
            # Absolute fallback to constant if something goes wrong
            object.__setattr__(context, 'seed', 42)
            logger.debug("Seed set to constant fallback: 42")

    @staticmethod
    def validate_audio_params(context: AudioGenerationContext):
        """Ensure all audio params have sane non-None defaults using context.voice_params.

        Assumes the bridge injected merged `voice_params` into the context.
        We do not re-fetch from config here to avoid divergence and extra I/O.
        """
        voice_params = getattr(context, 'voice_params', {}) or {}

        # Helper: Get from context (if explicitly set), then voice_params, else fall back to safe literals
        def sane_default(key: str, fallback_default: float) -> float:
            context_val = getattr(context, key, None)
            if isinstance(context_val, (int, float)):
                return float(context_val)
            voice_val = voice_params.get(key, None)
            if isinstance(voice_val, (int, float)):
                return float(voice_val)
            return float(fallback_default)

        # Set each param with sane_default
        context.exaggeration = sane_default('exaggeration', 0.5)
        context.temperature = sane_default('temperature', 0.7)
        context.cfg_weight = sane_default('cfg_weight', 0.45)
        context.min_p = sane_default('min_p', 0.05)
        context.top_p = sane_default('top_p', 1.0)
        context.repetition_penalty = sane_default('repetition_penalty', 1.2)

        # Ensure t3_params present (fallback dict if None)
        if context.t3_params is None:
            context.t3_params = {
                "generate_token_backend": "cudagraphs-manual",
                "stride_length": 4,
                "skip_when_1": True
            }
            logger.debug("Set default t3_params (was None)")

        logger.debug(
            f"Audio params validated: exagg={context.exaggeration}, temp={context.temperature}, cfg={context.cfg_weight}"
        )

    def handle_error(self, context: AudioGenerationContext, error: Exception) -> AudioGenerationContext:
        """REFACTORED: Delegate to base (silence)."""
        logger.error(f"Validation error: {error} – fallback to defaults")
        self.validate_audio_params(context)  # Re-ensure defaults on error
        return super().handle_error(context, error)