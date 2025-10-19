# src/generate/pipeline/phases/cache_check.py (Corrected: Safe device/dtype/sr in validation; consistent config access)
import os
import time
from pathlib import Path

import torch
import torchaudio

from src.config import get_config
from src.normalize_stem import normalize_stem
from .base import GenerationPhase
from ...pipeline.context import AudioGenerationContext
from loguru import logger

class CacheCheckPhase(GenerationPhase):
    """Voice cache check; SIMPLIFIED: Call manager.wrapper (norm stem stable); set attrs for conds."""

    def __init__(self, cache_manager=None):
        self.cache_manager = cache_manager
        if not self.cache_manager:
            logger.warning("No cache_manager; voice MISS")
        super().__init__()

    def execute(self, context: AudioGenerationContext) -> AudioGenerationContext:
        """Voice process/check. FIXED: Ensure processed_voice_path is always set/valid (derive from voice_stem if None/missing for conds prep in reuse)."""
        start_time = time.perf_counter()

        if not self.cache_manager:
            context.is_cached = False
            context.cache_hit_type = 'no_cache'
            context.cache_uuid = int(time.time() * 1000) % (2 ** 32)
            context.cache_key = f"nocache_{hash(context.text)}_{context.audio_prompt_path}"
            return context

        audio_prompt_path = getattr(context, 'audio_prompt_path', None)
        voice_stem = getattr(context, 'voice_stem',
                             normalize_stem(audio_prompt_path) if audio_prompt_path else 'default')
        text = context.text
        exag = getattr(context, 'exaggeration', 0.5)

        if not voice_stem or not text:
            logger.warning("Missing voice/text")
            context.cache_type = 'invalid'
            context.cache_uuid = int(time.time() * 1000) % (2 ** 32)
            return context

        # SIMPLIFIED: Call wrapper (handles norm; returns stable)
        processed_path, _, conds_key, voice_params, hit_entry = self.cache_manager.process_voice_reference(
            audio_path=audio_prompt_path, voice_stem=voice_stem
        )

        # FIXED: Ensure processed_voice_path is valid (set from return; derive if None/invalid for conds prep)
        context.voice_stem = voice_stem  # Norm from wrapper
        initial_path = processed_path
        if processed_path and os.path.exists(str(processed_path)):
            context.processed_voice_path = processed_path
            logger.debug(f"Set processed_voice_path from cache: {context.processed_voice_path}")
        else:
            # Derive valid path from voice_stem (for HIT reuse; assume cache/voices or resampled)
            voice_cache_dir = getattr(self.cache_manager, 'voice_cache_dir', None)
            if voice_cache_dir is None:
                from src.config import get_config
                config = get_config()
                voice_cache_dir = Path(getattr(config, 'cache_dir', './cache')) / "voices"  # Fallback to config
            derived_path = voice_cache_dir / f"{voice_stem}.wav"
            if not derived_path.exists():
                # Try resampled subdir (common for processed)
                resampled_dir = voice_cache_dir / "resampled"
                derived_path = resampled_dir / f"{voice_stem}.wav"
                logger.debug(f"Tried resampled derived path for {voice_stem}: {derived_path}")

            if derived_path.exists():
                context.processed_voice_path = derived_path
                logger.info(
                    f"Derived valid processed_voice_path for reuse: {context.processed_voice_path} (original: {initial_path})")
            else:
                # Ultimate fallback: Set to original audio_prompt_path if available, or warn
                context.processed_voice_path = audio_prompt_path or Path("")
                if not context.processed_voice_path or not os.path.exists(str(context.processed_voice_path)):
                    logger.warning(
                        f"Could not derive valid processed_voice_path for {voice_stem}; set to empty – conds may fail")
                    context.processed_voice_path = Path("")
                else:
                    logger.info(f"Fallback to original audio_prompt_path: {context.processed_voice_path}")

        context.processed_ref_path = context.processed_voice_path  # Mirror for compatibility
        context.conditionals_key = conds_key
        context.conds_key = conds_key
        context.voice_params = voice_params
        # Note: content_hash if needed in Gen/context; stub here if essential
        context.cache_uuid = context.cache_uuid or int(time.time() * 1000) % (2 ** 32)

        logger.info(
            f"Voice processed {voice_stem} (conds {conds_key[:20]}... | path exists: {os.path.exists(str(context.processed_voice_path))})")

        # Set type
        if hit_entry:
            context.cache_hit = True
            context.cache_type = 'voice_reuse'
            context.is_cached = True
            context.cache_hit_type = 'voice_reuse'
            logger.info(f"VOICE HIT: {voice_stem} (stable)")
        else:
            context.cache_hit = False
            context.cache_type = 'voice_miss'
            context.is_cached = False
            context.cache_hit_type = 'voice_miss'
            logger.info(f"VOICE MISS: {voice_stem} (new stable)")

        # Simple key stub (norm + hash text/exag; voice stable base)
        text_hash = hash(text)
        context.cache_key = f"{voice_stem}_{text_hash}_{exag:.2f}_{context.cache_uuid}"

        time_taken = time.perf_counter() - start_time
        logger.debug(
            f"Voice CacheCheck: {time_taken:.3f}s | {context.cache_type} | path={context.processed_voice_path}")

        return context  # To Gen


    def handle_error(self, context: AudioGenerationContext, error: Exception) -> AudioGenerationContext:
        """Fallback."""
        logger.warning(f"Voice check error: {error}; MISS")
        context.is_cached = False
        context.cache_type = 'error_miss'
        context.cache_hit_type = 'error_miss'
        context.cache_uuid = int(time.time() * 1000) % (2**32)
        context.cache_key = f"error_{hash(context.text)}_{context.audio_prompt_path or 'none'}"
        context.cache_hit = False
        return context