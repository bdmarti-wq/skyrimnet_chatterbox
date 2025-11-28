import os
from .base import BaseGenerationPhase
from ...pipeline.context import AudioGenerationContext
from loguru import logger

class VoiceProcessingPhase(BaseGenerationPhase):
    def __init__(self, cache_manager=None):
        self.cache_manager = cache_manager
        self.conditionals_cache = getattr(cache_manager, 'conditionals_cache', None) if cache_manager else None
        super().__init__()

    def _execute_core(self, context: AudioGenerationContext) -> AudioGenerationContext:
        """Voice prep: Atomic cache get/prepare/save via _get_or_prepare."""
        context.ensure_attrs()

        conds_key = context.conditionals_key or context.generate_cache_key()
        processed_path = context.processed_voice_path
        voice_stem = context.voice_stem
        exag = context.exaggeration

        if conds_key and self.conditionals_cache and processed_path and os.path.exists(processed_path):
            globals_dict = context.get_globals()
            device_str = str(globals_dict['device'])
            dtype = globals_dict['dtype']
            conds = self.conditionals_cache._get_or_prepare(
                context.model, processed_path, exag, device_str, dtype, conds_key
            )
            if conds is not None:
                context.conds = conds
                context.conds_key = conds_key
                context.conds_from_cache = True
                logger.info(f"Conds processed/restored for {voice_stem} via {conds_key[:20]}... (cache HIT or fresh compute)")
                return context
            else:
                logger.warning(f"Conds processing failed for {voice_stem} ({conds_key[:20]}...) – fallback to dummy")

        # Fallback: do NOT fabricate partial Conditionals – leave None so later phases can silence-fallback safely
        globals_dict = context.get_globals()
        context.model.conds = None
        context.conds = None
        context.conds_key = f"dummy_{voice_stem}"
        context.conds_from_cache = False
        logger.warning(
            f"Conds unavailable for '{voice_stem}'; proceeding without conds (silence fallback will be used)."
        )
        return context
    def handle_error(self, context: AudioGenerationContext, error: Exception) -> AudioGenerationContext:
        context.voice_params = {'exaggeration': 1.0}
        context.conditionals_key = f"error_{context.voice_stem or 'default'}"
        context.conds_key = context.conditionals_key
        return super().handle_error(context, error)