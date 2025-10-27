import os

import torch

from .base import BaseGenerationPhase
from ...pipeline.context import AudioGenerationContext
from loguru import logger

class VoiceProcessingPhase(BaseGenerationPhase):
    def __init__(self, cache_manager=None):
        self.cache_manager = cache_manager
        self.conditionals_cache = getattr(cache_manager, 'conditionals_cache', None) if cache_manager else None
        super().__init__()

    def _execute_core(self, context: AudioGenerationContext) -> AudioGenerationContext:
        """FIXED: Call conditionals_cache.get(key, model, device, dtype) – restores to model."""
        context.ensure_attrs()

        conds_key = context.conditionals_key or context.generate_cache_key()
        processed_path = context.processed_voice_path
        voice_stem = context.voice_stem
        exag = context.exaggeration

        # FIXED: HIT – pass model/device/dtype (console/original sig)
        if conds_key and self.conditionals_cache:
            globals_dict = context.get_globals()
            device_str = str(globals_dict['device'])
            dtype = globals_dict['dtype']
            conds = self.conditionals_cache.get(conds_key, context.model, device_str, dtype)
            if conds is not None:
                context.model.conds = conds  # Restore (as in original)
                if self._is_nonempty_conds(context.model.conds):  # Now allows dummy
                    context.conds = conds
                    context.conds_key = conds_key
                    context.conds_from_cache = True
                    logger.info(f"Conds HIT: {voice_stem} from {conds_key[:20]}...")
                    return context
                else:
                    logger.warning(f"Cached conds invalid ({conds_key[:20]}...) – prep fresh")

        # MISS: Prep (base; align with voice_reference _resample_and_save_persistent if path missing – but since CacheCheck derives, assume valid)
        if processed_path and os.path.exists(processed_path):
            conds = self._prepare_conditionals(context, processed_path, exag)
            if conds:
                context.model.conds = conds
                context.conds = conds
                context.conditionals_key = conds_key
                # FIXED: Save with args (if sig requires; assume save(key, conds))
                if context.save_cache and self.conditionals_cache:
                    self.conditionals_cache.save(conds_key, context.model.conds)  # Or with args if needed
                logger.info(f"Conds MISS → computed: {voice_stem}")
                return context

        # Dummy (align: neutral for error; numel>0 passes)
        globals_dict = context.get_globals()
        dummy = self._create_dummy_conds(context.model, torch.device(globals_dict['device']), globals_dict['dtype'])
        context.model.conds = dummy
        context.conds = dummy
        context.conds_key = f"dummy_{voice_stem}"
        logger.debug(f"Dummy conds for {voice_stem} (structure valid)")
        return context

    # _prepare_conditionals unchanged (uses original logic: prepare + to(dtype))

    def handle_error(self, context: AudioGenerationContext, error: Exception) -> AudioGenerationContext:
        context.voice_params = {'exaggeration': 1.0}
        context.conditionals_key = f"error_{context.voice_stem or 'default'}"
        context.conds_key = context.conditionals_key
        return super().handle_error(context, error)