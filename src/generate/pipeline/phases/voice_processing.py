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

        globals_dict = context.get_globals()
        dummy = self._create_dummy_conds(context.model, torch.device(globals_dict['device']), globals_dict['dtype'])
        context.model.conds = dummy
        context.conds = dummy
        context.conds_key = f"dummy_{voice_stem}"
        logger.debug(f"Dummy conds for {voice_stem} (structure valid)")
        return context

    def _prepare_conditionals(self, context, processed_path, exag):
        """Prepare conditionals from processed voice audio. Assumes this sets model.conds internally."""
        # Original implementation: e.g., model.prepare_conditionals(processed_path, exag), then conds = model.conds.to(dtype)
        # Note: Called internally by _get_or_prepare if MISS.
        raise NotImplementedError("Implement _prepare_conditionals based on original logic (e.g., model.prepare_conditionals + .to(dtype))")

    def _create_dummy_conds(self, model, device, dtype):
        """Create a dummy Conditionals object for fallback (zero embeddings)."""
        # Original implementation: e.g., dummy = type(model.conds)(); dummy.t3 = ... (empty tensors on device/dtype)
        # Returns a valid structure but empty (numel=0 or small zero tensor).
        dummy = model.conds.__class__()  # Assuming model.conds is a dataclass or similar
        dummy.speaker_emb = torch.zeros(1, dtype=dtype, device=device)
        # Add other attrs as per Conditionals structure (e.g., t3, etc.)
        return dummy

    def _is_nonempty_conds(self, conds):
        """Check if conds is valid/non-empty (optional, now handled by cache)."""
        # Original implementation: e.g., return hasattr(conds, 't3') and len(conds.t3.speaker_emb) > 0
        if conds is None:
            return False
        # Assuming conds is object; check key attrs
        return hasattr(conds, 'speaker_emb') and conds.speaker_emb.numel() > 0

    def handle_error(self, context: AudioGenerationContext, error: Exception) -> AudioGenerationContext:
        context.voice_params = {'exaggeration': 1.0}
        context.conditionals_key = f"error_{context.voice_stem or 'default'}"
        context.conds_key = context.conditionals_key
        return super().handle_error(context, error)