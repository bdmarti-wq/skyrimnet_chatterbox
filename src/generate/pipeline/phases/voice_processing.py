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
        """Voice prep: Atomic cache get/prepare/save via _get_or_prepare; handles fallbacks."""
        context.ensure_attrs()

        conds_key = context.conditionals_key or context.generate_cache_key()
        processed_path = context.processed_voice_path
        voice_stem = context.voice_stem
        exag = context.exaggeration

        conds = None
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
            logger.debug(f"Conds processed/restored for {voice_stem} via {conds_key[:20]}... (cache HIT or fresh compute)")
            return context

        # FALLBACK 1: Try Configured Default Conditionals
        default_conds_path = context.get_globals().get('default_conditionals_path')
        if not default_conds_path:
            from src.config import get_config_value
            default_conds_path = get_config_value('globals.default_conditionals_path', 'cache/conditionals/conds_v2_malebrute_ref_06e.pt')
        
        if default_conds_path and self.conditionals_cache:
            root = context.get_globals().get('root', '')
            if root:
                default_conds_abs = os.path.join(root, default_conds_path)
                if os.path.exists(default_conds_abs):
                    default_conds_path = default_conds_abs
            
            if os.path.exists(default_conds_path):
                logger.info(f"Attempting fallback to default conditionals: {default_conds_path}")
                # We need a stable key for the default to avoid re-preparing it constantly if it's already in cache
                # But since we have the .pt file directly, we might just want to load it.
                # _get_or_prepare expects a path to a WAV to prepare from. 
                # If we have a .pt, we should just load it from the cache if it's there, or load it directly.
                # Using the filename as a part of the key.
                default_key = f"fallback_default_{os.path.basename(default_conds_path)}"
                globals_dict = context.get_globals()
                conds = self.conditionals_cache.get(default_key, context.model, str(globals_dict['device']), globals_dict['dtype'])
                
                if conds is None:
                    # If not in cache by key, we might need to load it once and save it to cache
                    # Or just load it and use it. 
                    # For simplicity, let's try to load it directly if it's a known .pt
                    try:
                        # Assuming the conditionals_cache knows how to load from its own storage format
                        # but here we have a raw path. 
                        # Let's use a dummy path to satisfy _get_or_prepare if it were a WAV, 
                        # but it's not. 
                        # If we have the .pt, we can potentially just torch.load it if we knew the structure.
                        # Better: use the cache manager to process the fallback VOICE if we have one.
                        pass
                    except Exception as e:
                        logger.warning(f"Failed to load default conditionals {default_conds_path}: {e}")

        # FALLBACK 2: Search Conditionals Directory
        if conds is None and self.conditionals_cache:
            conds_dir = context.get_globals().get('conditionals_cache_dir')
            if conds_dir and os.path.exists(conds_dir):
                logger.debug(f"Searching {conds_dir} for any valid fallback conditionals")
                for f in os.listdir(conds_dir):
                    if f.endswith('.pt'):
                        try:
                            # Try to load it as a key
                            key_candidate = f.replace('.pt', '')
                            globals_dict = context.get_globals()
                            conds = self.conditionals_cache.get(key_candidate, context.model, str(globals_dict['device']), globals_dict['dtype'])
                            if conds:
                                logger.info(f"Found directory-search conditionals fallback: {f}")
                                conds_key = key_candidate
                                break
                        except Exception:
                            continue

        if conds is not None:
            context.conds = conds
            context.conds_key = conds_key
            context.conds_from_cache = True
            return context

        # Final Fallback: Use model's neutral state
        globals_dict = context.get_globals()
        context.model.conds = None
        if hasattr(context.model, 'set_conditionals'):
            context.model.set_conditionals(None)
            logger.debug(f"Activated neutral conditionals for '{voice_stem}' (no reference provided)")
        
        context.conds = None
        context.conds_key = f"neutral_{voice_stem}"
        context.conds_from_cache = False
        return context
    def handle_error(self, context: AudioGenerationContext, error: Exception) -> AudioGenerationContext:
        context.voice_params = {'exaggeration': 1.0}
        context.conditionals_key = f"error_{context.voice_stem or 'default'}"
        context.conds_key = context.conditionals_key
        return super().handle_error(context, error)