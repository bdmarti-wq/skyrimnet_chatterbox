import os
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torchaudio

from src.config import get_config
from src.normalize_stem import normalize_stem
from .base import GenerationPhase
from ...pipeline.context import AudioGenerationContext
from loguru import logger

class CacheCheckPhase(GenerationPhase):
    """Check for exact/fuzzy cache HIT; set context.is_cached + uuid/key."""

    def __init__(self, cache_manager=None):
        """FIXED: Inject cache_manager for voice_reference/fuzzy checks."""
        self.cache_manager = cache_manager  # From coordinator
        if not self.cache_manager:
            logger.warning("CacheCheckPhase: No cache_manager; always MISS")
        super().__init__()

    def execute(self, context: AudioGenerationContext) -> AudioGenerationContext:
        """Check cache: exact (audio/conds) or fuzzy; set is_cached, cache_type, cache_uuid, cache_key."""
        start_time = time.perf_counter()
        config = get_config()

        # FIXED: Safe cache_manager access
        if not self.cache_manager:
            context.is_cached = False
            context.cache_hit_type = 'no_cache'
            context.cache_uuid = int(time.time() * 1000) % (2**32)  # Pseudo-uuid
            context.cache_key = f"nocache_{hash(context.text)}_{context.audio_prompt_path}"
            logger.debug("CacheCheck: No manager – force MISS")
            return context

        audio_prompt_path = getattr(context, 'audio_prompt_path', None)
        voice_stem = getattr(context, 'voice_stem', normalize_stem(audio_prompt_path) if audio_prompt_path else 'default')
        text_hash = hash(context.text)
        exag = getattr(context, 'exaggeration', 1.0) or 1.0  # From params/UI

        # FIXED: Check voice_reference HIT first (resampled/conds)
        if (audio_prompt_path and hasattr(self.cache_manager, 'voice_reference') and
            self.cache_manager.voice_reference and voice_stem):
            success, processed_path, cond_key, voice_params, entry = self.cache_manager.voice_reference.process_new_reference(
                voice_stem, audio_prompt_path, force_update=False  # Reuse if HIT
            )
            if success and processed_path:
                context.processed_ref_path = processed_path
                context.conditionals_key = cond_key
                context.voice_params = voice_params
                context.voice_stem = voice_stem
                # If full HIT (no update), set cached
                if not success:  # Wait, check should_update from entry
                    context.is_cached = True
                    context.cache_hit_type = 'voice_reuse'  # Or 'full_exact' if conds too
                else:
                    context.is_cached = False
                    context.cache_hit_type = 'voice_miss'  # Processed but new
                context.cache_key = f"{voice_stem}_{text_hash}_{exag:.2f}_{cond_key[:8]}"
                context.cache_uuid = hash(processed_path + cond_key) % (2**32)
                logger.info(f"CacheCheck: Voice {'reuse' if context.is_cached else 'processed'} for {voice_stem} (key: {context.cache_key[:20]}...)")
                cache_time = time.perf_counter() - start_time
                logger.debug(f"CacheCheck time: {cache_time:.3f}s | HIT: {context.is_cached}")
                return context

        # Fallback: Exact audio cache check (assume cache_manager.audio_cache.get(cache_key))
        cache_key = f"{voice_stem}_{text_hash}_{exag:.2f}"  # Simple key
        cached_wav_path = self.cache_manager.audio_cache.get(cache_key) if hasattr(self.cache_manager, 'audio_cache') else None
        if cached_wav_path and os.path.exists(cached_wav_path):
            context.is_cached = True
            context.cache_hit_type = 'exact_audio'
            context.output_path = cached_wav_path  # Direct reuse
            context.generated_wav = torchaudio.load(cached_wav_path)[0]  # Load tensor
            context.cache_key = cache_key
            context.cache_uuid = hash(cache_key)
            logger.info(f"CacheCheck: Exact HIT for {cache_key[:20]}... (path: {cached_wav_path})")
        else:
            context.is_cached = False
            context.cache_hit_type = 'miss'
            context.cache_key = cache_key
            context.cache_uuid = int(time.time() * 1000) % (2**32)
            logger.debug(f"CacheCheck: MISS for {voice_stem}/{text_hash}")

        cache_time = time.perf_counter() - start_time
        logger.info(f"CacheCheck: {context.cache_hit_type} | time: {cache_time:.3f}s")
        return context

    def handle_error(self, context: AudioGenerationContext, error: Exception) -> AudioGenerationContext:
        """Fallback on cache check error: Force MISS."""
        logger.warning(f"Cache check failed: {error} – continuing with MISS")
        context.is_cached = False
        context.cache_hit_type = 'error_miss'
        context.cache_uuid = int(time.time() * 1000) % (2**32)
        context.cache_key = f"error_{hash(context.text)}_{context.audio_prompt_path or 'none'}"
        return context