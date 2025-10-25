"""
Generation Coordinator: Orchestrates the TTS pipeline phases with caching, error handling, and logging.
Handles warmup, phase execution, fallbacks, and performance metrics (RTF).
"""
import time
from pathlib import Path
from loguru import logger
from typing import Dict, Any, Optional

from src.config import get_config
from src.tts_model import get_model  # Warmup global in tts_model
from src.generate.cache import CacheManager
from src.generate.pipeline.phases.inputs_validation import InputsValidationPhase
from src.generate.pipeline.phases.check_cache import CacheCheckPhase
from src.generate.pipeline.phases.voice_processing import VoiceProcessingPhase
from src.generate.pipeline.phases.generation import GenerationPhase as TTSGenerationPhase
from src.generate.pipeline.phases.post_processing import PostProcessingPhase
from src.generate.pipeline.phases.output import OutputPhase
from src.generate.pipeline.context import AudioGenerationContext


class GenerationCoordinator:
    """Manages the full generation pipeline with phases and caching. FIXED: Bypass on audio HIT."""

    def __init__(self, config=None):
        """Assume valid config (raise if None); inject cache_manager once."""
        self.config = config or get_config()
        if self.config is None or not hasattr(self.config, 'app_config'):
            raise ValueError("Invalid config – cannot initialize coordinator")

        self.cache_manager = CacheManager(self.config)

        # Phases (inject cache_manager where needed)
        self.phases = [
            InputsValidationPhase(),
            CacheCheckPhase(self.cache_manager),
            VoiceProcessingPhase(self.cache_manager),
            TTSGenerationPhase(),
            PostProcessingPhase(),
            OutputPhase(self.cache_manager)
        ]
        self.model = get_model()  # From singleton
        logger.info("Coordinator initialized")

    def run(self, context: AudioGenerationContext) -> AudioGenerationContext:
        """Inject once; check skip_pipeline after CacheCheck (bypass Voice/Gen/Post on audio HIT)."""
        start_total = time.perf_counter()
        context.model = self.model
        context.cache_manager = self.cache_manager
        context.config = self.config

        phase_times = {}

        for i, phase in enumerate(self.phases):
            phase_name = phase.__class__.__name__
            phase_start = time.perf_counter()

            # Bypass on audio HIT (after CacheCheck; skip Voice/Gen/Post)
            if (hasattr(context, 'skip_pipeline') and context.skip_pipeline and
                i >= 2):  # From VoiceProcessingPhase (idx 2) onward
                logger.info(f"Skipping {phase_name} on full audio cache HIT (use cached_path)")
                phase_times[phase_name] = 0.0
                # Pre-set output for final Output if jumped
                if phase_name == 'OutputPhase' and hasattr(context, 'cached_path'):
                    context.output_path = context.cached_path
                continue

            context = phase.execute(context)
            phase_times[phase_name] = time.perf_counter() - phase_start

        total_time = time.perf_counter() - start_total
        self._log_pipeline_results(total_time, phase_times, context)
        logger.info(f"Pipeline complete: {total_time:.2f}s | Generated audio")
        return context

    def _fallback_context(self, context: AudioGenerationContext, error: Exception) -> AudioGenerationContext:
        """Use shared base _fallback_silence. Remove dupes."""
        logger.warning(f"Coordinator fallback due to {error}")
        # Fallback to first phase's handle_error or implement shared
        from src.generate.pipeline.phases.base import GenerationPhase
        return GenerationPhase()._fallback_silence(context, str(error))  # Static call

    def _log_pipeline_results(self, total_time: float, phase_times: Dict[str, float], context: AudioGenerationContext):
        """Use context globals/sr/audio_duration (DRY). Simplify conds stats."""
        globals_dict = context.get_globals()
        sr = globals_dict['sr']
        audio_dur = context.audio_duration
        rtf_total = audio_dur / total_time if total_time > 0 else float('inf')
        cache_type = getattr(context, 'cache_hit_type', 'miss')
        core_gen_time = phase_times.get('TTSGenerationPhase', 0)
        core_rtf = audio_dur / core_gen_time if core_gen_time > 0 else float('inf')

        log_msg = f"{cache_type.upper()} cycle: {total_time:.2f}s total | {sr//1000}kHz {audio_dur:.2f}s | RTF total: {rtf_total:.2f}x | core gen RTF: {core_rtf:.2f}x"
        for phase, t in phase_times.items():
            log_msg += f" | {phase}: {t:.2f}s"
        if cache_type != 'miss':
            log_msg += f" | cache: {cache_type}"

        # Conds stats (if cache_manager)
        if context.cache_manager and hasattr(context.cache_manager.conditionals_cache, 'get_stats'):
            stats = context.cache_manager.conditionals_cache.get_stats().get('stats', {})
            hits = stats.get('disk_hits', 0) + stats.get('memory_hits', 0)
            log_msg += f" | conds cache: hits={hits}, misses={stats.get('misses', 0)}"

        logger.info(log_msg)