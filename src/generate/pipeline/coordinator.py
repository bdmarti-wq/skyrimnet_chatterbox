"""
Generation Coordinator: Orchestrates the TTS pipeline phases with caching, error handling, and logging.
Handles warmup, phase execution, fallbacks, and performance metrics (RTF).
"""
import time
import os
import torch
from pathlib import Path
from loguru import logger
from typing import Dict, Any, Optional

from src.config import get_config
from src.tts_model import get_model, warmup_t3  # Warmup from tts_model
from src.generate.cache import CacheManager  # For injection
from src.generate.pipeline.phases.base import GenerationPhase  # If base needed
from src.generate.pipeline.phases.inputs_validation import InputsValidationPhase
from src.generate.pipeline.phases.check_cache import CacheCheckPhase
from src.generate.pipeline.phases.voice_processing import VoiceProcessingPhase
from src.generate.pipeline.phases.generation import GenerationPhase as TTSGenerationPhase
from src.generate.pipeline.phases.post_processing import PostProcessingPhase
from src.generate.pipeline.phases.output import OutputPhase
from src.generate.pipeline.context import AudioGenerationContext
from src.normalize_stem import normalize_stem


class GenerationCoordinator:
    """Manages the full generation pipeline with phases and caching."""

    def __init__(self, config=None):
        """Initialize coordinator with cache_manager and phases (injected). FIXED: Ensure config loaded (app_config not None)."""
        self.config = config or get_config()
        # FIXED: Force load/reload to ensure app_config is populated (handles missing/failed load)
        if self.config.app_config is None:
            logger.warning("Config app_config None; forcing load_config()")
            self.config.load_config()
        if self.config.app_config is None:  # Critical fallback
            logger.error("Config load failed completely; using minimal defaults")
            from src.config.models import AppConfig
            self.config.app_config = AppConfig()  # Direct instance

        # Now safe: Create cache_manager with populated config
        self.cache_manager = CacheManager(self.config)  # Pass loaded config

        # FIXED: Inject cache_manager into phases that need it
        self.inputs_validation_phase = InputsValidationPhase()  # Stateless
        self.cache_check_phase = CacheCheckPhase(self.cache_manager)  # Needs cache
        self.voice_processing_phase = VoiceProcessingPhase(self.cache_manager)  # Needs cache/conds
        self.generation_phase = TTSGenerationPhase()  # Model/stateful; assume has self.model
        self.post_processing_phase = PostProcessingPhase()  # Stateless
        self.output_phase = OutputPhase(self.cache_manager)  # Needs for fuzzy/exact

        # Phases list (order: validation → cache → voice → gen → post → output)
        self.phases = [
            self.inputs_validation_phase,
            self.cache_check_phase,
            self.voice_processing_phase,
            self.generation_phase,
            self.post_processing_phase,
            self.output_phase
        ]
        self.model = get_model()  # Assume from tts_model; warmup here if not
        logger.info("Coordinator initialized with injected cache_manager and loaded config")

    def _warmup_models(self):
        """Warmup T3 models for performance (graphs captured). FIXED: Deprecated – call only if needed (e.g., new model load); global in tts_model.py."""
        logger.info("Warming up T3 models for improved performance")
        start_time = time.perf_counter()
        warmup_t3(self.model)  # From tts_model
        warmup_time = time.perf_counter() - start_time
        logger.info(f"T3 warmup complete (manual graphs ready) | time: {warmup_time:.2f}s")

    def run(self, context: AudioGenerationContext) -> AudioGenerationContext:
        """Run full pipeline: Phases + timing/logs. FIXED: Safe attr access in fallbacks; REMOVED per-run warmup (global in model load)."""
        start_total = time.perf_counter()
        context.model = self.model  # Ensure model in context
        context.cache_manager = self.cache_manager  # Direct access if needed (fallback)
        # FIXED: Inject config if context.config None (for sr/multilingual props)
        if context.config is None:
            context.config = self.config

        # FIXED: Skip per-run warmup – already done in tts_model.py load() (global, one-time)
        # Optional: If context.needs_warmup (e.g., new multilingual): self._warmup_models()
        # self._warmup_models()  # REMOVED: Causes 4-6s overhead per gen

        phase_times = {}
        for phase in self.phases:
            phase_name = phase.__class__.__name__
            phase_start = time.perf_counter()
            logger.debug(f"Starting phase: {phase_name}")
            try:
                context = phase.execute(context)
            except Exception as phase_e:
                logger.error(f"Phase {phase_name} failed: {str(phase_e)}")
                # Try handle_error (phase-specific)
                try:
                    context = phase.handle_error(context, phase_e)
                except Exception as handle_e:
                    logger.error(f"Handle_error also failed for {phase_name}: {str(handle_e)}")
                    # Coordinator fallback: Silence + defaults
                    context = self._fallback_context(context, phase_e)
                phase_times[phase_name] = time.perf_counter() - phase_start
                continue  # Next phase

            phase_times[phase_name] = time.perf_counter() - phase_start

        total_time = time.perf_counter() - start_total
        self._log_pipeline_results(total_time, phase_times, context)
        logger.info(f"Pipeline complete: {total_time:.2f}s total | Generated audio")
        return context

    def _fallback_context(self, context: AudioGenerationContext, error: Exception) -> AudioGenerationContext:
        """Coordinated fallback: Silence wav + defaults; safe config. FIXED: Nested globals."""
        logger.warning(f"Full fallback due to {error}; silence tensor")
        globals_config = getattr(self.config.app_config, 'globals', None)  # Nested safe
        sr = getattr(globals_config, 'sr', 24000) if globals_config else 24000
        silence = torch.zeros(1, sr * 2, dtype=torch.float32, device='cpu')  # 2s @ sr
        context.generated_wav = silence
        context.output_path = ""  # Will temp save in output
        context.is_cached = False
        context.voice_ref_processed = False
        context.conditionals_key = "fallback_silence"
        # Set missing attrs safely
        if not hasattr(context, 'voice_params'):
            context.voice_params = {}
        if not hasattr(context, 'processed_ref_path'):
            context.processed_ref_path = ""
        context.audio_duration = 2.0  # For logs
        logger.info(f"Silence fallback created: {silence.shape} @ {sr}Hz")
        return context

    def _log_pipeline_results(self, total_time: float, phase_times: Dict[str, float], context: AudioGenerationContext):
        """Log performance: Total RTF, phase times, cache status. FIXED: Nested globals.sr; ADDED conds stats."""
        globals_config = getattr(self.config.app_config, 'globals', None)
        sr = getattr(globals_config, 'sr', 24000) if globals_config else 24000
        audio_dur = context.audio_duration  # Use attr (set by phases or property)
        rtf_total = audio_dur / total_time if total_time > 0 else float('inf')
        cache_type = getattr(context, 'cache_hit_type', 'miss')
        core_gen_time = phase_times.get('TTSGenerationPhase', 0)  # Assume class name
        core_rtf = audio_dur / core_gen_time if core_gen_time > 0 else float('inf')
        validation_time = phase_times.get('InputsValidationPhase', 0)

        log_msg = f"{cache_type.upper()} cycle: {total_time:.2f}s total | {sr//1000}kHz {audio_dur:.2f}s | RTF total: {rtf_total:.2f}x | core gen RTF: {core_rtf:.2f}x | validation: {validation_time:.2f}s"
        for phase, t in phase_times.items():
            log_msg += f" | {phase}: {t:.2f}s"
        if cache_type != 'miss':
            log_msg += f" | cache: {cache_type}"

        # NEW: Log conds cache stats (hits/misses for debugging)
        conds_stats = {}
        if hasattr(context, 'cache_manager') and hasattr(context.cache_manager, 'conditionals_cache'):
            conds_stats = context.cache_manager.conditionals_cache.get_stats().get('stats', {})
        log_msg += f" | conds cache: hits={conds_stats.get('disk_hits', 0)+conds_stats.get('memory_hits', 0)}, misses={conds_stats.get('misses', 0)}"

        logger.info(log_msg)