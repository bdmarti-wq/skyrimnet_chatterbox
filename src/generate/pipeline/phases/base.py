# src/generate/pipeline/base.py
import time
from typing import Optional, Type
from loguru import logger
from src.generate.pipeline.context import AudioGenerationContext

class GenerationPhase:
    """Base class for all audio generation pipeline phases."""

    # Should we process this phase even if cache hit occurred earlier?
    process_after_cache: bool = False

    def execute(self, context: AudioGenerationContext) -> AudioGenerationContext:
        """Execute the main functionality of the phase."""
        raise NotImplementedError("Subclasses must implement execute()")

    def handle_error(
        self,
        context: AudioGenerationContext,
        error: Exception
    ) -> AudioGenerationContext:
        """Handle errors from this phase with appropriate fallbacks."""
        logger.error(f"Error in {self.__class__.__name__}: {str(error)}")
        return context

    def _start_timer(self, context: AudioGenerationContext) -> float:
        """Start timing for this phase."""
        start_time = time.perf_counter()
        context.timing[f"{self.__class__.__name__.lower()}_start"] = time.time()
        return start_time

    def _record_timer(
        self,
        context: AudioGenerationContext,
        start_time: float,
        phase_name: str
    ) -> float:
        """Record elapsed time for this phase."""
        elapsed = time.perf_counter() - start_time
        context.step_times[phase_name] = elapsed
        context.timing[f"{phase_name}_elapsed"] = elapsed
        return elapsed