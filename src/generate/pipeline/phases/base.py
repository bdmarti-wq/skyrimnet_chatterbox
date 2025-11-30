import os
import time
from typing import Optional, Dict, Any
from loguru import logger
from src.generate.pipeline.context import AudioGenerationContext
import torch
from pathlib import Path
import torchaudio
from src.audio_paths import validate_user_audio
from src.generate.cache.conditionals_utils import is_valid_conditionals

class BaseGenerationPhase:
    """Base class for all audio generation pipeline phases. FIXED: _is_nonempty_conds allows dummy (structure OK)."""

    process_after_cache: bool = False

    def _validate(self, context: AudioGenerationContext) -> Optional[str]:
        """Shared validation. FIXED: Allow dummy structure."""
        if not context.text or not context.text.strip():
            return "Empty text input"

        context.ensure_attrs()

        if context.audio_prompt_path and not self.validate_path(context.audio_prompt_path, min_dur=1.0):
            return f"Invalid audio_prompt_path: {context.audio_prompt_path}"

        if context.processed_voice_path and context.processed_voice_path != "" and not self.validate_path(context.processed_voice_path, min_dur=3.0):
            return f"Invalid processed_voice_path: {context.processed_voice_path}"

        if not context.model:
            return "Missing model"

        if hasattr(context, 'conds') and context.conds is not None:
            if not self._is_nonempty_conds(context.conds):
                return "Empty conditionals"

        globals_dict = context.get_globals()
        if not globals_dict:
            return "Invalid globals"

        return None

    def execute(self, context: AudioGenerationContext) -> AudioGenerationContext:
        """Wrap with timing + _validate."""
        phase_name = self.__class__.__name__
        start_time = time.perf_counter()
        context.step_times[phase_name] = 0.0

        error = self._validate(context)
        if error:
            logger.warning(f"[PHASE:{phase_name}] validation failed: {error}")
            return self.handle_error(context, ValueError(error))

        logger.debug(f"[PHASE:{phase_name}] starting")
        try:
            context = self._execute_core(context)
        except Exception as phase_e:
            logger.error(f"[PHASE:{phase_name}] failed: {str(phase_e)}")
            context = self.handle_error(context, phase_e)

        elapsed = time.perf_counter() - start_time
        context.step_times[phase_name] = elapsed
        logger.debug(f"[PHASE:{phase_name}] done in {elapsed:.3f}s")
        return context

    def _execute_core(self, context: AudioGenerationContext) -> AudioGenerationContext:
        raise NotImplementedError("Subclasses must implement _execute_core()")

    def handle_error(self, context: AudioGenerationContext, error: Exception) -> AudioGenerationContext:
        logger.error(f"[PHASE:{self.__class__.__name__}] error: {str(error)} – silence fallback")
        return self._fallback_silence(context, str(error))

    def _fallback_silence(self, context: AudioGenerationContext, error_msg: str) -> AudioGenerationContext:
        globals_dict = context.get_globals()
        sr = globals_dict['sr']
        device = torch.device(globals_dict['device'])
        dtype = globals_dict['dtype']
        silence = self.create_silence(sr, 2.0, device, dtype)
        if hasattr(context, 'generated_wav'):
            context.generated_wav = silence
        if hasattr(context, 'processed_wav'):
            context.processed_wav = silence
        context.audio_duration = 2.0
        context.output_path = ""
        if hasattr(context, 'is_cached'):
            context.is_cached = False
        if hasattr(context, 'cache_hit_type'):
            context.cache_hit_type = 'error_fallback'
        if hasattr(context, 'conditionals_key') and not context.conditionals_key:
            context.conditionals_key = "silence_fallback"
        context.ensure_attrs()
        logger.warning(f"[PHASE:{self.__class__.__name__}] shared silence fallback ({silence.shape} @ {sr}Hz) due to: {error_msg}")
        return context

    @classmethod
    def create_silence(cls, sr: int, duration: float = 2.0, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float32) -> torch.Tensor:
        samples = int(sr * duration)
        return torch.zeros((1, samples), dtype=dtype, device=device)

    @classmethod
    def validate_path(cls, path: str, min_dur: float = 3.0) -> bool:
        """Shared audio path validator wrapper.

        Delegates to src.audio_paths.validate_user_audio for consistent behavior
        across UI bridge, caches, and phases.
        """
        return validate_user_audio(path, min_dur=min_dur)

    @classmethod
    def _is_nonempty_conds(cls, conds: Any) -> bool:
        """Delegate to shared conditionals validator (single source of truth)."""
        return is_valid_conditionals(conds)

    # Removed unused conditionals preparation helpers; conds are handled via caches/voice phase.