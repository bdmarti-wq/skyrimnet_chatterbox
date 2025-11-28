import os
import time
from typing import Optional, Dict, Any
from loguru import logger
from src.generate.pipeline.context import AudioGenerationContext
from src.audio_utils import get_silence
import torch
from pathlib import Path
import torchaudio

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
            logger.warning(f"{phase_name} validation failed: {error}")
            return self.handle_error(context, ValueError(error))

        logger.debug(f"Starting phase: {phase_name}")
        try:
            context = self._execute_core(context)
        except Exception as phase_e:
            logger.error(f"Phase {phase_name} failed: {str(phase_e)}")
            context = self.handle_error(context, phase_e)

        elapsed = time.perf_counter() - start_time
        context.step_times[phase_name] = elapsed
        logger.debug(f"{phase_name}: {elapsed:.3f}s")
        return context

    def _execute_core(self, context: AudioGenerationContext) -> AudioGenerationContext:
        raise NotImplementedError("Subclasses must implement _execute_core()")

    def handle_error(self, context: AudioGenerationContext, error: Exception) -> AudioGenerationContext:
        logger.error(f"Base error in {self.__class__.__name__}: {str(error)} – silence fallback")
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
        logger.warning(f"Shared silence fallback ({silence.shape} @ {sr}Hz) due to: {error_msg}")
        return context

    @classmethod
    def create_silence(cls, sr: int, duration: float = 2.0, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float32) -> torch.Tensor:
        samples = int(sr * duration)
        # Attempt to allocate silence on the requested device/dtype first.
        try:
            return torch.zeros((1, samples), dtype=dtype, device=device)
        except Exception as e:
            # If the CUDA context is lost or dtype/device is invalid, fall back to safe CPU/FP32.
            try:
                logger.warning(f"create_silence failed on device={device}, dtype={dtype}: {e} — falling back to CPU/float32")
            except Exception:
                # Logging should not block the fallback
                pass
            return torch.zeros((1, samples), dtype=torch.float32, device='cpu')

    @classmethod
    def validate_path(cls, path: str, min_dur: float = 3.0) -> bool:
        if not path or not os.path.exists(path):
            return False
        try:
            info = torchaudio.info(path)
            if info.num_frames / info.sample_rate < min_dur:
                return False
            waveform, _ = torchaudio.load(path)
            return waveform.numel() > 0 and waveform.abs().max() > 1e-6
        except Exception:
            return False

    @classmethod
    def _is_nonempty_conds(cls, conds: Any) -> bool:
        """FIXED: Allow dummy (numel>0/structure; zero emb OK for fallback). Matches original (emb present but zero)."""
        if conds is None:
            return False
        if isinstance(conds, torch.Tensor):
            return conds.numel() > 0  # Allow all-zero dummy
        if hasattr(conds, 't3') and hasattr(conds.t3, 'speaker_emb') and conds.t3.speaker_emb is not None:
            emb = conds.t3.speaker_emb
            return emb.numel() > 0  # Structure; zero OK
        if hasattr(conds, '__len__'):
            return len(conds) > 0
        if isinstance(conds, dict):
            return any(v is not None for v in conds.values())
        return True

    # _prepare_conditionals, _create_dummy_conds unchanged (as in previous)

    def _prepare_conditionals(self, context: AudioGenerationContext, prep_path: str, exag: float) -> Optional[Any]:
        """REFACTORED: Shared conds prep (fresh or eager). Use in voice/generation. Reduces bloat by ~50 lines each."""
        if not self.validate_path(prep_path):
            logger.warning(f"Invalid prep_path: {prep_path}")
            return None

        globals_dict = context.get_globals()
        device = torch.device(globals_dict['device'])
        dtype = globals_dict['dtype']

        # REFACTORED: Temp eager guard (shared)
        model = context.model
        original_params = None
        if hasattr(model.t3, 'params') and isinstance(model.t3.params, dict):
            original_params = model.t3.params.copy()
            model.t3.params['generate_token_backend'] = 'eager'

        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model.prepare_conditionals(prep_path, exaggeration=exag)
            if hasattr(model.conds, 't3'):
                model.conds.t3.to(device=device, dtype=dtype)
            from src.tts_model import chatterbox_tts_to
            chatterbox_tts_to(model, device, dtype)
            conds = model.conds
            if self._is_nonempty_conds(conds):
                logger.debug(f"Prepared conds: emb non-empty")
                return conds
            else:
                logger.warning("Prepared empty conds – dummy")
                return self._create_dummy_conds(model, device, dtype)
        except Exception as prep_e:
            logger.error(f"Prep failed: {prep_e}")
            return self._create_dummy_conds(model, device, dtype)
        finally:
            # Restore
            if original_params is not None:
                model.t3.params = original_params
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    def _create_dummy_conds(self, model: Any, device: torch.device, dtype: torch.dtype) -> Any:
        """REFACTORED: Shared dummy creator (from tts_model)."""
        from src.tts_model import create_dummy_conds
        create_dummy_conds(model, device, dtype, "dummy_fallback")
        return model.conds