# REFACTORED: Integrate _generate_audio_core into _execute_core; use base _prepare_conditionals/validate; context globals.
# Remove dupes: conds restore/prep in shared base; silence in base; logs simplified.
import time
from typing import Any, Dict, Optional

import torch
from src.seeding import set_seed
from .base import GenerationPhase
from ...pipeline.context import AudioGenerationContext
from loguru import logger
from src.tts_model import GEN_ACTIVE_LOCK, create_dummy_conds  # Minimal imports

class GenerationPhase(GenerationPhase):  # Inherit from base
    def _execute_core(self, context: AudioGenerationContext) -> AudioGenerationContext:
        """REFACTORED: Base validate; restore/prepare conds (shared); build args; generate w/ retry."""
        # Shared validate (base calls _validate)
        if not context.conditionals_key or context.model is None:
            logger.error("No conds or model – fallback silence")
            return self._fallback_silence(context, "No conds/model")

        context.ensure_attrs()  # Paths/seeds (DRY)

        # REFACTORED: Conds restore/validate (use base _is_nonempty_conds; prep if missing)
        conds = getattr(context, 'conds', None)
        if conds is not None and self._is_nonempty_conds(conds):
            context.model.conds = conds
            logger.debug("Restored conds from context")
        elif context.processed_voice_path and self.validate_path(str(context.processed_voice_path)):
            # Shared prep (base method)
            new_conds = self._prepare_conditionals(context, context.processed_voice_path, context.exaggeration)
            if new_conds:
                context.model.conds = new_conds
                context.conds = new_conds  # Preserve
            else:
                raise ValueError("Conds prep failed")
        else:
            # Shared dummy (base)
            globals_dict = context.get_globals()
            self._create_dummy_conds(context.model, torch.device(globals_dict['device']), globals_dict['dtype'])

        # Final conds check (shared base)
        if not self._is_nonempty_conds(context.model.conds):
            logger.error("Empty conds post-prep – fallback")
            return self._fallback_silence(context, "Empty conds")

        # Build args (REFACTORED)
        gen_args = self._build_generate_args(context)

        # Generate (integrate core; shared eager retry)
        if context.seed is not None:
            set_seed(context.seed)
        logger.debug(f"Gen args: text_len={len(context.text)}, conds_shapes={self._get_conds_shapes(context.model.conds)}")

        wav = self._generate_core(context.model, gen_args, context.t3_params)
        if wav is None or wav.numel() == 0:
            raise ValueError("Generated empty WAV")

        context.generated_wav = wav
        context.audio_duration = len(wav.squeeze(0)) / context.sr  # Auto via property fallback
        logger.info(f"Generation success: {context.audio_duration:.2f}s WAV @ {context.sr}Hz")

        # Preserve conds for downstream
        context.conds = context.model.conds
        return context

    def _generate_core(self, model: Any, gen_args: Dict, t3_params: Dict) -> Optional[torch.Tensor]:
        """REFACTORED: Integrated from _generate_audio_core; simplified eager retry; use base clear graphs."""
        gen_start = time.time()
        try:
            if 't3_params' not in gen_args:
                gen_args['t3_params'] = t3_params
            logger.debug("Calling model.generate...")
            return model.generate(**gen_args)
        except RuntimeError as graph_e:
            err_str = str(graph_e).lower()
            if any(term in err_str for term in ["graph", "capture", "offset", "stream is capturing"]):
                logger.warning(f"Graph/CUDA error: {graph_e} – clear & eager retry")
                # REFACTORED: Shared clear (base-like)
                if hasattr(model, 't3') and hasattr(model.t3, '_bucket_graphs'):
                    model.t3._bucket_graphs.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # Eager retry
                mod_args = gen_args.copy()
                mod_t3 = t3_params.copy()
                mod_t3['generate_token_backend'] = 'eager'
                mod_args['t3_params'] = mod_t3
                return model.generate(**mod_args)
            else:
                raise
        except Exception as e:
            logger.exception(f"Core gen failed: {e}")
            return None
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.debug(f"Gen complete: {time.time() - gen_start:.2f}s")

    def _build_generate_args(self, context: AudioGenerationContext) -> dict:
        """REFACTORED: Use ensured voice_params (dict); add path only if needed (guard)."""
        voice_params = context.voice_params  # Ensured dict
        generate_args = {
            'text': context.text,
            'exaggeration': context.exaggeration,  # Already merged in context
            'temperature': context.temperature,
            'cfg_weight': context.cfg_weight,
            'min_p': context.min_p,
            'top_p': context.top_p,
            'repetition_penalty': context.repetition_penalty,
            'language_id': context.language_id,
            't3_params': context.t3_params
        }

        # Override from voice_params (already dict, no guard needed)
        for key in ['temperature', 'top_p', 'repetition_penalty', 'exaggeration', 'min_p', 'cfg_weight']:
            if key in voice_params:
                generate_args[key] = voice_params[key]

        # REFACTORED: Path guard (only if conds None; "" if empty)
        prompt_path = context.audio_prompt_path or ""
        if context.model.conds is None and prompt_path.strip():
            generate_args['audio_prompt_path'] = prompt_path
            logger.debug(f"Added audio_prompt_path to args")

        return generate_args

    def _get_conds_shapes(self, conds: Any) -> dict:
        """Simplified log helper (internal; no dupes)."""
        shapes = {}
        if hasattr(conds, 'speaker_emb') and conds.speaker_emb is not None:
            shapes['speaker_emb'] = conds.speaker_emb.shape
        elif conds is not None:
            shapes['conds'] = f"Present (type: {type(conds).__name__})"
        else:
            shapes['conds'] = 'None'
        return shapes

    def handle_error(self, context: AudioGenerationContext, error: Exception) -> AudioGenerationContext:
        """REFACTORED: Delegate to base (silence + paths/attrs)."""
        logger.error(f"Generation error: {error} – silence fallback")
        # Ensure paths post-error (DRY)
        context.audio_prompt_path = getattr(context, 'audio_prompt_path', "") or ""
        context.processed_voice_path = getattr(context, 'processed_voice_path', "") or ""
        return super().handle_error(context, error)