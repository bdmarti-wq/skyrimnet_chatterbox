# REFACTORED: Use _build_generate_args helper for gen_args; set model.conds separately; no invalid kwargs.
# Silence fallback if no conds; retry only for graph errors; simplified logs.
import time
import torch
from typing import Dict, Optional, Any

from loguru import logger
from .base import GenerationPhase
from ...pipeline.context import AudioGenerationContext
from src.tts_model import GEN_ACTIVE_LOCK, create_dummy_conds  # Minimal imports

class GenerationPhase(GenerationPhase):  # Inherit from base
    def _execute_core(self, context: AudioGenerationContext) -> AudioGenerationContext:
        """Core TTS generation with conds. FIXED: Use _build_generate_args helper (proper args, no invalid kwargs)."""
        # Restore conds from cache if present (voice phase sets)
        if hasattr(context, 'conds') and context.conds is not None:
            context.model.conds = context.conds
            logger.debug("Restored conds from context")
        else:
            # Fallback stub if no conds (e.g., error)
            logger.warning("No conds in context; skipping gen (empty WAV)")
            context.processed_wav = torch.zeros((1, context.sr * 2), dtype=torch.float32, device=context.device)  # 2s silence
            return context

        # Build gen_args using helper (proper structure, no 'conditionals'/'seed' kwargs)
        gen_args = self._build_generate_args(context)

        # Log shapes/dims for debug (simple, no f-string issues)
        conds_desc = 'Present (type: Conditionals)' if context.model.conds else 'None'
        logger.debug(f"Conds shapes: {{'conds': '{conds_desc}'}}")
        logger.debug(f"Gen args: text_len={len(gen_args['text']) if gen_args['text'] else 0}, conds_shapes={{'conds': '{conds_desc}'}}")

        try:
            if len(gen_args['text'].strip()) == 0:
                raise ValueError("Empty text input")

            gen_start = time.perf_counter()
            context.processed_wav = self._generate_core(context.model, gen_args, context.t3_params)
            gen_time = time.perf_counter() - gen_start

            context.audio_duration = context.processed_wav.shape[1] / context.sr
            if context.audio_duration > 0:
                logger.info(f"Generation success: {context.audio_duration:.2f}s WAV @ {context.sr}Hz (gen time {gen_time:.2f}s)")
            else:
                raise ValueError("Generated empty WAV")

        except Exception as gen_e:
            logger.error(f"Core gen failed: {gen_e}")
            gen_args['text'] = gen_args['text'][:30] + '...' if len(gen_args['text']) > 30 else gen_args['text']
            msg = f"Gen error on '{gen_args['text']}' (exag={gen_args['exaggeration']}): {gen_e}"
            return self.handle_error(context, ValueError(msg))

        logger.debug("Gen complete")
        return context

    def _build_generate_args(self, context: AudioGenerationContext) -> dict:
        """Build args for model.generate() – uses context.voice_params; adds audio_prompt_path if conds None."""
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

        # Path guard (only if conds None; "" if empty)
        prompt_path = context.audio_prompt_path or ""
        if context.model.conds is None and prompt_path.strip():
            generate_args['audio_prompt_path'] = prompt_path
            logger.debug(f"Added audio_prompt_path to args")

        return generate_args

    def _generate_core(self, model: Any, gen_args: dict, t3_params: dict) -> Optional[torch.Tensor]:
        """REFACTORED: Integrated from _generate_audio_core; simplified retry; use base clear graphs."""
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

                # Eager retry (use original gen_args, override t3_params)
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

    def handle_error(self, context: AudioGenerationContext, error: Exception) -> AudioGenerationContext:
        """REFACTORED: Delegate to base (silence + paths/attrs)."""
        logger.error(f"Generation error: {error} – silence fallback")
        # Ensure paths post-error (DRY)
        context.audio_prompt_path = getattr(context, 'audio_prompt_path', "") or ""
        context.processed_voice_path = getattr(context, 'processed_voice_path', "") or ""
        return super().handle_error(context, error)