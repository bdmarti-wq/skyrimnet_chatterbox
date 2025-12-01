# REFACTORED: Use _build_generate_args helper for gen_args; set model.conds separately; no invalid kwargs.
# Silence fallback if no conds; retry only for graph errors; simplified logs.
import time
import torch
from typing import Dict, Optional, Any

from loguru import logger
from .base import BaseGenerationPhase
from ...pipeline.context import AudioGenerationContext
# Note: GEN_ACTIVE_LOCK and create_dummy_conds are not used directly in this phase.


class GenerationPhase(BaseGenerationPhase):
    def _execute_core(self, context: AudioGenerationContext) -> AudioGenerationContext:
        """Core TTS generation with conds. FIXED: Set generated_wav for PostProcessing; defer processed_wav."""
        # Restore conds from cache if present (voice phase sets)
        if hasattr(context, 'conds') and context.conds is not None:
            context.model.conds = context.conds
            logger.debug("Restored conds from context")
        else:
            # Fallback stub if no conds (e.g., error)
            logger.warning("No conds in context; skipping gen (empty WAV)")
            # Use shared silence helper for consistency – force CPU to avoid CUDA usage in fallback
            globals_dict = context.get_globals()
            sr = globals_dict['sr']
            dtype = globals_dict['dtype']
            context.processed_wav = self.create_silence(sr, 2.0, torch.device('cpu'), dtype)
            # For fallback, set generated_wav as empty too for post-consistency (CPU to avoid CUDA during failures)
            context.generated_wav = torch.zeros(0, dtype=torch.float32, device=torch.device('cpu'))  # Empty trigger
            return context

        # Log the exact text used for generation (helps debug padding/repeat logic)
        safe_text = context.text or ""
        try:
            preview = (safe_text[:120] + '...') if len(safe_text) > 120 else safe_text
            logger.info(f"Generating audio for text (len={len(safe_text)}): '{preview}'")
        except Exception:
            logger.info("Generating audio for text (len=?): <unprintable>")

        # Build gen_args using helper (proper structure, no 'conditionals'/'seed' kwargs)
        gen_args = self._build_generate_args(context)

        # Log shapes/dims for debug (simple, no f-string issues)
        conds_desc = 'Present (type: Conditionals)' if context.model.conds else 'None'
        logger.debug(f"Conds shapes: {{'conds': '{conds_desc}'}}")
        text_len = len(context.text.split())  # Estimate tokens for validation
        logger.debug(
            f"Gen args: text_len={text_len}, conds_shapes={{'conds': '{conds_desc}'}}")

        try:
            if len(context.text.strip()) == 0:  # Use context.text directly
                raise ValueError("Empty text input")

            gen_start = time.perf_counter()
            # UPDATED: Pass sr and text_len
            raw_output = self._generate_core(context.model, gen_args, context.t3_params, context.sr, text_len)
            gen_time = time.perf_counter() - gen_start

            if raw_output is None or raw_output.numel() == 0:
                raise ValueError("Generated empty/None WAV")

            # FIXED: Set generated_wav to raw output for PostProcessing to handle
            context.generated_wav = raw_output  # Raw tensor from model.generate
            # Do NOT set processed_wav here – let PostProcessing set it after processing

            context.audio_duration = context.generated_wav.shape[
                                         -1] / context.sr  # Use shape for duration (assume [1, N])
            if context.audio_duration > 0:
                logger.info(
                    f"Generation success: {context.audio_duration:.2f}s WAV @ {context.sr}Hz (gen time {gen_time:.2f}s)")
            else:
                raise ValueError("Computed duration <=0 despite valid WAV")

            # Previously: capture worker status when using external TTS worker process.
            # Rolled back: no external worker, so no worker status to capture.

        except Exception as gen_e:
            logger.error(f"Core gen failed: {gen_e}")
            gen_args['text'] = gen_args['text'][:30] + '...' if len(gen_args['text']) > 30 else gen_args['text']
            msg = f"Gen error on '{gen_args['text']}' (exag={gen_args['exaggeration']}): {gen_e}"
            # Rolled back: no external worker status to capture on failure.
            # FIXED: Set both for consistency in fallback
            # Ensure CPU to avoid touching CUDA in error paths
            context.generated_wav = torch.zeros(0, dtype=torch.float32, device=torch.device('cpu'))  # Empty trigger for post
            return self.handle_error(context, ValueError(msg))

        logger.debug("Gen complete")
        return context  # Post will process and set processed_wav

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

        # Rolled back: do not attach conds_key (no external worker uses it),
        # and avoid passing unknown kwargs to model.generate().
        
        return generate_args

    # Updated _generate_core (simplified: primary + clear + helpers; pass sr from _execute_core if needed)
    def _generate_core(self, model: Any, gen_args: dict, t3_params: dict, sr: int = 24000, text_len: int = 0) -> \
    Optional[torch.Tensor]:
        """ENHANCED: Primary gen + tiered retries via helpers. Cleaner flow for graph errors."""
        gen_start = time.time()

        # Estimate text_len if not provided (rough token count)
        if text_len == 0:
            text_len = len(gen_args['text'].split()) if gen_args.get('text') else 0
            logger.debug(f"Estimated text_len: {text_len}")

        # Primary: Standard generation (with validation)
        try:
            if 't3_params' not in gen_args:
                gen_args['t3_params'] = t3_params
            logger.debug("Calling model.generate...")
            # Rolled back: call model.generate directly (no external worker routing)
            output = model.generate(**gen_args)
            # UPDATED: Pass text_len to validation
            if self._validate_output(output, sr, text_len):
                return output
            else:
                raise ValueError("Invalid output from primary generation")
        except RuntimeError as graph_e:
            err_str = str(graph_e).lower()
            if any(term in err_str for term in ["graph", "capture", "offset", "stream is capturing"]):
                logger.warning(f"Graph/CUDA error: {graph_e} – enhanced clear & tiered retry")

                # Tier 1: Deeper clear (Torch 2.8+ / 50-series focus)
                orig_params = None
                if hasattr(model, 't3') and hasattr(model.t3, '_bucket_graphs'):
                    model.t3._bucket_graphs.clear()
                if hasattr(model, 't3') and hasattr(model.t3, 'params'):
                    orig_params = model.t3.params.copy()
                    if isinstance(model.t3.params, dict) and 'compile' not in model.t3.params:
                        model.t3.params['compile'] = False  # Disable if possible; harmless skip if key absent
                try:
                    import torch._dynamo as dynamo
                    dynamo.reset()  # Clears compiled caches
                except ImportError:
                    pass
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                        # Avoid synchronize during capture
                        if hasattr(torch.cuda, "is_current_stream_capturing") and not torch.cuda.is_current_stream_capturing():
                            torch.cuda.synchronize()
                    except Exception:
                        # Ignore sync errors; we are recovering from a graph failure
                        pass

                # Tier 2: Eager helper
                eager_result = self._retry_eager(model, gen_args, t3_params, sr, orig_params,
                                                 text_len)  # UPDATED: Pass text_len
                if eager_result is not None:
                    return eager_result

                # Tier 3: Safe helper
                safe_result = self._retry_safe(model, gen_args, t3_params, sr, orig_params,
                                               text_len)  # UPDATED: Pass text_len
                if safe_result is not None:
                    return safe_result

                # Exhaustion
                logger.error("All graph retries exhausted")
                return None
            else:
                raise  # Non-graph error
        except Exception as e:
            # Enhanced logging: include type and repr for empty-string exceptions
            try:
                etype = type(e).__name__
                logger.exception(f"Core gen failed [{etype}]: {repr(e)}")
            except Exception:
                logger.exception("Core gen failed (unprintable error)")
            return None
        finally:
            # Avoid CUDA maintenance calls while a stream is capturing
            if torch.cuda.is_available():
                try:
                    if hasattr(torch.cuda, "is_current_stream_capturing"):
                        if not torch.cuda.is_current_stream_capturing():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                    else:
                        # Older builds: best-effort empty cache only
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            logger.debug(f"Gen complete: {time.time() - gen_start:.2f}s")

    def _validate_output(self, output: Optional[torch.Tensor], sr: int, text_len: int = 0) -> bool:
        """Quick check for valid generated audio tensor. Allows low-energy but rejects NaNs/empty/graph artifacts."""
        if output is None or output.numel() == 0:
            logger.debug("Output None or empty")
            return False
        if torch.isnan(output).any() or torch.isinf(output).any():
            logger.debug("Output has NaN/Inf")
            return False
        samples = output.shape[-1]
        if text_len <= 2:
            min_samples = max(int(sr * 0.20), text_len * 80)  # 0.20s floor for very short texts
        else:
            min_samples = max(int(sr * 0.50), text_len * 80)
        if samples < min_samples:
            logger.debug(
                f"Output too short: samples={samples}, min={min_samples} (text_len={text_len}, shape={output.shape})")
            return False
        energy = torch.norm(output).item()
        if energy < 5e-5:  # Relaxed threshold; tune to 1e-5 if still too strict
            logger.debug(
                f"Output energy too low: {energy} (text_len={text_len}, samples={samples}, shape={output.shape}, min/max={output.min().item():.2e}/{output.max().item():.2e})")
            return False
        logger.debug(
            f"Output valid: energy={energy:.2e}, samples={samples}, shape={output.shape} (text_len={text_len})")
        return True

    # New helper: Tier 2 - Eager backend retry (extracted for clarity)
    def _retry_eager(self, model: Any, gen_args: dict, t3_params: dict, sr: int, orig_params: dict = None,
                     text_len: int = 0) -> Optional[torch.Tensor]:
        """Retry with eager backend (original logic, validated). Restores params if provided."""
        mod_args = gen_args.copy()
        mod_t3 = t3_params.copy()
        mod_t3['generate_token_backend'] = 'eager'
        mod_args['t3_params'] = mod_t3
        try:
            eager_output = model.generate(**mod_args)
            # UPDATED: Pass text_len to validation
            if self._validate_output(eager_output, sr, text_len):
                logger.debug("Eager retry succeeded")
                if orig_params:
                    model.t3.params = orig_params  # Restore
                return eager_output
            else:
                logger.warning("Eager retry invalid output")
            return None
        except Exception as eager_e:
            logger.warning(f"Eager retry failed: {eager_e}")
            return None
        finally:
            if orig_params:
                model.t3.params = orig_params

    # New helper: Tier 3 - Safe params fallback (exag=0 for offset dodge)
    def _retry_safe(self, model: Any, gen_args: dict, t3_params: dict, sr: int, orig_params: dict = None,
                    text_len: int = 0) -> Optional[torch.Tensor]:
        """Fallback with neutral params (exag=0, eager) to avoid dynamic offsets."""
        safe_args = gen_args.copy()
        safe_args['exaggeration'] = 0.0  # Neutral trigger avoidance
        safe_args['temperature'] = safe_args.get('temperature', 1.0)  # Stable default
        safe_t3 = t3_params.copy()
        safe_t3['generate_token_backend'] = 'eager'
        safe_args['t3_params'] = safe_t3
        try:
            safe_output = model.generate(**safe_args)
            # UPDATED: Pass text_len to validation
            if self._validate_output(safe_output, sr, text_len):
                logger.info("Safe params recovery succeeded (neutral exag)")
                if orig_params:
                    model.t3.params = orig_params
                return safe_output
            else:
                logger.warning("Safe params invalid output")
            return None
        except Exception as safe_e:
            logger.warning(f"Safe retry failed: {safe_e}")
            return None
        finally:
            if orig_params:
                model.t3.params = orig_params


    def handle_error(self, context: AudioGenerationContext, error: Exception) -> AudioGenerationContext:
        """REFACTORED: Delegate to base (silence + paths/attrs); ensure generated_wav for post fallback."""
        logger.error(f"Generation error: {error} – silence fallback")
        # Ensure paths post-error (DRY)
        context.audio_prompt_path = getattr(context, 'audio_prompt_path', "") or ""
        context.processed_voice_path = getattr(context, 'processed_voice_path', "") or ""
        # FIXED: Set generated_wav empty for post to fallback gracefully
        context.generated_wav = torch.zeros(0, dtype=torch.float32, device=context.device)
        # Base handles processed_wav if needed, but post will override
        return super().handle_error(context, error)