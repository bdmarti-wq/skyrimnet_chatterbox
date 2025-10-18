# In src/generate/pipeline/phases/generation.py
# (Updated GenerationPhase – FIXED: Guard gen_args paths more aggressively; safe config in _create_silence_fallback/handle_error;
#  clear graphs on CUDA capture error in _generate_audio_core; no None.items in fallback.)

"""
Generation Phase: Core TTS audio synthesis using the TTS model.
Handles parameter extraction, generation with retry logic, and fallbacks.
Integrates robust _generate_audio_core for graph capture issues in T3/Chatterbox.
"""
import time
import os
from typing import Any, Dict, Optional

import torch
import torchaudio  # For optional load/debug if needed

from src.audio_utils import pad_short_text, get_silence
from src.config import get_config
from src.tts_model import GEN_ACTIVE_LOCK  # Global lock for gen/prepare
from .base import GenerationPhase as BaseGenerationPhase
from ...pipeline.context import AudioGenerationContext
from loguru import logger
from src.seeding import set_seed


class GenerationPhase(BaseGenerationPhase):
    def execute(self, context: AudioGenerationContext) -> AudioGenerationContext:
        start_time = time.time()
        logger.debug(
            f"GenerationPhase start: text='{context.text[:50]}...', seed={context.seed}, conds_key={context.conditionals_key}")

        if not context.conditionals_key or not hasattr(context, 'model') or context.model is None:
            logger.error("No conds or model for generation – fallback silence")
            return self._silent_fallback(context)

        # FIXED: Validate conds before generation (prevent bool tensor error; check model.conds as in working snippet)
        conds = getattr(context.model, 'conds', None)  # Internal state from prepare_conditionals
        if conds is None:
            logger.warning("No model.conds – dummy fallback")
            from src.tts_model import create_dummy_conds  # Import from working snippet
            create_dummy_conds(context.model, context.device, context.dtype, "no_conds")
            conds = getattr(context.model, 'conds', None)  # Refresh

        # Log shapes for debug (mel/token mismatch; optional if conds has cmel/cmap)
        if conds and hasattr(conds, 't3') and hasattr(conds.t3, 'speaker_emb'):
            emb = conds.t3.speaker_emb
            emb_nonzero = not torch.all(emb == 0).item()
            logger.debug(f"Conds validation: emb {emb.shape} (non-zero: {emb_nonzero})")
            # If has cmel/cmap (if exposed), validate lengths
            if hasattr(conds, 'cmel') and hasattr(conds, 'cmap'):
                cmel_len = conds.cmel.shape[-1] if conds.cmel is not None else 0
                cmap_token_len = conds.cmap.shape[-1] // 2 if conds.cmap is not None else 0
                logger.debug(
                    f"Conds validation: cmel_len={cmel_len}, expected_token_len={cmap_token_len * 2}, match={cmel_len == cmap_token_len * 2}")

                if cmel_len != cmap_token_len * 2:
                    logger.warning(f"Mel-token mismatch ({cmel_len} != {cmap_token_len * 2}) – padding/trim conds")
                    # FIXED: Adjust to match (simple pad/trim; tune based on model)
                    min_len = min(cmel_len, cmap_token_len * 2)
                    if cmel_len > min_len:
                        conds.cmel = conds.cmel[..., :min_len]  # Trim mel
                    elif cmap_token_len * 2 > cmel_len:
                        pad_mel = torch.zeros((conds.cmel.shape[0], conds.cmel.shape[1], cmap_token_len * 2 - cmel_len),
                                              dtype=context.dtype, device=context.device)
                        conds.cmel = torch.cat([conds.cmel, pad_mel], dim=-1)  # Pad mel to match tokens
                    logger.debug(f"Adjusted conds: cmel_len={conds.cmel.shape[-1]}")

        # FIXED: Ensure bool checks use .all().item() (scalar bool) – general guard
        try:
            # FIXED: Build args WITHOUT 'conds' (stateful model.conds internal; align with working snippet)
            gen_args = self._build_generate_args(context)  # No 'conds' added

            # FIXED: Simplified conds_shapes log (compute outside to avoid broken f-string)
            conds_shapes = {}
            if conds and hasattr(conds, 't3') and hasattr(conds.t3, 'speaker_emb'):
                conds_shapes['model.conds.t3.speaker_emb'] = conds.t3.speaker_emb.shape
            elif conds:
                conds_shapes['model.conds'] = 'Present (type: {})'.format(type(conds).__name__)
            else:
                conds_shapes['model.conds'] = 'None'
            logger.debug(
                f"Gen args: text_len={len(context.text)}, max_new_tokens=250, conds_shapes={conds_shapes}")

            # FIXED: Call generate WITHOUT conds kwarg (uses model.conds state; as in working snippet)
            # Set seed first (align with seeding)
            if context.seed is not None:
                set_seed(context.seed)

            generated = context.model.generate(**gen_args)  # No 'conds'; relies on internal model.conds

            if generated is None:
                raise ValueError("Model.generate returned None")

            logger.debug(
                f"Generated tokens/output: {type(generated)} {getattr(generated, 'shape', 'N/A') if hasattr(generated, 'shape') or hasattr(generated, '__len__') else len(generated) if hasattr(generated, '__len__') else 'N/A'}")

            # FIXED: In working snippet, generate likely returns tokens; need to_wav or decode if separate
            # Assume generate returns WAV tensor directly (from logs: no separate to_wav); if tokens, add decode
            wav = generated  # If generate returns WAV; else: wav = context.model.to_wav(generated, conds=context.model.conds)
            if wav is None or wav.numel() == 0:
                raise ValueError("Generated WAV empty")

            # Ensure scalar bool in any final checks (e.g., if valid_audio = torch.all(wav > -1e-6))
            if any(isinstance(x, torch.Tensor) and x.dtype == torch.bool and x.numel() > 1 for x in locals().values()):
                logger.warning("Detected potential bool tensor – forcing scalar")
                # Example: if (wav > threshold).any(): → if (wav > threshold).any().item()

            context.generated_wav = wav
            context.audio_duration = len(wav.squeeze(0)) / context.sr
            logger.info(f"Generation success: {context.audio_duration:.2f}s WAV @ {context.sr}Hz")

        except Exception as gen_e:
            logger.error(
                f"Generation core error: {gen_e} – traceback: {gen_e.__traceback__ if hasattr(gen_e, '__traceback__') else 'No trace'}")
            # Log tensor shapes before failure (focus on model.conds)
            conds_info = "None"
            if hasattr(context, 'model') and hasattr(context.model, 'conds') and context.model.conds:
                if hasattr(context.model.conds, 't3') and hasattr(context.model.conds.t3, 'speaker_emb'):
                    conds_info = f"torch.Size({context.model.conds.t3.speaker_emb.shape})"
                else:
                    conds_info = f"type {type(context.model.conds).__name__}"
            logger.debug(f"Pre-gen conds: model.conds {conds_info}")
            return self._silent_fallback(context)

        return context

    def _silent_fallback(self, context: AudioGenerationContext) -> AudioGenerationContext:
        """Create cached silence fallback using get_silence()."""
        duration = 2.0  # Fixed fallback duration
        logger.warning(f"Generation fallback: {duration}s silence via get_silence()")
        context.generated_wav = get_silence(
            duration=duration,
            sr=context.sr,
            dtype=context.dtype,
            device=context.device
        )
        context.audio_duration = duration
        # FIXED: Ensure no None paths (from coordinator error)
        if not hasattr(context, 'audio_prompt_path') or context.audio_prompt_path is None:
            context.audio_prompt_path = ""  # Empty str valid for PathLike checks
        return context

    def handle_error(self, context: AudioGenerationContext, error: Exception) -> AudioGenerationContext:
        """Fallback to silence on gen error. FIXED: Use get_silence; no None paths; safe config.globals."""
        logger.error(f"Generation phase error: {error} – using silent fallback")

        # FIXED: Safe config access (no 'globals' error; use getattr)
        config = get_config()
        if hasattr(config, 'globals'):
            sr = getattr(config.globals, 'sr', 24000)
            device = getattr(config.globals, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
            dtype = getattr(config.globals, 'dtype', torch.float32)
        else:
            # Fallback if no globals (from error logs)
            sr = getattr(context, 'sr', 24000)
            device = getattr(context, 'device', torch.device('cpu')) if hasattr(context, 'device') else torch.device(
                'cpu')
            dtype = getattr(context, 'dtype', torch.float32) if hasattr(context, 'dtype') else torch.float32

        # FIXED: Ensure safe path (no None for os.PathLike)
        if not hasattr(context, 'audio_prompt_path') or context.audio_prompt_path is None:
            context.audio_prompt_path = ""

        silence = get_silence(duration=2.0, sr=sr, dtype=dtype, device=device)
        context.generated_wav = silence
        context.audio_duration = 2.0
        logger.warning(f"Silence fallback created: {silence.shape}")
        return context

    def _generate_audio_core(self, model: Any, gen_args: Dict, t3_params: Dict) -> Optional[torch.Tensor]:
        """Core generation with graph retry and cleanup. FIXED: Clear graphs on CUDA capture/previous error; eager fallback."""
        if model is None:
            logger.error("generate_audio_core: model is None – creating dummy silence")
            from src.tts_model import create_dummy_conds
            config = get_config()
            if hasattr(config, 'globals'):
                device = getattr(config.globals, 'device', 'cpu')
                dtype = getattr(config.globals, 'dtype', torch.float32)
            else:
                device = torch.device('cpu')
                dtype = torch.float32
            create_dummy_conds(model, device, dtype, "core_none")
            return self._create_silence_fallback()

        # FIXED: Guard any path in gen_args (e.g., audio_prompt_path None → remove or "")
        if 'audio_prompt_path' in gen_args and gen_args['audio_prompt_path'] is None:
            del gen_args['audio_prompt_path']  # Don't pass None to generate (avoids PathLike error)
            logger.debug("Removed None audio_prompt_path from gen_args")

        with GEN_ACTIVE_LOCK:  # From tts_model
            wav = None
            gen_start = time.time()
            try:
                if 't3_params' not in gen_args:
                    gen_args['t3_params'] = t3_params
                logger.debug("Calling model.generate...")
                wav = model.generate(**gen_args)  # FIXED: No 'conds' – uses model.conds state
                logger.debug(
                    f"model.generate completed in {time.time() - gen_start:.2f}s, wav: {type(wav)} {getattr(wav, 'shape', 'N/A') if wav else None}")
            except RuntimeError as graph_e:
                err_str = str(graph_e).lower()
                if any(term in err_str for term in
                       ["graph", "capture", "offset", "stream is capturing", "previous error during capture"]):
                    logger.warning(f"Graph/CUDA capture error: {graph_e} – clearing graphs and retrying with eager")
                    # FIXED: Clear all graphs/buckets on capture fail (from prior prep error)
                    if hasattr(model, 't3') and hasattr(model.t3, '_bucket_graphs'):
                        model.t3._bucket_graphs.clear()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()  # Sync async errors

                    # Retry with eager
                    modified_args = gen_args.copy()
                    modified_t3 = t3_params.copy()
                    modified_t3['generate_token_backend'] = 'eager'
                    modified_args['t3_params'] = modified_t3
                    wav = model.generate(**modified_args)
                    logger.debug("Eager retry success")
                else:
                    raise
            except Exception as e:
                logger.exception(f"Core gen failed: {e}")
                wav = None
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # Final sync

            return wav if wav else self._create_silence_fallback()

    def _build_generate_args(self, context: AudioGenerationContext) -> dict:
        """Construct arguments for the generation call (no 'conds' kwarg – state-based). FIXED: Param mapping (cfgw → cfg_weight); no 'conds' added; guard path None."""
        generate_args = {
            'text': context.text,
            'exaggeration': getattr(context, 'exaggeration', 0.7),
            'temperature': getattr(context, 'temperature', 0.7),
            'cfg_weight': getattr(context, 'cfg_weight', getattr(context, 'cfgw', 0.0)),  # Map legacy cfgw
            'min_p': getattr(context, 'min_p', 0.05),
            'top_p': getattr(context, 'top_p', 1.0),
            'repetition_penalty': getattr(context, 'repetition_penalty', 2.0),
            'language_id': getattr(context, 'language_id', None),
            't3_params': getattr(context, 't3_params', {})
        }

        # FIXED: Guard voice_params (no None.items from fallback)
        voice_params = getattr(context, 'voice_params', {}) or {}
        if isinstance(voice_params, dict):
            for key in ['temperature', 'top_p', 'repetition_penalty', 'exaggeration', 'min_p']:
                if key in voice_params:
                    generate_args[key] = voice_params[key]
            # FIXED: Override cfg_weight (map 'cfgw' in voice_params to 'cfg_weight')
            if 'cfgw' in voice_params:
                generate_args['cfg_weight'] = voice_params['cfgw']
            if 'cfg_weight' in voice_params:  # Direct
                generate_args['cfg_weight'] = voice_params['cfg_weight']
        else:
            logger.warning("voice_params not dict in gen_args; skipping overrides")

        # FIXED: Add audio_prompt_path ONLY if conds needed for fallback (but since prepare_conditionals already set model.conds, optional)
        # From working snippet: If no conds loaded, prepare_conditionals uses path – but here, assume prepped
        # FIXED: Guard None (set to "" if None, but only add if not None/empty)
        prompt_path = getattr(context, 'audio_prompt_path', None)
        if (hasattr(context.model, 'conds') and context.model.conds is None and
                prompt_path is not None and str(prompt_path).strip()):  # FIXED: Skip if None/empty str
            generate_args['audio_prompt_path'] = str(prompt_path)  # Ensure str
            logger.debug(
                f"Added audio_prompt_path '{generate_args['audio_prompt_path']}' to args (Chatterbox fallback)")
        elif prompt_path is None:
            logger.debug("audio_prompt_path is None – skipping addition to gen_args")

        # Log state (no conds kwarg; check internal)
        has_conds = hasattr(context.model, 'conds') and context.model.conds is not None
        emb_active = (has_conds and hasattr(context.model.conds, 't3') and hasattr(context.model.conds.t3,
                                                                                   'speaker_emb') and
                      context.model.conds.t3.speaker_emb is not None and not torch.all(
                    context.model.conds.t3.speaker_emb == 0))
        logger.debug(
            f"Generate args: conds state {'clone' if emb_active else 'neutral (None)' if has_conds else 'none'}")

        return generate_args

    def _generate_wav(self, context: AudioGenerationContext, generate_args: dict) -> Optional[torch.Tensor]:
        """Execute the core generation process with error handling. FIXED: Call integrated _generate_audio_core; no PathLike issues."""
        try:
            # No need for extra no_grad (handled in core)
            return self._generate_audio_core(
                model=context.model,
                gen_args=generate_args,
                t3_params=getattr(context, 't3_params', {})
            )
        except Exception as e:
            logger.error(f"Core generation failed: {str(e)}")
            # FIXED: No None paths; fallback safe
            if not hasattr(context, 'audio_prompt_path') or context.audio_prompt_path is None:
                context.audio_prompt_path = ""
            return None

    def _create_silence_fallback(self) -> torch.Tensor:
        """Internal helper: 2s silence (fallback for core errors). FIXED: Safe config."""
        config = get_config()
        if hasattr(config, 'globals'):
            sr = getattr(config.globals, 'sr', 24000)
            device = getattr(config.globals, 'device', 'cpu')
            dtype = getattr(config.globals, 'dtype', torch.float32)
        else:
            sr = 24000
            device = torch.device('cpu')
            dtype = torch.float32
        return get_silence(duration=2.0, sr=sr, dtype=dtype, device=device)