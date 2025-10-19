# src/generate/pipeline/phases/generation.py (Fixed: Remove t3.to() call to avoid 'dtype' error; import create_dummy_conds at top; safe emb move if needed)
"""
Generation Phase: Core TTS audio synthesis using the TTS model.
Handles parameter extraction, generation with retry logic, and fallbacks.
Integrates robust _generate_audio_core for graph capture issues in T3/Chatterbox.
"""
import time
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from src.audio_utils import pad_short_text, get_silence
from src.config import get_config
from src.tts_model import GEN_ACTIVE_LOCK, restore_graphs_for_bucket, create_dummy_conds  # FIXED: Import at top for scope
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

        # FIXED: Early guard paths (set "" if None/empty – prevents fspath in gen_args/internal; consistent)
        if not hasattr(context, 'audio_prompt_path') or context.audio_prompt_path is None or not str(
                context.audio_prompt_path).strip():
            context.audio_prompt_path = ""
        if not hasattr(context, 'processed_voice_path') or context.processed_voice_path is None:
            context.processed_voice_path = ""

        # FIXED: Use context.conds if preserved from VoiceProcessing (skip fresh prep – recover 0.1s)
        # Log entry state for debug
        has_context_conds = hasattr(context, 'conds') and context.conds is not None
        model_conds_none = getattr(context.model, 'conds', None) is None
        logger.debug(f"Generation entry: has_context_conds={has_context_conds}, model_conds_none={model_conds_none}")

        conds = None
        if has_context_conds:
            conds = context.conds  # Restore preserved conds
            logger.debug(f"Attempting restore from context.conds: type={type(conds).__name__ if conds else 'None'}")

            # FIXED: Assign to model IMMEDIATELY for state-based gen (before validation)
            if conds is not None:
                context.model.conds = conds  # Key: Restore model state here
                logger.debug("Assigned context.conds to model.conds")

                # FIXED: Validate AFTER assign (on model.conds – checks assigned state; direct emb access, no t3)
                if self._is_nonempty_conds(context.model.conds):
                    # Diagnostics: Log if partial (no force fresh; emb non-zero is sufficient)
                    try:
                        # FIXED: Direct access to speaker_emb on conds (T3Cond has it directly, no .t3 sub-object)
                        if hasattr(context.model.conds, 'speaker_emb') and context.model.conds.speaker_emb is not None:
                            emb = context.model.conds.speaker_emb
                            emb_nonzero = not torch.all(emb == 0).item()
                            emb_mean = emb.mean().item()
                            logger.debug(
                                f"Restored model.conds emb: shape={emb.shape}, mean={emb_mean:.4f} (non-zero: {emb_nonzero})")
                        else:
                            logger.warning(
                                "Restored model.conds missing speaker_emb – but validation passed, proceeding")
                    except Exception as diag_e:
                        logger.warning(f"Diagnostics failed on restored conds: {diag_e} – proceeding")
                else:
                    logger.warning("Post-restore model.conds invalid – force fresh prep")

                    # FIXED: Skip redundant dummy; directly force fresh (avoids unnecessary dummy creation)
                    conds = None
                    context.model.conds = None  # Clear for fresh prep
            else:
                logger.warning("Preserved conds is None – fallback to model.conds")
                conds = getattr(context.model, 'conds', None)
        else:
            logger.warning("No context.conds – check VoiceProcessing handoff")
            conds = getattr(context.model, 'conds', None)

        # FIXED: Combine fallbacks (no duplicate dummy; fresh only if no valid conds post-restore)
        # REMOVED: Separate post-assign validation – now unified below to avoid duplicate blocks
        if conds is None or context.model.conds is None or not self._is_nonempty_conds(context.model.conds):
            logger.warning(f"No valid model.conds after restore (context: {has_context_conds}) – fresh prep from path")
            processed_path = getattr(context, 'processed_voice_path', None)
            if processed_path and os.path.exists(str(processed_path)):
                # Fresh prep logic unchanged
                voice_params = getattr(context, 'voice_params', {})
                exag = voice_params.get('exaggeration', 1.0)
                config = get_config()
                if hasattr(config, 'app_config') and hasattr(config.app_config, 'globals'):
                    device = config.app_config.globals.device
                    dtype = config.app_config.globals.dtype
                else:
                    device = getattr(context, 'device', torch.device('cpu'))
                    dtype = getattr(context, 'dtype', torch.float32)
                try:
                    context.model.prepare_conditionals(str(processed_path), exaggeration=exag)
                    if hasattr(context.model, 'conds') and context.model.conds is not None:
                        conds = context.model.conds
                        logger.debug("Fresh real conds prepped from path")
                        # FIXED: Set key if none
                        if not context.conditionals_key:
                            voice_stem = getattr(context, 'voice_stem', 'default')
                            context.conditionals_key = f"fresh_{voice_stem}_{int(time.time()) % 10000}"
                        # FIXED: Preserve for downstream
                        context.conds = conds
                    else:
                        raise ValueError("Prep returned None conds")
                except Exception as prep_e:
                    logger.error(f"Fresh prep failed {prep_e} – dummy fallback")
                    # Dummy as final fallback (unchanged)
                    config = get_config()
                    if hasattr(config, 'app_config') and hasattr(config.app_config, 'globals'):
                        device = config.app_config.globals.device
                        dtype = config.app_config.globals.dtype
                    else:
                        device = getattr(context, 'device',
                                         torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                        dtype = getattr(context, 'dtype', torch.bfloat16)
                    create_dummy_conds(context.model, device, dtype, "no_path_dummy")
                    conds = getattr(context.model, 'conds', None)
                    logger.debug("Dummy conds set (no valid path)")
                    # Set key on dummy too
                    if not context.conditionals_key:
                        voice_stem = getattr(context, 'voice_stem', 'dummy')
                        context.conditionals_key = f"dummy_{voice_stem}_{int(time.time()) % 10000}"
                    if conds is not None:
                        context.conds = conds
            else:
                logger.error("No path for fresh prep – full dummy fallback")
                # Ultimate dummy (unchanged)
                config = get_config()
                if hasattr(config, 'app_config') and hasattr(config.app_config, 'globals'):
                    device = config.app_config.globals.device
                    dtype = config.app_config.globals.dtype
                else:
                    device = getattr(context, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                    dtype = getattr(context, 'dtype', torch.bfloat16)
                create_dummy_conds(context.model, device, dtype, "ultimate_dummy")
                conds = getattr(context.model, 'conds', None)
                logger.debug("Ultimate dummy conds set")

        # FIXED: Final validation simplified (post-val/prep; direct on model.conds; no redundant if or t3 access)
        emb_present = self._is_nonempty_conds(context.model.conds)
        if emb_present:
            # Diagnostics for partial/full T3
            try:
                if hasattr(context.model.conds,
                           'cond_prompt_speech_tokens') and context.model.conds.cond_prompt_speech_tokens is not None:
                    logger.debug("T3 full (has cond_prompt_speech_tokens)")
                else:
                    logger.debug("T3 partial (no cond_prompt_speech_tokens)")
            except Exception as t3_e:
                logger.debug(f"T3 check failed: {t3_e}")
            logger.debug("Final pre-gen: model.conds valid – proceed to generation")
        else:
            logger.error("No valid conds post-all – silent fallback")
            return self._silent_fallback(context)

        # Proceed to gen (emb non-zero → real enough; catch any T3 error in try below)
        logger.debug("Model.conds valid → allow gen")

        try:
            # FIXED: Build args WITHOUT 'conds' (stateful model.conds internal; align with working snippet)
            gen_args = self._build_generate_args(context)  # No 'conds' added

            # FIXED: Simplified conds_shapes log (direct access to speaker_emb on model.conds)
            conds_shapes = {}
            if hasattr(context.model.conds, 'speaker_emb') and context.model.conds.speaker_emb is not None:
                conds_shapes['model.conds.speaker_emb'] = context.model.conds.speaker_emb.shape
            elif hasattr(context.model, 'conds') and context.model.conds is not None:
                conds_shapes['model.conds'] = f'Present (type: {type(context.model.conds).__name__})'
            else:
                conds_shapes['model.conds'] = 'None'
            logger.debug(f"Gen args: text_len={len(context.text)}, max_new_tokens=250, conds_shapes={conds_shapes}")

            # FIXED: Call generate WITHOUT conds kwarg (uses model.conds state; as in working snippet)
            if context.seed is not None:
                set_seed(context.seed)

            # FIXED: Log before generate for debug
            logger.debug("Calling model.generate with restored model.conds")

            generated = context.model.generate(**gen_args)  # FIXED: Relies on model.conds (no kwarg)

            if generated is None:
                raise ValueError("Model.generate returned None")

            logger.debug(
                f"Generated tokens/output: {type(generated)} {getattr(generated, 'shape', 'N/A') if hasattr(generated, 'shape') else len(generated) if hasattr(generated, '__len__') else 'N/A'}")

            # FIXED: Assume generate returns WAV (logs: no to_wav); if tokens, add decode/to_wav
            wav = generated  # Direct WAV
            if wav is None or wav.numel() == 0:
                raise ValueError("Generated WAV empty")

            context.generated_wav = wav
            context.audio_duration = len(wav.squeeze(0)) / context.sr
            logger.info(f"Generation success: {context.audio_duration:.2f}s WAV @ {context.sr}Hz")

            # FIXED: Preserve for downstream phases (e.g., PostProcessing if needed)
            if conds is not None:
                context.conds = conds

        except Exception as gen_e:
            logger.error(
                f"Generation core error: {gen_e} – traceback: {gen_e.__traceback__ if hasattr(gen_e, '__traceback__') else 'No trace'}")
            # FIXED: Log conds pre-fail (use model.conds for consistency)
            conds_info = "None"
            if hasattr(context, 'model') and hasattr(context.model, 'conds') and context.model.conds:
                if hasattr(context.model.conds, 'speaker_emb'):
                    emb = context.model.conds.speaker_emb
                    conds_info = f"torch.Size({emb.shape}) non-zero: {not torch.all(emb == 0).item()}"
                else:
                    conds_info = f"type {type(context.model.conds).__name__}"
            logger.debug(f"Pre-fail conds (post-restore): model.conds {conds_info}")
            return self.handle_error(context, gen_e)  # Generation-specific (zeros, no cache)

        return context



    # _build_generate_args unchanged (it's fine)
    def _build_generate_args(self, context: AudioGenerationContext) -> dict:
        """Construct arguments for the generation call (no 'conds' kwarg – state-based)."""
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
        # FIXED: Guard None (set to "" if None, but only add if not None/empty; remove/del if invalid – no fspath error in generate)
        prompt_path = getattr(context, 'audio_prompt_path', "")
        if hasattr(context.model, 'conds') and context.model.conds is None and prompt_path and str(prompt_path).strip():
            generate_args['audio_prompt_path'] = str(prompt_path)  # FIXED: Ensure str (valid PathLike)
            logger.debug(f"Added audio_prompt_path '{generate_args['audio_prompt_path']}' to args (Chatterbox fallback)")
        elif not prompt_path or not str(prompt_path).strip():
            # FIXED: Explicitly don't add if None/empty (avoids passing None/"" to generate/fspath crash)
            if 'audio_prompt_path' in generate_args:
                del generate_args['audio_prompt_path']
            logger.debug("audio_prompt_path is None/empty – skipping addition to gen_args")

        # Log state (no conds kwarg; check internal)
        has_conds = hasattr(context.model, 'conds') and context.model.conds is not None
        emb_active = (has_conds and hasattr(context.model.conds, 't3') and hasattr(context.model.conds.t3, 'speaker_emb') and
                      context.model.conds.t3.speaker_emb is not None and not torch.all(context.model.conds.t3.speaker_emb == 0))
        logger.debug(f"Generate args: conds state {'clone' if emb_active else 'neutral (None)' if has_conds else 'none'}")

        return generate_args

    # Other methods (_generate_audio_core, _is_nonempty_conds, _create_silence_fallback, _silent_fallback, handle_error) unchanged
    def _generate_audio_core(self, model: Any, gen_args: Dict, t3_params: Dict) -> Optional[torch.Tensor]:
        """Core generation with graph retry and cleanup."""
        if model is None:
            logger.error("generate_audio_core: model is None – creating dummy silence")
            config = get_config()
            if hasattr(config, 'app_config') and hasattr(config.app_config, 'globals'):
                device = getattr(config.app_config.globals, 'device', 'cpu')
                dtype = getattr(config.app_config.globals, 'dtype', torch.float32)
            else:
                device = torch.device('cpu')
                dtype = torch.float32
            create_dummy_conds(model, device, dtype, "core_none")
            return self._create_silence_fallback()

        # FIXED: Guard any path in gen_args (e.g., audio_prompt_path None → remove or "")
        if 'audio_prompt_path' in gen_args and gen_args['audio_prompt_path'] is None:
            if 'audio_prompt_path' in gen_args:
                del gen_args['audio_prompt_path']  # FIXED: Don't pass None to generate (avoids fspath error)
            logger.debug("Removed None audio_prompt_path from gen_args")

        # FIXED: Early mock T3 check (if persists post-conds, fallback – no attr error in generate)
        if hasattr(model, 't3') and not hasattr(model.t3, 'cond_prompt_speech_tokens'):
            logger.warning("MockT3 detected in core (check conds prep) – silence fallback")
            return self._create_silence_fallback()

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

    def _is_nonempty_conds(self, conds: Any) -> bool:
        """Validate non-empty conds (emb non-zero). FIXED: No to() check; emb.numel() > 0 for size."""
        if conds is None:
            return False
        if hasattr(conds, 't3') and hasattr(conds.t3, 'speaker_emb') and conds.t3.speaker_emb is not None:
            emb = conds.t3.speaker_emb
            return emb.numel() > 0 and not torch.all(emb == 0)
        # Fallback for other (len/tensor/dict non-empty)
        if isinstance(conds, torch.Tensor):
            return conds.numel() > 0
        if hasattr(conds, '__len__') and len(conds) > 0:
            return True
        if isinstance(conds, dict) and any(v is not None for v in conds.values()):
            return True
        return False

    # _create_silence_fallback and _silent_fallback unchanged (they're fine)
    def _create_silence_fallback(self) -> torch.Tensor:
        """Internal helper: 2s silence (fallback for core errors). FIXED: Use config.app_config.globals."""
        config = get_config()
        if hasattr(config, 'app_config') and hasattr(config.app_config, 'globals'):
            sr = getattr(config.app_config.globals, 'sr', 24000)
            device = getattr(config.app_config.globals, 'device', 'cpu')
            dtype = getattr(config.app_config.globals, 'dtype', torch.float32)
        else:
            sr = 24000
            device = torch.device('cpu')
            dtype = torch.float32
        return get_silence(duration=2.0, sr=sr, dtype=dtype, device=device)

    def _silent_fallback(self, context: AudioGenerationContext) -> AudioGenerationContext:
        """Create cached silence fallback using zeros. FIXED: Direct torch.zeros (no get_silence/cache fspath)."""
        duration = 2.0
        config = get_config()
        sr = config.app_config.globals.sr if hasattr(config, 'app_config') and hasattr(config.app_config,
                                                                                       'globals') else 24000
        dtype = config.app_config.globals.dtype if hasattr(config, 'app_config') and hasattr(config.app_config,
                                                                                             'globals') else torch.float32
        device = config.app_config.globals.device if hasattr(config, 'app_config') and hasattr(config.app_config,
                                                                                               'globals') else torch.device(
            'cpu')
        samples = int(sr * duration)
        context.generated_wav = torch.zeros((1, samples), dtype=dtype, device=device)
        context.audio_duration = duration
        # FIXED: Paths "" (no None)
        context.audio_prompt_path = getattr(context, 'audio_prompt_path', "") or ""
        context.processed_voice_path = getattr(context, 'processed_voice_path', "") or ""
        logger.warning(f"Silent fallback zeros: {context.generated_wav.shape} @ {sr}Hz")
        return context

    def handle_error(self, context: AudioGenerationContext, error: Exception) -> AudioGenerationContext:
        """Fallback to silence on gen error. FIXED: Import guard for get_config (no 'not defined' error); safe config access."""
        logger.error(f"Generation phase error: {error} – using silent fallback")

        # FIXED: Import guard for get_config (if not imported at file top)
        try:
            from src.config import get_config
        except ImportError as imp_e:
            logger.warning(f"get_config import failed in handle_error: {imp_e} – ultimate fallback (hardcoded params)")
            # Ultimate fallback without config
            sr = getattr(context, 'sr', 24000)
            device = torch.device('cpu')  # Safe
            dtype = torch.float32
        else:
            config = get_config()
            if hasattr(config, 'app_config') and hasattr(config.app_config, 'globals'):
                sr = getattr(config.app_config.globals, 'sr', 24000)
                device = getattr(config.app_config.globals, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
                dtype = getattr(config.app_config.globals, 'dtype', torch.float32)
            else:
                sr = getattr(context, 'sr', 24000)
                device = getattr(context, 'device', torch.device('cpu')) if hasattr(context,
                                                                                    'device') else torch.device('cpu')
                dtype = getattr(context, 'dtype', torch.float32) if hasattr(context, 'dtype') else torch.float32

        # FIXED: Ensure safe paths (no None for os.PathLike in downstream)
        if not hasattr(context, 'audio_prompt_path') or context.audio_prompt_path is None:
            context.audio_prompt_path = ""
        if not hasattr(context, 'processed_voice_path') or context.processed_voice_path is None:
            context.processed_voice_path = ""

        # FIXED: Use direct zeros (no get_silence – avoids any fspath if cached)
        samples = int(sr * 2.0)  # 2s
        silence = torch.zeros((1, samples), dtype=dtype, device=device)
        context.generated_wav = silence
        context.audio_duration = 2.0
        logger.warning(f"Silence fallback created: {silence.shape}")
        return context

    # _generate_wav removed – use integrated _generate_audio_core for consistency (as in your original code)
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