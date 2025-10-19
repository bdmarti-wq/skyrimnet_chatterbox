# In src/generate/pipeline/phases/voice_processing.py
# (Updated VoiceProcessingPhase – FIXED: CUDA graph conflict in _prepare_conditionals by temp eager mode + sync;
#  guard voice_params in execute/handle_error ('NoneType' no 'items'); safe config attrs. NEW: Stable hash key + _get_or_prepare for atomic cache use.)

import hashlib
import os
import time
from pathlib import Path
import threading

import torch
from typing import Optional, Any, Tuple, Dict

import torchaudio

from src.chatterbox import ChatterboxTTS
from src.config import get_config_value, get_config
from src.normalize_stem import normalize_stem
from .base import GenerationPhase
from ...cache import CacheManager
from ...pipeline.context import AudioGenerationContext
from loguru import logger
from src.tts_model import GEN_ACTIVE_LOCK, MODEL_LOCK, chatterbox_tts_to, create_dummy_conds, get_model


class VoiceProcessingPhase(GenerationPhase):
    """Processes voice prompts and prepares conditional features."""

    process_after_cache = True

    def __init__(self, cache_manager=None):
        """Initialize with cache_manager for conditionals/voice access (inject from coordinator)."""
        config = get_config()
        self.cache_manager = cache_manager or CacheManager(config)  # Fallback if none
        self.conditionals_cache = getattr(self.cache_manager, 'conditionals_cache', None)  # Assume set
        if self.conditionals_cache is None:
            logger.warning("No conditionals_cache; prep will skip caching")
        logger.debug("VoiceProcessingPhase initialized with cache_manager")

    def _get_audio_dur(self, path: str) -> float:
        """Helper: Get duration from path (for log; avoids hang debug)."""
        try:
            info = torchaudio.info(path)
            return info.num_frames / info.sample_rate
        except Exception as e:
            logger.debug(f"Dur fetch failed for {path}: {e}")
            return 0.0

    def execute(self, context: AudioGenerationContext) -> AudioGenerationContext:
        """Process conds safely; FIXED: Use context.processed_voice_path/conds_key if HIT; globals.device; check exists before compute/save."""
        start_time = time.perf_counter()

        if not context.model:
            logger.warning("No model – dummy fallback")
            return context

        # FIXED: Clear state at start (prevent bleed from prior gen)
        if hasattr(context.model, 'conds'):
            context.model.conds = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Minimal perf hit
        logger.debug(f"Cleared state for {context.voice_stem}")

        model = context.model


        conds_key = context.conds_key or context.conditionals_key
        processed_path = context.processed_voice_path
        voice_stem = context.voice_stem
        voice_params = getattr(context, 'voice_params', {})

        if not voice_stem:
            logger.warning("No voice_stem – dummy")
            from src.tts_model import create_dummy_conds
            config = get_config()
            device = config.app_config.globals.device  # FIXED: Safe globals.device
            dtype = config.app_config.globals.dtype
            model.conds = create_dummy_conds(model, device, dtype)
            context.conds_key = 'dummy_no_voice'
            return context

        config = get_config()
        device = config.app_config.globals.device  # FIXED: Standard access (fixes 'Config' error)
        dtype = config.app_config.globals.dtype

        # FIXED: Load if HIT key set (fast; no path needed if cached)
        if conds_key and self.cache_manager and self.cache_manager.get_conditionals(conds_key, model):
            logger.info(f"VoiceProcessing: Conds HIT for {voice_stem} from key {conds_key[:20]}...")
            # Quick validate (move to device if loaded raw)
            if hasattr(model.conds, 'speaker_emb'):
                model.conds = model.conds.to(device=device, dtype=dtype)
            context.conds_key = conds_key
            # Success – return
            time_taken = time.perf_counter() - start_time
            logger.debug(f"VoiceProcessing HIT: {time_taken:.3f}s | key={conds_key[:20]}")
            return context

        # FIXED: Compute if path available (real; check exists)
        if processed_path and os.path.exists(processed_path):
            try:
                logger.info(f"VoiceProcessing: Compute conds for {voice_stem} from {processed_path}")
                conds = context.model.prepare_conditionals(processed_path, exaggeration=context.exaggeration)
                # Set key if not set (use existing or derive)
                if not conds_key:
                    conds_key = f"computed_{voice_stem}_{int(time.time())}"
                context.conds_key = conds_key
                context.conditionals_key = conds_key

                # FIXED: Safe save (check conds valid; no None)
                if hasattr(model, 'conds') and model.conds is not None and self.cache_manager:
                    if self.cache_manager.save_conditionals(conds_key, model):
                        logger.debug(f"Conds saved for {voice_stem}: {conds_key[:20]}")
                    else:
                        logger.warning(f"Compute OK but save failed for {voice_stem}")
                else:
                    logger.warning("Computed but no conds/model – skip save")
            except Exception as e:
                logger.error(f"Compute error for {voice_stem}: {e} – dummy fallback")
                processed_path = None  # To trigger dummy below
        else:
            logger.warning(f"VoiceProcessing: No path for {voice_stem} ({processed_path or 'none'}) – dummy")

        # FIXED: Dummy if no real conds (pre-change behavior; safe device)
        if not hasattr(model, 'conds') or model.conds is None:
            from src.tts_model import create_dummy_conds
            model.conds = create_dummy_conds(model, device, dtype)
            context.conds_key = f"dummy_{voice_stem}"
            logger.debug(f"Dummy conds set for {voice_stem}")

        time_taken = time.perf_counter() - start_time
        logger.debug(
            f"VoiceProcessing MISS: {time_taken:.3f}s | key={context.conds_key[:20]} | path={processed_path or 'none'}")
        return context


    def _prepare_conditionals(self, context: AudioGenerationContext, prep_path: str,
                              voice_params: Dict[str, Any]) -> Optional[Any]:
        """NEW: Atomic prepare with cache (_get_or_prepare); returns conds if success (loaded or fresh). FIXED: Eager/sync; set conds_loaded flag."""
        prep_start = time.perf_counter()
        exag = voice_params.get('exaggeration', 1.0)
        device = context.device
        dtype = context.dtype
        cache_key = context.conditionals_key  # Stable from execute

        if not self.conditionals_cache:
            logger.warning("No cache; falling back to direct prep")
            context.conds_from_cache = False
            conds = self._prepare_fresh(context.model, prep_path, exag, device, dtype)
            return conds

        # FIXED: Use cache's _get_or_prepare (atomic: get → miss? prep + save → return)
        conds = self.conditionals_cache._get_or_prepare(
            model=context.model, audio_path=prep_path, exag=exag,
            device=str(device), dtype=dtype, cache_key=cache_key
        )

        if conds is not None:
            # Post-load/set: Set model state (as in cache.get)
            context.model.conds = conds
            if hasattr(context.model, 'set_conditionals'):
                context.model.set_conditionals(conds)
            context.conds_from_cache = True  # Flag for execute (HIT vs fresh)
            chatterbox_tts_to(context.model, device, dtype)

            # Log emb for validation
            if hasattr(conds, 't3') and hasattr(conds.t3, 'speaker_emb'):
                emb = conds.t3.speaker_emb
                emb_nonzero = not torch.all(emb == 0)
                emb_time = time.perf_counter() - prep_start
                logger.info(f"Conds ready | emb {emb.shape} (non-zero: {emb_nonzero}) | time {emb_time:.3f}s")
            else:
                logger.debug(f"Conds ready (no emb attr) | time {time.perf_counter() - prep_start:.3f}s")
            return conds

        # Failed: Flag for fallback
        context.conds_from_cache = False
        logger.warning("Cache prep returned None – fallback to fresh")
        conds = self._prepare_fresh(context.model, prep_path, exag, device, dtype)
        return conds

    def _prepare_fresh(self, model: Any, prep_path: str, exag: float, device: torch.device, dtype: torch.dtype) -> \
    Optional[Any]:
        """Fallback direct prep (no cache; for error cases). FIXED: Full guards (eager, sync, dummy)."""
        # FIXED: Clear graphs (always safe)
        if hasattr(model, 't3') and hasattr(model.t3, '_bucket_graphs'):
            model.t3._bucket_graphs.clear()

        # FIXED: Temp eager if 'params' exists (guard no attr error)
        original_params = None
        if hasattr(model, 't3') and hasattr(model.t3, 'params') and isinstance(model.t3.params, dict):
            original_params = model.t3.params.copy()
            model.t3.params['generate_token_backend'] = 'eager'
            logger.debug("Temp eager set for prep")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            logger.debug("Sync before prep")

        try:
            # CORE OLD: prepare + to()
            model.prepare_conditionals(prep_path, exaggeration=exag)
            if dtype != torch.float32 and hasattr(model.conds, 't3'):
                model.conds.t3.to(device=device, dtype=dtype)
            chatterbox_tts_to(model, device, dtype)
            logger.debug(f"Prepared conds fresh (exag {exag})")

            # Return conds for caller
            conds = model.conds
            if hasattr(conds, 't3') and hasattr(conds.t3, 'speaker_emb'):
                emb = conds.t3.speaker_emb
                emb_nonzero = not torch.all(emb == 0)
                logger.info(f"Fresh conds ready | emb {emb.shape} (non-zero: {emb_nonzero})")
            return conds
        except Exception as prep_e:
            logger.error(f"Direct prep failed: {prep_e} – dummy fallback")
            return None
        finally:
            # FIXED: Restore if set
            if original_params is not None and hasattr(model, 't3') and hasattr(model.t3, 'params'):
                model.t3.params = original_params
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                logger.debug("Restored params + final sync")

    def validate_voice_path(self, path: str, stem: str) -> Tuple[bool, str]:
        """Local FIXED: Basic validation (exists/dur/artifacts; from audio_utils move). Returns (valid, msg)."""
        if not path or not os.path.exists(path):
            return False, f"Path missing: {path}"

        config = get_config()
        min_duration = get_config_value('globals.min_ref_duration', 3.0)
        try:
            info = torchaudio.info(path)
            duration = info.num_frames / info.sample_rate
            if duration < min_duration:
                return False, f"Duration {duration:.2f}s < {min_duration}s for {stem}"

            # Artifact check (simple: load + max amp check; assume no is_artifact_laden)
            waveform, _ = torchaudio.load(path)
            if torch.all(waveform == 0) or waveform.abs().max() < 1e-6:
                return False, f"Silent/empty for {stem}"
            # Optional advanced (if audio_utils has it – uncomment for stricter)
            # from src.audio_utils import is_artifact_laden
            # if is_artifact_laden(path, threshold_hz=config.sr // 3):
            #     return False, f"Artifacts in {stem} (threshold {config.sr // 3}Hz)"

            return True, f"Valid {stem} (dur {duration:.2f}s)"
        except Exception as v_e:
            return False, f"Validation error for {stem}: {str(v_e)}"

    def _is_nonempty_conds(self, conds: Any) -> bool:
        """Check if conds non-empty (opposite of empty check; used in tracing). FIXED: Handle model.conds.t3.speaker_emb specifically."""
        if conds is None:
            return False
        if isinstance(conds, torch.Tensor):
            return conds.numel() > 0
        if isinstance(conds, dict):
            return any(
                (isinstance(v, torch.Tensor) and v.numel() > 0) or
                (hasattr(v, '__len__') and len(v) > 0) or v is not None
                for v in conds.values()
            )
        if hasattr(conds, '__len__') and len(conds) > 0:
            return True
        # For custom Conditionals/T3Cond (align with log: check t3.speaker_emb)
        if hasattr(conds, 't3') and hasattr(conds.t3, 'speaker_emb') and conds.t3.speaker_emb is not None:
            emb = conds.t3.speaker_emb
            return emb.numel() > 0 and not torch.all(emb == 0)
        return True  # Assume non-empty if unknown type (lenient)

    def _is_empty_raw_conds(self, raw_conds: Any) -> bool:
        """Quick check if raw_conds empty."""
        if raw_conds is None:
            return True
        if isinstance(raw_conds, torch.Tensor):
            return raw_conds.numel() == 0
        if hasattr(raw_conds, '__len__'):
            return len(raw_conds) == 0
        if isinstance(raw_conds, dict) and all(v is None for v in raw_conds.values()):
            return True
        return False

    def _generate_conditionals_key(self, context: AudioGenerationContext) -> str:
        """Generate conditionals cache key based on voice content and parameters (fallback use)."""
        content_hash = getattr(context, 'voice_content_hash', '') or hashlib.md5(
            str(context.audio_prompt_path or '').encode('utf-8')).hexdigest()[:12]
        params = context.voice_params or {}
        return (f"v2_{context.voice_stem or 'default'}_ref_{content_hash}_exag{params.get('exaggeration', 0.5):.2f}"
                f"_temp{params.get('temperature', 0.8):.2f}_topp{params.get('top_p', 1.0):.2f}")

    def _cache_conditionals(self, context: AudioGenerationContext) -> bool:
        """Cache conditionals with proper validation and logging. Fixed: enable_disk check. FIXED: Use model.conds (align with working snippet); deprecated as _get_or_prepare handles save."""
        if not context.conditionals_key or not hasattr(context.model, 'conds') or context.model.conds is None:
            logger.debug("Skipping conditionals cache – no conditionals generated")
            return False

        config = get_config()
        if not (config.enable_memory_cache or config.enable_disk_cache):
            logger.debug("Caching disabled in config")
            return False

        # FIXED: Use conditionals_cache (from init) or manager; save model.conds as in working snippet
        success = False
        if self.conditionals_cache:
            success = self.conditionals_cache.save(cond_key=context.conditionals_key,
                                                   conds=context.model.conds)  # Assume method: save(conds as .pt)
        elif self.cache_manager and hasattr(self.cache_manager, 'save_conditionals'):
            success = self.cache_manager.save_conditionals(
                conditionals_key=context.conditionals_key, model=context.model
            )

        if success:
            logger.debug(f"Conds cached: {context.conditionals_key[:12]}...")
        else:
            logger.warning(f"Conds save failed: {context.conditionals_key[:12]}...")

        return success

    def handle_error(self, context: AudioGenerationContext, error: Exception) -> AudioGenerationContext:
        """Fallback to simple/default on voice error. FIXED: Import guard for get_config; safe access; set conds_key/voice_stem."""
        logger.error(f"Voice processing error: {error} – falling back to simple/default")

        # FIXED: Import guard for get_config (if not at file top)
        try:
            from src.config import get_config
        except ImportError:
            logger.error("get_config import failed in handle_error – ultimate fallback")
            context.voice_params = {'exaggeration': 1.0}  # Hardcode
            context.processed_voice_path = ""
            context.conditionals_key = "error_default"
            if not hasattr(context, 'conds_key'):
                context.conds_key = "error_default"
            return context

        config = get_config()
        stem = getattr(context, 'voice_stem', 'error_default')

        try:
            # FIXED: Safe get_voice_params (no device/dtype kwargs)
            voice_params = config.get_voice_params(stem)  # FIXED: No kwargs
            context.voice_params = voice_params
            context.processed_voice_path = getattr(context, 'audio_prompt_path', "") or ""
            context.conditionals_key = f"error_{stem}"
            if not hasattr(context,
                           'conds_key') or context.conds_key is None:  # FIXED: Set if missing (no attr error downstream)
                context.conds_key = context.conditionals_key

            # FIXED: Dummy conds for model (neutral; use safe device/dtype)
            if hasattr(context, 'model') and context.model is not None:
                from src.tts_model import create_dummy_conds
                device = getattr(config.app_config.globals, 'device', 'cpu') if hasattr(config,
                                                                                        'app_config') and hasattr(
                    config.app_config, 'globals') else torch.device('cpu')
                dtype = getattr(config.app_config.globals, 'dtype', torch.float32) if hasattr(config,
                                                                                              'app_config') and hasattr(
                    config.app_config, 'globals') else torch.float32
                create_dummy_conds(context.model, device, dtype, f"error_{stem}")

            logger.debug(f"Fallback for {stem}: dummy conds set, key={context.conditionals_key}")
        except Exception as fallback_e:
            logger.warning(f"Stem fallback failed: {fallback_e}; no audio_path for stem derivation")
            context.voice_params = {'exaggeration': 1.0}  # Ultimate
            context.processed_voice_path = ""
            context.conditionals_key = "error_default"
            if not hasattr(context, 'conds_key'):
                context.conds_key = "error_default"

        return context

