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
        """If cache miss, prep conds on processed_path (reuse if HIT). FIXED: None-safe exag format; stable cond_key with hash."""
        start_time = time.perf_counter()
        config = get_config()
        stem = context.voice_stem or normalize_stem(context.audio_prompt_path or '')
        processed_path = getattr(context, 'processed_ref_path', context.audio_prompt_path or '')  # Safe getattr

        if not processed_path or not os.path.exists(processed_path):
            logger.warning(f"No valid path for {stem} – dummy conds")
            create_dummy_conds(context.model, config.device, config.dtype, "no_path")
            context.conditionals_key = f"dummy_{stem}"
            context.voice_ref_processed = False
            voice_time = time.perf_counter() - start_time
            logger.info(f"Voice time: {voice_time:.3f}s (dummy for {stem})")
            return context

        # Pad text pre-prep
        from src.audio_utils import pad_short_text
        # FIXED: Guard voice_params (prevent None.items in handle_error fallback)
        voice_params = getattr(context, 'voice_params', {}) or {}
        if isinstance(voice_params, dict):
            context.text = pad_short_text(context.text, voice_params)
        else:
            logger.warning("voice_params not dict; defaulting empty")
            context.text = pad_short_text(context.text, {})

        # NEW: Stable cond_key using voice_content_hash (from voice_reference; fallback MD5 on path for reuse)
        content_hash = getattr(context, 'voice_content_hash', None)
        if content_hash is None:
            # Fallback: Compute MD5 on path (stable across uploads if content same)
            try:
                with open(processed_path, 'rb') as f:
                    content_hash = hashlib.md5(f.read()).hexdigest()[:12]
                context.voice_content_hash = content_hash  # Store for downstream
            except Exception as hash_e:
                logger.warning(f"Hash fallback failed: {hash_e}; using timestamp")
                content_hash = f"hash_{int(time.time() % 1000000)}"[:12]
        exag = voice_params.get('exaggeration', 1.0)
        if exag is None:  # FIXED: Handle None
            exag = 1.0
            logger.warning("Exaggeration None; default 1.0")
        temp = voice_params.get('temperature', 0.8)
        topp = voice_params.get('top_p', 1.0)
        cond_key = f"hash_{content_hash}_exag{exag:.2f}_temp{temp:.2f}_topp{topp:.2f}"  # TODO review do we need stem?
        context.conditionals_key = cond_key

        # FIXED: Prep with cache integration (get_or_prep atomic; HIT skips prep)
        prep_time = time.perf_counter()
        conds_loaded = False
        if self.conditionals_cache:
            conds = self._prepare_conditionals(context, processed_path, voice_params)
            if conds is not None:
                conds_loaded = True
        else:
            # No cache: Always fresh prep (use fallback)
            conds = self._prepare_fresh(context.model, processed_path, exag, context.device, context.dtype)
            if conds is not None:
                conds_loaded = True

        prep_time = time.perf_counter() - prep_time
        status = "reuse" if conds_loaded else "prep"
        if conds_loaded and self.conditionals_cache:
            status = "cache_reuse" if hasattr(context,
                                              'conds_from_cache') and context.conds_from_cache else "fresh_prep"
        logger.info(
            f"Prepared conds on {processed_path} for {stem} (key: {cond_key[:20]}...) | time {prep_time:.3f}s ({status})")

        if conds is None:
            # Failed: Dummy fallback
            logger.warning(f"Prep failed for {stem} – dummy conds")
            create_dummy_conds(context.model, context.device, context.dtype, f"prep_fail_{stem}")
            conds_loaded = False

        # FIXED: Ensure voice_params is always dict
        context.voice_params = voice_params or {}
        context.voice_ref_processed = True
        voice_time = time.perf_counter() - start_time
        logger.info(
            f"Voice time: {voice_time:.3f}s ({'reuse' if conds_loaded else 'prep'} for {stem})")
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
        """Handle errors with fallback (safe sr). FIXED: Guard voice_params (no None.items); safe config.device/globals; dummy conds; path safety."""
        logger.error(f"Voice processing error: {str(error)} – falling back to simple/default")
        context.processed_voice_path = context.audio_prompt_path or ""
        try:
            context.voice_stem = normalize_stem(context.processed_voice_path) or 'default'
        except Exception as stem_e:
            logger.warning(f"Stem fallback failed: {stem_e}")
            context.voice_stem = 'default'

        config = get_config()
        # FIXED: Safe config attrs (nested/flat; fallback to context/defaults – no 'device' error)
        if hasattr(config, 'globals'):
            device = getattr(config.globals, 'device', None) or getattr(config, 'device', None) or \
                     (context.device if hasattr(context, 'device') else torch.device(
                         'cuda' if torch.cuda.is_available() else 'cpu'))
            dtype = getattr(config.globals, 'dtype', None) or getattr(config, 'dtype', None) or \
                    (context.dtype if hasattr(context, 'dtype') else torch.bfloat16)
            sr = getattr(config.globals, 'sr', None) or getattr(config, 'sr', None) or \
                 (context.sr if hasattr(context, 'sr') else 24000)
        else:
            # Flat config or no attribs
            device = getattr(config, 'device', context.device if hasattr(context, 'device') else torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'))
            dtype = getattr(config, 'dtype', context.dtype if hasattr(context, 'dtype') else torch.bfloat16)
            sr = getattr(config, 'sr', context.sr if hasattr(context, 'sr') else 24000)

        # FIXED: Additional guard if still None (rare)
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if dtype is None:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        if sr is None or not isinstance(sr, int):
            sr = 24000

        # FIXED: Guard voice_params (prevent None.items in get_voice_params or access)
        voice_params = getattr(context, 'voice_params', {}) or {}
        if not isinstance(voice_params, dict):
            voice_params = {}  # Force dict
        context.voice_params = config.get_voice_params(context.voice_stem,
                                                       {'exaggeration': 1.0}) or voice_params  # Safe fallback

        context.conditionals_key = f"fallback_{context.voice_stem}_{int(time.time() % 10000)}"
        context.voice_ref_processed = False

        # FIXED: Use create_dummy_conds (from working snippet); guard model exists
        if hasattr(context, 'model') and context.model is not None:
            create_dummy_conds(context.model, device, dtype, f"error_{context.voice_stem}")

        # FIXED: Silence fallback with get_silence (safe sr/device; CPU for persistence)
        from src.audio_utils import get_silence
        if not hasattr(context, 'generated_wav') or context.generated_wav is None:
            context.generated_wav = get_silence(duration=2.0, sr=sr, dtype=torch.float32,
                                                device=torch.device('cpu')).to(device, dtype)

        # FIXED: Ensure path not None (fixes PathLike in downstream gen/output)
        if not hasattr(context, 'audio_prompt_path') or context.audio_prompt_path is None:
            context.audio_prompt_path = ""  # Valid str for os/Path

        logger.debug("Error fallback complete; proceeding with default (dummy conds + silence)")
        return context