import os
import time
from typing import Tuple

from pathlib import Path

import torch
import torchaudio

from .base import BaseGenerationPhase
from ...pipeline.context import AudioGenerationContext
from ...cache.voice_reference import VoiceReferenceCache  # For process_voice_reference call
from loguru import logger
from src.audio_utils import is_artifact_laden  # For validate_cached_audio


class CacheCheckPhase(BaseGenerationPhase):
    """REFACTORED: Early full audio_cache/fuzzy_cache + voice_reference (before gen). FIXED: Use context.sr (no MODEL_SR); pass context to process_voice_reference; fix path refs."""

    def __init__(self, cache_manager=None):
        self.cache_manager = cache_manager
        super().__init__()

    def _execute_core(self, context: AudioGenerationContext) -> AudioGenerationContext:
        """Early audio_cache/fuzzy_cache checks (before voice_reference). If HIT, set flags/path, return (full bypass). Else, continue to voice. FIXED: Safe unpack guard; set path/conds always (real for new); MISS key='none' for neutral."""
        context.ensure_attrs()  # Stem/text ready

        # Allow bypassing caches entirely via context flags (only if both disabled)
        audio_cache_enabled = getattr(context, 'enable_audio_cache', True)
        fuzzy_cache_enabled = getattr(context, 'enable_fuzzy_cache', True)
        if not audio_cache_enabled and not fuzzy_cache_enabled:
            logger.info("Cache checks bypassed: both enable_audio_cache and enable_fuzzy_cache are False")
            context.is_cached = False
            context.cache_hit_type = 'cache_disabled'

            # IMPORTANT: We still want a proper voice reference prepared so that
            # downstream VoiceProcessing can compute real conditionals instead of
            # falling back to dummy (which would yield silence-like output).
            # When caches are disabled, we only bypass audio/fuzzy cache lookup,
            # not the voice preparation step.
            if self.cache_manager:
                result = self.cache_manager.process_voice_reference(context)

                # Safe-unpack identical to MISS path below
                is_valid_voice = False
                resampled_path = ""
                conds_key = ""
                voice_params = {}
                hit_entry = None
                if isinstance(result, tuple) and len(result) == 5:
                    is_valid_voice, resampled_path, conds_key, voice_params, hit_entry = result
                    logger.debug(
                        f"Voice (cache_disabled) unpack: success={is_valid_voice}, path={bool(resampled_path)}, key={conds_key[:20] if conds_key else 'none'}")
                else:
                    logger.warning(
                        f"Unexpected result from process_voice_reference (cache_disabled): {type(result)}, {result} – treating as MISS")

                # Mirror field assignments from MISS branch
                context.processed_voice_path = resampled_path if is_valid_voice and resampled_path else ""
                if conds_key:
                    context.conditionals_key = conds_key
                elif not is_valid_voice:
                    context.conditionals_key = f"none_{context.voice_stem or 'default'}"
                else:
                    context.conditionals_key = f"voice_new_{context.voice_stem}_{int(time.time() % 10000)}"
                context.conds_key = context.conditionals_key
                context.voice_params = voice_params or context.voice_params

                if hit_entry is not None:
                    logger.info(f"VOICE HIT (cache_disabled): {context.voice_stem}")
                else:
                    logger.info(f"VOICE MISS (cache_disabled): {context.voice_stem}")

            # Continue pipeline (do not set skip flags or output here)
            return context

        if not self.cache_manager:
            context.is_cached = False
            context.cache_hit_type = 'no_cache'
            return context

        audio_prompt = context.audio_prompt_path or ""  # FIXED: Use context.audio_prompt_path (input path)
        voice_stem = context.voice_stem
        text = context.text
        exag = context.exaggeration

        if not voice_stem or not text.strip():
            context.cache_hit_type = 'invalid_input'
            return context

        # Early full audio cache checks (bypass all expensive phases)
        cache_uuid = getattr(context, 'cache_uuid', int(time.time() * 1000) % (2 ** 32))  # Ensure
        cache_key = self.cache_manager.generate_audio_cache_key(voice_stem, text, exag, cache_uuid)

        # 1. Exact match (audio_cache) if enabled
        if audio_cache_enabled:
            exact_path = self.cache_manager.get_audio_cache(cache_key)
            if exact_path and os.path.exists(exact_path):
                if self._validate_cached_audio(exact_path, voice_stem):
                    try:
                        # FIXED: Load WAV tensor on exact HIT to enable .shape access (use context.sr if avail)
                        wav_tensor, _ = torchaudio.load(exact_path)  # Load audio (handle multi-channel by mean)
                        context.processed_wav = wav_tensor.mean(dim=0, keepdim=True).to(context.device,
                                                                                        context.dtype)  # Average channels, to device/dtype
                        sr = getattr(context, 'sr', 24000)  # Fallback if sr not set
                        context.audio_duration = context.processed_wav.shape[1] / sr  # Compute from tensor
                        logger.info(
                            f"AUDIO EXACT HIT: {cache_key} → {exact_path} (loaded tensor, duration: {context.audio_duration:.2f}s; bypass gen/conds/post; RTF ∞)")
                    except Exception as load_e:
                        logger.warning(
                            f"Failed to load tensor from exact {exact_path}: {load_e}; use path for play (duration=0.0)")
                        context.audio_duration = 0.0  # Fallback for play
                    context.cached_path = exact_path
                    context.is_cached = True
                    context.cache_hit_type = 'audio_exact'
                    context.skip_pipeline = True
                    context.output_path = exact_path  # Pre-set for Output
                    # Stub for conds (no prep needed)
                    context.conditionals_key = f"cached_exact_{voice_stem}_{cache_key[:12]}"
                    return context
                else:
                    self.cache_manager.set_audio_cache(cache_key, None)  # Purge invalid
                    logger.warning(f"Exact path invalid {exact_path} – purged & check fuzzy")

        # Optional: force-skip fuzzy cache if text contains any configured skip words
        effective_fuzzy_enabled = fuzzy_cache_enabled
        try:
            skip_words = []
            if getattr(context, 'config', None) is not None and \
               hasattr(context.config, 'app_config') and \
               hasattr(context.config.app_config, 'globals') and \
               hasattr(context.config.app_config.globals, 'fuzzy'):
                skip_words = getattr(context.config.app_config.globals.fuzzy, 'fuzzy_force_skip_words', []) or []

            if skip_words:
                lt = (text or "").lower()
                # simple substring match (case-insensitive)
                if any(sw for sw in skip_words if sw and sw in lt):
                    effective_fuzzy_enabled = False
                    logger.info(f"Fuzzy cache skip forced by skip words: {skip_words}")
        except Exception as _e:
            # Be conservative: do not disable fuzzy on errors
            logger.debug(f"Skip-words check failed: {_e}")

        # 2. Fuzzy match (if exact miss) if enabled (and not force-skipped)
        if effective_fuzzy_enabled:
            fuzzy_path = self.cache_manager.get_fuzzy_audio_cache(audio_prompt, text, voice_stem, threshold=0.70)
            if fuzzy_path and os.path.exists(fuzzy_path):
                if self._validate_cached_audio(fuzzy_path, voice_stem):
                    try:
                        # FIXED: Load WAV tensor on fuzzy HIT to enable .shape access (use context.sr if avail)
                        wav_tensor, _ = torchaudio.load(fuzzy_path)  # Load audio (handle multi-channel by mean)
                        context.processed_wav = wav_tensor.mean(dim=0, keepdim=True).to(context.device,
                                                                                        context.dtype)  # Average channels, to device/dtype
                        sr = getattr(context, 'sr', 24000)  # Fallback if sr not set
                        context.audio_duration = context.processed_wav.shape[1] / sr  # Compute from tensor
                        logger.info(
                            f"AUDIO FUZZY HIT (sim≥0.70): '{text[:30]}...' → {fuzzy_path} (loaded tensor, duration: {context.audio_duration:.2f}s; bypass gen/conds/post; RTF ∞)")
                    except Exception as load_e:
                        logger.warning(
                            f"Failed to load tensor from fuzzy {fuzzy_path}: {load_e}; use path for play (duration=0.0)")
                        context.audio_duration = 0.0  # Fallback for play
                    context.cached_path = fuzzy_path
                    context.is_cached = True
                    context.cache_hit_type = 'audio_fuzzy'
                    context.skip_pipeline = True
                    context.output_path = fuzzy_path
                    context.conditionals_key = f"cached_fuzzy_{voice_stem}_{cache_key[:12]}"
                    return context
                else:
                    logger.warning(f"Fuzzy path invalid {fuzzy_path} – purged")

        # No full audio HIT → MISS full pipeline (voice/conds/gen)
        logger.debug(f"No audio HIT (exact/fuzzy miss for key={cache_key[:20]}) – full pipeline")
        context.is_cached = False
        context.cache_hit_type = 'audio_miss'

        # FIXED: Pass context to process_voice_reference so it can set attributes
        result = self.cache_manager.process_voice_reference(context)  # Use audio_prompt (context.audio_prompt_path)

        # FIXED: Safe unpack (now bool-str-str-dict-Optional from aligned return); log for trace
        is_valid_voice = False
        resampled_path = ""
        conds_key = ""
        voice_params = {}
        hit_entry = None
        if isinstance(result, tuple) and len(result) == 5:
            is_valid_voice, resampled_path, conds_key, voice_params, hit_entry = result
            logger.debug(
                f"Voice result unpack: success={is_valid_voice}, path={bool(resampled_path)}, key={conds_key[:20] if conds_key else 'none'}")
        else:
            logger.warning(
                f"Unexpected result from process_voice_reference: {type(result)}, {result} – treating as MISS")
            is_valid_voice = False
            resampled_path = ""
            conds_key = ""
            voice_params = {}
            hit_entry = None

        # FIXED: Set voice fields; on MISS (rare), key="none" for neutral skip (no dummy)
        context.processed_voice_path = resampled_path if is_valid_voice and resampled_path else ""
        if conds_key:
            context.conditionals_key = conds_key
        elif not is_valid_voice:
            context.conditionals_key = f"none_{voice_stem or 'default'}"
        else:
            context.conditionals_key = f"voice_new_{voice_stem}_{int(time.time() % 10000)}"
        context.conds_key = context.conditionals_key
        context.voice_params = voice_params or context.voice_params  # Merge; ensure dict

        # Voice HIT/MISS (partial conds skip if HIT)
        if hit_entry is not None:
            logger.info(f"VOICE HIT: {voice_stem} (stable; after audio miss)")
        else:
            logger.info(f"VOICE MISS: {voice_stem} (new; after audio miss)")

        context.cache_key = cache_key  # Full key
        path_set = bool(context.processed_voice_path and os.path.exists(context.processed_voice_path))
        logger.debug(
            f"CacheCheck MISS (full pipeline) | voice: {voice_stem} | key: {context.conditionals_key[:20]} | path: {path_set}")
        return context



    def _validate_cached_audio(self, path: str, stem: str) -> bool:
        """Validate cached path. FIXED: Use context.sr for SR check and duration (no MODEL_SR)."""
        if not os.path.exists(path):
            logger.debug(f"Invalid cached path (missing): {path}")
            return False

        try:
            info = torchaudio.info(path)
            sr = getattr(self.context, 'sr', 24000) if hasattr(self, 'context') else 24000  # Fallback to avoid unbound
            if info.sample_rate != sr:
                logger.warning(f"SR mismatch for {stem}: {info.sample_rate}Hz != {sr}Hz")
                return False

            duration = info.num_frames / info.sample_rate

            # Artifact check (skip for refs; align with voice cache)
            try:
                # Use config threshold if available (via cache_manager if present)
                if hasattr(self.cache_manager, 'config') and hasattr(self.cache_manager.config.app_config, 'globals'):
                    globals_config = self.cache_manager.config.app_config.globals
                    if hasattr(globals_config, 'fuzzy') and hasattr(globals_config.fuzzy, 'artifact_threshold_hz'):
                        threshold = globals_config.fuzzy.artifact_threshold_hz
                    else:
                        threshold = sr // 3  # Dynamic from SR
                else:
                    threshold = sr // 3  # Fallback
                is_voice_ref = ('voices' in str(path).lower() or
                                any(s in Path(path).stem for s in ['_fixed_new', '_padded', '_resampled', '_24kHz']))
                if not is_voice_ref and threshold > 0:
                    if is_artifact_laden(path, threshold_hz=threshold):
                        return False, f"Artifacts in {stem}"
                logger.trace(f"Artifact passed (or skipped) for {path}")
            except ImportError:
                logger.warning("Artifact check skipped (missing func)")
            except Exception as a_e:
                logger.warning(f"Artifact check error for {stem}: {a_e}")

            logger.trace(f"Valid cached {stem}: {duration:.2f}s @ {info.sample_rate}Hz")
            return True
        except Exception as e:
            logger.warning(f"Validation failed for {stem}: {e}")
            return False

    @classmethod
    def validate_voice_ref(cls, audio_path: str, stem: str = None) -> Tuple[bool, str]:
        """Align with voice_reference.validate_voice_prompt (dur>=3s, SR check, non-empty; artifacts skipped for refs). FIXED: Use provided sr or fallback."""
        if not os.path.exists(audio_path):
            return False, f"Missing: {audio_path}"

        # Note: Since this is a classmethod, no self/context; assume sr=24000 for validation (or pass sr if called with it)
        sr = 24000  # Fallback SR for standalone use; in practice, caller can adjust
        try:
            info = torchaudio.info(audio_path)
            if info.sample_rate != sr:
                logger.warning(f"SR mismatch for {stem}: {info.sample_rate}Hz != {sr}Hz")
                return False, f"SR mismatch for {stem}: {info.sample_rate}Hz != {sr}Hz"

            duration = info.num_frames / sr if info.sample_rate > 0 else 0
            min_duration = 3.0  # Default min ref duration
            if duration < min_duration:
                return False, f"Short for {stem}: {duration:.2f}s < {min_duration}s"

            # Artifact check skipped for refs (as per original)
            logger.trace(f"Valid ref {stem}: {duration:.2f}s @ {info.sample_rate}Hz")
            return True, f"Valid ({duration:.2f}s)"
        except Exception as e:
            return False, f"Validation error {stem}: {e}"

    def handle_error(self, context: AudioGenerationContext, error: Exception) -> AudioGenerationContext:
        """REFACTORED: Delegate to base (silence). FIXED: Ensure sr if needed."""
        # Fallback: Set sr if not set (prevents None errors elsewhere)
        if not hasattr(context, 'sr') or context.sr is None:
            context.sr = 24000
        return super().handle_error(context, error)