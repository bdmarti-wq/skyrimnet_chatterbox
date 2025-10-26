import os
import time
from pathlib import Path

import torch
import torchaudio

from .base import GenerationPhase
from ...pipeline.context import AudioGenerationContext
from loguru import logger
from src.audio_utils import is_artifact_laden  # For validate_cached_audio


class CacheCheckPhase(GenerationPhase):
    def __init__(self, cache_manager=None):
        self.cache_manager = cache_manager
        super().__init__()

    def _execute_core(self, context: AudioGenerationContext) -> AudioGenerationContext:
        """Early audio_cache/fuzzy_cache checks (before voice_reference). If HIT, set flags/path, return (full bypass). Else, continue to voice. FIXED: Load tensor on HIT (avoids None.shape); guard duration on MISS."""
        context.ensure_attrs()  # Stem/text ready

        if not self.cache_manager:
            context.is_cached = False
            context.cache_hit_type = 'no_cache'
            return context

        audio_prompt = context.audio_prompt_path or ""
        voice_stem = context.voice_stem
        text = context.text
        exag = context.exaggeration

        if not voice_stem or not text.strip():
            context.cache_hit_type = 'invalid_input'
            return context

        # Early full audio cache checks (bypass all expensive phases)
        cache_uuid = getattr(context, 'cache_uuid', int(time.time() * 1000) % (2 ** 32))  # Ensure
        cache_key = self.cache_manager.generate_audio_cache_key(voice_stem, text, exag, cache_uuid)


        # 1. Exact match (audio_cache)
        exact_path = self.cache_manager.get_audio_cache(cache_key)
        if exact_path and os.path.exists(exact_path):
            if self._validate_cached_audio(exact_path, voice_stem):
                try:
                    # FIXED: Load WAV tensor on exact HIT to enable .shape access (avoids None error in logs/post-process)
                    wav_tensor, _ = torchaudio.load(exact_path)  # Load audio (handle multi-channel by mean)
                    context.processed_wav = wav_tensor.mean(dim=0, keepdim=True).to(context.device,
                                                                                    context.dtype)  # Average channels, to device/dtype
                    context.audio_duration = context.processed_wav.shape[1] / context.sr  # Compute from tensor
                    logger.info(
                        f"AUDIO EXACT HIT: {cache_key[:20]} → {exact_path} (loaded tensor, duration: {context.audio_duration:.2f}s; bypass gen/conds/post; RTF ∞)")
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

        # 2. Fuzzy match (if exact miss)
        fuzzy_path = self.cache_manager.get_fuzzy_audio_cache(audio_path=audio_prompt, text=text, stem=voice_stem,
                                                              threshold=0.70)
        if fuzzy_path and os.path.exists(fuzzy_path):
            if self._validate_cached_audio(fuzzy_path, voice_stem):
                try:
                    # FIXED: Load WAV tensor on fuzzy HIT to enable .shape access (avoids None error in logs/post-process)
                    wav_tensor, _ = torchaudio.load(fuzzy_path)  # Load audio (handle multi-channel by mean)
                    context.processed_wav = wav_tensor.mean(dim=0, keepdim=True).to(context.device,
                                                                                    context.dtype)  # Average channels, to device/dtype
                    context.audio_duration = context.processed_wav.shape[1] / context.sr  # Compute from tensor
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

        # Continue to voice_reference (partial conds skip)
        result = self.cache_manager.process_voice_reference(audio_path=audio_prompt, voice_stem=voice_stem)

        # FIXED: Robust parse (by position/type, ignores empty '' in tuple – logs for debug)
        logger.debug(
            f"Voice result: {len(result) if isinstance(result, (list, tuple)) else 'invalid type {type(result)}'} items, types={[type(x) for x in result] if isinstance(result, (list, tuple)) else [type(result)]}")
        processed_path = None
        conds_key = None
        voice_params = {}
        hit_entry = None
        if isinstance(result, (list, tuple)) and len(result) == 5:
            processed_path, empty, conds_key, voice_params, hit_entry = result  # Align: (path, '', key, dict, entry)
            if empty != '':  # Rare non-empty
                logger.warning(f"Unexpected empty field: '{empty}' – assuming tuple align")
        elif isinstance(result, (list, tuple)) and len(result) == 4:
            processed_path, conds_key, voice_params, hit_entry = result  # Skip inferred empty
        elif isinstance(result, (list, tuple)) and len(result) == 3:
            processed_path, conds_key, voice_params = result
        else:
            logger.warning("Invalid result type/len from process_voice_reference (not tuple or empty) – MISS")

        if conds_key and isinstance(conds_key, str):
            logger.debug(
                f"Parsed voice: path={processed_path}, key={conds_key[:20]}..., params type={type(voice_params)}")
        else:
            logger.warning("Invalid voice parse – MISS (processed as MISS)")

        # Derive/validate path (unchanged)
        if not processed_path or not os.path.exists(processed_path):
            globals_dict = context.get_globals()
            cache_root = globals_dict.get('cache_dir', Path('./cache'))
            voice_dir = cache_root / "voices"
            voice_dir.mkdir(parents=True, exist_ok=True)
            derived = voice_dir / f"{voice_stem}.wav"
            if derived.exists() and self.validate_path(str(derived)):
                processed_path = derived
                logger.debug(f"Derived voice path: {derived}")
            else:
                resampled_dir = voice_dir / "resampled"
                resampled_dir.mkdir(parents=True, exist_ok=True)
                resampled = resampled_dir / f"{voice_stem}_24000Hz.wav"
                if resampled.exists() and self.validate_path(str(resampled)):
                    processed_path = resampled
                    logger.debug(f"Derived resampled: {resampled}")

        if processed_path:
            is_valid, msg = self.validate_voice_ref(processed_path, voice_stem)
            if not is_valid:
                logger.warning(f"Derived path invalid {voice_stem}: {msg} – MISS")
                processed_path = None  # Don't set invalid
            else:
                # FIXED: Optional load of ref tensor for .shape access (if ref accessed later; non-crashing)
                try:
                    wav_tensor, _ = torchaudio.load(processed_path)
                    context.processed_ref_path = wav_tensor.mean(dim=0, keepdim=True).to(context.device,
                                                                                         context.dtype)  # Token tensor for ref
                    logger.debug(f"Loaded voice ref tensor: {processed_path}")
                except Exception as load_e:
                    logger.warning(f"Failed to load voice ref tensor {processed_path}: {load_e}; use path only")

        # FIXED: Guard duration computation (processed_wav None on MISS; set in Gen/Post)
        if context.processed_wav is not None and hasattr(context.processed_wav, 'shape'):
            context.audio_duration = context.processed_wav.shape[1] / context.sr
        else:
            context.audio_duration = 0.0  # Avoid crash; compute later
            logger.debug("Guarded duration set=0.0 (MISS; wav set in Gen)")

        context.processed_voice_path = str(processed_path) if processed_path else ""
        context.processed_ref_path = context.processed_voice_path
        context.conditionals_key = conds_key or f"stub_{voice_stem}_{int(time.time() % 10000)}"
        context.conds_key = context.conditionals_key
        context.voice_params = voice_params  # Dict ensured; fallback {} if str

        # Voice HIT/MISS (partial conds skip if HIT)
        if hit_entry is not None:
            context.is_cached = True  # Partial voice
            context.cache_hit_type = 'voice_reuse'  # Overrides audio_miss
            logger.info(f"VOICE HIT: {voice_stem} (stable; after audio miss)")
        else:
            context.is_cached = False
            context.cache_hit_type = 'voice_miss'
            logger.info(f"VOICE MISS: {voice_stem} (new stable; after audio miss)")

        context.cache_key = cache_key  # Full key
        logger.debug(
            f"CacheCheck MISS (full pipeline) | voice: {context.cache_hit_type} | path: {bool(processed_path)}")
        return context


    def _validate_cached_audio(self, path: str, stem: str) -> bool:
        """Validate cached path (exists + no artifacts + subpath). FIXED: Attribute access for globals_dict (no .get)."""
        if not path or not os.path.exists(path):
            logger.debug(f"Invalid cached path (missing): {path}")
            return False

        p = Path(path).resolve()
        # FIXED: Direct attr access (namedtuple/object, fallback Path('./cache'))
        try:
            cache_root = self.cache_manager.config.app_config.globals.cache_dir if hasattr(
                self.cache_manager.config.app_config.globals, 'cache_dir') else Path('./cache')
        except AttributeError:
            cache_root = Path('./cache')
        base_dir = (cache_root / "audio" / "output").resolve()
        if not p.is_relative_to(base_dir):
            logger.warning(f"Cached path not in subpath {base_dir}: {path}")
            return False

        # Artifact check (purge if laden; align threshold)
        threshold_hz = 7000  # From config/fuzzy
        if is_artifact_laden(path, threshold_hz=threshold_hz):
            logger.warning(f"Artifact in cached audio {path} (stem={stem}; thresh={threshold_hz}Hz) – invalid")
            return False

        logger.trace(f"Validated cached audio: {path} (stem={stem}; clean)")
        return True

    @classmethod
    def validate_voice_ref(cls, audio_path: str, stem: str) -> tuple[bool, str]:
        """Align with voice_reference.validate_voice_prompt (dur>=3s, SR check, non-empty; artifacts skipped for refs)."""
        if not audio_path or not os.path.exists(audio_path):
            return False, f"Missing: {audio_path}"

        try:
            info = torchaudio.info(audio_path)
            duration = info.num_frames / info.sample_rate
            if duration < 3.0:  # min_ref_duration
                return False, f"Short {stem}: {duration:.2f}s < 3s"
            if info.sample_rate != 24000:
                logger.warning(f"SR mismatch {stem}: {info.sample_rate}Hz != 24000Hz")
            waveform, _ = torchaudio.load(audio_path)
            if waveform.numel() == 0 or torch.max(torch.abs(waveform)) <= 1e-6:
                return False, f"Empty/silent {stem}"
            # Artifacts skipped for voices (as in original)
            logger.trace(f"Valid ref {stem}: {duration:.2f}s @ {info.sample_rate}Hz")
            return True, f"Valid ({duration:.2f}s)"
        except Exception as e:
            return False, f"Validation error {stem}: {e}"

    def handle_error(self, context: AudioGenerationContext, error: Exception) -> AudioGenerationContext:
        context.is_cached = False
        context.cache_hit_type = 'error_miss'
        return super().handle_error(context, error)