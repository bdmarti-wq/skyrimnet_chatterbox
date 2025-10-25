import os
import time
from pathlib import Path

import torch
import torchaudio

from .base import GenerationPhase
from ...pipeline.context import AudioGenerationContext
from loguru import logger


class CacheCheckPhase(GenerationPhase):
    def __init__(self, cache_manager=None):
        self.cache_manager = cache_manager
        super().__init__()

    def _execute_core(self, context: AudioGenerationContext) -> AudioGenerationContext:
        """FIXED: Robust unpack/result parsing from process_voice_reference (5-tuple: success/bool, path, key, params_dict, hit_entry). Defaults if misalign."""
        context.ensure_attrs()  # Now guards voice_params to dict

        if not self.cache_manager:
            context.is_cached = False
            context.cache_hit_type = 'no_cache'
            return context

        audio_prompt = context.audio_prompt_path or ""
        voice_stem = context.voice_stem
        text = context.text

        if not voice_stem or not text.strip():
            context.cache_hit_type = 'invalid_input'
            return context

        # FIXED: Get result, parse safely (aligns with voice_reference 5-tuple)
        result = self.cache_manager.process_voice_reference(audio_path=audio_prompt, voice_stem=voice_stem)

        # Parse: Assume [success?, path, key, params, hit_entry] – index safely
        processed_path = None
        conds_key = None
        voice_params = {}
        hit_entry = None

        if isinstance(result, (list, tuple)):
            if len(result) >= 1:
                processed_path = result[
                    1 if len(result) > 1 and isinstance(result[0], bool) else 0]  # Skip bool if present
            if len(result) >= 2:
                conds_key = result[2 if len(result) > 2 and isinstance(result[0], bool) else 1] if isinstance(
                    result[2 if len(result) > 2 else 1], str) else None
            if len(result) >= 3:
                param_idx = 3 if len(result) > 3 and isinstance(result[0], bool) else 2
                raw_params = result[param_idx]
                voice_params = raw_params if isinstance(raw_params, dict) else {}  # Ensure dict
            if len(result) >= 4:
                hit_idx = 4 if len(result) > 4 and isinstance(result[0], bool) else 3
                hit_entry = result[hit_idx]

        if conds_key and isinstance(conds_key, str):
            logger.debug(f"Parsed: path={processed_path}, key={conds_key[:20]}..., params type={type(voice_params)}")
        else:
            logger.warning("Invalid parse from process_voice_reference – MISS")

        # Derive/validate path (same as before)
        if not processed_path or not os.path.exists(processed_path):
            globals_dict = context.get_globals()
            cache_root = globals_dict.get('cache_dir', Path('./cache'))
            voice_dir = cache_root / "voices"
            voice_dir.mkdir(parents=True, exist_ok=True)
            derived = voice_dir / f"{voice_stem}.wav"
            if derived.exists() and self.validate_path(str(derived)):
                processed_path = derived
                logger.debug(f"Derived path: {derived}")
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
                logger.warning(f"Path invalid {voice_stem}: {msg} – MISS")
                processed_path = None

        context.processed_voice_path = str(processed_path) if processed_path else ""
        context.processed_ref_path = context.processed_voice_path
        context.conditionals_key = conds_key or f"stub_{voice_stem}_{int(time.time() % 10000)}"
        context.conds_key = context.conditionals_key
        context.voice_params = voice_params  # Already dict from parse

        # HIT/MISS
        if hit_entry:
            context.is_cached = True
            context.cache_hit_type = 'voice_reuse'
            logger.info(f"VOICE HIT: {voice_stem} (stable)")
        else:
            context.is_cached = False
            context.cache_hit_type = 'voice_miss'
            logger.info(f"VOICE MISS: {voice_stem} (new stable)")

        context.cache_key = context.generate_cache_key(voice_stem, text, context.exaggeration)
        logger.debug(f"CacheCheck: {context.cache_hit_type} | path exists: {bool(processed_path)}")
        return context

    @classmethod
    def validate_voice_ref(cls, audio_path: str, stem: str) -> tuple[bool, str]:
        """FIXED: Align with voice_reference.validate_voice_prompt (dur>=3s, SR check, non-empty; artifacts skipped for refs)."""
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