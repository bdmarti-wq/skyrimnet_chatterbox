# REFACTORED: Use context.get_globals(); shared validate_path/create_silence; simplify fallbacks (no dup sr/device fetches). Queue only if HIT method.
import os
import time
from typing import Optional

import torch
import torchaudio
from pathlib import Path

from src.config import get_config
from .base import GenerationPhase
from ...cache import CacheManager
from ...pipeline.context import AudioGenerationContext
from loguru import logger

class OutputPhase(GenerationPhase):
    """REFACTORED: Always save; use base validate_path; shared silence/save fallback."""

    process_after_cache = True

    def __init__(self, cache_manager=None):
        self.cache_manager = cache_manager or CacheManager(get_config())
        self.audio_cache = getattr(self.cache_manager, 'audio_cache', None)
        if self.audio_cache is None:
            logger.warning("No audio_cache; saves only")
        super().__init__()

    def _execute_core(self, context: AudioGenerationContext) -> AudioGenerationContext:
        """REFACTORED: Guard empty wav (shared silence); save to output_dir; verify; queue if MISS."""
        # REFACTORED: Shared empty wav guard + silence
        if not hasattr(context, 'processed_wav') or context.processed_wav is None or context.processed_wav.numel() == 0:
            silence = self.create_silence(context.sr, 2.0, context.device, torch.float32)
            context.processed_wav = silence
            logger.warning("Created silence for empty processed_wav")

        # Dir (use context cache_dir or fallback)
        globals_dict = context.get_globals()
        cache_root = globals_dict.get('cache_dir', Path('./cache'))
        output_dir = getattr(self.cache_manager, 'output_dir', cache_root / "audio" / "output")
        output_dir.mkdir(parents=True, exist_ok=True)

        # REFACTORED: Filename (use context to ensure stem/text)
        context.ensure_attrs()
        sr_int = context.sr
        voice_stem = context.voice_stem
        text_hash = abs(hash(context.text or '')) % 1000000
        timestamp = int(time.time() * 1000) % 10000  # Unique
        base_name = f"{voice_stem}_{text_hash}_{timestamp}_{sr_int}kHz.wav"
        output_path = output_dir / base_name

        # Save (CPU float32)
        wav_cpu = context.processed_wav.cpu().float()
        torchaudio.save(str(output_path), wav_cpu, sr_int)

        # Verify (shared-like: exists + size)
        if output_path.exists() and output_path.stat().st_size > 100:  # Min size
            context.output_path = str(output_path.absolute())
            size_kb = output_path.stat().st_size / 1024
            logger.info(f"Saved/verified WAV: {context.output_path}, size={size_kb:.1f}KB")
        else:
            raise OSError("Save failed (no file/zero-size)")

        # Queue (only MISS; if method)
        if not context.is_cached and self.audio_cache and hasattr(self.audio_cache, 'async_cache_postgen'):
            try:
                cache_key = getattr(context, 'cache_key', f"postgen_{text_hash}")
                self.audio_cache.async_cache_postgen(cache_key, context.output_path, context.text, voice_stem)
                logger.debug(f"Queued postgen for {base_name}")
            except Exception as queue_e:
                logger.warning(f"Queue failed: {queue_e} – saved only")

        return context

    def handle_error(self, context: AudioGenerationContext, error: Exception) -> AudioGenerationContext:
        """REFACTORED: Try fallback save (shared silence to output_dir); ultimate temp; delegate to base if fails."""
        logger.error(f"Output failed: {error} – fallback save")
        voice_stem = getattr(context, 'voice_stem', 'fallback')
        fallback_path = self._save_fallback_to_dir(voice_stem, context.sr, output_dir_from_context=context)
        if fallback_path:
            context.output_path = fallback_path
            # Queue if possible (same as core)
            if not context.is_cached and self.audio_cache and hasattr(self.audio_cache, 'async_cache_postgen'):
                try:
                    cache_key = getattr(context, 'cache_key', f"fb_{int(time.time())}")
                    self.audio_cache.async_cache_postgen(cache_key, fallback_path, context.text, voice_stem)
                except Exception:
                    pass
            return context

        # Ultimate temp (simplified; no env bloat)
        temp_path = self._create_temp_silence(context.sr)
        context.output_path = temp_path
        logger.warning(f"Ultimate temp fallback: {temp_path}")
        return super().handle_error(context, error)  # Base silence if needed

    def _save_fallback_to_dir(self, voice_stem: str, sr: int, output_dir: Path) -> Optional[str]:
        """REFACTORED: Shared fallback save to dir (no dupe sr/device; use create_silence)."""
        output_dir.mkdir(parents=True, exist_ok=True)
        sr_final = int(sr)
        silence = self.create_silence(sr_final, 2.0, torch.device('cpu'), torch.float32)
        timestamp = int(time.time())
        fallback_path = output_dir / f"{voice_stem}_fb_{timestamp}_{sr_final}kHz.wav"
        try:
            torchaudio.save(str(fallback_path), silence, sr_final)
            if fallback_path.exists() and fallback_path.stat().st_size > 0:
                return str(fallback_path.absolute())
        except Exception as save_e:
            logger.error(f"Fallback save error: {save_e}")
        return None

    def _create_temp_silence(self, sr: int) -> str:
        """REFACTORED: Temp silence (hardcode/minimal; use create_silence)."""
        import tempfile
        sr_final = int(sr)
        silence = self.create_silence(sr_final, 2.0)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            torchaudio.save(tmp.name, silence, sr_final)
        return tmp.name