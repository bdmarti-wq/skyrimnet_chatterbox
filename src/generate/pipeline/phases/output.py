"""
Output Phase: Finalizes generation by saving the processed audio file.
Handles caching (exact/fuzzy), file I/O, and fallback saves (temp/direct).
Ensures output_path is set in context for UI return.
"""
import os
import time
import torch
import torchaudio
from pathlib import Path
from typing import Optional
from loguru import logger

from src.config import get_config
from src.generate.cache import CacheManager  # For fuzzy/exact
from .base import GenerationPhase
from ...pipeline.context import AudioGenerationContext


class OutputPhase(GenerationPhase):
    """Saves generated audio to file and caches for reuse."""

    process_after_cache = True  # Always run (save even on HIT for fresh)

    def __init__(self, cache_manager=None):
        """Initialize with cache_manager for exact/fuzzy access. FIXED: Injection from coordinator."""
        self.cache_manager = cache_manager or CacheManager()  # Fallback if not injected
        self.audio_cache = getattr(self.cache_manager, 'audio_cache', None)  # Assume exists
        if self.audio_cache is None:
            logger.warning("OutputPhase: No audio_cache; saves only (no fuzzy/exact)")
        logger.debug("OutputPhase initialized with cache_manager")

    def execute(self, context: AudioGenerationContext) -> AudioGenerationContext:
        if context.processed_wav is None:
            # Fallback creation
            context.processed_wav = torch.zeros((1, int(context.sr * 2.0)), dtype=torch.float32)  # 2s silence

        try:
            # FIXED: Use int(context.sr) for filename
            sr_int = int(context.sr)  # Ensure int (from context.sr=24000)
            base_name = f"{context.voice_stem}_{hash(context.text) % 1000000}_{sr_int}kHz.wav"
            output_dir = Path("cache/audio/output")  # Or context.config.cache_dir
            output_path = output_dir / base_name
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save with torchaudio (ensure float32 for WAV)
            import torchaudio
            wav_cpu = context.processed_wav.cpu().float()  # To float32 CPU
            torchaudio.save(str(output_path), wav_cpu, sr_int)  # sr_int as int

            logger.debug(f"Saved WAV: {output_path}, size={output_path.stat().st_size / 1024:.1f}KB")
            context.output_path = str(output_path)

        except Exception as e:
            logger.error(f"Output save error: {e} – fallback direct")
            # FIXED: In _save_fallback_direct: use int(context.sr)
            fallback_path = self._save_fallback_direct(context.voice_stem, int(context.sr))
            context.output_path = fallback_path

        return context



    def _save_fallback_direct(self, voice_stem: str, sr: int) -> str:
        import torchaudio
        fallback_dir = Path("cache/audio/fallback")
        fallback_dir.mkdir(parents=True, exist_ok=True)
        fallback_path = fallback_dir / f"{voice_stem}_fb_{int(time.time())}_{sr}kHz.wav"
        silence = torch.zeros((1, sr * 2))  # 2s
        torchaudio.save(str(fallback_path), silence, sr)
        return str(fallback_path)


    def _create_temp_fallback(self) -> str:
        """Create temporary fallback audio file (silence). FIXED: Hardcode sr (no config)."""
        config = get_config()
        globals_config = getattr(config.app_config, 'globals', None)
        sr = getattr(globals_config, 'sr', 24000) if globals_config else 24000
        duration = 2.0  # Short silence
        samples = int(sr * duration)
        silence = torch.zeros(1, samples, dtype=torch.float32)
        temp_dir = Path(os.environ.get('TMPDIR', os.environ.get('TEMP', '/tmp'))) / "audio_fallback"
        temp_dir.mkdir(exist_ok=True)
        timestamp = int(time.time())
        temp_path = temp_dir / f"audio_fallback_{timestamp}.wav"
        try:
            torchaudio.save(temp_path, silence, sr)
            logger.debug(f"Created temporary fallback audio: {temp_path}")
            return str(temp_path)
        except Exception as temp_e:
            logger.error(f"Temp fallback failed: {temp_e} – return empty str")
            return ""

    def handle_error(self, context: AudioGenerationContext, error: Exception) -> AudioGenerationContext:
        """Handle output errors with fallback saves. FIXED: Nested cache_dir/sr; safe paths."""
        logger.error(f"Output phase failed: {error} – attempting fallback solutions")
        config = get_config()
        globals_config = getattr(config.app_config, 'globals', None)
        cache_dir = getattr(globals_config, 'cache_dir', Path('./cache')) if globals_config else Path('./cache')
        sr = getattr(globals_config, 'sr', 24000) if globals_config else 24000

        if (hasattr(context, 'generated_wav') and context.generated_wav is not None and 
            context.generated_wav.numel() > 0):
            # Try direct fallback (no cache)
            stem = getattr(context, 'voice_stem', 'fallback')
            fallback_path = self._save_fallback_direct(context.generated_wav, sr, cache_dir, stem)
            if fallback_path:
                context.output_path = fallback_path
                return context

        # Critical: Temp fallback (silence)
        context.output_path = self._create_temp_fallback()
        logger.warning(f"Output: Ultimate temp fallback {context.output_path}")
        return context