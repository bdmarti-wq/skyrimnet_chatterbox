# src/generate/pipeline/phases/output.py (Clean: Separate handle_error for OutputPhase only – has cache_manager/audio_cache; no duplicates)
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
from src.audio_utils import get_silence  # For silence fallback
from .base import GenerationPhase
from ...pipeline.context import AudioGenerationContext


class OutputPhase(GenerationPhase):
    """Saves generated audio to file and caches for reuse."""

    process_after_cache = True  # Always run (save even on HIT for fresh)

    def __init__(self, cache_manager=None):
        """Initialize with cache_manager for exact/fuzzy access. FIXED: Injection from coordinator."""
        self.cache_manager = cache_manager or CacheManager(get_config())  # Fallback if not injected (pass config)
        self.audio_cache = getattr(self.cache_manager, 'audio_cache', None)  # Assume exists
        if self.audio_cache is None:
            logger.warning("OutputPhase: No audio_cache; saves only (no fuzzy/exact)")
        logger.debug("OutputPhase initialized with cache_manager")

    def execute(self, context: AudioGenerationContext) -> AudioGenerationContext:
        """Save generated audio to file and optionally queue for exact/fuzzy cache. FIXED: Use cache_manager.output_dir (subpath safe); str(abs path) for fspath; queue on MISS only."""
        if context.processed_wav is None or (hasattr(context.processed_wav, 'numel') and context.processed_wav.numel() == 0):
            # FIXED: Guard empty/None wav → create silence (use get_silence; safe sr/device from app_config.globals/context)
            config = get_config()
            if hasattr(config, 'app_config') and hasattr(config.app_config, 'globals'):
                sr = config.app_config.globals.sr
                device = config.app_config.globals.device
                dtype = config.app_config.globals.dtype
            else:
                sr = getattr(context, 'sr', 24000)
                device = getattr(context, 'device', torch.device('cpu'))
                dtype = getattr(context, 'dtype', torch.float32)
            context.processed_wav = get_silence(duration=2.0, sr=sr, dtype=dtype, device=device)

        try:
            # FIXED: Use cache_manager.output_dir for safe subpath (resolves to "cache/audio/output"; fallback app_config.globals)
            config = get_config()
            if hasattr(config, 'app_config') and hasattr(config.app_config, 'globals'):
                globals_config = config.app_config.globals
                cache_root = getattr(globals_config, 'cache_dir', Path('./cache'))
            else:
                globals_config = None
                cache_root = Path('./cache')
            if self.cache_manager and hasattr(self.cache_manager, 'output_dir'):
                output_dir = self.cache_manager.output_dir  # Injected; full Path("cache/audio/output")
            else:
                output_dir = cache_root / "audio" / "output"  # FIXED: Ensure subpath "cache/audio/output"
            output_dir.mkdir(parents=True, exist_ok=True)

            # FIXED: Safe filename (int sr; str stem/text hash – no None)
            sr_int = int(context.sr) if context.sr else 24000
            voice_stem = getattr(context, 'voice_stem', 'default')
            text_hash = abs(hash(context.text or '')) % 1000000
            base_name = f"{voice_stem}_{text_hash}_{sr_int}kHz.wav"
            output_path = output_dir / base_name

            # FIXED: Save with torchaudio (CPU float32; ensure non-None wav/sr)
            wav_cpu = context.processed_wav.cpu().float() if hasattr(context.processed_wav, 'cpu') else context.processed_wav
            if wav_cpu is None or wav_cpu.numel() == 0:
                raise ValueError("Invalid WAV for save (empty/None)")
            torchaudio.save(str(output_path), wav_cpu, sr_int)  # str for fspath

            # FIXED: Verify save (exists/size >0); log
            if output_path.exists() and output_path.stat().st_size > 0:
                size_kb = output_path.stat().st_size / 1024
                logger.debug(f"Saved WAV: {output_path}, size={size_kb:.1f}KB")
                context.output_path = str(output_path.absolute())  # FIXED: Abs str for fuzzy/UI (safe PathLike)
            else:
                raise OSError("Save succeeded but no file (zero-size)")

            # FIXED: Queue to exact/fuzzy cache ONLY on MISS (non-cached gen; safe: check path str/exists; skip if no async method)
            if not getattr(context, 'is_cached', True) and self.audio_cache and context.output_path and os.path.exists(context.output_path):
                try:
                    # Optional: If audio_cache has async_cache_postgen
                    if hasattr(self.audio_cache, 'async_cache_postgen'):
                        self.audio_cache.async_cache_postgen(
                            cache_key=getattr(context, 'cache_key', f"postgen_{text_hash}"),
                            audio_path=context.output_path,
                            text=context.text or '',
                            voice_stem=voice_stem
                        )
                        logger.debug(f"Queued postgen to audio_cache for {base_name} (fuzzy/exact)")
                        # Optional: Inc stats if available (align with conditionals)
                        if hasattr(self.audio_cache, 'stats') and 'queued' in self.audio_cache.stats:
                            self.audio_cache.stats['queued'] = getattr(self.audio_cache.stats, 'queued', 0) + 1
                    else:
                        logger.debug("No async_cache_postgen method; saved only (add for fuzzy/exact)")
                except Exception as queue_e:
                    logger.warning(f"Postgen queue failed for {base_name}: {queue_e} – saved only")
            else:
                logger.trace("No queue: Cached gen or no audio_cache")

        except Exception as e:
            logger.error(f"Output save error: {e} – fallback direct")
            # FIXED: Call fallback with guards (int sr; safe dir)
            fallback_path = self._save_fallback_direct(voice_stem=getattr(context, 'voice_stem', 'fallback'), sr=int(context.sr))
            if fallback_path:
                context.output_path = fallback_path
                # Optional queue on fallback if MISS (rare; check exists)
                if not getattr(context, 'is_cached', True) and self.audio_cache and os.path.exists(fallback_path):
                    try:
                        if hasattr(self.audio_cache, 'async_cache_postgen'):
                            self.audio_cache.async_cache_postgen(
                                cache_key=getattr(context, 'cache_key', f"fallback_{int(time.time())}"),
                                audio_path=fallback_path,
                                text=context.text or '',
                                voice_stem=getattr(context, 'voice_stem', 'fallback')
                            )
                            logger.debug("Queued fallback to audio_cache")
                        else:
                            logger.trace("Fallback saved (no queue method)")
                    except Exception as queue_e:
                        logger.warning(f"Fallback queue failed: {queue_e}")
            else:
                context.output_path = self._create_temp_fallback()  # Ultimate temp

        return context

    def _save_fallback_direct(self, voice_stem: str, sr: int) -> str:
        """Save fallback WAV directly (no cache queue). FIXED: Use output_dir subpath; guard None wav/sr → silence/default."""
        config = get_config()
        if hasattr(config, 'app_config') and hasattr(config.app_config, 'globals'):
            globals_config = config.app_config.globals
            cache_root = getattr(globals_config, 'cache_dir', Path('./cache'))
        else:
            globals_config = None
            cache_root = Path('./cache')
        if self.cache_manager and hasattr(self.cache_manager, 'output_dir'):
            fallback_dir = self.cache_manager.output_dir  # "cache/audio/output"
        else:
            fallback_dir = cache_root / "audio" / "output"  # FIXED: Subpath safe
        fallback_dir.mkdir(parents=True, exist_ok=True)

        # FIXED: Guard sr (int/default); create silence if no wav
        sr_final = int(sr) if sr and isinstance(sr, (int, float)) else 24000
        if hasattr(self, 'context') and hasattr(self.context, 'processed_wav') and self.context.processed_wav is not None:
            wav_to_save = self.context.processed_wav.cpu().float()
        else:
            # FIXED: Silence fallback (safe sr/device; from audio_utils)
            from src.audio_utils import get_silence
            wav_to_save = get_silence(duration=2.0, sr=sr_final, dtype=torch.float32, device=torch.device('cpu'))

        if wav_to_save is None or wav_to_save.numel() == 0:
            raise ValueError("Fallback WAV invalid (empty/None)")

        timestamp = int(time.time())
        fallback_path = fallback_dir / f"{voice_stem}_fb_{timestamp}_{sr_final}kHz.wav"
        try:
            torchaudio.save(str(fallback_path), wav_to_save, sr_final)  # FIXED: str for fspath; int sr
            if fallback_path.exists() and fallback_path.stat().st_size > 0:
                logger.debug(f"Fallback direct saved: {fallback_path}")
                return str(fallback_path.absolute())  # FIXED: Abs str for queue/UI
            else:
                logger.warning("Fallback save: No file created")
        except Exception as save_e:
            logger.error(f"Fallback direct error: {save_e}")
        return ""  # Empty on fail (call temp next)

    def _create_temp_fallback(self) -> str:
        """Create temporary fallback audio file (silence). FIXED: Hardcode sr (no config); safe env TMPDIR/TEMP; str(abs)."""
        config = get_config()
        if hasattr(config, 'app_config') and hasattr(config.app_config, 'globals'):
            sr = config.app_config.globals.sr
        else:
            sr = 24000
        sr_final = int(sr)  # FIXED: Ensure int

        duration = 2.0  # Short silence
        from src.audio_utils import get_silence  # Assume import (align with fallback)
        silence = get_silence(duration=duration, sr=sr_final, dtype=torch.float32, device=torch.device('cpu'))
        if silence is None or silence.numel() == 0:
            logger.error("Temp silence creation failed")
            return ""

        # FIXED: Safe temp dir (cross-OS; mkdir)
        temp_dir = Path(os.environ.get('TMPDIR', os.environ.get('TEMP', os.environ.get('TMP', '/tmp')))) / "audio_fallback"
        temp_dir.mkdir(exist_ok=True)
        timestamp = int(time.time())
        temp_path = temp_dir / f"audio_fallback_{timestamp}.wav"
        try:
            torchaudio.save(str(temp_path), silence, sr_final)  # FIXED: str for fspath
            if temp_path.exists():
                logger.debug(f"Created temporary fallback audio: {temp_path}")
                return str(temp_path.absolute())  # FIXED: Abs str for safety
        except Exception as temp_e:
            logger.error(f"Temp fallback failed: {temp_e} – return empty str")
        return ""

    def handle_error(self, context: AudioGenerationContext, error: Exception) -> AudioGenerationContext:
        """Handle output errors with fallback saves. FIXED: Use output_dir subpath; safe paths (str/abs); optional purge on invalid."""
        logger.error(f"Output phase failed: {error} – attempting fallback solutions")
        config = get_config()
        if hasattr(config, 'app_config') and hasattr(config.app_config, 'globals'):
            globals_config = config.app_config.globals
            cache_root = getattr(globals_config, 'cache_dir', Path('./cache'))
            sr = getattr(globals_config, 'sr', 24000)
        else:
            globals_config = None
            cache_root = Path('./cache')
            sr = 24000
        if self.cache_manager and hasattr(self.cache_manager, 'output_dir'):
            output_dir = self.cache_manager.output_dir  # FIXED: Subpath safe
        else:
            output_dir = cache_root / "audio" / "output"  # Default subpath
        sr_final = int(sr)

        if (hasattr(context, 'processed_wav') and context.processed_wav is not None and 
            context.processed_wav.numel() > 0):
            # FIXED: Try direct fallback (safe dir/wav; as in execute)
            stem = getattr(context, 'voice_stem', 'fallback')
            fallback_path = self._save_fallback_direct(stem, sr_final)  # Now uses output_dir
            if fallback_path and os.path.exists(fallback_path):
                context.output_path = fallback_path
                # FIXED: Queue if MISS and valid (rare error case; safe exists/str)
                if not getattr(context, 'is_cached', True) and self.audio_cache:
                    try:
                        if hasattr(self.audio_cache, 'async_cache_postgen'):
                            self.audio_cache.async_cache_postgen(
                                cache_key=getattr(context, 'cache_key', f"error_fallback_{int(time.time())}"),
                                audio_path=context.output_path,
                                text=context.text or '',
                                voice_stem=stem
                            )
                            logger.debug(f"Queued error fallback to audio_cache")
                        else:
                            logger.trace("Error fallback saved (no queue method)")
                    except Exception as queue_e:
                        logger.warning(f"Error fallback queue failed: {queue_e}")
                return context

        # FIXED: Ultimate temp fallback (silence; safe sr)
        context.output_path = self._create_temp_fallback()
        if not context.output_path:
            context.output_path = ""  # Empty str ultimate

        # FIXED: Optional purge (if audio_cache and prior output invalid – e.g., zero-size from silence error)
        if self.audio_cache and hasattr(context, 'output_path') and context.output_path and os.path.exists(context.output_path):
            output_p = Path(context.output_path)
            if output_p.stat().st_size == 0 or "silence" in str(output_p).lower():  # Heuristic invalid
                try:
                    if hasattr(self.audio_cache, 'purge_invalid'):
                        self.audio_cache.purge_invalid(output_p)  # Assume method; or manual del
                    elif hasattr(self.audio_cache, 'delete'):
                        self.audio_cache.delete(getattr(context, 'cache_key', str(output_p.name)))
                    logger.debug(f"Purged invalid fallback from audio_cache: {output_p.name}")
                except Exception as purge_e:
                    logger.warning(f"Purge failed: {purge_e}")

        logger.warning(f"Output: Ultimate fallback {context.output_path or 'empty'}")
        return context