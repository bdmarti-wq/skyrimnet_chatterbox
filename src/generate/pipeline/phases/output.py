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
    """Saves generated audio to file and caches for reuse. FIXED: Direct set_audio_cache after save (sync, log)."""

    process_after_cache = True  # Always run (verify even on HIT)

    def __init__(self, cache_manager=None):
        self.cache_manager = cache_manager or CacheManager(get_config())  # Fallback
        self.audio_cache = getattr(self.cache_manager, 'audio_cache', None)
        if self.audio_cache is None:
            logger.warning("No audio_cache; saves only")
        super().__init__()

    def _execute_core(self, context: AudioGenerationContext) -> AudioGenerationContext:
        """If cached_path set (from audio/fuzzy HIT), verify/use it (no save/post). Else, normal save + exact cache set."""
        # Early HIT path (bypass save/gen)
        if hasattr(context, 'cached_path') and context.cached_path and os.path.exists(context.cached_path):
            cached_p = Path(context.cached_path)
            if cached_p.stat().st_size > 100:  # Min size
                # FIXED: Attr access for base_dir (no .get)
                try:
                    cache_root = context.config.app_config.globals.cache_dir if hasattr(context.config.app_config,
                                                                                        'globals') and hasattr(
                        context.config.app_config.globals, 'cache_dir') else Path('./cache')
                except AttributeError:
                    cache_root = Path('./cache')
                base_dir = (cache_root / "audio" / "output").resolve()
                if cached_p.is_relative_to(base_dir):
                    context.output_path = str(cached_p.absolute())
                    logger.info(f"Used cached path on HIT: {context.output_path} (no save)")
                    return context
                else:
                    logger.warning(f"Cached path not in output dir {base_dir}: {context.cached_path} – regenerate")

        # No HIT or invalid → normal save
        if context.processed_wav is None or context.processed_wav.numel() == 0:
            silence = self.create_silence(context.sr, 2.0,
                                          context.device if hasattr(context, 'device') else torch.device('cpu'),
                                          torch.float32)
            context.processed_wav = silence
            logger.warning("Created silence for empty processed_wav")

        globals_dict = context.get_globals()
        cache_root = globals_dict.get('cache_dir', Path('./cache'))
        output_dir = cache_root / "audio" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        context.ensure_attrs()
        sr_int = context.sr
        voice_stem = context.voice_stem
        text_hash = abs(hash(context.text or '')) % 1000000
        timestamp = int(time.time() * 1000) % 10000
        base_name = f"{voice_stem}_{text_hash}_{timestamp}_{sr_int}kHz.wav"
        output_path = output_dir / base_name

        wav_cpu = context.processed_wav.cpu().float()
        torchaudio.save(str(output_path), wav_cpu, sr_int)

        if output_path.exists() and output_path.stat().st_size > 100:
            context.output_path = str(output_path.absolute())
            size_kb = output_path.stat().st_size / 1024
            logger.info(f"Saved WAV: {context.output_path}, size={size_kb:.1f}KB")

            # FIXED: Direct exact cache set (sync, on MISS) – pass voice_stem
            is_miss = not getattr(context, 'is_cached', True)
            if is_miss and self.cache_manager:
                cache_key = getattr(context, 'cache_key', f"postgen_{text_hash}")
                self.cache_manager.set_audio_cache(cache_key, str(output_path.absolute()))

            # Queue fuzzy index (async, on MISS)
            if is_miss and self.cache_manager:
                try:
                    self.cache_manager.index_audio_for_fuzzy(context.text, context.output_path, voice_stem)
                    logger.debug(f"Queued fuzzy index for {base_name}")
                except Exception as queue_e:
                    logger.warning(f"Fuzzy index failed: {queue_e}")

            # Legacy async postgen queue (if needed, but direct above preferred)
            if (is_miss and self.audio_cache and hasattr(self.audio_cache, 'async_cache_postgen')):
                try:
                    self.audio_cache.async_cache_postgen(
                        cache_key=cache_key,
                        audio_path=context.output_path,
                        text=context.text or '',
                        voice_stem=voice_stem
                    )
                    logger.debug(f"Queued legacy postgen for {base_name} (direct set already done)")
                except Exception as queue_e:
                    logger.warning(f"Legacy queue failed: {queue_e}")
        else:
            raise OSError("Save failed (zero-size)")

        return context

    def _save_fallback_direct(self, voice_stem: str, sr: int) -> str:
        """Save fallback WAV directly (no cache queue)."""
        globals_dict = get_config().app_config.globals if hasattr(get_config(), 'app_config') else {}
        cache_root = getattr(globals_dict, 'cache_dir', Path('./cache')) if globals_dict else Path('./cache')
        fallback_dir = cache_root / "audio" / "output"
        fallback_dir.mkdir(parents=True, exist_ok=True)

        sr_final = int(sr) if sr else 24000
        if hasattr(self, 'context') and hasattr(self.context, 'processed_wav') and self.context.processed_wav is not None:
            wav_to_save = self.context.processed_wav.cpu().float()
        else:
            from src.audio_utils import get_silence
            wav_to_save = get_silence(duration=2.0, sr=sr_final, dtype=torch.float32, device=torch.device('cpu'))

        if wav_to_save is None or wav_to_save.numel() == 0:
            raise ValueError("Fallback WAV invalid (empty/None)")

        timestamp = int(time.time())
        fallback_path = fallback_dir / f"{voice_stem}_fb_{timestamp}_{sr_final}kHz.wav"
        try:
            torchaudio.save(str(fallback_path), wav_to_save, sr_final)
            if fallback_path.exists() and fallback_path.stat().st_size > 0:
                logger.debug(f"Fallback direct saved: {fallback_path}")
                return str(fallback_path.absolute())
            else:
                logger.warning("Fallback save: No file created")
        except Exception as save_e:
            logger.error(f"Fallback direct error: {save_e}")
        return ""

    def _create_temp_fallback(self) -> str:
        """Create temporary fallback audio file (silence)."""
        globals_dict = get_config().app_config.globals if hasattr(get_config(), 'app_config') else {}
        sr = getattr(globals_dict, 'sr', 24000) if globals_dict else 24000
        sr_final = int(sr)

        duration = 2.0
        from src.audio_utils import get_silence
        silence = get_silence(duration=duration, sr=sr_final, dtype=torch.float32, device=torch.device('cpu'))
        if silence is None or silence.numel() == 0:
            logger.error("Temp silence creation failed")
            return ""

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            torchaudio.save(tmp.name, silence, sr_final)
            if Path(tmp.name).exists():
                logger.debug(f"Created temporary fallback audio: {tmp.name}")
                return str(Path(tmp.name).absolute())  # Abs str
        return ""

    def handle_error(self, context: AudioGenerationContext, error: Exception) -> AudioGenerationContext:
        """Handle output errors with fallback saves."""
        logger.error(f"Output phase failed: {error} – attempting fallback solutions")
        globals_dict = context.get_globals()
        cache_root = globals_dict.get('cache_dir', Path('./cache'))
        sr = globals_dict.get('sr', 24000)
        output_dir = cache_root / "audio" / "output"
        sr_final = int(sr)

        if (hasattr(context, 'processed_wav') and context.processed_wav is not None and
            context.processed_wav.numel() > 0):
            stem = getattr(context, 'voice_stem', 'fallback')
            fallback_path = self._save_fallback_direct(stem, sr_final)
            if fallback_path and os.path.exists(fallback_path):
                context.output_path = fallback_path
                if not getattr(context, 'is_cached', True) and self.cache_manager:
                    try:
                        if hasattr(self.cache_manager.audio_cache, 'async_cache_postgen'):
                            self.cache_manager.audio_cache.async_cache_postgen(
                                cache_key=getattr(context, 'cache_key', f"error_fallback_{int(time.time())}"),
                                audio_path=context.output_path,
                                text=context.text or '',
                                voice_stem=stem
                            )
                            logger.debug(f"Queued error fallback to audio_cache")
                        self.cache_manager.index_audio_for_fuzzy(context.text, fallback_path, stem)
                    except Exception as queue_e:
                        logger.warning(f"Error fallback queue/index failed: {queue_e}")
                return context

        context.output_path = self._create_temp_fallback()
        if not context.output_path:
            context.output_path = ""

        # Optional purge (if prior invalid)
        if self.cache_manager and hasattr(context, 'output_path') and context.output_path and os.path.exists(context.output_path):
            output_p = Path(context.output_path)
            if output_p.stat().st_size == 0 or "silence" in str(output_p).lower():
                try:
                    if hasattr(self.cache_manager.audio_cache, 'purge_invalid'):
                        self.cache_manager.audio_cache.purge_invalid(output_p)
                    logger.debug(f"Purged invalid fallback from audio_cache: {output_p.name}")
                except Exception as purge_e:
                    logger.warning(f"Purge failed: {purge_e}")

        logger.warning(f"Output: Ultimate fallback {context.output_path or 'empty'}")
        return context