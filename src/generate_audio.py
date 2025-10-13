import gc
import threading
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from pathlib import Path
import torch
from time import perf_counter_ns
import torchaudio
from typing import Optional, Dict, Any, Tuple, Callable

from .config import get_config, get_config_value
from .monitor import monitor_resources
from .audio_utils import apply_post_processing
from .cache import (
    try_audio_cache, get_cache_key, get_or_queue_voice_process,
    validate_voice_path, create_dummy_conds, load_conditionals_cache, save_conditionals_cache,
    get_cache_stats, check_and_update_ref, save_torchaudio_wav
)
from .fuzzy_cache import try_fuzzy_audio_cache, FUZZY_QUEUE
from loguru import logger  # FIXED: Use loguru consistently (remove logging import/getLogger)
from .normalize_stem import  normalize_stem


def get_voice_stem(audio_prompt_path: Optional[str]) -> str:
    """Helper: Derive normalized voice stem once (DRY for stem extraction)."""
    return normalize_stem(audio_prompt_path) if audio_prompt_path else 'default'

def set_seed(seed: int):
    """
    Set  seeds for reproducible generation.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# Verify save_torchaudio_wav signature to prevent future errors
def _validate_save_function():
    """Ensure save_torchaudio_wav has expected signature"""
    import inspect
    sig = inspect.signature(save_torchaudio_wav)
    params = list(sig.parameters.keys())

    required_params = ['wav', 'sr', 'uuid', 'cache']
    if not all(param in params for param in required_params):
        logger.critical("save_torchaudio_wav has incorrect signature! Required params: " 
                        f"{required_params}, Found: {params}")
        # Patch the function if needed
        original_save = save_torchaudio_wav

        def patched_save(wav, sr, uuid, cache=False, audio_path=None, cache_key=None, text=None):
            return original_save(wav, sr, uuid=uuid, cache=cache,
                                audio_path=audio_path, cache_key=cache_key, text=text)

        globals()['save_torchaudio_wav'] = patched_save
        logger.info("Applied patch to save_torchaudio_wav function")

# Call during module initialization
# try:
#    _validate_save_function()
# except Exception as e:
#    logger.error(f"Save function validation failed: {str(e)}")


# ADD THIS AT MODULE LEVEL (after imports)
SYSTEM_SILENCE_PATH = None
SILIENCE_CACHE = {}

def initialize_silence_assets():
    """Creates persistent silence assets at startup - call this during app init"""
    global SYSTEM_SILENCE_PATH

    try:
        from .config import get_config
        config = get_config()
        sr = config.app_config.globals.sr

        silence_dir = Path(config.app_config.cache.root) / "fallbacks"
        silence_dir.mkdir(parents=True, exist_ok=True)

        # Create multiple durations for different fallback scenarios
        for duration in [0.5, 1.0, 2.0]:
            path = silence_dir / f"silence_{int(duration*1000)}ms.wav"
            if not path.exists():
                wav = AudioFallbackManager.create_silence(sr, duration_s=duration, device='cpu')
                torchaudio.save(str(path), wav, sample_rate=sr)

            # Cache tensor for immediate access
            SILIENCE_CACHE[duration] = wav
            if duration == 2.0:  # Default becomes our system silence
                SYSTEM_SILENCE_PATH = str(path)

        logger.info(f"Initialized fallback silence assets in {silence_dir}")
        return True
    except Exception as e:
        logger.critical(f"FAILED to initialize silence assets: {str(e)}")
        # Create minimal fallback in current dir as absolute last resort
        fallback_path = Path("./system_silence.wav")
        if not fallback_path.exists():
            torchaudio.save(
                str(fallback_path),
                torch.zeros(1, 24000*2, dtype=torch.float32),
                sample_rate=24000
            )
        SYSTEM_SILENCE_PATH = str(fallback_path)
        return False

# Call this during application startup
# initialize_silence_assets()

# ADD THESE MISSING HELPERS (replace existing stubs)

def _sanitize_text_input(text: str) -> str:
    """Comprehensive text sanitization with safety fallbacks"""
    if text is None:
        logger.warning("Received None text input - using empty string")
        return ""

    if not isinstance(text, str):
        logger.warning(f"Non-string text input (type={type(text)}) - converting safely")
        try:
            return str(text)
        except:
            return "text conversion failed"

    # Clean control characters
    clean_text = ''.join(c for c in text if c.isprintable() or c in ['\n', '\r', '\t'])

    if not clean_text.strip():
        logger.debug("Sanitized text is empty - adding minimal content")
        return "[empty]"

    return clean_text


def _create_empty_text_fallback(cache_uuid: int) -> str:
    """Enhanced fallback with multiple safety layers"""
    sr = get_config().app_config.globals.sr

    try:
        # Try using pre-cached short silence
        if 0.5 in SILIENCE_CACHE:
            return str(save_torchaudio_wav(
                SILIENCE_CACHE[0.5].clone(),  # Clone to prevent tensor ownership issues
                sr,
                uuid=cache_uuid,
                filename_prefix="empty_text_",
                cache=False
            ))
    except:
        pass  # Fall through to alternatives

    # Final fallback
    return str(save_torchaudio_wav(
        AudioFallbackManager.create_silence(sr, duration_s=0.5),
        sr,
        uuid=cache_uuid,
        filename_prefix="empty_text_",
        cache=False
    ))


def _create_model_failure_fallback(cache_uuid: int) -> str:
    try:
        config = get_config()
        sr = config.app_config.globals.sr
    except:
        sr = 24000

    try:
        if 1.0 in SILIENCE_CACHE:
            path = save_torchaudio_wav(
                SILIENCE_CACHE[1.0].clone(),
                sr,
                uuid=cache_uuid,
                cache=False
            )
            logger.error("Returning 1-second silence due to model failure")
            return str(path)
    except Exception as e:
        logger.debug(f"Cache-based fallback failed: {str(e)}")

    # Absolute last resort
    logger.critical("ALL SILENCE FALLBACKS EXHAUSTED - CREATING TEMPORARY FILE")
    fallback_path = Path(f"./model_fail_{cache_uuid}.wav")
    try:
        torchaudio.save(
            str(fallback_path),
            torch.zeros(1, sr, dtype=torch.float32),
            sample_rate=sr
        )
    except:
        # Try alternative location
        fallback_path = Path(f"/tmp/model_fail_{cache_uuid}.wav")
        torchaudio.save(
            str(fallback_path),
            torch.zeros(1, sr, dtype=torch.float32),
            sample_rate=sr
        )
    return str(fallback_path)


def _create_context_failure_fallback(cache_uuid: int, original_text: str) -> str:
    """Context creation failure fallback with safe config access"""
    try:
        config = get_config()
        sr = config.app_config.globals.sr
    except:
        sr = 24000  # Fallback sample rate

    # Use safe cache path resolution
    try:
        cache_root = Path(config.app_config.globals.cache_dir)
    except:
        try:
            cache_root = Path(config.app_config.globals.cache)
        except:
            cache_root = Path("./fallback_cache")

    try:
        # Create diagnostics directory if possible
        diagnostics_dir = cache_root / "failure_diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

        # Save diagnostics
        with open(diagnostics_dir / f"context_fail_{cache_uuid}.log", "w") as f:
            f.write(f"Failed text input: '{original_text[:200]}...'\n")
            f.write(f"Timestamp: {perf_counter_ns()}\n")
            import traceback
            f.write("Traceback:\n" + traceback.format_exc())
    except Exception as e:
        logger.debug(f"Diagnostic logging failed: {str(e)}")

    # Return appropriate silence fallback
    try:
        if 1.0 in SILIENCE_CACHE:
            return str(save_torchaudio_wav(
                SILIENCE_CACHE[1.0].clone(),
                sr,
                uuid=cache_uuid,
                cache=False
            ))
    except:
        pass

    # Final fallback with safe parameters
    return str(save_torchaudio_wav(
        AudioFallbackManager.create_silence(sr, duration_s=1.0),
        sr,
        uuid=cache_uuid,
        cache=False
    ))


def _create_final_safety_fallback(cache_uuid: int, text: str) -> str:
    """Absolute last resort fallback when all else fails"""
    sr = 24000  # Hardcoded default if config unavailable

    try:
        # Try multiple fallback mechanisms
        if SYSTEM_SILENCE_PATH and Path(SYSTEM_SILENCE_PATH).exists():
            logger.critical("USING SYSTEM SILENCE AS FINAL FALLBACK")
            return SYSTEM_SILENCE_PATH

        if 2.0 in SILIENCE_CACHE:
            return str(save_torchaudio_wav(
                SILIENCE_CACHE[2.0].clone(),
                sr,
                uuid=cache_uuid,
                filename_prefix="final_safety_",
                cache=False
            ))
    except:
        pass

    # Emergency direct creation
    logger.critical("CREATING EMERGENCY SILENCE FILE IN CURRENT DIRECTORY")
    emergency_path = Path(f"./emergency_fail_{cache_uuid}.wav")
    try:
        torchaudio.save(
            str(emergency_path),
            torch.zeros(1, sr * 2, dtype=torch.float32),
            sample_rate=sr
        )
        return str(emergency_path)
    except:
        # Only possible if disk is completely full/unwritable
        logger.critical("FATAL: CANNOT CREATE ANY FILES - RETURNING DUMMY STRING")
        return f"/dev/null/fallback_{cache_uuid}"

class AudioFallbackManager:
    @staticmethod
    def create_silence(sr: int, duration_s: float = 2.0, device: str = 'cpu') -> torch.Tensor:
        """Standard silence tensor for any generation failure point."""
        return torch.zeros(1, int(sr * duration_s), dtype=torch.float32, device=device)

    @staticmethod
    def silence_if_empty(wav: Optional[torch.Tensor], sr: int) -> torch.Tensor:
        """Guards against empty tensors while preserving device/dtype."""
        if wav is None or wav.numel() == 0:
            logger.warning("Empty audio tensor detected – injecting silence fallback")
            return AudioFallbackManager.create_silence(sr)
        return wav


@contextmanager
def measure_time(description: str, log_level: str = "DEBUG") -> Callable[[], float]:
    """Standardized timing with consistent logging."""
    start = perf_counter_ns()
    yield lambda: (perf_counter_ns() - start) / 1_000_000_000
    elapsed = (perf_counter_ns() - start) / 1_000_000_000
    getattr(logger, log_level.lower())(f"[Timing] {description}: {elapsed:.3f}s")


@dataclass
class AudioGenerationContext:
    """Unified state for entire generation workflow."""
    model: Any
    config: Any  # Respective config object
    text: str
    audio_prompt_path: Optional[str]
    cache_uuid: int
    exaggeration: float
    temperature: float
    cfgw: float
    min_p: float
    top_p: float
    repetition_penalty: float
    language_id: str
    seed_num: int
    enable_memory_cache: bool
    enable_disk_cache: bool
    device: torch.device
    dtype: torch.dtype
    sr: int
    multilingual: bool
    voice_stem: str
    voice_params: Dict[str, Any]
    t3_params: Dict[str, Any]
    original_text: str
    total_start_time: int

    @classmethod
    @classmethod
    def from_args(cls, **kwargs):
        from .config import get_config
        config = get_config()
        sr = config.app_config.globals.sr

        # Create a clean copy of kwargs to avoid mutation issues
        clean_kwargs = kwargs.copy()

        # Handle text safely FIRST, before any other processing
        text = clean_kwargs.pop('text', '')
        if text is None:
            logger.warning("Received None text input – using empty string")
            text = ''
        if not isinstance(text, str):
            logger.warning(f"Text input is not string (type={type(text)}), converting to string")
            text = str(text)

        # Coerce numerical parameters from the remaining kwargs
        coerce_params = {
            'exaggeration': 0.5,
            'temperature': 0.8,
            'cfgw': 0.0,
            'min_p': 0.05,
            'top_p': 1.0,
            'repetition_penalty': 1.2,
            'seed_num': 42
        }

        for param, default in coerce_params.items():
            if param in clean_kwargs:
                try:
                    clean_kwargs[param] = float(clean_kwargs[param]) if param != 'seed_num' else int(
                        clean_kwargs[param])
                except (TypeError, ValueError):
                    logger.warning(f"Invalid {param} value, using default {default}")
                    clean_kwargs[param] = default

        # Extract audio prompt path safely
        audio_prompt_path = clean_kwargs.get('audio_prompt_path')
        voice_stem = get_voice_stem(audio_prompt_path)
        voice_params = config.get_merged_audio_params(voice_name=voice_stem)

        return cls(
            config=config,
            device=torch.device(config.app_config.globals.device),
            dtype=config.app_config.globals.dtype,
            sr=sr,
            multilingual=config.app_config.globals.multilingual,
            voice_stem=voice_stem,
            voice_params=voice_params,
            t3_params={
                "generate_token_backend": "cudagraphs-manual",
                "stride_length": 4,
                "skip_when_1": True
            },
            total_start_time=perf_counter_ns(),
            text=text,
            original_text=text,
            **clean_kwargs  # Now safe from duplicate text
        )

    @property
    def cache(self) -> bool:
        return self.enable_disk_cache or self.enable_memory_cache

    def get_timing_log(self, phase: str, duration: float, notes: str = "") -> str:
        return f"{phase} completed in {duration:.3f}s{f' ({notes})' if notes else ''}"

class AudioCacheLookup:
    """Encapsulates all cache retrieval strategies."""

    def execute(self, ctx: AudioGenerationContext) -> Optional[Tuple[str, str, float]]:
        """Performs sequential cache lookups with timing."""
        if not ctx.audio_prompt_path:
            return None

        try_fuzzy = get_config_value('app_config.globals.fuzzy.enable_fuzzy.cache', True)
        start = perf_counter_ns()

        # Exact match (UUID-sensitive)
        exact_path = try_audio_cache(
            ctx.audio_prompt_path,
            ctx.text,
            ctx.exaggeration,
            cache_uuid=ctx.cache_uuid
        )
        if exact_path:
            elapsed = (perf_counter_ns() - start) / 1_000_000_000
            logger.debug(f"Exact cache hit: {exact_path} (UUID={ctx.cache_uuid}) in {elapsed:.3f}s")
            return exact_path, "Full audio", elapsed

        # Fuzzy match (voice-stem based)
        if not try_fuzzy:
            return None

        voice_stem = get_voice_stem(ctx.audio_prompt_path)
        fuzzy_path = try_fuzzy_audio_cache(
            ctx.audio_prompt_path,
            ctx.text,
            stem=voice_stem
        )
        if fuzzy_path:
            elapsed = (perf_counter_ns() - start) / 1_000_000_000
            logger.debug(f"Fuzzy cache hit: {fuzzy_path} (stem={voice_stem}) in {elapsed:.3f}s")
            return fuzzy_path, "Fuzzy audio", elapsed

        return None


def prepare_voice_conditions(ctx: AudioGenerationContext) -> Optional[str]:
    """Handles voice processing and conditional preparation with unified error handling."""
    if ctx.model is None:
        logger.error("Model unavailable – creating dummy conditionals")
        create_dummy_conds(None, ctx.device, ctx.dtype, "model_none")
        return None

    # Process audio prompt if provided
    if ctx.audio_prompt_path:
        voice_stem = ctx.voice_stem
        logger.debug(f"Starting voice processing (stem={voice_stem}, uuid={ctx.cache_uuid})")

        with measure_time("Voice processing"):
            processed_path = get_or_queue_voice_process(
                ctx.audio_prompt_path,
                ctx.model,
                ctx.device,
                ctx.dtype,
                stem=voice_stem,
                exaggeration=ctx.exaggeration,
                quiet=not ctx.cache
            )

        if not processed_path or not Path(processed_path).exists():
            logger.warning(f"Voice processing failed for {ctx.audio_prompt_path} – using dummy conds")
            create_dummy_conds(ctx.model, ctx.device, ctx.dtype, "process_fail")
            return None

        # Validation (with automatic correction)
        with measure_time("Path validation"):
            valid, _ = validate_voice_path(processed_path, stem=voice_stem)
            if not valid:
                logger.debug(f"Fixing invalid path: {processed_path}")
                processed_path = check_and_update_ref(processed_path, ctx.exaggeration, stem=voice_stem)

    # Conditional loading/caching
    with measure_time("Conditionals handling"):
        return _handle_conditionals(ctx, processed_path if ctx.audio_prompt_path else None)


def _handle_conditionals(ctx: AudioGenerationContext, audio_path: Optional[str]) -> Optional[str]:
    """Manages conditional cache retrieval and preparation."""
    if not audio_path:
        create_dummy_conds(ctx.model, ctx.device, ctx.dtype, "no_audio")
        logger.info("No audio prompt – using dummy conditionals")
        return None

    cache_params = {'language_id': ctx.language_id, 'cache_uuid': ctx.cache_uuid}
    cache_key = get_cache_key(audio_path, ctx.cache_uuid, ctx.exaggeration, params=cache_params)

    if cache_key and ctx.cache:
        if load_conditionals_cache(cache_key, ctx.model, ctx.device, ctx.dtype,
                                 ctx.enable_memory_cache, ctx.enable_disk_cache):
            logger.info(f"Conditionals cache HIT: {cache_key[:8]}... "
                       f"(stem={ctx.voice_stem}, uuid={ctx.cache_uuid})")
            return audio_path

    # Prepare and cache conditionals
    try:
        ctx.model.prepare_conditionals(audio_path, exaggeration=ctx.exaggeration)
        if ctx.dtype != torch.float32:
            ctx.model.conds.t3.to(device=ctx.device, dtype=ctx.dtype)

        if cache_key and ctx.cache:
            save_conditionals_cache(
                cache_key,
                ctx.model.conds,
                model=ctx.model,
                device=ctx.device,
                dtype=ctx.dtype,
                enable_memory_cache=ctx.enable_memory_cache,
                enable_disk_cache=ctx.enable_disk_cache
            )
            logger.info(f"Prepared and cached conditionals: {cache_key[:8]}... "
                       f"(stem={ctx.voice_stem}, uuid={ctx.cache_uuid})")
        return audio_path
    except Exception as e:
        logger.error(f"Conditionals preparation failed: {str(e)}")
        create_dummy_conds(ctx.model, ctx.device, ctx.dtype, "cond_prep_fail")
        return None


@contextmanager
def gpu_operations():
    """Guaranteed cleanup after GPU operations with error handling."""
    try:
        yield
    except RuntimeError as e:
        logger.warning(f"GPU operation failed: {str(e)}")
        torch.cuda.empty_cache()
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


async def execute_generation(ctx: AudioGenerationContext) -> torch.Tensor:
    """Core generation with proper async flow."""
    if ctx.model is None:
        logger.error("Model unavailable during generation phase")
        return AudioFallbackManager.create_silence(ctx.sr)

    generate_args = {
        'text': ctx.text,
        'exaggeration': ctx.exaggeration,
        'temperature': ctx.temperature,
        'cfg_weight': ctx.cfgw,
        'min_p': ctx.min_p,
        'top_p': ctx.top_p,
        'repetition_penalty': ctx.repetition_penalty,
        'language_id': ctx.language_id,
        't3_params': ctx.t3_params
    }

    with measure_time("Core generation"), gpu_operations():
        set_seed(ctx.seed_num)

        try:
            with torch.no_grad():
                return _safe_generate(ctx.model, generate_args, ctx.t3_params)
        except Exception as e:
            logger.exception("Generation failed – attempting recovery")
            return _handle_generation_error(ctx, e, generate_args)


def _safe_generate(model: Any, generate_args: Dict, t3_params: Dict) -> torch.Tensor:
    """Wrapped generation call with basic safety."""
    try:
        return model.generate(**generate_args)
    except RuntimeError as e:
        if any(kw in str(e).lower() for kw in ["graph", "capture", "offset"]):
            raise
        logger.warning(f"Generation error (non-graph): {str(e)}")
        raise


def _handle_generation_error(ctx: AudioGenerationContext, exc: Exception,
                            generate_args: Dict) -> torch.Tensor:
    """Recovery flow for different error types."""
    try:
        # Graph-related errors: reset and retry
        if any(kw in str(exc).lower() for kw in ["graph", "capture", "offset"]):
            logger.warning("Graph error detected – resetting T3 graphs and retrying")
            if hasattr(ctx.model, 't3') and hasattr(ctx.model.t3, '_bucket_graphs'):
                ctx.model.t3._bucket_graphs.clear()

            # Retry with eager backend
            retry_args = {
                **generate_args,
                't3_params': {**generate_args['t3_params'], 'generate_token_backend': 'eager'}
            }
            with torch.no_grad():
                return ctx.model.generate(**retry_args)

        # Memory-related errors: cleanup and retry
        if "memory" in str(exc).lower() or "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            with torch.no_grad():
                return ctx.model.generate(**generate_args)

        # All other errors: fallback to silence
        logger.error(f"Unhandled generation error: {str(exc)}")
        return AudioFallbackManager.create_silence(ctx.sr)

    except Exception as recovery_exc:
        logger.critical(f"Generation recovery failed: {str(recovery_exc)}")
        return AudioFallbackManager.create_silence(ctx.sr)


async def process_audio_result(ctx: AudioGenerationContext, wav: torch.Tensor) -> torch.Tensor:
    """Fixed: Properly handles async post-processing with safety guards."""
    wav = AudioFallbackManager.silence_if_empty(wav, ctx.sr)

    with measure_time("Tensor standardization"):
        # Ensure 2D mono format for processing
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        elif wav.dim() == 2 and wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

    with measure_time("Post-processing"):
        try:
            processed = await _execute_post_processing(wav, ctx)
        except Exception as e:
            logger.error(f"Post-processing failed: {str(e)} – raw passthru")
            processed = wav.cpu().numpy()

    return _normalize_post_result(processed, ctx.sr)


async def _execute_post_processing(wav: torch.Tensor, ctx: AudioGenerationContext) -> np.ndarray:
    """Properly awaits the async post-processing function."""
    post_result = await apply_post_processing(
        wav,
        ctx.sr,
        ctx.voice_params
    )

    if isinstance(post_result, torch.Tensor):
        return post_result.detach().cpu().numpy().squeeze()
    if isinstance(post_result, np.ndarray):
        return post_result.squeeze()

    raise ValueError(f"Unexpected post-processing result type: {type(post_result)}")


def _normalize_post_result(result: Any, sr: int) -> torch.Tensor:
    """Ensures consistent post-processed tensor format."""
    if isinstance(result, np.ndarray):
        if result.ndim > 1:
            result = result.mean(axis=1) if result.shape[1] > 1 else result.squeeze()
        if len(result) == 0:
            logger.warning("Empty post-processed audio – using silence")
            result = np.zeros(sr * 2)
        return torch.from_numpy(result).unsqueeze(0)

    raise TypeError(f"Invalid post-processing result format: {type(result)}")


def save_generation_result(ctx: AudioGenerationContext, processed_wav: torch.Tensor) -> str:
    """Handles saving with consistent metadata handling."""
    wav_file = _save_and_cache(ctx, processed_wav)

    # Update fuzzy cache tracking
    from .normalize_stem import normalize_stem
    if ctx.audio_prompt_path:
        normalized_stem = normalize_stem(ctx.voice_stem)
        FUZZY_QUEUE.put((ctx.text, wav_file, normalized_stem))

    _log_generation_metrics(ctx, processed_wav)
    _log_cache_statistics()

    return wav_file

def save_and_cache_output(
        wav: torch.Tensor,
        audio_prompt_path: Optional[str],  # Now explicitly named 'audio_prompt_path' for unpadded
        cache_uuid: int,
        text: str,
        exaggeration: float,
        params: Dict[str, Any],
        enable_memory_cache: bool = True,
        enable_disk_cache: bool = True,
        sr: int = 24000
) -> str:
    """
    Helper: Save WAV, cache exact audio if applicable.
    FIXED: Compute cache_key upfront (uses text/exag); pass to save_torchaudio_wav for I/O + cache set.
    """
    cache = enable_disk_cache or enable_memory_cache
    # FIXED: Always compute full key with real params (text/exag/uuid/audio)
    if audio_prompt_path:
        full_cache_key = get_cache_key(audio_path=audio_prompt_path, uuid=cache_uuid, exaggeration=exaggeration,
                                       text=text)
        # Enhanced log: Use normalized stem from voice_path param (now unpadded)
        norm_stem = get_voice_stem(audio_prompt_path)
        logger.debug(
            f"Generated audio cache_key: {full_cache_key} (stem={norm_stem}, text='{text[:20]}...', uuid_hex={hex(cache_uuid)[:10]}...)")

        # FIXED: Pass pre-computed key and text to save (no text/exag in I/O; text for filename uniqueness)
        wave_file = str(save_torchaudio_wav(wav.cpu(), sr, audio_path=audio_prompt_path, uuid=cache_uuid,
                                            cache_key=full_cache_key, text=text, cache=cache))  # Pass text!

        logger.debug(f"Audio saved to cache dir: {Path(wave_file).parent}, cache={cache}")
    else:
        # Fallback for no prompt (e.g., dummy): Use fallback key
        fallback_key = get_cache_key(audio_path="default", uuid=cache_uuid, exaggeration=exaggeration, text=text)
        wave_file = str(save_torchaudio_wav(wav.cpu(), sr, audio_path=None, uuid=cache_uuid,
                                            cache_key=fallback_key, text=text, cache=cache))
        logger.debug(f"Audio fallback saved: {wave_file}, skip audio cache (no prompt)")

    return wave_file


def _save_and_cache(ctx: AudioGenerationContext, processed_wav: torch.Tensor) -> str:
    """Saves audio to appropriate location with consistent naming."""
    with measure_time("Save operation"):
        if ctx.audio_prompt_path:
            # Use original path (unpadded) for cache key consistency
            return save_and_cache_output(
                wav=processed_wav.to(torch.float32).cpu(),
                audio_prompt_path=ctx.audio_prompt_path,
                cache_uuid=ctx.cache_uuid,
                text=ctx.text,
                exaggeration=ctx.exaggeration,
                params=ctx.voice_params,
                enable_memory_cache=ctx.enable_memory_cache,
                enable_disk_cache=ctx.enable_disk_cache,
                sr=ctx.sr
            )

        # Fallback for edge cases
        fallback_key = get_cache_key(
            audio_path="default",
            uuid=ctx.cache_uuid,
            exaggeration=ctx.exaggeration,
            text=ctx.text
        )
        return save_torchaudio_wav(
            AudioFallbackManager.create_silence(ctx.sr),
            ctx.sr,
            uuid=ctx.cache_uuid,
            cache_key=fallback_key,
            text=ctx.text,
            cache=ctx.cache
        )


def _log_generation_metrics(ctx: AudioGenerationContext, wav: torch.Tensor) -> None:
    """Consolidated metrics logging with error guards."""
    try:
        audio_length = wav.shape[-1] / ctx.sr if wav.numel() > 0 else 0.0
        total_time = (perf_counter_ns() - ctx.total_start_time) / 1_000_000_000

        if audio_length > 0:
            rtf = total_time / audio_length
            logger.info(f"Generated {audio_length:.2f}s audio (RTF: {rtf:.2f}x, {1/rtf:.2f}x real-time)")

        logger.info(f"Processing completed in {total_time:.2f}s")
    except Exception as e:
        logger.warning(f"Failed to log generation metrics: {str(e)}")


def _log_cache_statistics() -> None:
    stats = get_cache_stats()
    from .fuzzy_cache import FUZZY_AUDIO_DICT

    # ACTUAL FUZZY COUNTS
    total_entries = 0
    stem_breakdown = {}
    if FUZZY_AUDIO_DICT:
        for stem, entries in FUZZY_AUDIO_DICT.items():
            count = len(entries)
            total_entries += count
            stem_breakdown[stem] = count

    # Format stem breakdown (limit to first 5 for log readability)
    breakdown_str = ", ".join([f"{s}={c}" for s, c in list(stem_breakdown.items())[:5]])
    if len(stem_breakdown) > 5:
        breakdown_str += f", ... (+{len(stem_breakdown)-5} more)"

    logger.info(
        "Cache stats | "
        f"mem: {stats['memory_cache_size']}, "
        f"disk: {stats['disk_files']}, "
        f"audio: {stats['audio_cache_size']}, "
        f"fuzzy: {total_entries} (stems: {breakdown_str or 'none'})"
    )


async def _check_cache(ctx: AudioGenerationContext) -> Optional[str]:
    """Async cache lookup with proper error handling."""
    if cache_hit := AudioCacheLookup().execute(ctx):
        audio_path, hit_type, lookup_time = cache_hit
        audio_duration = torchaudio.info(audio_path).num_frames / ctx.sr
        logger.info(f"{hit_type} cache HIT: \"{ctx.text[:50]}...\" – skipping generation")
        logger.info(f"Reused {audio_duration:.2f}s audio (lookup: {lookup_time:.3f}s)")
        return audio_path
    return None


async def _execute_full_generation_pipeline(ctx: AudioGenerationContext) -> str:
    """Complete generation workflow with robust error handling."""
    try:
        # Correctly propagate async through all phases
        with measure_time("Preparation phase"):
            valid_path = prepare_voice_conditions(ctx)

        wav = await execute_generation(ctx)
        processed_wav = await process_audio_result(ctx, wav)
        return save_generation_result(ctx, processed_wav)

    except Exception as e:
        logger.critical(
            f"Pipeline execution failed: {str(e)}\n"
            f"Context: voice='{ctx.voice_stem}', "
            f"text='{ctx.text[:30]}...'",
            exc_info=True
        )
        return await _handle_generation_failure(ctx, e)


async def _handle_generation_failure(ctx: AudioGenerationContext, exc: Exception) -> str:
    """Unified failure recovery with progressive fallbacks."""
    # First attempt: clean-up and try post-processing on silence
    try:
        logger.debug("Attempting fallback with basic silence")
        torch.cuda.empty_cache()
        fallback_wav = AudioFallbackManager.create_silence(ctx.sr)
        processed = await process_audio_result(ctx, fallback_wav)
        return save_generation_result(ctx, processed)

    except Exception as post_exc:
        logger.warning(f"Post-processing fallback failed: {str(post_exc)}")

        # Second attempt: raw silence without post-processing
        try:
            logger.debug("Attempting raw silence fallback")
            raw_silence = AudioFallbackManager.create_silence(ctx.sr)
            return str(save_torchaudio_wav(
                raw_silence,
                ctx.sr,
                uuid=ctx.cache_uuid,
                cache=ctx.cache
            ))

        except Exception as save_exc:
            logger.error(f"Save operation failed in fallback: {str(save_exc)}")

            # Final attempt: create file in safe location
            try:
                fallback_path = Path(f"./fallback_{ctx.cache_uuid}.wav")
                fallback_wav = AudioFallbackManager.create_silence(ctx.sr, duration_s=1.0)
                torchaudio.save(
                    str(fallback_path),
                    fallback_wav,
                    sample_rate=ctx.sr
                )
                logger.critical(f"Using emergency fallback: {fallback_path}")
                return str(fallback_path)

            except Exception as final_exc:
                logger.exception("CRITICAL: ALL STANDARD FALLBACKS FAILED - ACTIVATING EMERGENCY PROTOCOLS")
                # Try using our pre-initialized system silence
                if SYSTEM_SILENCE_PATH:
                    try:
                    # Verify file still exists
                        if Path(SYSTEM_SILENCE_PATH).exists():
                            logger.critical(f"Using system fallback silence: {SYSTEM_SILENCE_PATH}")
                            return SYSTEM_SILENCE_PATH
                    except:
                        pass

                # Try creating in multiple potential safe locations
                for prefix in ['./', '/tmp/', 'C:/Windows/Temp/', str(Path.home())]:
                    try:
                        fallback_path = Path(f"{prefix}emergency_fallback_{ctx.cache_uuid}.wav")
                        fallback_wav = AudioFallbackManager.create_silence(ctx.sr, duration_s=1.0)
                        torchaudio.save(str(fallback_path), fallback_wav, sample_rate=ctx.sr)
                        logger.critical(f"Created emergency fallback at: {fallback_path}")
                        return str(fallback_path)
                    except:
                        continue
                # Absolute final fallback
                logger.critical("SYSTEM IN CRITICAL STATE - NO FILE WRITES POSSIBLE")
                return _create_final_safety_fallback(ctx.cache_uuid, ctx.text)


@monitor_resources(enable=True, log_level="INFO")
async def generate_audio(
    model,
    text: str,
    audio_prompt_path: Optional[str] = None,
    exaggeration: float = 0.5,
    cache_uuid: int = 0,
    temperature: float = 0.8,
    cfgw: float = 0,
    min_p: float = 0.05,
    top_p: float = 1.0,
    repetition_penalty: float = 1.2,
    language_id: str = "en",
    seed_num: int = 42,
    enable_memory_cache: bool = True,
    enable_disk_cache: bool = True
) -> str:
    """Main entry with comprehensive error shielding."""
    start_time = perf_counter_ns()

    try:
        # Initial validation with guardrails
        if not text:
            logger.warning("No text provided – returning minimal silence")
            return _create_empty_text_fallback(cache_uuid)

        # Model resolution with redundancy
        resolved_model = _resolve_model(model)
        if resolved_model is None:
            logger.error("CRITICAL: No model available after resolution attempts")
            return _create_model_failure_fallback(cache_uuid)

        # Context initialization with safety checks
        try:
            ctx = AudioGenerationContext.from_args(
                model=resolved_model,
                text=_sanitize_text_input(text),
                audio_prompt_path=audio_prompt_path,
                cache_uuid=cache_uuid,
                exaggeration=exaggeration,
                temperature=temperature,
                cfgw=cfgw,
                min_p=min_p,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                language_id=language_id,
                seed_num=seed_num,
                enable_memory_cache=enable_memory_cache,
                enable_disk_cache=enable_disk_cache
            )
        except Exception as ctx_exc:
            logger.error(f"Context creation failed: {str(ctx_exc)}")
            return _create_context_failure_fallback(cache_uuid, text)

        # Execute with final safety net
        return await _execute_pipeline_with_final_catch(ctx)

    except Exception as top_exc:
        logger.critical(
            f"UNHANDLED EXCEPTION IN GENERATE_AUDIO (uuid={cache_uuid}):\n"
            f"{str(top_exc)}",
            exc_info=True
        )
        return _create_final_safety_fallback(cache_uuid, text)
    finally:
        total_time = (perf_counter_ns() - start_time) / 1_000_000_000
        logger.debug(f"generate_audio completed in {total_time:.3f}s")


# Helper fallback implementations
def _create_empty_text_fallback(cache_uuid: int) -> str:
    try:
        config = get_config()
        sr = config.app_config.globals.sr
    except:
        sr = 24000

    try:
        if 0.5 in SILIENCE_CACHE:
            return str(save_torchaudio_wav(
                SILIENCE_CACHE[0.5].clone(),
                sr,
                uuid=cache_uuid,
                cache=False
            ))
    except Exception as e:
        logger.debug(f"Short silence fallback failed: {str(e)}")

    return str(save_torchaudio_wav(
        AudioFallbackManager.create_silence(sr, duration_s=0.5),
        sr,
        uuid=cache_uuid,
        cache=False
    ))

async def _execute_pipeline_with_final_catch(ctx: AudioGenerationContext) -> str:
    """Execution wrapper with last-resort exception handling."""
    try:
        if cache_hit := await _check_cache(ctx):
            return cache_hit
        return await _execute_full_generation_pipeline(ctx)
    except Exception as pipeline_exc:
        logger.warning(f"Pipeline execution failed: {str(pipeline_exc)}")
        return await _handle_generation_failure(ctx, pipeline_exc)


def _resolve_model(model) -> Optional[Any]:
    """Safely resolve model with multiple fallback strategies."""
    if model is not None and hasattr(model, 'generate'):
        return model

    try:
        from src.tts_model import ModelManager
        return ModelManager.get_instance(allow_reinit=True).get_model()
    except Exception as model_exc:
        logger.error(f"Model manager resolution failed: {str(model_exc)}")

    try:
        # Last-ditch effort to load default model
        from src.tts_model import load_default_model
        return load_default_model()
    except Exception:
        return None