import threading
from contextlib import contextmanager

import numpy as np
from pathlib import Path
import torch
from time import perf_counter_ns
import torchaudio
from typing import Optional, Dict, Any, Tuple

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

GEN_ACTIVE_LOCK = threading.RLock()  # Global for gen/prepare


def set_seed(seed: int):
    """
    Set  seeds for reproducible generation.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_voice_stem(audio_prompt_path: Optional[str]) -> str:
     """Helper: Derive normalized voice stem once (DRY for stem extraction)."""
     return normalize_stem(audio_prompt_path) if audio_prompt_path else 'default'


def calculate_rtf(time_taken_s: float, audio_duration_s: float) -> float:
     """Helper: Compute RTF safely (DRY; avoid divide-by-zero)."""
     return time_taken_s / audio_duration_s if audio_duration_s > 0 else float('inf')

# Text padding for short/vocalise (pre-TTS; smooths garble via pauses)
# Gated text padding (pre-TTS; uses merged voice params for tunables/gates)
def pad_short_text(text: str, params: Dict[str, Any]) -> str:
    """
    Pad short text based on merged params (enable/gates/tunables from config).
    :param text: Input text.
    :param params: Merged audio params dict (from CONFIG.get_merged_audio_params).
    :return: Padded text or original.
    """
    enable = params.get('enable_text_padding', True)
    if not enable:
        logger.debug("Text padding skipped (enable=False)")
        return text
    ellipses = params.get('text_ellipses_count', 2)
    max_len = params.get('max_short_word_len', 3)
    patterns = params.get('vocalise_patterns', ['ah', 'oh', 'aah', 'mmm'])

    words = text.strip().split()
    if (len(words) == 1 and len(words[0]) <= max_len and words[0].lower() in patterns) or '...' in text:
        pad = '.' * (3 * ellipses)  # 3 dots per ...
        padded = f"{pad} {text.strip()} {pad}".strip()
        logger.debug(f"Text pad applied: '{text}' → '{padded}' (count={ellipses}; patterns={patterns})")
        return padded
    return text


def _generate_audio_core(
    model: Any,
    generate_args: Dict[str, Any],
    t3_params: Dict[str, Any]
) -> torch.Tensor:
    """
    Core: Just model.generate + graph retry. Returns wav; no del/cleanup (handled caller).
    :param model: Loaded TTS model.
    :param generate_args: Dict of args for model.generate.
    :param t3_params: Dict of t3-specific params.
    :return: Generated WAV tensor.
    """
    if model is None:
        logger.error("_generate_audio_core: model is None – creating dummy silence")
        return torch.zeros(1, 24000 * 2, dtype=torch.float32, device='cpu')  # 2s silence fallback (default sr)

    with GEN_ACTIVE_LOCK:  # Serialize vs async prepare (prevent graph race)
        wav = None
        try:
            wav = model.generate(**generate_args)
        except RuntimeError as graph_e:
            if "graph" in str(graph_e).lower() or "capture" in str(graph_e).lower() or "offset" in str(graph_e).lower():
                logger.warning(f"Graph corrupt: {graph_e} – resetting t3 graphs and retrying")
                if hasattr(model, 't3') and hasattr(model.t3, '_bucket_graphs'):
                    model.t3._bucket_graphs.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                t3_params_temp = t3_params.copy()
                t3_params_temp['generate_token_backend'] = 'eager'  # Force eager on retry
                generate_args_temp = generate_args.copy()
                generate_args_temp['t3_params'] = t3_params_temp
                wav = model.generate(**generate_args_temp)
            else:
                raise
        except Exception as e:  # FIXED: Broader catch for compile errors (e.g., cuFFT)
            logger.exception(f"Core gen failed (fallback silence)")
            wav = None  # Trigger silence
        finally:  # FIXED: Ensure cleanup if exception (leaks)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return wav if wav is not None else torch.zeros(1, 24000 * 2, dtype=torch.float32, device='cpu')



def try_reuse_audio(
    text: str,
    audio_prompt_path: Optional[str],
    exaggeration: float,
    cache_uuid: int = None,  # NEW: Add param
    params: Dict[str, Any] = None,
    try_fuzzy: bool = True
) -> Optional[Tuple[str, str]]:
    if not audio_prompt_path:
        return None
    # Exact: Pass cache_uuid (or "audio_reuse" if None)
    exact_path = try_audio_cache(audio_prompt_path, text, exaggeration, cache_uuid=cache_uuid)
    if exact_path:
        return exact_path, "Full audio"
    # Fuzzy: Unchanged (doesn't need UUID)
    fuzzy_path = None
    if try_fuzzy:
        if not audio_prompt_path:
            logger.debug("Fuzzy skipped: No audio_prompt_path")
            return None
        voice_stem = get_voice_stem(audio_prompt_path)
        logger.debug(f"Query stem derived: '{voice_stem}' from path '{audio_prompt_path}' (using cache module)")
        fuzzy_path = try_fuzzy_audio_cache(audio_prompt_path, text, stem=voice_stem)
    if fuzzy_path:
        return fuzzy_path, "Fuzzy audio"
    return None


@contextmanager
def no_grad_context(device: str = 'cuda'):
    """Context for no_grad (memory/perf opt). DRY."""
    if device != 'cuda':
        yield
        return
    with torch.no_grad():
        yield


def prepare_voice_and_conds(
    model: Any,
    audio_prompt_path: Optional[str],
    cache_uuid: int,
    exaggeration: float,
    language_id: str,
    enable_memory_cache: bool,
    enable_disk_cache: bool,
    device: torch.device,
    dtype: torch.dtype,
    sr: int,
    multilingual: bool,
    voice_stem: str = 'default'
) -> Optional[str]:
    """
    Helper: Process voice, validate, load/prepare conds; returns valid path or None.
    :param model: Loaded TTS model.
    :param audio_prompt_path: Path to audio prompt.
    :param cache_uuid: Cache UUID.
    :param exaggeration: Exaggeration value.
    :param language_id: Language ID.
    :param enable_memory_cache: Enable memory cache.
    :param enable_disk_cache: Enable disk cache.
    :param device: Torch device.
    :param dtype: Torch dtype.
    :param sr: Sample rate.
    :param multilingual: Multilingual mode.
    :param voice_stem: Voice stem (for cache/logging).
    :return: Valid path or None.
    """
    if model is None:
        logger.error("prepare_voice_and_conds: model is None – creating dummy conds")
        create_dummy_conds(None, device, dtype, "model_none")  # Dummy ignores model
        return None  # Trigger dummy

    original_path = audio_prompt_path
    logger.debug(f"Derived voice_stem: '{voice_stem}' (cache_uuid={cache_uuid}; orig_path={audio_prompt_path})")

    if audio_prompt_path is not None:
        # Pass voice_stem as stem (not cache_uuid); cache_uuid for key only
        fixed_from_process = get_or_queue_voice_process(
            audio_prompt_path, model, device, dtype, stem=voice_stem, exaggeration=exaggeration,  # stem=voice_stem
            quiet=(not enable_memory_cache and not enable_disk_cache)
        )
        audio_prompt_path = fixed_from_process
        if not audio_prompt_path or not Path(audio_prompt_path).exists():
            logger.warning(f"Process failed for {original_path} – using dummy")
            create_dummy_conds(model, device, dtype, "process_fail")
            return None

    if audio_prompt_path:
        valid, _ = validate_voice_path(audio_prompt_path, stem=voice_stem)  # Pass voice_stem
        logger.debug(f"Path after process: {audio_prompt_path}, valid: {valid}, stem: {voice_stem}")
        if not valid:
            logger.debug(f"Re-fix invalid path: {audio_prompt_path}")
            audio_prompt_path = check_and_update_ref(audio_prompt_path, exaggeration, stem=voice_stem)
        else:
            logger.debug(f"Valid path from process: {audio_prompt_path} - no resample")

        # Conds (use voice_stem if no cache hit)
        cache_params = {'language_id': language_id, 'cache_uuid': cache_uuid}
        cache_key = get_cache_key(audio_prompt_path, cache_uuid, exaggeration, params=cache_params)
        conditionals_loaded = False
        if cache_key and (enable_memory_cache or enable_disk_cache):
            if load_conditionals_cache(cache_key, model, device, dtype, enable_memory_cache, enable_disk_cache):
                conditionals_loaded = True
                logger.info(f"Conditionals cache HIT: {cache_key[:8]}... (stem={voice_stem}, uuid={cache_uuid})")
        if not conditionals_loaded:
            model.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
            if dtype != torch.float32:
                model.conds.t3.to(device=device, dtype=dtype)
            if cache_key and (enable_memory_cache or enable_disk_cache):
                save_conditionals_cache(cache_key, model.conds, model=model, device=device, dtype=dtype,
                                        enable_memory_cache=enable_memory_cache, enable_disk_cache=enable_disk_cache)
                logger.info(
                    f"Prepared and cached conditionals: {cache_key[:8]}... (stem={voice_stem}, uuid={cache_uuid})")
        return audio_prompt_path
    else:
        create_dummy_conds(model, device, dtype, "no_audio")
        logger.info("No audio prompt – using dummy conditionals")
        return None


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


def prepare_generation_params(text: str, audio_prompt_path: Optional[str], exaggeration: float, temperature: float, cfgw: float, min_p: float, top_p: float, repetition_penalty: float, language_id: str, seed_num: int, config) -> Tuple[Dict[str, Any], str, Dict[str, Any], str]:
     """Helper: Coerce and merge gen params (DRY for setup). Returns (params_dict, voice_stem, voice_params)."""
     # merge sanitize and set defaults here
     # Coerce params
     exaggeration = float(exaggeration)  # ... (copy coercions from original function)
     temperature = float(temperature)
     cfgw = float(cfgw)
     min_p = float(min_p)
     top_p = float(top_p)
     repetition_penalty = float(repetition_penalty)
     seed_num = int(seed_num or 42)

     voice_stem = get_voice_stem(audio_prompt_path)  # Use existing helper
     voice_params = config.get_merged_audio_params(voice_name=voice_stem)
     text = pad_short_text(text, voice_params)

     t3_params = {
         "generate_token_backend": "cudagraphs-manual",
         # TODO from config ? FIXED: Use inductor for fused speed (source)
         "stride_length": 4,  # Parallel tokens (source)
         "skip_when_1": True
     }

     # if get_config_value('app_config.globals.multilingual', False):
     #    generate_args["language_id"] = language_id

     generate_args = {
         'text': text, 'exaggeration': exaggeration, 'temperature': temperature, 'cfg_weight': cfgw,  'min_p': min_p, 'top_p': top_p,
         'repetition_penalty': repetition_penalty, 'language_id': language_id, 't3_params': t3_params
     }

     return generate_args, voice_stem, voice_params, text


@monitor_resources(enable=True, log_level="INFO")
async def generate_audio(model, text: str, audio_prompt_path: Optional[str], exaggeration: float = 0.5,
                         cache_uuid: int = 0,
                         temperature: float = 0.8, cfgw: float = 0, min_p: float = 0.05, top_p: float = 1.0,
                         repetition_penalty: float = 1.2, language_id: str = "en", seed_num: int = 42,
                         enable_memory_cache: bool = True, enable_disk_cache: bool = True) -> str:
    """
    Main orchestration: Validate, cache checks, prep, gen, post-process.
    Assumes model/device/dtype from globals; cleaned sig (no dead params).
    """
    func_start_time = perf_counter_ns()  # FIXED: Overall start
    from src.tts_model import ModelManager  # Ensure
    cache = enable_disk_cache or enable_memory_cache # TODO review

    if model is None or isinstance(model, str) or not hasattr(model, 'generate'):
        logger.warning(f"Invalid model to generate_audio: type={type(model).__name__} ({model}) – fetching from manager")
        real_model = ModelManager.get_instance().get_model()
        if real_model is None:
            logger.error("No model available – silence fallback")
            sr = 24000  # Default
            return str(save_torchaudio_wav(torch.zeros(1, sr * 2), sr, uuid=cache_uuid, cache=cache))  # 2s silence
        model = real_model
        logger.debug(f"generate_audio using real model: {type(model).__name__} on {getattr(model, 'device', 'N/A')}")

    config = get_config()
    device = torch.device(config.app_config.globals.device)  # Coerce to torch.device
    dtype = config.app_config.globals.dtype
    sr = config.app_config.globals.sr
    multilingual = config.app_config.globals.multilingual

    if not text:
        logger.warning("No text – using dummy")
        create_dummy_conds(model, device, dtype, "no_text")
        dummy_path = str(save_torchaudio_wav(torch.zeros(1, sr * 2), sr, uuid=cache_uuid, audio_path=None, cache=False))  # FIXED: Use sr
        return dummy_path

    original_text = text
    # get safe, cleaned, merged voice_params, and casted values
    generate_args, voice_stem, voice_params, text = prepare_generation_params(text, audio_prompt_path, exaggeration, temperature,
                                                                    cfgw, min_p, top_p, repetition_penalty, language_id,
                                                                    seed_num, config)
    set_seed(generate_args.get('seed_num', 42))
    logger.debug(f"Set seed: {generate_args.get('seed_num')}")

    if get_config_value('warmup_t3', True) and hasattr(model, 'generate') and not (hasattr(model, 'optimized') and model.optimized):
        dummy_warm = model.generate("Warm-up text.")  # Short; triggers if no cache hit
    elif hasattr(model, 'optimized') and model.optimized:
        logger.debug("T3 warmup skipped: Model pre-optimized")

    reuse_start = perf_counter_ns()
    try_fuzzy = get_config_value('app_config.globals.fuzzy.enable_fuzzy.cache', True)  # Use get_value (consistent)
    reuse_result = try_reuse_audio(generate_args.get('text'), audio_prompt_path, generate_args.get('exaggeration'), cache_uuid=cache_uuid, try_fuzzy=try_fuzzy) if audio_prompt_path else None
    reuse_time_ms = (perf_counter_ns() - reuse_start) / 1_000_000
    if reuse_result:
        audio_reuse_path, hit_type = reuse_result
        wav_reused, _ = torchaudio.load(audio_reuse_path)
        wav_length = wav_reused.shape[-1] / sr  # Use sr
        logger.info(
            f"{hit_type} cache HIT: \"{text[:50]}...\" ({hit_type.lower()}-match) for {generate_args.get('voice_stem')} – skipping gen (uuid={cache_uuid}; reuse: {reuse_time_ms:.0f}ms)")
        logger.info(f"Reused {hit_type.lower()} audio: {wav_length:.2f}s in ~0s")
        if audio_prompt_path:
            FUZZY_QUEUE.put((text, audio_reuse_path, voice_stem))
        total_time_ms = (perf_counter_ns() - func_start_time) / 1_000_000
        logger.info(f"Full cycle: HIT in {total_time_ms:.0f}ms (infinite speed!)")
        return audio_reuse_path

    logger.debug(f"Reuse MISS (took {reuse_time_ms:.0f}ms) – proceeding to full gen")

    prep_start = perf_counter_ns()
    valid_path = prepare_voice_and_conds(
        model, audio_prompt_path, cache_uuid, generate_args.get('exaggeration', 0.7), language_id,
        enable_memory_cache, enable_disk_cache, device, dtype, sr, multilingual, generate_args.get('voice_stem')
    )
    prep_time_ms = (perf_counter_ns() - prep_start) / 1_000_000
    logger.debug(f"Prep/conds: {prep_time_ms:.0f}ms")


    gen_start = perf_counter_ns()  # FIXED: Core gen start
    with no_grad_context(device=device):  # FIXED: Pass torch.device (no str)
        wav = _generate_audio_core(model, generate_args, generate_args.get('t3_params')) #TODO review t3
    gen_time_s = (perf_counter_ns() - gen_start) / 1_000_000_000  # FIXED: Core time
    logger.debug(f"Core gen time: {gen_time_s:.2f}s | raw wav shape: {wav.shape if wav is not None else 'None'}")

    if wav is None or wav.numel() == 0:
        logger.warning(f"Empty gen for '{text[:50]}...' – fallback silence")
        # FIXED: Centralize silence; CPU/fp32 for torchaudio save
        def _silence_tensor(sr: int, duration_s: float = 2.0) -> torch.Tensor:
            return torch.zeros(1, int(sr * duration_s), dtype=torch.float32, device='cpu')  # FIXED: CPU fallback (save-safe)
        wav = _silence_tensor(sr)

    post_start = perf_counter_ns()
    # FIXED: Ensure 2D mono input
    if wav.dim() == 1:
        wav_tensor = wav.unsqueeze(0)
    elif wav.dim() == 2 and wav.shape[0] == 1:
        wav_tensor = wav
    else:
        if wav.dim() > 1:
            wav = torch.mean(wav, dim=0)
        wav_tensor = wav.unsqueeze(0)
    logger.debug(f"Post input shape: {wav_tensor.shape} (stem={voice_stem})")

    try:
        post_result = await apply_post_processing(wav_tensor, sr, voice_params)
        if isinstance(post_result, torch.Tensor):
            wav_np = post_result.detach().cpu().numpy().squeeze() if post_result.dim() > 1 else post_result.cpu().numpy()
        elif isinstance(post_result, np.ndarray):
            wav_np = post_result
        else:
            raise ValueError(f"Post returned invalid type {type(post_result)}")
    except Exception as post_e:
        logger.error(f"Post failed for {voice_stem}: {post_e} – raw passthru")
        wav_np = wav_tensor.squeeze(0).cpu().numpy()

    # FIXED: Ensure 1D np (standardize)
    if len(wav_np.shape) > 1:
        if wav_np.shape[0] == 1:
            wav_np = wav_np.squeeze(0)
        else:
            wav_np = np.mean(wav_np, axis=1 if wav_np.shape[1] > 1 else 0)

    if len(wav_np) == 0:
        logger.warning(f"Post empty for {voice_stem} – silence fallback")
        wav_np = np.zeros(sr * 2)

    processed_wav = torch.from_numpy(wav_np).unsqueeze(0).to(device, dtype)
    post_time_ms = (perf_counter_ns() - post_start) / 1_000_000

    # FIXED: Guard None in format (error source)
    speaking_rate = voice_params.get('speaking_rate', 1.0) or 1.0  # Fallback float
    notch_gain_db = voice_params.get('notch_gain_db', None)
    notch_str = f"{notch_gain_db:.1f}dB" if notch_gain_db is not None else 'N/A'
    logger.info(f"Post for {voice_stem}: {post_time_ms:.0f}ms (rate={speaking_rate:.2f}, notch={notch_str})")

    func_end_time = perf_counter_ns()
    total_duration_s = (func_end_time - func_start_time) / 1_000_000_000
    audio_dur_s = len(wav_np) / sr  # FIXED: Generated length (post-processed)
    rtf_total = calculate_rtf(total_duration_s, audio_dur_s)
    rtf_core = calculate_rtf(gen_time_s, audio_dur_s)
    logger.info(  # FIXED: RTF log at INFO (end)
        f"Generated {audio_dur_s:.2f}s audio (total RTF: {rtf_total:.2f}x, core RTF: {rtf_core:.2f}x)")

    # FIXED: Conditional cleanup (always; safe)
    if torch.cuda.is_available():
        del wav  # FIXED: Del even if reused (idempotent)
        del wav_tensor  # NEW: Extra del for post
        torch.cuda.empty_cache()

    logger.info(f"Processed: {audio_dur_s:.2f}s {sr/1000:.0f}kHz in {total_duration_s:.2f}s (speed: {audio_dur_s/total_duration_s:.2f}x; gen: {gen_time_s:.2f}s, post: {post_time_ms/1000:.3f}s)")

    save_start = perf_counter_ns()
    save_wav = processed_wav.to(torch.float32).cpu()
    # FIXED: Use original unpadded audio_prompt_path for save (unpadded key/prefix); valid_path for prep only
    save_voice_path = audio_prompt_path  # Unpadded original for consistent cache key
    logger.debug(f"Save using unpadded voice_path: '{save_voice_path}' (valid_path was '{valid_path}')")
    wave_file = save_and_cache_output(
        save_wav, save_voice_path, cache_uuid, generate_args.get('text'), generate_args.get('exaggeration'), generate_args, enable_memory_cache, enable_disk_cache, sr
    )
    save_time_ms = (perf_counter_ns() - save_start) / 1_000_000
    logger.debug(f"Save/cache: {save_time_ms:.0f}ms → {wave_file}")

    if reuse_result is None and (valid_path or audio_prompt_path):
        FUZZY_QUEUE.put((text, wave_file, voice_stem))

    stats = get_cache_stats()
    logger.info(f"Cache post-gen: mem={stats['memory_cache_size']}, disk={stats['disk_files']}, audio={stats['audio_cache_size']}, fuzzy={stats.get('fuzzy_size', 'N/A')}")

    total_time_ms = (perf_counter_ns() - func_start_time) / 1_000_000
    logger.info(f"MISS cycle: {total_time_ms/1000:.2f}s (reuse={reuse_time_ms:.0f}ms, prep={prep_time_ms:.0f}ms, gen={gen_time_s:.2f}s, post={post_time_ms:.0f}ms, save={save_time_ms:.0f}ms)")

    return wave_file