# src/generate_audio.py
import logging
import threading

from pathlib import Path
import torch
from time import perf_counter_ns
import torchaudio
from typing import Optional, Dict, Any, Tuple
from config import ENABLE_MEMORY_CACHE, ENABLE_DISK_CACHE, DEVICE, DTYPE, MULTILINGUAL  # Import actual globals (fixes ... placeholders)
from .cache import (
    try_audio_cache, set_audio_cache, get_cache_key, get_or_queue_voice_process,
    validate_voice_path, create_dummy_conds, load_conditionals_cache, save_conditionals_cache,
    get_cache_stats, check_and_update_ref, save_torchaudio_wav
)
from .fuzzy_cache import try_fuzzy_audio_cache, FUZZY_QUEUE

logger = logging.getLogger(__name__)
GEN_ACTIVE_LOCK = threading.RLock()  # Global for gen/prepare

def set_seed(seed: int):
    """
    Set random seeds for reproducible generation.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def _generate_audio_core(model, generate_args: Dict[str, Any], t3_params: Dict[str, Any]) -> torch.Tensor:
    """Core: Just model.generate + graph retry. Returns wav; no del/cleanup."""
    with GEN_ACTIVE_LOCK:  # NEW: Serialize vs async prepare (prevent graph race)
        wav = None
        try:
            wav = model.generate(**generate_args)
        except RuntimeError as graph_e:
            if "graph" in str(graph_e).lower() or "capture" in str(graph_e).lower() or "offset" in str(graph_e).lower():
                logger.warning(f"Graph corrupt: {graph_e} – resetting t3 graphs and retrying")
                if hasattr(model, 't3') and hasattr(model.t3, '_bucket_graphs'):
                    model.t3._bucket_graphs.clear()
                    torch.cuda.empty_cache()
                t3_params_temp = t3_params.copy()
                t3_params_temp['generate_token_backend'] = 'eager'  # Force eager on retry
                generate_args_temp = generate_args.copy()
                generate_args_temp['t3_params'] = t3_params_temp
                wav = model.generate(**generate_args_temp)
            else:
                raise
        return wav

def try_reuse_audio(text: str, audio_prompt_path: Optional[str], exaggeration: float, params: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """Try exact then fuzzy; return (path, hit_type) or None. Handles None path.
    Fixed: Pass audio_prompt_path as audio_path for stem extraction; text as text_input; explicit stem kwarg."""
    if not audio_prompt_path:
        return None
    # Exact
    exact_path = try_audio_cache(audio_prompt_path, text, exaggeration, params=params)
    if exact_path:
        return exact_path, "Full audio"
    # Fuzzy: Extract stem explicitly (normalize, remove '_fixed' etc.)
    voice_stem = Path(audio_prompt_path).stem.replace('_fixed', '').split('_')[0]  # e.g., 'dlc1seranavoice' (robust)
    fuzzy_path = try_fuzzy_audio_cache(audio_prompt_path, text, stem=voice_stem)  # Correct: audio_path=voice path (for fallback extract), text_input=text, stem=voice_stem
    if fuzzy_path:
        return fuzzy_path, "Fuzzy audio"
    return None


def prepare_voice_and_conds(model, audio_prompt_path: Optional[str], cache_uuid: int, exaggeration: float,
                            language_id: str, enable_memory_cache: bool, enable_disk_cache: bool,
                            device: torch.device, dtype: torch.dtype) -> Optional[str]:
    """Helper: Process voice, validate, load/prepare conds; returns valid path or None."""
    original_path = audio_prompt_path
    # NEW: Derive voice_stem from path (e.g., 'nwsjennavoice' from temp/nwsjennavoice.wav)
    voice_stem = Path(audio_prompt_path).stem.replace('_fixed', '').split('_')[
        0] if audio_prompt_path else None  # Prefix like 'nwsjenna'
    logger.debug(f"Derived voice_stem from '{audio_prompt_path}': '{voice_stem}' (cache_uuid={cache_uuid})")

    if audio_prompt_path is not None:
        # FIXED: Pass voice_stem as stem (not cache_uuid); cache_uuid for key only
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


def save_and_cache_output(wav: torch.Tensor, model, audio_prompt_path: Optional[str], cache_uuid: int,
                          text: str, exaggeration: float, params: Dict[str, Any], enable_memory_cache: bool,
                          enable_disk_cache: bool) -> str:
    """Helper: Save WAV, cache exact audio if applicable."""
    if audio_prompt_path:
        full_cache_key = get_cache_key(audio_prompt_path, cache_uuid, exaggeration, params={
            'text': text, 'cfgw': params.get('cfgw'), 'temperature': params.get('temperature'),
            'min_p': params.get('min_p'), 'top_p': params.get('top_p'),
            'repetition_penalty': params.get('repetition_penalty'),
            'language_id': params.get('language_id')
        })
        wave_file = str(save_torchaudio_wav(wav.cpu(), model.sr, audio_path=audio_prompt_path, uuid=cache_uuid))
        set_audio_cache(full_cache_key, wave_file)
    else:
        wave_file = str(save_torchaudio_wav(wav.cpu(), model.sr, audio_path=None, uuid=cache_uuid))
    return wave_file

def generate_audio(model, text: str, audio_prompt_path: Optional[str], exaggeration: float = 0.5, cache_uuid: int = 0,
                   temperature: float = 0.8, cfgw: float = 0, min_p: float = 0.05, top_p: float = 1.0,
                   repetition_penalty: float = 1.2, language_id: str = "en", seed_num: int = 42,  # DEFAULT: Fixed 42 for consistency
                   enable_memory_cache: bool = True, enable_disk_cache: bool = True) -> str:
    """
    Main orchestration: Validate, cache checks, prep, gen, post-process.
    Assumes model/device/dtype from globals; cleaned sig (no dead params).
    """
    if not text:
        logger.warning("No text – using dummy")
        create_dummy_conds(model, DEVICE, DTYPE, "no_text")
        return str(save_torchaudio_wav(torch.zeros(1, 24000), 24000, uuid=cache_uuid))  # Short dummy

    # Float conversions
    exaggeration = float(exaggeration)
    temperature = float(temperature)
    cfgw = float(cfgw)
    min_p = float(min_p)
    top_p = float(top_p)
    repetition_penalty = float(repetition_penalty)

    params = {
        'cfgw': cfgw, 'temperature': temperature, 'min_p': min_p, 'top_p': top_p,
        'repetition_penalty': repetition_penalty, 'language_id': language_id
    }

    func_start_time = perf_counter_ns()  # Overall timer

    # Logging (ONLY here—no dup in shell)
    stem = Path(audio_prompt_path).stem if audio_prompt_path else "No ref audio"
    logger.info(f"generate called for: \"{text}\", {stem}, uuid: {cache_uuid}, exaggeration: {exaggeration}")
    logger.info(f"Parameters - temp: {temperature}, min_p: {min_p}, top_p: {top_p}, rep_penalty: {repetition_penalty}, cfg_weight: {cfgw}")

    # FIXED: Always set seed (default 42 if 0; ensures consistent accents)
    if seed_num == 0:
        seed_num = 42  # Default for reproducibility
    set_seed(int(seed_num))
    logger.debug(f"Set seed: {seed_num} (for consistent voices/accents)")

    reuse_start = perf_counter_ns()  # Time reuse check
    # Reuse check (updated helper)
    reuse_result = try_reuse_audio(text, audio_prompt_path, exaggeration, params) if audio_prompt_path else None
    reuse_time_ms = (perf_counter_ns() - reuse_start) / 1_000_000
    if reuse_result:
        audio_reuse_path, hit_type = reuse_result
        # Load/log (local torchaudio)
        wav_reused, sr = torchaudio.load(audio_reuse_path)
        wav_length = wav_reused.shape[-1] / sr
        logger.info(f"{hit_type} cache HIT: \"{text[:20]}\" ({hit_type.lower()}-match) for {Path(audio_prompt_path).stem} – skipping gen (uuid={cache_uuid}; reuse: {reuse_time_ms:.2f}ms)")
        logger.info(f"Reused {hit_type.lower()} audio: {wav_length:.2f}s in ~0s")
        # Enqueue fuzzy...
        if audio_prompt_path:
            voice_stem = Path(audio_prompt_path).stem.replace('_fixed', '').split('_')[0]
            logger.debug(f"Enqueued for fuzzy enrich (HIT): \"{text[:20]}\" (norm stem={voice_stem})")
            FUZZY_QUEUE.put((text, audio_reuse_path, voice_stem))
        total_time_ms = (perf_counter_ns() - func_start_time) / 1_000_000
        logger.info(f"Full cycle: HIT in {total_time_ms:.2f}ms (infinite speed!)")
        return audio_reuse_path

    logger.debug(f"Reuse MISS (took {reuse_time_ms:.2f}ms) – proceeding to full gen")

    # Prep voice/conds (helper; handles None → dummy)
    prep_start = perf_counter_ns()
    valid_path = prepare_voice_and_conds(
        model, audio_prompt_path, cache_uuid, exaggeration, language_id, enable_memory_cache, enable_disk_cache, DEVICE, DTYPE
    )
    prep_time_ms = (perf_counter_ns() - prep_start) / 1_000_000
    logger.debug(f"Prep/conds: {prep_time_ms:.2f}ms")

    # Conds prep time log (legacy)
    conditional_start_time = perf_counter_ns()
    logger.info(f"Conditionals prepared. Time: {(conditional_start_time - func_start_time) / 1_000_000:.4f}ms")

    # Build args/t3_params
    t3_params = {
        "generate_token_backend": "cudagraphs-manual",
        "stride_length": 4,
        "skip_when_1": True,
    }
    generate_args = {
        "text": text,
        "exaggeration": exaggeration,
        "temperature": temperature,
        "cfg_weight": cfgw,
        "min_p": min_p,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "t3_params": t3_params,
    }
    if MULTILINGUAL:
        generate_args["language_id"] = language_id

    # Core gen (time it)
    gen_start = perf_counter_ns()
    # NEW: Ensure seed before gen (consistent even on retry)
    set_seed(int(seed_num))  # Re-set post-prep (safe)
    wav = _generate_audio_core(model, generate_args, t3_params)
    gen_time_s = (perf_counter_ns() - gen_start) / 1_000_000_000
    logger.debug(f"Core gen time: {gen_time_s:.2f}s")

    # Post-gen timings/log
    func_end_time = perf_counter_ns()
    total_duration_s = (func_end_time - func_start_time) / 1_000_000_000
    wav_length = wav.shape[-1] / model.sr
    logger.info(
        f"Generated audio: {wav_length:.2f}s {model.sr / 1000:.2f}kHz in {total_duration_s:.2f}s. Speed: {wav_length / total_duration_s:.2f}x (gen: {gen_time_s:.2f}s, seed: {seed_num})")

    # Save/cache
    save_start = perf_counter_ns()
    wave_file = save_and_cache_output(
        wav, model, valid_path or audio_prompt_path, cache_uuid, text, exaggeration, params, enable_memory_cache, enable_disk_cache
    )
    save_time_ms = (perf_counter_ns() - save_start) / 1_000_000
    logger.debug(f"Save/cache: {save_time_ms:.2f}ms")

    # Enqueue fuzzy (only on true MISS; use normalized stem from valid_path)
    if not reuse_result and (valid_path or audio_prompt_path):
        voice_stem = Path(valid_path or audio_prompt_path).stem.replace('_fixed', '').split('_')[0]
        logger.debug(f"Enqueued for fuzzy index (MISS): \"{text[:20]}\" (norm stem={voice_stem})")
        FUZZY_QUEUE.put((text, wave_file, voice_stem))

    # Stats
    stats = get_cache_stats()
    logger.info(
        f"Cache stats post-gen: {stats['memory_cache_size']} mem, {stats['disk_files']} disk, {stats['audio_cache_size']} audio, fuzzy: {stats.get('fuzzy_size', 'N/A')}")

    # Cleanup (post-save)
    del wav
    torch.cuda.empty_cache()
    total_time_ms = (perf_counter_ns() - func_start_time) / 1_000_000
    logger.info(f"Full MISS cycle: {total_time_ms / 1000:.2f}s (reuse: {reuse_time_ms:.0f}ms, prep: {prep_time_ms:.0f}ms, gen: {gen_time_s:.2f}s, save: {save_time_ms:.0f}ms, seed: {seed_num})")
    return wave_file