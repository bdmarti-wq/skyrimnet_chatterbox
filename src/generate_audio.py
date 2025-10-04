# src/generate_audio.py
import logging
from pathlib import Path
import torch
from time import perf_counter_ns
import torchaudio
from typing import Optional, Dict, Any, Tuple
from config import ENABLE_MEMORY_CACHE, ENABLE_DISK_CACHE, DEVICE, DTYPE, MULTILINGUAL  # Import actual globals (fixes ... placeholders)
from .cache import (
    try_audio_cache, set_audio_cache, get_cache_key, get_or_queue_voice_process,
    validate_voice_path, create_dummy_conds, load_conditionals_cache, save_conditionals_cache,
    get_cache_stats, try_fuzzy_audio_cache, _fuzzy_queue, check_and_update_ref, save_torchaudio_wav
)

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """
    Set random seeds for reproducible generation.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def _generate_audio_core(model, generate_args: Dict[str, Any], t3_params: Dict[str, Any]) -> torch.Tensor:
    """Core: Just model.generate + graph retry. Returns wav; no del/cleanup."""
    wav = None
    try:
        wav = model.generate(**generate_args)
    except RuntimeError as graph_e:
        if "graph" in str(graph_e).lower() or "capture" in str(graph_e).lower():
            logger.warning(f"Graph corrupt: {graph_e} – resetting t3 graphs and retrying")
            if hasattr(model, 't3') and hasattr(model.t3, '_bucket_graphs'):
                model.t3._bucket_graphs.clear()
                torch.cuda.empty_cache()
            t3_params_temp = t3_params.copy()
            t3_params_temp['generate_token_backend'] = 'eager'
            generate_args_temp = generate_args.copy()
            generate_args_temp['t3_params'] = t3_params_temp
            wav = model.generate(**generate_args_temp)
        else:
            raise
    return wav  # No del—caller handles post-use

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
    if audio_prompt_path is not None:
        fixed_from_process = get_or_queue_voice_process(
            audio_prompt_path, model, device, dtype, cache_uuid, exaggeration,
            quiet=(not enable_memory_cache and not enable_disk_cache)
        )
        audio_prompt_path = fixed_from_process
        if not audio_prompt_path or not Path(audio_prompt_path).exists():
            logger.warning(f"Process failed for {original_path} – using dummy")
            create_dummy_conds(model, device, dtype, "process_fail")
            return None

    if audio_prompt_path:
        valid, _ = validate_voice_path(audio_prompt_path)
        logger.debug(f"Path after process: {audio_prompt_path}, valid: {valid}")
        if not valid:
            logger.debug(f"Re-fix invalid path: {audio_prompt_path}")
            audio_prompt_path = check_and_update_ref(audio_prompt_path, exaggeration)
        else:
            logger.debug(f"Valid path from process: {audio_prompt_path} - no resample")

        # Conds
        cache_params = {'language_id': language_id, 'cache_uuid': cache_uuid}
        cache_key = get_cache_key(audio_prompt_path, cache_uuid, exaggeration, params=cache_params)
        conditionals_loaded = False
        if cache_key and (enable_memory_cache or enable_disk_cache):
            if load_conditionals_cache(cache_key, model, device, dtype, enable_memory_cache, enable_disk_cache):
                conditionals_loaded = True
                logger.info(f"Conditionals cache HIT: {cache_key[:8]}... (uuid={cache_uuid})")
        if not conditionals_loaded:
            model.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
            if dtype != torch.float32:
                model.conds.t3.to(device=device, dtype=dtype)  # Explicit device too (safe)
            if cache_key and (enable_memory_cache or enable_disk_cache):
                save_conditionals_cache(cache_key, model.conds, model=model, device=device, dtype=dtype,
                                        enable_memory_cache=enable_memory_cache, enable_disk_cache=enable_disk_cache)
                logger.info(f"Prepared and cached conditionals: {cache_key[:8]}... (uuid={cache_uuid})")
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
                   repetition_penalty: float = 1.2, language_id: str = "en", seed_num: int = 0,
                   enable_memory_cache: bool = True, enable_disk_cache: bool = True) -> str:
    """
    Main orchestration: Validate, cache checks, prep, gen, post-process.
    Assumes model/device/dtype from globals; cleaned sig (no dead params).
    """
    # Invalid: Handle None gracefully (original no-audio dummy)
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

    func_start_time = perf_counter_ns()

    # Logging (ONLY here—no dup in shell)
    stem = Path(audio_prompt_path).stem if audio_prompt_path else "No ref audio"
    logger.info(f"generate called for: \"{text}\", {stem}, uuid: {cache_uuid}, exaggeration: {exaggeration}")
    logger.info(f"Parameters - temp: {temperature}, min_p: {min_p}, top_p: {top_p}, rep_penalty: {repetition_penalty}, cfg_weight: {cfgw}")

    # Seed
    if seed_num != 0:
        set_seed(int(seed_num))

    # Reuse check (updated helper)
    reuse_result = try_reuse_audio(text, audio_prompt_path, exaggeration, params) if audio_prompt_path else None
    if reuse_result:
        audio_reuse_path, hit_type = reuse_result
        # Load/log (local torchaudio)
        wav_reused, sr = torchaudio.load(audio_reuse_path)
        wav_length = wav_reused.shape[-1] / sr
        logger.info(f"{hit_type} cache HIT: \"{text[:20]}\" ({hit_type.lower()}-match) for {Path(audio_prompt_path).stem} – skipping gen (uuid={cache_uuid})")
        logger.info(f"Reused {hit_type.lower()} audio: {wav_length:.2f}s in ~0s (infinite speed!)")
        # Enqueue fuzzy (low-priority enrichment on HIT; use normalized stem)
        if audio_prompt_path:
            voice_stem = Path(audio_prompt_path).stem.replace('_fixed', '').split('_')[0]  # Normalize: 'dlc1seranavoice'
            logger.debug(f"Enqueued for fuzzy enrich (HIT): \"{text[:20]}\" (norm stem={voice_stem})")
            _fuzzy_queue.put((text, audio_reuse_path, voice_stem))
        return audio_reuse_path

    logger.debug("No audio reuse – proceeding to full gen")

    # Prep voice/conds (helper; handles None → dummy)
    valid_path = prepare_voice_and_conds(
        model, audio_prompt_path, cache_uuid, exaggeration, language_id, enable_memory_cache, enable_disk_cache, DEVICE, DTYPE
    )

    # Conds prep time log
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

    # Core gen
    wav = _generate_audio_core(model, generate_args, t3_params)

    # Post-gen timings/log
    func_end_time = perf_counter_ns()
    total_duration_s = (func_end_time - func_start_time) / 1_000_000_000
    wav_length = wav.shape[-1] / model.sr
    logger.info(
        f"Generated audio: {wav_length:.2f}s {model.sr / 1000:.2f}kHz in {total_duration_s:.2f}s. Speed: {wav_length / total_duration_s:.2f}x")

    # Save/cache
    wave_file = save_and_cache_output(
        wav, model, valid_path or audio_prompt_path, cache_uuid, text, exaggeration, params, enable_memory_cache, enable_disk_cache
    )

     # Enqueue fuzzy (only on true MISS; use normalized stem from valid_path)
    if not reuse_result and (valid_path or audio_prompt_path):
        voice_stem = Path(valid_path or audio_prompt_path).stem.replace('_fixed', '').split('_')[0]  # Normalize: 'dlc1seranavoice'
        logger.debug(f"Enqueued for fuzzy index (MISS): \"{text[:20]}\" (norm stem={voice_stem})")
        _fuzzy_queue.put((text, wave_file, voice_stem))

    # Stats
    stats = get_cache_stats()
    logger.info(
        f"Cache stats post-gen: {stats['memory_cache_size']} mem, {stats['disk_files']} disk, {stats['audio_cache_size']} audio")

    # Cleanup (post-save)
    del wav
    torch.cuda.empty_cache()
    return wave_file