import functools
import logging
import threading

import numpy as np
from pathlib import Path
import torch
from time import perf_counter_ns, time
import torchaudio
from typing import Optional, Dict, Any, Tuple

from .config import CONFIG
from .audio_utils import apply_post_processing
from .cache import (
    try_audio_cache, set_audio_cache, get_cache_key, get_or_queue_voice_process,
    validate_voice_path, create_dummy_conds, load_conditionals_cache, save_conditionals_cache,
    get_cache_stats, check_and_update_ref, save_torchaudio_wav
)
from .fuzzy_cache import try_fuzzy_audio_cache, FUZZY_QUEUE  # Added missing import

logger = logging.getLogger(__name__)
GEN_ACTIVE_LOCK = threading.RLock()  # Global for gen/prepare


def set_seed(seed: int):
    """
    Set random seeds for reproducible generation.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# Text padding for short/vocalise (pre-TTS; smooths garble via pauses)
# Gated text padding (pre-TTS; uses merged params for tunables/gates)
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
    patterns = params.get('vocalise_patterns', ['ah', 'oh', 'aah'])

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
    Core: Just model.generate + graph retry. Returns wav; no del/cleanup.
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
        return wav


def try_reuse_audio(
    text: str,
    audio_prompt_path: Optional[str],
    exaggeration: float,
    params: Dict[str, Any],
    try_fuzzy: bool = True
) -> Optional[Tuple[str, str]]:
    """
    Try exact then fuzzy; return (path, hit_type) or None. Handles None path.
    :param text: Input text.
    :param audio_prompt_path: Path to audio prompt.
    :param exaggeration: Exaggeration value.
    :param params: Merged params dict.
    :param try_fuzzy: Whether to try fuzzy cache.
    :return: (path, hit_type) or None.
    """
    if not audio_prompt_path:
        return None
    # Exact
    exact_path = try_audio_cache(audio_prompt_path, text, exaggeration, params=params)
    if exact_path:
        return exact_path, "Full audio"
    # Fuzzy: Extract stem explicitly (normalize, remove '_fixed' etc.)
    fuzzy_path = None
    if try_fuzzy:
        # FIXED: Derive full_stem consistently (strip '_voice' suffix without split for full base)
        full_stem = Path(audio_prompt_path).stem.replace('_fixed', '').replace('_padded', '').replace('_resampled', '').replace('_ui_resampled', '')  # e.g., 'cs_coralyn_voice'
        voice_stem = full_stem[:-6] if full_stem.endswith('_voice') else full_stem  # Strip '_voice' suffix (e.g., "cs_coralyn_voice" → "cs_coralyn")
        if len(voice_stem) < 3:
            voice_stem = full_stem  # Ensure full
        logger.debug(f"Query stem derived: '{voice_stem}' from path '{audio_prompt_path}'")  # FIXED: Added debug log
        fuzzy_path = try_fuzzy_audio_cache(audio_prompt_path, text, stem=voice_stem)  # audio_path=voice path, text_input=text, stem=voice_stem
    if fuzzy_path:
        return fuzzy_path, "Fuzzy audio"
    return None


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
    model: Any,
    audio_prompt_path: Optional[str],
    cache_uuid: int,
    text: str,
    exaggeration: float,
    params: Dict[str, Any],
    enable_memory_cache: bool,
    enable_disk_cache: bool,
    sr: int
) -> str:
    """
    Helper: Save WAV, cache exact audio if applicable.
    :param wav: Processed WAV tensor (2D mono).
    :param model: Loaded model (for sr if needed, but pass sr param).
    :param audio_prompt_path: Audio prompt path.
    :param cache_uuid: Cache UUID.
    :param text: Text.
    :param exaggeration: Exaggeration.
    :param params: Merged params.
    :param enable_memory_cache: Enable memory cache.
    :param enable_disk_cache: Enable disk cache.
    :param sr: Sample rate.
    :return: Saved path str.
    """
    if audio_prompt_path:
        full_cache_key = get_cache_key(audio_prompt_path, cache_uuid, exaggeration, params={
            'text': text, 'cfgw': params.get('cfgw'), 'temperature': params.get('temperature'),
            'min_p': params.get('min_p'), 'top_p': params.get('top_p'),
            'repetition_penalty': params.get('repetition_penalty'),
            'language_id': params.get('language_id')
        })
        wave_file = str(save_torchaudio_wav(wav.cpu(), sr, audio_path=audio_prompt_path, uuid=cache_uuid))
        set_audio_cache(full_cache_key, wave_file)
    else:
        wave_file = str(save_torchaudio_wav(wav.cpu(), sr, audio_path=None, uuid=cache_uuid))
    return wave_file


async def generate_audio(model, text: str, audio_prompt_path: Optional[str], exaggeration: float = 0.5,
                         cache_uuid: int = 0,
                         temperature: float = 0.8, cfgw: float = 0, min_p: float = 0.05, top_p: float = 1.0,
                         repetition_penalty: float = 1.2, language_id: str = "en", seed_num: int = 42,
                         enable_memory_cache: bool = True, enable_disk_cache: bool = True) -> str:
    """
    Main orchestration: Validate, cache checks, prep, gen, post-process.
    Assumes model/device/dtype from globals; cleaned sig (no dead params).
    """
    device = CONFIG.device
    dtype = CONFIG.dtype
    sr = CONFIG.sr  # 24000 from config.py
    multilingual = CONFIG.multilingual  # False by default
    # FIXED: Derive full_stem consistently (strip '_voice' suffix without split for full base)
    full_stem = Path(audio_prompt_path).stem.replace('_fixed', '').replace('_padded', '').replace('_resampled', '').replace('_ui_resampled', '')  # e.g., 'cs_coralyn_voice'
    voice_stem = full_stem[:-6] if full_stem.endswith('_voice') else full_stem  # Strip '_voice' suffix (e.g., "cs_coralyn_voice" → "cs_coralyn")
    if len(voice_stem) < 3:
        voice_stem = full_stem  # Ensure full
    logger.debug(f"Main: Derived voice_stem: '{voice_stem}' from path '{audio_prompt_path}'")  # FIXED: Added debug log

    if not text:
        logger.warning("No text – using dummy")
        create_dummy_conds(model, CONFIG.device, CONFIG.dtype, "no_text")
        dummy_path = str(save_torchaudio_wav(torch.zeros(1, 24000), 24000, uuid=cache_uuid))  # Short dummy path
        return dummy_path

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
    logger.info(
        f"Parameters - temp: {temperature}, min_p: {min_p}, top_p: {top_p}, rep_penalty: {repetition_penalty}, cfg_weight: {cfgw}")

    # FIXED: Use consistent voice_stem for params (full base, no split[0])
    original_text = text
    text = pad_short_text(text, CONFIG.get_merged_audio_params(voice_name=voice_stem))  # Was split[0] → 'cs'; now 'cs_coralyn'
    logger.debug(
        f"TTS text: '{text}' (padded={len(text) > len(original_text) if 'original_text' in locals() else False})")

    # FIXED: Always set seed (default 42 if 0; ensures consistent accents)
    if seed_num == 0:
        seed_num = 42  # Default for reproducibility
    set_seed(int(seed_num))
    logger.debug(f"Set seed: {seed_num} (for consistent voices/accents)")

    reuse_start = perf_counter_ns()  # Time reuse check
    # Reuse check (updated helper)
    try_fuzzy = CONFIG.get_value('fuzzy_enable', False)
    reuse_result = try_reuse_audio(text, audio_prompt_path, exaggeration, params, try_fuzzy) if audio_prompt_path else None
    reuse_time_ms = (perf_counter_ns() - reuse_start) / 1_000_000
    if reuse_result:
        audio_reuse_path, hit_type = reuse_result
        # Load/log (local torchaudio)
        wav_reused, sr = torchaudio.load(audio_reuse_path)
        wav_length = wav_reused.shape[-1] / sr
        logger.info(
            f"{hit_type} cache HIT: \"{text[:20]}\" ({hit_type.lower()}-match) for {Path(audio_prompt_path).stem} – skipping gen (uuid={cache_uuid}; reuse: {reuse_time_ms:.2f}ms)")
        logger.info(f"Reused {hit_type.lower()} audio: {wav_length:.2f}s in ~0s")
        # FIXED: Enqueue fuzzy with consistent voice_stem (matching query derivation)
        if audio_prompt_path:
            logger.debug(f"Enqueued for fuzzy enrich (HIT): \"{text[:20]}\" (norm stem={voice_stem})")
            FUZZY_QUEUE.put((text, audio_reuse_path, voice_stem))  # Now uses full 'cs_coralyn'
        total_time_ms = (perf_counter_ns() - func_start_time) / 1_000_000
        logger.info(f"Full cycle: HIT in {total_time_ms:.2f}ms (infinite speed!)")
        return audio_reuse_path  # Str path (early return)

    logger.debug(f"Reuse MISS (took {reuse_time_ms:.2f}ms) – proceeding to full gen")

    # Prep voice/conds (helper; handles None → dummy)
    prep_start = perf_counter_ns()
    valid_path = prepare_voice_and_conds(
        model, audio_prompt_path, cache_uuid, exaggeration, language_id,
        enable_memory_cache, enable_disk_cache, device, dtype, sr, multilingual, voice_stem  # Pass consistent voice_stem
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
    if CONFIG.multilingual:
        generate_args["language_id"] = language_id

    # Core gen (time it)
    gen_start = perf_counter_ns()
    # NEW: Ensure seed before gen (consistent even on retry)
    set_seed(int(seed_num))  # Re-set post-prep (safe)
    wav = _generate_audio_core(model, generate_args, t3_params)
    gen_time_s = (perf_counter_ns() - gen_start) / 1_000_000_000
    logger.debug(f"Core gen time: {gen_time_s:.2f}s | raw wav shape: {wav.shape if wav is not None else 'None'}")

    if wav is None or wav.numel() == 0:
        logger.warning(f"Empty gen for '{text[:50]}...' – fallback silence")
        wav = torch.zeros(1, CONFIG.sr * 2, dtype=CONFIG.dtype, device=CONFIG.device)  # 2D silence

    # Post-gen timings/log (derive stem for voice-specific params)
    # FIXED: Use consistent voice_stem for params (full base, no split[0])
    stem = voice_stem  # Reuse the derived voice_stem
    merged_params = CONFIG.get_merged_audio_params(voice_name=stem)  # Now 'cs_coralyn', not 'cs'
    post_start = perf_counter_ns()
    # FIXED: Ensure 2D input for post ([1, samples] mono)
    if wav.dim() == 1:
        wav_tensor = torch.unsqueeze(wav, 0)  # [1, samples]
    elif wav.dim() == 2 and wav.shape[0] == 1:
        wav_tensor = wav  # Already good
    else:
        logger.warning(f"Unexpected wav shape {wav.shape} for post – squeezing to mono 2D")
        if wav.dim() > 1:
            wav = torch.mean(wav, dim=0)  # Avg channels → 1D
        wav_tensor = torch.unsqueeze(wav, 0)  # [1, samples]
    logger.debug(f"Post input shape: {wav_tensor.shape} (stem={stem})")

    try:
        # FIXED: Await async post; handle np/torch return
        post_result = await apply_post_processing(wav_tensor, CONFIG.sr, merged_params)
        if isinstance(post_result, torch.Tensor):
            logger.debug(f"Post returned tensor {post_result.shape} – to np")
            wav_np = post_result.detach().cpu().numpy().squeeze() if post_result.dim() > 1 else post_result.cpu().numpy()
        elif isinstance(post_result, np.ndarray):
            wav_np = post_result
        else:
            raise ValueError(f"Post returned invalid type {type(post_result)}")
        logger.debug(f"Post np: {wav_np.shape} (dtype={wav_np.dtype})")
    except Exception as post_e:
        logger.error(f"Post failed for {stem}: {post_e} – raw passthru")
        wav_np = wav_tensor.squeeze(0).cpu().numpy()  # Raw 1D fallback

    # Ensure 1D np
    if len(wav_np.shape) > 1:
        if wav_np.shape[0] == 1:
            wav_np = wav_np.squeeze(0)
        else:
            wav_np = np.mean(wav_np, axis=1 if wav_np.shape[1] > 1 else 0)  # Stereo to mono

    logger.debug(f"Final wav_np: {wav_np.shape} (len={len(wav_np)})" )

    # Fallback empty
    if len(wav_np) == 0:
        logger.warning(f"Post empty for {stem} – silence fallback")
        wav_np = np.zeros(CONFIG.sr * 2)

    # To tensor (2D for save)
    processed_wav = torch.from_numpy(wav_np).unsqueeze(0).to(CONFIG.device, CONFIG.dtype)
    post_time_ms = (perf_counter_ns() - post_start) / 1_000_000
    logger.info(f"Post for {stem}: {post_time_ms:.2f}ms (rate={merged_params.get('speaking_rate', 1.0)}, notch={merged_params.get('notch_gain_db', 'N/A')}dB)")

    func_end_time = perf_counter_ns()
    total_duration_s = (func_end_time - func_start_time) / 1_000_000_000
    wav_length = len(wav_np) / CONFIG.sr
    logger.info(f"Processed: {wav_length:.2f}s {CONFIG.sr/1000:.0f}kHz in {total_duration_s:.2f}s (speed: {wav_length/total_duration_s:.2f}x; gen: {gen_time_s:.2f}s, post: {post_time_ms/1000:.3f}s)")

    # Save/cache (2D float32 CPU)
    save_start = perf_counter_ns()
    save_wav = processed_wav.to(torch.float32).cpu()  # 2D mono
    wave_file = save_and_cache_output(save_wav, model, valid_path or audio_prompt_path, cache_uuid, text, exaggeration, params, enable_memory_cache, enable_disk_cache, sr)
    save_time_ms = (perf_counter_ns() - save_start) / 1_000_000
    logger.debug(f"Save/cache (processed): {save_time_ms:.2f}ms → {wave_file}")

    # Enqueue fuzzy (MISS only)
    if reuse_result is None and (valid_path or audio_prompt_path):
        # FIXED: Enqueue fuzzy with consistent voice_stem (matching query derivation)
        logger.debug(f"Enqueued fuzzy MISS: \"{text[:20]}\" (stem={voice_stem})")
        FUZZY_QUEUE.put((text, wave_file, voice_stem))

    # Stats
    stats = get_cache_stats()
    logger.info(f"Cache post-gen: mem={stats['memory_cache_size']}, disk={stats['disk_files']}, audio={stats['audio_cache_size']}, fuzzy={stats.get('fuzzy_size', 'N/A')}")

    # Cleanup
    del wav
    torch.cuda.empty_cache()
    total_time_ms = (perf_counter_ns() - func_start_time) / 1_000_000
    logger.info(f"MISS cycle: {total_time_ms/1000:.2f}s (reuse={reuse_time_ms:.0f}ms, prep={prep_time_ms:.0f}ms, gen={gen_time_s:.2f}s, post={post_time_ms:.0f}ms, save={save_time_ms:.0f}ms)")

    return wave_file  # Str path