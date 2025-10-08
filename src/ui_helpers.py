"""
ui_helpers.py - FIXED: Import SkyrimNetConfig class + create singleton instance (no 'CONFIG' export error).
Supports CONFIG methods (load_voice_params_clamped, update_voice_parameter, etc.).
Handlers accept *args (Gradio multi-inputs).
UPDATED: Added safe helpers, prop lists, wrappers (safe_generate_wrapper, test_voice_wrapper, etc.),
cross-tab functions, and full hidden_generate_api for modular UI refactor. Enhanced existing handlers
for full param sets (e.g., all sliders in global/voice).
NEW: Generic safe_to_* converters (for arbitrary val - handles lists/None in runtime, e.g., from load_voice_params_for_edit).
FIXED: Inlined stub_wav_path function (self-contained stub for fallback audio - temp silent WAV).
FIXED: Removed duplicate wrappers; fixed test_voice_wrapper typo (no JSON return).
"""

import json
from loguru import logger
import random
import asyncio
import os
import torchaudio
from pathlib import Path
import tempfile
import torch  # For stub_wav_path (if needed for tensor)
from .config import CONFIG



# NEW: Safe helpers for scalar conversion (fixes list/None issues in sliders) - prop-based
def safe_float_value(prop, default=0.0):
    """Helper to fix list/None values from config parser - ensures float scalars for sliders."""
    val = getattr(CONFIG, prop, None)
    if val is None:
        return CONFIG.CONSTANTS.get(f'{prop.upper()}', default)
    if isinstance(val, (list, tuple)):
        val = val[0] if val else CONFIG.CONSTANTS.get(f'{prop.upper()}', default)
    return float(val)

def safe_int_value(prop, default=0):
    """Helper for int values (e.g., tokens)."""
    val = getattr(CONFIG, prop, None)
    if val is None:
        return CONFIG.CONSTANTS.get(f'{prop.upper()}', default)
    if isinstance(val, (list, tuple)):
        val = val[0] if val else CONFIG.CONSTANTS.get(f'{prop.upper()}', default)
    return int(val)

def safe_bool_value(prop, default=False):
    """Helper for bools/flags (checkboxes)."""
    val = getattr(CONFIG, prop, None)
    if val is None:
        return default
    if isinstance(val, (list, tuple)):
        val = val[0] if val else default
    return bool(val)

def safe_str_value(prop, default=''):
    """Helper for strings (e.g., methods, levels)."""
    val = getattr(CONFIG, prop, None)
    if val is None:
        return default
    if isinstance(val, (list, tuple)):
        val = val[0] if val else default
    return str(val)

# NEW: Generic converters for arbitrary runtime values (e.g., from load_voice_params_for_edit) - handles nested lists/None
def safe_to_float(val, default=0.0):
    """Generic: Convert val to float, handling list/tuple (take [0], recursive), None → default."""
    if val is None:
        return default
    if isinstance(val, (list, tuple)):
        if val:
            return safe_to_float(val[0], default)  # Recursive for nested (e.g., [ [1.0] ])
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        logger.warning(f"safe_to_float failed on {val} (type: {type(val)}), using default {default}")
        return default

def safe_to_int(val, default=0):
    """Generic: Like safe_to_float but to int."""
    if val is None:
        return default
    if isinstance(val, (list, tuple)):
        if val:
            return safe_to_int(val[0], default)
        return default
    try:
        return int(float(val))  # Allow float → int
    except (ValueError, TypeError):
        logger.warning(f"safe_to_int failed on {val}, using default {default}")
        return default

def safe_to_bool(val, default=False):
    """Generic: Like above, to bool (truthy/falsy)."""
    if val is None:
        return default
    if isinstance(val, (list, tuple)):
        if val:
            return safe_to_bool(val[0], default)
        return default
    return bool(val)

def safe_to_str(val, default=''):
    """Generic: To str."""
    if val is None:
        return default
    if isinstance(val, (list, tuple)):
        if val:
            return safe_to_str(val[0], default)
        return default
    return str(val)

# NEW: Prop types for JSON safety in init_load/refresh
float_props = ['exaggeration', 'cfg_weight', 'temperature', 'min_p', 'top_p', 'repetition_penalty', 'speaking_rate', 'eq_gain_db', 'max_gain', 'target_max', 'noise_floor_db', 'trim_threshold_db', 'eq_cutoff_hz', 'fade_ms', 'notch_low', 'notch_high', 'notch_gain_db', 'notch_gain_db_for_stretch']
int_props = ['max_new_tokens', 'min_new_tokens', 'max_cache_len']
bool_props = ['enable_memory_cache', 'enable_disk_cache', 'enable_denoising', 'enable_pre_adjustment', 'enable_post_processing', 'enable_light_stretch', 'enable_smoothing', 'enable_spectral_gating', 'notch_enabled', 'hp_enabled', 'timings_enabled', 'enable_timing_logs', 'log_step_times']
str_props = ['normalize_method', 'logging_level']


def update_api_status():
    """Return API/model status string."""
    try:
        model_ready = "Ready" if CONFIG.model else "Not Loaded"
        return f"API Status: Model={model_ready} | Voices={len(CONFIG.get_all_voices())} | Modified={CONFIG.is_modified}"
    except Exception as e:
        logger.error(f"Update status failed: {e}")
        return "Status: Error (check logs)"


def load_voice_params_ui(voice):
    """Load clamped params for test sliders (8-tuple). UPDATED: Use safe_to_* for outputs."""
    try:
        rate, eq, gain, target, noise, trim, notch, hp = CONFIG.load_voice_params_clamped(voice)
        params = CONFIG.get_voice_parameters(voice)
        status = "Loaded voice params"
        # Wrap with generics for safety (clean: uses new helpers)
        return (safe_to_float(rate), safe_to_float(eq), safe_to_float(gain), safe_to_float(target),
                safe_to_float(noise), safe_to_float(trim), safe_to_bool(notch), safe_to_bool(hp),
                params, status)
    except Exception as e:
        logger.error(f"Load voice params failed for {voice}: {e}")
        # Fallback with safe_to_*
        return (safe_to_float(1.0), safe_to_float(-8.0), safe_to_float(2.0), safe_to_float(0.6),
                safe_to_float(-30.0), safe_to_float(-28.0), safe_to_bool(True), safe_to_bool(True), {}, "Load failed")


def load_voice_params_for_edit(voice):
    """Load clamped params for edit sliders (8-tuple + info). UPDATED: Use safe_to_* for outputs."""
    try:
        rate, eq, gain, target, noise, trim, notch, hp = CONFIG.load_voice_params_clamped(voice)
        info = CONFIG.get_voice_info(voice)
        # Wrap with generics (clean integration)
        return (safe_to_float(rate), safe_to_float(eq), safe_to_float(gain), safe_to_float(target),
                safe_to_float(noise), safe_to_float(trim), safe_to_bool(notch), safe_to_bool(hp),
                safe_to_str(info))
    except Exception as e:
        logger.error(f"Load edit params failed for {voice}: {e}")
        return (safe_to_float(1.0), safe_to_float(-8.0), safe_to_float(2.0), safe_to_float(0.6),
                safe_to_float(-30.0), safe_to_float(-28.0), safe_to_bool(False), safe_to_bool(False),
                safe_to_str("Default voice - Load failed"))


def update_voice_params_ui(voice, rate, eq, gain, target, noise, trim, notch, hp, temp=None, cfg=None, min_p=None, top_p=None, rep=None, exagg=None):
    """Apply voice updates (clamped, memory-only). UPDATED: Use safe_to_* on inputs; return safe types."""
    try:
        # Use generics on inputs (clean: handles if passed list from UI/event)
        rate = safe_to_float(rate)
        eq = safe_to_float(eq)
        gain = safe_to_float(gain)
        target = safe_to_float(target)
        noise = safe_to_float(noise)
        trim = safe_to_float(trim)
        notch = safe_to_bool(notch)
        hp = safe_to_bool(hp)
        temp = safe_to_float(temp)
        cfg = safe_to_float(cfg)
        min_p = safe_to_float(min_p)
        top_p = safe_to_float(top_p)
        rep = safe_to_float(rep)
        exagg = safe_to_float(exagg)

        # Processing params (core) - unchanged, but now inputs safe
        CONFIG.update_voice_parameter(voice, 'speaking_rate', CONFIG.clamp_value('speaking_rate', rate))
        CONFIG.update_voice_parameter(voice, 'eq_gain_db', CONFIG.clamp_value('eq_gain_db', eq))
        CONFIG.update_voice_parameter(voice, 'max_gain', CONFIG.clamp_value('max_gain', gain))
        CONFIG.update_voice_parameter(voice, 'target_max', CONFIG.clamp_value('target_max', target))
        CONFIG.update_voice_parameter(voice, 'noise_floor_db', CONFIG.clamp_value('noise_floor_db', noise))
        CONFIG.update_voice_parameter(voice, 'trim_threshold_db', CONFIG.clamp_value('trim_threshold_db', trim))
        CONFIG.update_voice_parameter(voice, 'notch_enabled', notch)
        CONFIG.update_voice_parameter(voice, 'hp_enabled', hp)

        # Gen overrides (optional) - safe inputs
        if temp is not None:
            CONFIG.update_voice_parameter(voice, 'temperature', CONFIG.clamp_value('temperature', temp))
        if cfg is not None:
            CONFIG.update_voice_parameter(voice, 'cfg_weight', CONFIG.clamp_value('cfg_weight', cfg))
        if min_p is not None:
            CONFIG.update_voice_parameter(voice, 'min_p', CONFIG.clamp_value('min_p', min_p))
        if top_p is not None:
            CONFIG.update_voice_parameter(voice, 'top_p', CONFIG.clamp_value('top_p', top_p))
        if rep is not None:
            CONFIG.update_voice_parameter(voice, 'repetition_penalty', CONFIG.clamp_value('repetition_penalty', rep))
        if exagg is not None:
            CONFIG.update_voice_parameter(voice, 'exaggeration', CONFIG.clamp_value('exaggeration', exagg))

        CONFIG._is_modified = True  # Flag as dirty
        status = "Voice params updated (memory-only)"
        info = CONFIG.get_voice_info(voice)
        return status, update_api_status(), safe_to_str(info)  # Safe str for Markdown
    except Exception as e:
        logger.error(f"Update voice failed for {voice}: {e}")
        return "Update failed", update_api_status(), safe_to_str("Error applying changes")


def handle_global_flags_change(mem_cache, disk_cache, denoise, pre_adjust=None, post_proc=None, light_stretch=None, smoothing=None, spectral_gating=None, notch=None, hp=None, timings=None, timing_logs=None, step_times=None):
    """Handle global flags checkbox changes. UPDATED: More flags via kwargs for flexibility."""
    try:
        # Use safe_to_* on inputs (clean safety)
        mem_cache = safe_to_bool(mem_cache)
        disk_cache = safe_to_bool(disk_cache)
        denoise = safe_to_bool(denoise)
        pre_adjust = safe_to_bool(pre_adjust)
        post_proc = safe_to_bool(post_proc)
        light_stretch = safe_to_bool(light_stretch)
        smoothing = safe_to_bool(smoothing)
        spectral_gating = safe_to_bool(spectral_gating)
        notch = safe_to_bool(notch)
        hp = safe_to_bool(hp)
        timings = safe_to_bool(timings)
        timing_logs = safe_to_bool(timing_logs)
        step_times = safe_to_bool(step_times)

        CONFIG.flags['enable_memory_cache'] = mem_cache
        CONFIG.flags['enable_disk_cache'] = disk_cache
        CONFIG.flags['enable_denoising'] = denoise
        if pre_adjust is not None:
            CONFIG.flags['enable_pre_adjustment'] = pre_adjust
        if post_proc is not None:
            CONFIG.flags['enable_post_processing'] = post_proc
        if light_stretch is not None:
            CONFIG.flags['enable_light_stretch'] = light_stretch
        if smoothing is not None:
            CONFIG.flags['enable_smoothing'] = smoothing
        if spectral_gating is not None:
            CONFIG.flags['enable_spectral_gating'] = spectral_gating
        if notch is not None:
            CONFIG.flags['notch_enabled'] = notch
        if hp is not None:
            CONFIG.flags['hp_enabled'] = hp
        if timings is not None:
            CONFIG.flags['timings_enabled'] = timings
        if timing_logs is not None:
            CONFIG.flags['enable_timing_logs'] = timing_logs
        if step_times is not None:
            CONFIG.flags['log_step_times'] = step_times
        # Sync to instance
        CONFIG.enable_memory_cache = mem_cache
        CONFIG.enable_disk_cache = disk_cache
        CONFIG.enable_denoising = denoise
        logger.debug(f"Global flags updated: mem={mem_cache}, disk={disk_cache}, denoise={denoise} (extras provided)")
        return "Flags applied", update_api_status()
    except Exception as e:
        logger.error(f"Global flags change failed: {e}")
        return "Flags update failed", update_api_status()


def handle_global_params_change(*args):
    """Apply global param changes (clamped). UPDATED: *args for full sliders (e.g., rate, eq, gain, ...); use safe_to_*."""
    try:
        param_map = {
            0: ('speaking_rate', float), 1: ('eq_gain_db', float), 2: ('max_gain', float), 3: ('target_max', float),
            4: ('noise_floor_db', float), 5: ('trim_threshold_db', float), 6: ('normalize_method', str),
            7: ('eq_cutoff_hz', float), 8: ('fade_ms', float), 9: ('notch_low', float), 10: ('notch_high', float),
            11: ('notch_gain_db', float), 12: ('notch_gain_db_for_stretch', float), 13: ('temperature', float),
            14: ('exaggeration', float), 15: ('cfg_weight', float), 16: ('min_p', float), 17: ('top_p', float),
            18: ('repetition_penalty', float), 19: ('max_new_tokens', int), 20: ('min_new_tokens', int), 21: ('max_cache_len', int)
        }
        for i, value in enumerate(args):
            if i in param_map:
                key, conv_type = param_map[i]
                # Safe convert (handles list input)
                converted = safe_to_float(value) if conv_type == float else safe_to_int(value) if conv_type == int else safe_to_str(value)
                clamped = CONFIG.clamp_value(key, converted)
                setattr(CONFIG, key, clamped)
                CONFIG.defaults[key] = converted  # Sync raw if needed
        CONFIG._is_modified = True
        logger.debug(f"Global params applied from {len(args)} inputs")
        return "Global params updated", update_api_status()
    except Exception as e:
        logger.error(f"Global params change failed: {e}")
        return "Params update failed", update_api_status()


def handle_token_limits_change(max_tokens, min_tokens, cache_len):
    """Handle token limits. UPDATED: Includes min_tokens; safe_to_int."""
    try:
        max_tokens = safe_to_int(max_tokens)
        min_tokens = safe_to_int(min_tokens)
        cache_len = safe_to_int(cache_len)
        CONFIG.max_new_tokens = max(CONFIG.CONSTANTS['MAX_NEW_TOKENS_MIN'], min(CONFIG.CONSTANTS['MAX_NEW_TOKENS_MAX'], max_tokens))
        CONFIG.min_new_tokens = max(CONFIG.CONSTANTS['MIN_NEW_TOKENS_MIN'], min(CONFIG.CONSTANTS['MIN_NEW_TOKENS_MAX'], min_tokens))
        CONFIG.max_cache_len = max(CONFIG.CONSTANTS['MAX_CACHE_LEN_MIN'], min(CONFIG.CONSTANTS['MAX_CACHE_LEN_MAX'], cache_len))
        CONFIG.defaults['max_new_tokens'] = CONFIG.max_new_tokens
        CONFIG.defaults['min_new_tokens'] = CONFIG.min_new_tokens
        CONFIG.defaults['max_cache_len'] = CONFIG.max_cache_len
        logger.debug(f"Tokens updated: max={CONFIG.max_new_tokens}, min={CONFIG.min_new_tokens}, cache={CONFIG.max_cache_len}")
        return "Token limits applied", update_api_status()
    except Exception as e:
        logger.error(f"Token limits change failed: {e}")
        return "Limits update failed", update_api_status()


def create_global_param_handler(param_type):
    """Create handler for individual global slider changes (e.g., rate.change). UPDATED: Uses safe_to_float/int."""
    def handler(*args, **kwargs):
        # args[0] = changed value (Gradio passes first input as primary); ignore others
        if args:
            changed_value = args[0]
            logger.debug(f"Global {param_type} changed to: {changed_value} (args len={len(args)})")
            # Partial apply with safe clamp
            if param_type in float_props:
                safe_val = safe_to_float(changed_value)
            elif param_type in int_props:
                safe_val = safe_to_int(changed_value)
            else:
                safe_val = safe_to_str(changed_value)
            setattr(CONFIG, param_type, safe_val)
        return f"{param_type.capitalize()} updated (partial—apply globals for full sync)", update_api_status()
    return handler


def save_all_config():
    """Save to file (with backup)."""
    try:
        success, msg = CONFIG.save_config(create_backup=True)
        if success:
            logger.info("Full config saved to file")
        return msg, update_api_status() if success else "Save failed—check logs"
    except Exception as e:
        logger.error(f"Save config failed: {e}")
        return "Save error", update_api_status()


def reset_all_config():
    """Reset to original state."""
    try:
        success, msg = CONFIG.reset_to_original()
        if success:
            logger.info("Config reset to original")
        return msg, update_api_status() if success else "Reset failed"
    except Exception as e:
        logger.error(f"Reset config failed: {e}")
        return "Reset error", update_api_status()


def reload_all_config():
    """Reload from file."""
    try:
        CONFIG.reload()
        logger.info("Config reloaded from file")
        return "Reloaded successfully", update_api_status()
    except Exception as e:
        logger.error(f"Reload config failed: {e}")
        return "Reload error", update_api_status()


def apply_json_changes(json_str):
    """Apply memory state from JSON. UPDATED: Applies clamped values for props."""
    try:
        data = json.loads(json_str) if json_str else {}
        success, msg = CONFIG.import_memory_state(data)
        if success and 'clamped_values' in data:
            # Apply clamped (e.g., from UI edit)
            for k, v in data['clamped_values'].items():
                if k in float_props:
                    CONFIG.update_voice_parameter('default', k, safe_float_value(k, safe_to_float(v)))  # Global as default
                elif k in int_props:
                    CONFIG.update_voice_parameter('default', k, safe_int_value(k, safe_to_int(v)))
                elif k in bool_props:
                    CONFIG.update_voice_parameter('default', k, safe_bool_value(k, safe_to_bool(v)))
                elif k in str_props:
                    CONFIG.update_voice_parameter('default', k, safe_str_value(k, safe_to_str(v)))
        if success:
            logger.info("JSON memory state imported")
        return json_str, safe_to_str(msg), update_api_status() if success else "Apply failed—invalid JSON"
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode failed: {e}")
        return json_str, "Invalid JSON format", update_api_status()
    except Exception as e:
        logger.error(f"JSON apply failed: {e}")
        return json_str, "Apply error", update_api_status()


def refresh_config_json():
    """Generate current config JSON for display. UPDATED: Includes clamped_values using prop lists."""
    try:
        clamped = {
            k: (safe_float_value(k) if k in float_props else
                safe_int_value(k) if k in int_props else
                safe_str_value(k) if k in str_props else
                safe_bool_value(k)) for k in CONFIG._clamped_keys if hasattr(CONFIG, k)
        }
        state = {
            'tts_name': CONFIG.tts_name,
            'sr': CONFIG.sr,
            'multilingual': CONFIG.multilingual,
            'voices': CONFIG.get_all_voices(),
            'modified': CONFIG.is_modified,
            'clamped_values': clamped,
            'defaults': {k: v for k, v in CONFIG.defaults.items() if k in ['max_new_tokens', 'speaking_rate', 'exaggeration']},  # Partial for UI
            'flags': {k: v for k, v in CONFIG.flags.items() if 'enable' in k},
            'memory_state_keys': list(CONFIG._memory_state.keys())
        }
        return json.dumps(state, indent=2, default=str)
    except Exception as e:
        logger.error(f"Refresh JSON failed: {e}")
        return json.dumps({"error": "JSON generation failed"}, indent=2)



# NEW: stub_wav_path (persistent fallback - creates/reuses stub_silent.wav in output_temp/)
# Generates a 3s silence WAV for error cases; checks existence before creating.
# Returns (path, status); optional empty_dict=True for 3rd {} value (e.g., for test_wrapper params)
STUB_DIR = Path("output_temp")
STUB_PATH = STUB_DIR / "stub_silent.wav"

def stub_wav_path(empty_dict=False):
    """Generate/reuse a persistent silent WAV file for UI fallbacks (3s silence @ 24kHz)."""
    try:
        # Ensure dir exists
        STUB_DIR.mkdir(exist_ok=True)

        # Check if stub exists and is valid (size >0)
        if STUB_PATH.exists() and STUB_PATH.stat().st_size > 0:
            try:
                info = torchaudio.info(STUB_PATH)  # Verify format
                duration = info.num_frames / info.sample_rate
                if duration == 3.0:  # Expected duration
                    status = f"Reused stub: {STUB_PATH.name} ({duration:.1f}s silence @ {info.sample_rate}Hz) - Model not ready"
                    path = str(STUB_PATH)
                    logger.debug(f"Stub reused: {path}")
                    if empty_dict:
                        return path, status, {}
                    return path, status
            except Exception as e:
                logger.warning(f"Invalid stub file {STUB_PATH}: {e} - regenerating")

        # Create the stub if missing/invalid
        sr = CONFIG.sr if hasattr(CONFIG, 'sr') else 24000  # Fallback to common TTS SR
        duration = 3.0  # 3s silence
        samples = int(sr * duration)
        silence = torch.zeros(1, samples, dtype=torch.float32)  # Mono silence

        torchaudio.save(str(STUB_PATH), silence, sr, format='wav')
        info = torchaudio.info(str(STUB_PATH))
        status = f"Created stub: {STUB_PATH.name} ({duration:.1f}s silence @ {sr}Hz) - Model not ready"
        path = str(STUB_PATH)
        logger.info(f"Stub created/persisted: {path}")

        if empty_dict:
            return path, status, {}
        return path, status

    except Exception as e:
        logger.error(f"Stub WAV creation failed: {e}")
        # Fallback to non-existent path (UI will show no audio + error)
        fallback_path = "stub_silent.wav"  # Relative, non-existent indicator
        error_status = "Stub unavailable - check permissions/logs"
        if empty_dict:
            return fallback_path, error_status, {}
        return fallback_path, error_status


def test_voice_wrapper(test_text, test_voice, tts_state):
    from ui import generate_audio_test
    """Simplified wrapper for ui.py dummy (matches 3 inputs → 3 outputs: audio, status, voices)."""
    try:
        # Type safety
        test_text = safe_to_str(test_text)
        test_voice = safe_to_str(test_voice) if test_voice else 'default'
        # tts_name = safe_to_str(tts_state.value) if tts_state else 'default'

        if not CONFIG.model:
            path, status = stub_wav_path()
            voices = CONFIG.get_all_voices() or ['default']
            return path, status, voices

        # Update params for voice (safe_to_*)
        rate = safe_float_value('speaking_rate')
        eq = safe_float_value('eq_gain_db')
        gain = safe_float_value('max_gain')
        target = safe_float_value('target_max')
        noise = safe_float_value('noise_floor_db')
        trim = safe_float_value('trim_threshold_db')
        notch = safe_bool_value('notch_enabled')
        hp = safe_bool_value('hp_enabled')
        seed = random.randint(0, 2**32 - 1)

        CONFIG.update_voice_parameter(test_voice, 'speaking_rate', rate)
        CONFIG.update_voice_parameter(test_voice, 'eq_gain_db', eq)
        CONFIG.update_voice_parameter(test_voice, 'max_gain', gain)
        CONFIG.update_voice_parameter(test_voice, 'target_max', target)
        CONFIG.update_voice_parameter(test_voice, 'noise_floor_db', noise)
        CONFIG.update_voice_parameter(test_voice, 'trim_threshold_db', trim)
        CONFIG.update_voice_parameter(test_voice, 'notch_enabled', notch)
        CONFIG.update_voice_parameter(test_voice, 'hp_enabled', hp)

        kwargs = {'text': test_text, 'seed_num': seed, 'voice_name': test_voice}
        result = generate_audio_test(**kwargs)
        path, status_raw, params_dict = (result[0], result[1], result[2]) if len(result) == 3 else (result, None, {})
        path = str(path) if path else stub_wav_path()[0]
        status = safe_to_str(status_raw) if status_raw is not None else "✅ Test generated"
        if isinstance(status_raw, (int, float)):
            status = f"✅ Test | Code: {status_raw}"
        # status += f" | Voice: {test_voice} | TTS: {tts_name}"
        voices = CONFIG.get_all_voices() or ['default']
        logger.info(f"Test wrapper success: {status}")
        return path, status, voices
    except Exception as e:
        logger.error(f"Test wrapper failed: {e}")
        voices = ['default']
        return stub_wav_path()[0], f"Error: {str(e)}", voices

