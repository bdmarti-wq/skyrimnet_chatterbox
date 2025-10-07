#src/ui.py
import asyncio
import os
import random  # For seed fallback
import warnings

import gradio as gr
from loguru import logger
from pathlib import Path

# CONFIG singleton
from config import SkyrimNetConfig, ENABLE_DISK_CACHE
from src.cache import ENABLE_MEMORY_CACHE
from src.model import load_model

CONFIG = SkyrimNetConfig()

# ui_helpers (full: safe_*, handlers, stub_wav_path exclusive for fallbacks)
from src.ui_helpers import (
    update_api_status, safe_to_float, safe_to_int, safe_to_bool, safe_to_str, safe_float_value,
    load_voice_params_for_edit, handle_global_params_change, refresh_config_json, apply_json_changes,
    save_all_config, reset_all_config, reload_all_config, update_voice_params_ui,
    load_voice_params_ui, handle_token_limits_change, create_global_param_handler, stub_wav_path, test_voice_wrapper
)


# Hidden API supports communication with SkyrimNet via a Zonos bridge
from src.bridge_ui import setup_bridge_api

# generation_utils (single try: cores only; log fail - no local stubs/fallbacks)
try:
    from src.generate_audio import generate_audio
    logger.info("generation_audio imported successfully: generate_internal pipeline ready")
except ImportError as e:
    logger.error(f"generation_audio import failed: {e} - Fix deps (e.g., pip install librosa scipy) or path (src/cache_utils)")
    raise

# Warnings- Review These
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", message=r"Reference mel length is not equal to 2 \* reference token length\.", category=UserWarning)
warnings.filterwarnings("ignore", message="torchaudio._backend.utils.info has been deprecated")  # Suppress stub_wav_path warning

def generate_audio_test(model, text, language_id="en", audio_prompt_path=None, exaggeration=0.5, temperature=0.8, seed_num=0,
             cfgw=0, min_p=0.05, top_p=1.0, repetition_penalty=1.2, cache_uuid=0):
    """
    UI shell: Keeps server-compatible sig. Minimal setup, delegates to generate_audio.
    """
    if model is None:
        model = load_model()

    if not text:
        logger.warning("No text provided – returning empty")
        return ""  # Or dummy path

    if seed_num != 0:
        from src.generate_audio import set_seed  # Lazy
        set_seed(int(seed_num))

    # Lazy import + call (no global exposure)
    from src.generate_audio import generate_audio
    result = generate_audio(
        model, text, audio_prompt_path,  # None as-is (no "" coercion)
        float(exaggeration), int(cache_uuid),
        float(temperature), float(cfgw), float(min_p), float(top_p), float(repetition_penalty),
        language_id, int(seed_num),
        ENABLE_MEMORY_CACHE, ENABLE_DISK_CACHE
    )
    return result



# Tab Load Functions (lazy: CONFIG direct, no State – user clicks "Load/Refresh" to populate)
def load_test_tab():
    """Load Voice Test tab components from CONFIG (lazy on btn click)."""
    voice_choices = CONFIG.get_all_voices() or ['default']
    if 'default' not in voice_choices:
        voice_choices.append('default')
    test_text_value = "Testing shared config voice parameters."
    params = {
        'speaking_rate': CONFIG.get_value('speaking_rate', 1.0),
        'eq_gain_db': CONFIG.get_value('eq_gain_db', -8.0),
        'max_gain': CONFIG.get_value('max_gain', 2.0),
        'target_max': CONFIG.get_value('target_max', 0.6),
        'noise_floor_db': CONFIG.get_value('noise_floor_db', -30.0),
        'trim_threshold_db': CONFIG.get_value('trim_threshold_db', -28.0),
        'notch_enabled': CONFIG.get_value('notch_enabled', True),
        'hp_enabled': CONFIG.get_value('hp_enabled', True),
    }
    test_params = {"Loaded defaults": params}  # Sample dict
    test_status_value = "Values loaded from CONFIG – test with custom params if needed."
    logger.debug("Voice Test tab loaded from CONFIG")
    return (
        gr.update(choices=voice_choices),  # test_voice_dropdown
        test_text_value,  # test_text
        42,  # test_seed
        params['speaking_rate'], params['eq_gain_db'], params['max_gain'], params['target_max'],
        params['noise_floor_db'], params['trim_threshold_db'], params['notch_enabled'], params['hp_enabled'],
        test_params, test_status_value
    )

def load_editor_tab():
    """Load Voice Editor tab from CONFIG (lazy)."""
    voice_choices = CONFIG.get_all_voices() or ['default']
    if 'default' not in voice_choices:
        voice_choices.append('default')
    # Defaults from CONFIG (global, as voice-specific needs select first)
    params = {
        'speaking_rate': CONFIG.get_value('speaking_rate', 1.0),
        'eq_gain_db': CONFIG.get_value('eq_gain_db', -8.0),
        'max_gain': CONFIG.get_value('max_gain', 2.0),
        'target_max': CONFIG.get_value('target_max', 0.6),
        'noise_floor_db': CONFIG.get_value('noise_floor_db', -30.0),
        'trim_threshold_db': CONFIG.get_value('trim_threshold_db', -28.0),
        'temperature': CONFIG.get_value('temperature', 0.65),
        'exaggeration': CONFIG.get_value('exaggeration', 1.0),
        'cfg_weight': CONFIG.get_value('cfg_weight', 0.45),
        'min_p': CONFIG.get_value('min_p', 0.1),
        'top_p': CONFIG.get_value('top_p', 1.0),
        'repetition_penalty': CONFIG.get_value('repetition_penalty', 1.5),
        'notch_enabled': CONFIG.get_value('notch_enabled', True),
        'hp_enabled': CONFIG.get_value('hp_enabled', True),
    }
    voice_info_md = "Select a voice and click Load to edit specific params (globals shown now)."
    voice_status = "Ready – load voice-specific values."
    logger.debug("Voice Editor tab loaded from CONFIG globals")
    return (
        gr.update(choices=voice_choices),  # edit_voice_dropdown
        params['speaking_rate'], params['eq_gain_db'], params['max_gain'], params['target_max'],
        params['noise_floor_db'], params['trim_threshold_db'], params['temperature'], params['exaggeration'],
        params['cfg_weight'], params['min_p'], params['top_p'], params['repetition_penalty'],
        params['notch_enabled'], params['hp_enabled'],  # voice_* components
        voice_info_md, voice_status
    )

def load_global_tab():
    """Load Global Config tab from CONFIG (lazy)."""
    global_dict = {
        'speaking_rate': safe_to_float(CONFIG.get_value('speaking_rate', 1.0)),
        'eq_gain_db': safe_to_float(CONFIG.get_value('eq_gain_db', -8.0)),
        'max_gain': safe_to_float(CONFIG.get_value('max_gain', 2.0)),
        'target_max': safe_to_float(CONFIG.get_value('target_max', 0.6)),
        'noise_floor_db': safe_to_float(CONFIG.get_value('noise_floor_db', -30.0)),
        'trim_threshold_db': safe_to_float(CONFIG.get_value('trim_threshold_db', -28.0)),
        'eq_cutoff_hz': safe_to_float(CONFIG.get_value('eq_cutoff_hz', 300.0)),
        'fade_ms': safe_to_float(CONFIG.get_value('fade_ms', 50.0)),
        'notch_low': safe_to_float(CONFIG.get_value('notch_low', 100.0)),
        'notch_high': safe_to_float(CONFIG.get_value('notch_high', 5000.0)),
        'notch_gain_db': safe_to_float(CONFIG.get_value('notch_gain_db', -20.0)),
        'notch_gain_db_for_stretch': safe_to_float(CONFIG.get_value('notch_gain_db_for_stretch', -10.0)),
        'temperature': safe_to_float(CONFIG.get_value('temperature', 0.65)),
        'exaggeration': safe_to_float(CONFIG.get_value('exaggeration', 1.0)),
        'cfg_weight': safe_to_float(CONFIG.get_value('cfg_weight', 0.45)),
        'min_p': safe_to_float(CONFIG.get_value('min_p', 0.1)),
        'top_p': safe_to_float(CONFIG.get_value('top_p', 1.0)),
        'repetition_penalty': safe_to_float(CONFIG.get_value('repetition_penalty', 1.5)),
        'max_new_tokens': safe_to_int(CONFIG.get_value('max_new_tokens', 1499)),
        'min_new_tokens': safe_to_int(CONFIG.get_value('min_new_tokens', 1)),
        'max_cache_len': safe_to_int(CONFIG.get_value('max_cache_len', 1024)),
        'normalize_method': safe_to_str(CONFIG.get_value('normalize_method', 'rms')),
        'logging_level': safe_to_str(CONFIG.get_value('logging_level', 'INFO')),
        'enable_memory_cache': safe_to_bool(CONFIG.get_value('enable_memory_cache', True)),
        'enable_disk_cache': safe_to_bool(CONFIG.get_value('enable_disk_cache', False)),
        'enable_denoising': safe_to_bool(CONFIG.get_value('enable_denoising', True)),
    }
    global_status = "Globals loaded from CONFIG – edit and apply."
    logger.debug("Global Config tab loaded from CONFIG")
    return (
        global_dict['speaking_rate'], global_dict['eq_gain_db'], global_dict['max_gain'], global_dict['target_max'],
        global_dict['noise_floor_db'], global_dict['trim_threshold_db'], global_dict['eq_cutoff_hz'], global_dict['fade_ms'],
        global_dict['notch_low'], global_dict['notch_high'], global_dict['notch_gain_db'], global_dict['notch_gain_db_for_stretch'],
        global_dict['temperature'], global_dict['exaggeration'], global_dict['cfg_weight'],
        global_dict['min_p'], global_dict['top_p'], global_dict['repetition_penalty'],
        global_dict['max_new_tokens'], global_dict['min_new_tokens'], global_dict['max_cache_len'],
        global_dict['normalize_method'], global_dict['logging_level'],
        global_dict['enable_memory_cache'], global_dict['enable_disk_cache'], global_dict['enable_denoising'],
        global_status  # All global components
    )

def create_ui():
    """State-free UI: Tabs with lazy load (user clicks 'Load/Refresh' for CONFIG values). Generate default."""
    with gr.Blocks(title="SkyrimNet Chatterbox", theme=gr.themes.Soft()) as demo:
        # Global Status (top-level, updated via btns)
        api_status_md = gr.Markdown(value=update_api_status())

        gr.Markdown("# SkyrimNet Chatterbox TTS UI\nSimplified tabbed interface for generation, testing, and config editing.", elem_id="title-md")

        # Tabs (Generate default; others lazy-load via btns)
        with gr.Tabs(selected="generate") as tabs:
            # Tab 1: Generate Audio (no state/lazy – always ready)
            with gr.TabItem("🎤 Generate Audio", id="generate", elem_id="tab-generate"):
                with gr.Row():
                    text_input = gr.Textbox(label="Input Text", placeholder="Enter text to generate...", lines=3, value="")
                    # Dropdown (static defaults; refresh via manual if needed)
                    generate_voice_choices = CONFIG.get_all_voices() or ['default']
                    if 'default' not in generate_voice_choices:
                        generate_voice_choices.append('default')
                    generate_voice_dropdown = gr.Dropdown(
                        label="Voice", choices=generate_voice_choices, value='default', allow_custom_value=False
                    )
                ref_audio = gr.Audio(label="Reference Audio (optional)", sources="upload", type="filepath")
                generate_btn = gr.Button("Generate Audio", variant="primary")
                audio_output = gr.Audio(label="Generated Audio")
                generate_status = gr.Textbox(label="Status", interactive=False, value="Ready")

                # Events (UI-only, no state)
                generate_btn.click(
                    fn=safe_generate_async,
                    inputs=[text_input, generate_voice_dropdown, ref_audio],
                    outputs=[audio_output, generate_status],
                    js="", show_progress=True, concurrency_limit=1
                )

                # Voice .change (update self from CONFIG – no state)
                def update_generate_voice_choices(voice):
                    choices = CONFIG.get_all_voices() or ['default']
                    if 'default' not in choices:
                        choices.append('default')
                    return gr.update(choices=choices, value=safe_to_str(voice, 'default'))

                generate_voice_dropdown.change(
                    fn=update_generate_voice_choices,
                    inputs=[generate_voice_dropdown],
                    outputs=[generate_voice_dropdown],
                    js="", show_progress=False
                )

            # Tab 2: Voice Test (lazy load btn)
            with gr.TabItem("🔊 Voice Test", id="test", elem_id="tab-test"):
                gr.Markdown("### 🔬 Test Voice Parameters (Click Load to sync from CONFIG)")
                load_test_btn = gr.Button("Load/Refresh Values", variant="secondary")  # Lazy trigger

                with gr.Row():
                    with gr.Column(scale=1):
                        # Dropdown (static; updated on load btn)
                        test_voice_dropdown = gr.Dropdown(
                            label="Test Voice", choices=['default'], value='default', allow_custom_value=False
                        )
                        test_text = gr.Textbox(value="Testing shared config voice parameters.", label="Test Text", lines=3)
                        test_seed = gr.Number(value=42, label="Test Seed", minimum=0, maximum=99999)
                        with gr.Group():
                            gr.Markdown("### Voice Processing Parameters")
                            test_rate = gr.Slider(0.5, 2.0, 1.0, step=0.05, label="🗣️ Speaking Rate")
                            test_eq = gr.Slider(-20.0, 0.0, -8.0, step=0.5, label="🎛️ EQ Gain (dB)")
                            test_gain = gr.Slider(1.0, 5.0, 2.0, step=0.1, label="📈 Max Gain")
                            test_target = gr.Slider(0.0, 1.0, 0.6, step=0.05, label="🎯 Target Max")
                            test_noise = gr.Slider(-60.0, -20.0, -30.0, step=1.0, label="🔇 Noise Floor")
                            test_trim = gr.Slider(-40.0, -20.0, -28.0, step=1.0, label="✂️ Trim Threshold")
                            with gr.Row():
                                test_notch = gr.Checkbox(value=True, label="🛡️ Notch Filter")
                                test_hp = gr.Checkbox(value=True, label="🔊 High-Pass")
                        test_btn = gr.Button("🎵 Test Generation", variant="primary")

                    with gr.Column(scale=2):
                        test_audio = gr.Audio(label="Test Audio Output")
                        test_status = gr.Markdown(value="Click Load/Refresh then Test")
                        test_params = gr.JSON(value={}, label="Parameters Applied")

                # Load btn (lazy: populates from CONFIG)
                load_test_btn.click(
                    fn=load_test_tab,
                    inputs=[],  # No inputs
                    outputs=[test_voice_dropdown, test_text, test_seed, test_rate, test_eq, test_gain, test_target, test_noise, test_trim, test_notch, test_hp, test_params, test_status],
                    js="", show_progress=False
                )

                # Test btn (UI-only)
                test_btn.click(
                    fn=test_voice_wrapper,
                    inputs=[test_voice_dropdown, test_text, test_seed, test_rate, test_eq, test_gain, test_target, test_noise, test_trim, test_notch, test_hp],
                    outputs=[test_audio, test_status, test_params],
                    js="", show_progress=True, concurrency_limit=1
                )

                # Voice .change (update self from CONFIG)
                def update_test_voice_choices(voice):
                    choices = CONFIG.get_all_voices() or ['default']
                    if 'default' not in choices:
                        choices.append('default')
                    return gr.update(choices=choices, value=safe_to_str(voice, 'default'))

                test_voice_dropdown.change(
                    fn=update_test_voice_choices,
                    inputs=[test_voice_dropdown],
                    outputs=[test_voice_dropdown],
                    js="", show_progress=False
                )

            # Tab 3: Voice Editor (lazy load btn)
            with gr.TabItem("🎤 Voice Editor", id="editor", elem_id="tab-editor"):
                gr.Markdown("### Edit Per-Voice Params (Click Load to sync globals, then select voice and load specific)")
                load_editor_btn = gr.Button("Load/Refresh Values", variant="secondary")  # Lazy

                edit_voice_dropdown = gr.Dropdown(
                    label="Select Voice to Edit", choices=['default'], value='default', allow_custom_value=False
                )
                load_voice_btn = gr.Button("Load Selected Voice Params")  # Per-voice after dropdown

                # Inputs (static defaults; updated on load)
                with gr.Column():
                    voice_rate_num = gr.Number(label="🗣️ Speaking Rate (x)", value=1.0, precision=2)
                    voice_eq_num = gr.Number(label="🎛️ EQ Gain (dB)", value=-8.0, precision=1)
                    voice_gain_num = gr.Number(label="📈 Max Gain (x)", value=2.0, precision=1)
                    voice_target_num = gr.Number(label="🎯 Target Max", value=0.6, precision=2)
                    voice_noise_num = gr.Number(label="🔇 Noise Floor (dB)", value=-30.0)
                    voice_trim_num = gr.Number(label="✂️ Trim Threshold (dB)", value=-28.0)
                    voice_temp_num = gr.Number(label="🌡️ Temperature", value=0.65, precision=2)
                    voice_exagg_num = gr.Number(label="🎭 Exaggeration", value=1.0, precision=2)
                    voice_cfg_num = gr.Number(label="⚖️ CFG Weight", value=0.45, precision=2)
                    voice_min_p_num = gr.Number(label="⚡ Min P", value=0.1, precision=2)
                    voice_top_p_num = gr.Number(label="🔝 Top P", value=1.0, precision=2)
                    voice_rep_num = gr.Number(label="🔄 Repetition Penalty", value=1.5, precision=1)
                    voice_notch_cb = gr.Checkbox(label="🛡️ Enable Notch Filter", value=True)
                    voice_hp_cb = gr.Checkbox(label="🔊 Enable High-Pass", value=True)

                voice_apply_btn = gr.Button("Apply Voice Changes", variant="secondary")
                voice_info_md = gr.Markdown(value="Click Load/Refresh for globals, select voice, then Load for specifics.")
                voice_status = gr.Textbox(label="Status", interactive=False, value="Ready – load values")

                # Global load btn (lazy globals)
                load_editor_btn.click(
                    fn=load_editor_tab,
                    inputs=[],  # No inputs
                    outputs=[edit_voice_dropdown, voice_rate_num, voice_eq_num, voice_gain_num, voice_target_num, voice_noise_num,
                             voice_trim_num, voice_temp_num, voice_exagg_num, voice_cfg_num, voice_min_p_num,
                             voice_top_p_num, voice_rep_num, voice_notch_cb, voice_hp_cb, voice_info_md, voice_status],
                    js="", show_progress=False
                )

                # Per-voice load (after dropdown + global load)
                def load_voice_to_display(voice):
                    voice = safe_to_str(voice) or 'default'
                    current_voices = CONFIG.get_all_voices() or ['default']
                    if voice not in current_voices:
                        voice = current_voices[0] if current_voices else 'default'
                    try:
                        rate, eq, gain, target, noise, trim, notch, hp, info = load_voice_params_for_edit(voice)
                        gen_params = CONFIG.get_voice_parameters(voice) or {}
                        info_md = f"**Loaded '{voice}'** | Info: {safe_to_str(info)}"
                        status = "Params loaded – edit and apply."
                        logger.info(f"Voice {voice} loaded via helpers")
                        return (
                            safe_to_float(rate), safe_to_float(eq), safe_to_float(gain), safe_to_float(target),
                            safe_to_float(noise), safe_to_float(trim),
                            gen_params.get('temperature', safe_float_value('temperature')),
                            gen_params.get('exaggeration', safe_float_value('exaggeration')),
                            gen_params.get('cfg_weight', safe_float_value('cfg_weight')),
                            gen_params.get('min_p', safe_float_value('min_p')),
                            gen_params.get('top_p', safe_float_value('top_p')),
                            gen_params.get('repetition_penalty', safe_float_value('repetition_penalty')),
                            safe_to_bool(notch), safe_to_bool(hp),
                            info_md, status
                        )
                    except Exception as e:
                        logger.error(f"Load voice failed: {e}")
                        info_md = f"Load failed for {voice}: {str(e)} (using defaults)"
                        status = "Error – using defaults."
                        return (
                            1.0, -8.0, 2.0, 0.6, -30.0, -28.0,
                            safe_float_value('temperature'), safe_float_value('exaggeration'),
                            safe_float_value('cfg_weight'), safe_float_value('min_p'),
                            safe_float_value('top_p'), safe_float_value('repetition_penalty'),
                            True, True, info_md, status
                        )

                load_voice_btn.click(
                    fn=load_voice_to_display,
                    inputs=[edit_voice_dropdown],
                    outputs=[voice_rate_num, voice_eq_num, voice_gain_num, voice_target_num, voice_noise_num,
                             voice_trim_num, voice_temp_num, voice_exagg_num, voice_cfg_num, voice_min_p_num,
                             voice_top_p_num, voice_rep_num, voice_notch_cb, voice_hp_cb, voice_info_md, voice_status],
                    js="", show_progress=False
                )

                # Apply (UI-only; updates CONFIG directly)
                def apply_voice_params(voice, rate, eq, gain, target, noise, trim, temp, exagg, cfg, min_p, top_p, rep, notch, hp):
                    voice = safe_to_str(voice) or 'default'
                    rate = safe_to_float(rate)
                    eq = safe_to_float(eq)
                    gain = safe_to_float(gain)
                    target = safe_to_float(target)
                    noise = safe_to_float(noise)
                    trim = safe_to_float(trim)
                    temp = safe_to_float(temp)
                    exagg = safe_to_float(exagg)
                    cfg = safe_to_float(cfg)
                    min_p = safe_to_float(min_p)
                    top_p = safe_to_float(top_p)
                    rep = safe_to_float(rep)
                    notch = safe_to_bool(notch)
                    hp = safe_to_bool(hp)
                    try:
                        status, api_status, info = update_voice_params_ui(
                            voice, rate, eq, gain, target, noise, trim, notch, hp, temp, cfg, min_p, top_p, rep, exagg
                        )
                        voice_status.value = status  # Direct (no State)
                        voice_info_md.value = f"**Applied to {voice}** | {safe_to_str(info)}"
                        api_status_md.value = api_status
                        logger.info(f"Voice {voice} applied via helpers")
                        return status, f"**Updated {voice}** | {safe_to_str(info)}", api_status
                    except Exception as e:
                        status = f"Apply failed: {str(e)}"
                        logger.error(status)
                        return status, f"Error applying {voice}", update_api_status()

                voice_apply_btn.click(
                    fn=apply_voice_params,
                    inputs=[edit_voice_dropdown, voice_rate_num, voice_eq_num, voice_gain_num, voice_target_num, voice_noise_num,
                            voice_trim_num, voice_temp_num, voice_exagg_num, voice_cfg_num, voice_min_p_num,
                            voice_top_p_num, voice_rep_num, voice_notch_cb, voice_hp_cb],
                    outputs=[voice_status, voice_info_md, api_status_md],
                    js="", show_progress=False
                )

                # Editor .change (update self from CONFIG)
                def update_edit_voice_choices(voice):
                    choices = CONFIG.get_all_voices() or ['default']
                    if 'default' not in choices:
                        choices.append('default')
                    return gr.update(choices=choices, value=safe_to_str(voice, 'default'))

                edit_voice_dropdown.change(
                    fn=update_edit_voice_choices,
                    inputs=[edit_voice_dropdown],
                    outputs=[edit_voice_dropdown],
                    js="", show_progress=False
                )

            # Tab 5: Global Config Editor (lazy load btn, with manual reload)
            with gr.TabItem("⚙️ Global Config Editor", id="config", elem_id="tab-config"):
                gr.Markdown("### Edit Global Parameters (Click Load to sync from CONFIG; use Reload after external changes)")
                load_global_btn = gr.Button("Load/Refresh Values", variant="secondary")  # Lazy

                with gr.Column():
                    speaking_rate_num = gr.Number(label="🗣️ Speaking Rate (x)", value=1.0, precision=2)
                    eq_gain_num = gr.Number(label="🎛️ EQ Gain (dB)", value=-8.0, precision=1)
                    max_gain_num = gr.Number(label="📈 Max Gain (x)", value=2.0, precision=1)
                    target_max_num = gr.Number(label="🎯 Target Max", value=0.6, precision=2)
                    noise_floor_num = gr.Number(label="🔇 Noise Floor (dB)", value=-30.0)
                    trim_threshold_num = gr.Number(label="✂️ Trim Threshold (dB)", value=-28.0)
                    eq_cutoff_num = gr.Number(label="📡 EQ Cutoff (Hz)", value=300.0)
                    fade_ms_num = gr.Number(label="🎭 Fade (ms)", value=50.0)
                    notch_low_num = gr.Number(label="Notch Low (Hz)", value=100.0)
                    notch_high_num = gr.Number(label="Notch High (Hz)", value=5000.0)
                    notch_gain_num = gr.Number(label="Notch Gain (dB)", value=-20.0)
                    notch_stretch_num = gr.Number(label="Notch for Stretch (dB)", value=-10.0)

                with gr.Column():
                    temperature_num = gr.Number(label="🌡️ Temperature", value=0.65, precision=2)
                    exaggeration_num = gr.Number(label="🎭 Exaggeration", value=1.0, precision=2)
                    cfg_weight_num = gr.Number(label="⚖️ CFG Weight", value=0.45, precision=2)
                    min_p_num = gr.Number(label="⚡ Min P", value=0.1, precision=2)
                    top_p_num = gr.Number(label="🔝 Top P", value=1.0, precision=2)
                    repetition_penalty_num = gr.Number(label="🔄 Repetition Penalty", value=1.5, precision=1)

                with gr.Column():
                    max_tokens_num = gr.Number(label="Max New Tokens", value=1499)
                    min_tokens_num = gr.Number(label="Min New Tokens", value=1)
                    cache_len_num = gr.Number(label="Max Cache Length", value=1024)
                    normalize_dropdown = gr.Dropdown(label="🎚️ Normalize Method", choices=['peak', 'rms', 'ebu'], value='rms')
                    logging_dropdown = gr.Dropdown(label="📢 Logging Level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], value='INFO')
                    enable_cache_cb = gr.Checkbox(label="✅ Enable Memory Cache", value=True)
                    enable_disk_cb = gr.Checkbox(label="💾 Enable Disk Cache", value=False)
                    enable_denoise_cb = gr.Checkbox(label="🧹 Enable Denoising", value=True)

                global_apply_btn = gr.Button("Apply Global Changes (Update CONFIG)", variant="secondary")
                global_status = gr.Textbox(label="Status", interactive=False, value="Click Load/Refresh to populate from CONFIG")

                # Manual reload btn (full config from file + auto-load tab values)
                reload_btn = gr.Button("🔄 Reload Config from File", variant="secondary")

                # Load btn (lazy: populates all from CONFIG)
                load_global_btn.click(
                    fn=load_global_tab,
                    inputs=[],  # No inputs
                    outputs=[speaking_rate_num, eq_gain_num, max_gain_num, target_max_num, noise_floor_num,
                             trim_threshold_num, eq_cutoff_num, fade_ms_num, notch_low_num, notch_high_num,
                             notch_gain_num, notch_stretch_num, temperature_num, exaggeration_num, cfg_weight_num,
                             min_p_num, top_p_num, repetition_penalty_num, max_tokens_num, min_tokens_num, cache_len_num,
                             normalize_dropdown, logging_dropdown, enable_cache_cb, enable_disk_cb, enable_denoise_cb, global_status],
                    js="", show_progress=False
                )

                # Reload (from file + auto-load)
                def on_reload():
                    reload_result = reload_all_config()
                    status = reload_result[0] if reload_result else "Reloaded from file"
                    api_status = update_api_status()
                    # Auto-load tab after reload
                    load_global_tab()  # Calls internally – but since outputs chained, trigger via separate click-like
                    # (For simplicity, reload btn just updates status; user clicks Load for full repop)
                    logger.info("Config reloaded from file")
                    return status, api_status

                reload_btn.click(
                    fn=on_reload,
                    inputs=[],  # No inputs
                    outputs=[global_status, api_status_md],
                    js="", show_progress=True
                )

                # Apply (UI-only; updates CONFIG)
                def apply_global_params(speaking_rate, eq_gain, max_gain, target_max, noise_floor, trim_threshold, eq_cutoff, fade_ms, notch_low, notch_high, notch_gain, notch_stretch, temperature, exaggeration, cfg_weight, min_p, top_p, repetition_penalty, max_tokens, min_tokens, cache_len, normalize_method, logging_level, enable_memory_cache, enable_disk_cache, enable_denoising):
                    try:
                        params = handle_global_params_change(
                            speaking_rate, eq_gain, max_gain, target_max, noise_floor, trim_threshold, normalize_method,
                            eq_cutoff, fade_ms, notch_low, notch_high, notch_gain, notch_stretch,
                            temperature, exaggeration, cfg_weight, min_p, top_p, repetition_penalty,
                            max_tokens, min_tokens, cache_len
                        )
                        handle_token_limits_change(max_tokens, min_tokens, cache_len)
                        CONFIG.enable_memory_cache = safe_to_bool(enable_memory_cache)
                        CONFIG.enable_disk_cache = safe_to_bool(enable_disk_cache)
                        CONFIG.enable_denoising = safe_to_bool(enable_denoising)
                        CONFIG.normalize_method = safe_to_str(normalize_method)
                        CONFIG.logging_level = safe_to_str(logging_level).upper()
                        status = "Globals applied to CONFIG (memory-only – save to persist; refresh other tabs to see changes)"
                        api_status = update_api_status()
                        logger.info("Global config applied via helpers")
                        return status, api_status
                    except Exception as e:
                        logger.error(f"Global apply failed: {e}")
                        return f"Apply failed: {str(e)}", update_api_status()

                global_apply_btn.click(
                    fn=apply_global_params,
                    inputs=[speaking_rate_num, eq_gain_num, max_gain_num, target_max_num, noise_floor_num,
                            trim_threshold_num, eq_cutoff_num, fade_ms_num, notch_low_num, notch_high_num,
                            notch_gain_num, notch_stretch_num, temperature_num, exaggeration_num, cfg_weight_num,
                            min_p_num, top_p_num, repetition_penalty_num, max_tokens_num, min_tokens_num, cache_len_num,
                            normalize_dropdown, logging_dropdown, enable_cache_cb, enable_disk_cb, enable_denoise_cb],
                    outputs=[global_status, api_status_md],
                    js="", show_progress=False
                )

                # Slider .change (partial – optional, UI-only)
                def on_speaking_rate_change(value):
                    create_global_param_handler("speaking_rate")(value)
                    return global_status.value, update_api_status()  # Direct update

                speaking_rate_num.change(
                    fn=on_speaking_rate_change,
                    inputs=[speaking_rate_num],
                    outputs=[global_status, api_status_md],
                    show_progress=False
                )
                # Similar for temperature_num, min_p_num (keep as-is; extend if needed)

                with gr.Accordion("📝 JSON Editor", open=False):
                    gr.Markdown("### Advanced: View/Edit Full Config JSON (Refresh, edit, Apply)")
                    current_json_tb = gr.Textbox(label="Current Config JSON", lines=10, interactive=False)
                    refresh_json_btn = gr.Button("🔄 Refresh JSON", variant="secondary")

                    memory_json_tb = gr.Textbox(label="Edit/Apply Memory State JSON", lines=8, placeholder="Paste or edit JSON here...", interactive=True)
                    apply_json_btn = gr.Button("Apply JSON Changes", variant="secondary")
                    json_status = gr.Textbox(label="JSON Status", interactive=False, value="Ready – refresh to load")

                    def refresh_and_copy():
                        json_str = refresh_config_json()
                        status = "Current config JSON loaded – edit below if needed."
                        return json_str, json_str, status

                    refresh_json_btn.click(
                        fn=refresh_and_copy,
                        inputs=[],  # No inputs
                        outputs=[current_json_tb, memory_json_tb, json_status],
                        js="", show_progress=False
                    )

                    apply_json_btn.click(
                        fn=apply_json_changes,
                        inputs=[memory_json_tb],
                        outputs=[memory_json_tb, json_status, api_status_md],
                        js="", show_progress=False
                    )

                # Controls (UI-only)
                with gr.Row():
                    save_btn = gr.Button("💾 Save All Changes to Config File", variant="primary")
                    reset_btn = gr.Button("🔄 Reset All to Original Values", variant="stop")

                controls_status = gr.Textbox(label="Controls Status", interactive=False, value="Ready to save/reset")

                save_btn.click(
                    fn=save_all_config,
                    outputs=[controls_status, api_status_md],
                    js="", show_progress=True
                )
                reset_btn.click(
                    fn=reset_all_config,
                    outputs=[controls_status, api_status_md],
                    js="", show_progress=True
                )

        # Attach Hidden API (to demo; uses Generate tab outputs – unchanged)
        setup_bridge_api(demo, audio_output, generate_status)

        logger.info("State-free Tab UI: Generate default; lazy load via 'Load/Refresh' btns (user-driven, no errors)")
        return demo

import tempfile  # Already? Add if not

# FIXED: Async wrapper (await generate_internal; yield progress for queue overlap)
async def safe_generate_async(text, voice, ref_wav, exagger=None, cfg_w=None, lang="en", temp=None, min_p=None, top_p=None, rep_pen=None, seed=None):
    """Async UI Wrapper: Awaits generate_internal (GPU gen + thread post/I/O); yields status."""
    try:
        logger.info(f"safe_generate_async inputs: text='{text}', voice='{voice}', ref_wav type={type(ref_wav)}")
        text = safe_to_str(text, "")
        voice = safe_to_str(voice, 'default')
        ref_path = None

        if not text.strip():
            yield None, "Enter text to generate."
            return

        yield None, "Preparing..."  # Initial yield (UI responsive)

        # Handle uploaded ref (Gradio dict or temp str/bytes – always valid post-preprocess)
        if isinstance(ref_wav, dict):
            ref_path = safe_to_str(ref_wav.get('path') or ref_wav.get('name'), None)
            if ref_path and ('gradio' in ref_path or '/tmp/' in ref_path):  # Confirm temp upload
                logger.info(f"Uploaded ref: {ref_path}")
            else:
                ref_path = None  # Invalid dict
        elif isinstance(ref_wav, str):
            # Temp upload str (post-cache)
            if os.path.exists(ref_wav) and ('gradio' in ref_wav or '/tmp/' in ref_wav):
                ref_path = ref_wav
                logger.info(f"Temp ref: {ref_path}")
            else:
                logger.warning(f"Invalid ref str: {ref_wav} – no upload")
                ref_path = None
        elif isinstance(ref_wav, bytes):  # FIXED: Handle bytes directly (Gradio Audio type="bytes")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(ref_wav)
                ref_path = tmp.name
                logger.info(f"Bytes ref saved temp: {ref_path}")

        # Fallback: CONFIG voice WAV (but validate: exists, not project/root/dir) – Unchanged
        if not ref_path:
            candidate_path = CONFIG.get_voice_wav_path(voice)
            if candidate_path and os.path.isfile(candidate_path):  # File only (not dir/project)
                # Extra safety: Skip if looks like project root (e.g., contains script name or .venv)
                if any(bad in candidate_path.lower() for bad in ['skyrimnet_chatterbox', '.venv', os.getcwd()]):
                    logger.warning(f"Skipping bad fallback path (project dir?): {candidate_path}")
                    ref_path = None
                else:
                    ref_path = candidate_path
                    logger.info(f"Using valid default ref for '{voice}': {ref_path}")
            else:
                logger.info(f"No valid default ref for '{voice}' (path invalid/missing) – Default synthesis")
                ref_path = None

        lang = safe_to_str(lang, "en")

        # UI params or CONFIG fallback (no clamping—internal merges) – Unchanged
        final_exagger = safe_to_float(exagger, CONFIG.get_value('exaggeration', 1.0))
        final_cfg_w = safe_to_float(cfg_w, CONFIG.get_value('cfg_weight', 0.45))
        final_temp = safe_to_float(temp, CONFIG.get_value('temperature', 0.65))
        final_min_p = safe_to_float(min_p, CONFIG.get_value('min_p', 0.1))
        final_top_p = safe_to_float(top_p, CONFIG.get_value('top_p', 1.0))
        final_rep = safe_to_float(rep_pen, CONFIG.get_value('repetition_penalty', 1.5))
        if seed is None:
            seed_num = random.randint(0, 2**31 - 1)
        else:
            seed_num = safe_to_int(seed, 0)

        # Voice/Ref (stem for cloning; default if invalid) – Unchanged
        voice_name = voice
        if ref_path and os.path.exists(ref_path):
            voice_name = Path(ref_path).stem
            if voice_name in ['tmp', 'unknown', 'stub', '']:
                voice_name = voice or 'default'
            logger.info(f"Cloning from ref: {ref_path} → voice_name={voice_name}")
        else:
            logger.warning(f"No valid ref for '{voice}' - Using default synthesis")

        # Cache UUID (simple seed-based) – Unchanged
        cache_uuid = seed_num if seed_num > 0 else 0

        # Minimal post_overrides (audio params from CONFIG; merge in internal) – Unchanged
        post_overrides = {
            'speaking_rate': float(CONFIG.get_value('speaking_rate', 1.0)),
            # Extend e.g., 'eq_gain_db': CONFIG.get_value('eq_gain_db', 0.0) if UI slider added
        }

        yield None, "Generating audio..."  # Progress during gen

        # FIXED: Direct await generate_internal (async pipeline; no asyncio.run – event loop safe)
        path = await generate_audio(
            text, lang, ref_path, final_exagger, final_temp, seed_num, final_cfg_w,
            final_min_p, final_top_p, final_rep, cache_uuid, voice_name, post_overrides
        )

        yield None, "Processing and saving..."  # Yield during post/save (overlapped)

        # Validate output – Unchanged
        status = f"✅ Generated | Voice: {voice_name} | Seed: {seed_num} | Text: {text[:20]}..."
        if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
            logger.warning(f"Invalid path from internal: {path} - Falling back to stub")
            stub_path, stub_status = stub_wav_path()  # From ui_helpers
            logger.warning(f"Invalid path from internal: {path} - Falling back to stub ({stub_status})")
            status = f"❌ Invalid output | Voice: {voice_name} | {stub_status}"
            path = stub_path

        # Cleanup temp ref if bytes
        if isinstance(ref_wav, bytes) and ref_path:
            try:
                os.unlink(ref_path)
                logger.debug(f"Temp ref cleaned: {ref_path}")
            except:
                pass

        yield path, status  # Final yield (audio + success)

        logger.info(f"safe_generate_async: {status} (UI async await → internal)")
    except Exception as e:
        logger.error(f"safe_generate_async error: {e}")
        stub_path, stub_status = stub_wav_path()
        status = f"Wrapper error: {e} | {stub_status}"
        yield stub_path, status


if __name__ == "__main__":
    demo = create_ui()
    # TODO test concurrency
    demo.queue(concurrency_limit=3,  # Max simultaneous gens (e.g., 4 short ones OK; test GPU mem)
               max_size=9)  # Queue max jobs (prevents overload)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        enable_queue=True,
        max_threads=40,
        show_error=True,
        api_open=True
    )