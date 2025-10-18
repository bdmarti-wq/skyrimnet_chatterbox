# src/ui.py
import os
import gradio as gr
from loguru import logger
from pathlib import Path

# CONFIG singleton
from src.config import get_config, get_config_value
from src.generate_tab import create_generate_tab
from src.tts_model import get_model, ModelManager

# Path helpers
from src.ui_helpers import (
    update_api_status, stub_wav_path, test_voice_generation,
)

# Hidden API supports communication with SkyrimNet via a Zonos bridge
from src.generate.ui_interface import generate_audio_ui, setup_bridge_api

# Warnings cleanup
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", message="torchaudio._backend.utils.info has been deprecated")

# Required for proper config handling
from src.config.models import AppConfig

def create_ui(
    cache_manager=None,
    pipeline=None,
    config: AppConfig = None
):
    """State-free UI: Tabs with lazy load (user clicks 'Load/Refresh' for CONFIG values). Generate default."""
    # Ensure config is available
    config = config or get_config()

    with gr.Blocks(title="SkyrimNet Chatterbox", theme=gr.themes.Soft()) as demo:
        # Global Status (top-level, updated via btns)
        api_status_md = gr.Markdown(value=update_api_status())

        gr.Markdown("# SkyrimNet Chatterbox TTS UI\nSimplified tabbed interface for generation, testing, and config editing.", elem_id="title-md")

        # Generate Tab - Create it first so we can get its outputs for bridge API
        audio_output, generate_status = create_generate_tab()
        logger.debug("Generate Audio tab initialized")

        # Tabs container
        with gr.Tabs(selected="generate") as tabs:
            # Voice Test Tab (separate from Generate)
            with gr.TabItem("🔊 Voice Test", id="test", elem_id="tab-test"):
                gr.Markdown("### 🔬 Test Voice Parameters (Click Load to sync from CONFIG)")
                load_test_btn = gr.Button("Load/Refresh Values", variant="secondary")

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
                    fn=lambda: load_test_tab(config),
                    inputs=[],  # No inputs
                    outputs=[test_voice_dropdown, test_text, test_seed, test_rate, test_eq, test_gain, test_target, test_noise, test_trim, test_notch, test_hp, test_params, test_status],
                    js="", show_progress=False
                )

                # Test btn (UI-only)
                test_btn.click(
                    fn=lambda *args: test_voice_generation(config, *args),
                    inputs=[test_voice_dropdown, test_text, test_seed],
                    outputs=[test_audio, test_status, test_params],
                    js="", show_progress=True, concurrency_limit=1
                )

                # Voice change handler
                test_voice_dropdown.change(
                    fn=lambda voice: update_test_voice_choices(config, voice),
                    inputs=[test_voice_dropdown],
                    outputs=[test_voice_dropdown],
                    js="", show_progress=False
                )

            # Voice Editor Tab (minimal implementation for now)
            with gr.TabItem("🎤 Voice Editor", id="editor", elem_id="tab-editor"):
                gr.Markdown("### Edit Per-Voice Params (Simple implementation for testing)")
                voice_dropdown = gr.Dropdown(
                    label="Select Voice",
                    choices=['default'] + config.get_all_voices(),
                    value='default',
                    allow_custom_value=False
                )
                load_btn = gr.Button("Load Voice Parameters")
                save_btn = gr.Button("Save Voice Parameters", variant="primary")

                # Editor fields
                voice_rate = gr.Slider(0.5, 2.0, 1.0, label="Speaking Rate")
                voice_temp = gr.Slider(0.1, 1.5, 0.8, label="Temperature")
                voice_exagg = gr.Slider(0.1, 2.0, 0.75, label="Exaggeration")

                voice_status = gr.Markdown("Load a voice to edit parameters")

                # Load voice params when dropdown changes
                def load_voice_params(voice_name):
                    voice_name = voice_name or 'default'
                    voice = config.get_voice_params(voice_name)
                    return [
                        voice.get('speaking_rate', 1.0),
                        voice.get('temperature', 0.8),
                        voice.get('exaggeration', 0.75)
                    ]

                voice_dropdown.change(
                    fn=load_voice_params,
                    inputs=[voice_dropdown],
                    outputs=[voice_rate, voice_temp, voice_exagg]
                )

                # Save params to config
                def save_voice_params(voice_name, rate, temp, exagg):
                    voice_name = voice_name
                    config.update_voice_parameter(voice_name, 'speaking_rate', rate)
                    config.update_voice_parameter(voice_name, 'temperature', temp)
                    config.update_voice_parameter(voice_name, 'exaggeration', exagg)
                    return f"Saved voice parameters for {voice_name}"

                save_btn.click(
                    fn=save_voice_params,
                    inputs=[voice_dropdown, voice_rate, voice_temp, voice_exagg],
                    outputs=[voice_status]
                )

        # Attach Hidden API to generate tab outputs - critical for remote connection
        setup_bridge_api(demo, audio_output, generate_status, config=config)

        logger.info("Initialize state-free Tab UI successfully. Bridge API connected.")
        return demo

# Tab Load Functions - simplified to match new UI structure
def load_test_tab(config):
    """Load Voice Test tab components from CONFIG (lazy on btn click)."""
    voice_choices = config.get_all_voices() or ['default']
    if 'default' not in voice_choices:
        voice_choices.append('default')

    # Get audio parameters from config with proper hierarchy
    audio_cfg = config.app_config.audio
    txs_cfg = config.app_config.tts

    # Defaults from CONFIG (global, as voice-specific needs select first)
    params = {
        'speaking_rate': audio_cfg.speaking_rate,
        'eq_gain_db': audio_cfg.eq_gain_db,
        'max_gain': audio_cfg.max_gain,
        'target_max': audio_cfg.target_max,
        'noise_floor_db': audio_cfg.noise_floor_db,
        'trim_threshold_db': audio_cfg.trim_threshold_db,
        'notch_enabled': audio_cfg.notch_enabled,
        'hp_enabled': audio_cfg.hp_enabled,
    }

    return (
        gr.update(choices=voice_choices),
        "Testing shared config voice parameters.",
        42,
        params['speaking_rate'], params['eq_gain_db'], params['max_gain'], params['target_max'],
        params['noise_floor_db'], params['trim_threshold_db'], params['notch_enabled'], params['hp_enabled'],
        {},  # test_params
        "Values loaded from CONFIG – test with custom params if needed."
    )

def update_test_voice_choices(config, voice):
    """Handle voice dropdown changes with proper config reference."""
    choices = config.get_all_voices() or ['default']
    if 'default' not in choices:
        choices.append('default')
    return gr.update(choices=choices, value=voice or 'default')

def test_voice_generation(config, voice_name, test_text, seed):
    """
    Handle voice test generation with proper config reference.
    """
    try:
        logger.info(f"Voice test requested: voice='{voice_name}', text='{test_text[:50]}...', seed={seed}")

        # Get config values using hierarchical structure
        txs_cfg = config.app_config.tts
        audio_cfg = config.app_config.audio

        # Generate UUID for cache (simple deterministic)
        cache_uuid = abs(hash(f"{voice_name}_{test_text}_{seed}")) % (2**31)

        # Call the UX-compatible audio generation interface
        result = generate_audio_ui(
            model_choice=None,
            text=test_text,
            language="en",
            speaker_audio=None,
            prefix_audio=None,
            e1=None, e2=None, e3=None, e4=None, e5=None, e6=None, e7=None, e8=None,
            vq_single=None, fmax=None, pitch_std=None,
            speaking_rate=None,
            dnsmos_ovrl=None,
            speaker_noised=False,
            cfg_scale=txs_cfg.cfg_weight,
            top_p_param=txs_cfg.top_p,
            top_k=None,
            min_p_param=txs_cfg.min_p,
            linear_temp=txs_cfg.temperature,
            confidence_rep=txs_cfg.repetition_penalty,
            quadratic_exagg=txs_cfg.exaggeration,
            uuid_seed=cache_uuid,
            randomize_seed_toggle=False,
            unconditional_keys_list=None
        )

        # Extract the path from result
        audio_path = result[0] if isinstance(result, (list, tuple)) and result else None

        # Prepare used params for display
        actual_params = {
            'voice': voice_name,
            'text': test_text,
            'seed': seed,
            'temperature': txs_cfg.temperature,
            'cfg_weight': txs_cfg.cfg_weight,
            'exaggeration': txs_cfg.exaggeration,
            'cache_uuid': cache_uuid,
            'status': result[1] if isinstance(result, (list, tuple)) and len(result) > 1 else "Unknown status"
        }

        status = "✅ Test generated successfully" if audio_path and Path(audio_path).exists() else "⚠️ Generated but audio not found"
        return audio_path, status, actual_params

    except Exception as e:
        logger.exception("Voice test failed")
        fallback_path, fallback_status = stub_wav_path()
        return None, f"❌ Test generation failed: {str(e)} | {fallback_status}", {}

if __name__ == "__main__":
    # For standalone testing
    demo = create_ui()
    demo.queue(
        concurrency_limit=3,
        max_size=9
    ).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        enable_queue=True,
        max_threads=40,
        show_error=True,
        api_open=True
    )