"""
Self-contained Generate Audio tab implementation for the new pipeline.

This tab follows the standard workflow:
1. User uploads reference audio
2. User enters text
3. User clicks generate
4. System handles the entire pipeline process

The key difference from the old implementation:
- Uses the bridge API compatible 29-parameter interface
- Integrates with the CacheManager and GenerationPipeline
- No direct access to pipeline internals from UI
- Proper async handling with status updates
"""

import os
import random
import tempfile
from pathlib import Path
import warnings

import gradio as gr
from loguru import logger

# Core application imports (only UI interface needed)
from src.config import get_config, get_config_value
from src.generate.ui_interface import generate_audio_ui

# UI helper functions
from src.ui_helpers import stub_wav_path, update_api_status

# Suppress known warnings for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", message=r"Reference mel length is not equal to 2 \* reference token length\.", category=UserWarning)
warnings.filterwarnings("ignore", message="torchaudio._backend.utils.info has been deprecated")

def create_generate_tab():
    """
    Creates the Generate Audio tab as a standalone component.

    Returns:
        tuple: (audio_output, generate_status)
            - audio_output: The audio output component for API integration
            - generate_status: The status component for API integration
    """
    with gr.Column():
        # Main UI components
        text_input = gr.Textbox(
            label="Input Text",
            placeholder="Enter text to generate speech...",
            lines=3,
            value=""
        )

        ref_audio = gr.Audio(
            label="Voice Reference (WAV file)",
            sources="upload",
            type="filepath",
            format="wav",
            interactive=True
        )

        # REPLACE info with separate Markdown (Gradio <3.37 compatibility)
        voice_help = gr.Markdown("Upload your own voice sample (5-10 seconds of clear speech)")

        voice_status = gr.Markdown("🔊 Upload a reference audio file to begin")

        generate_btn = gr.Button(
            "Generate Audio",
            variant="primary",
            interactive=False
        )

        audio_output = gr.Audio(
            label="Generated Speech",
            type="filepath"
        )

        generate_status = gr.Textbox(
            label="Status",
            interactive=False,
            value="Pipeline ready - waiting for input"
        )

    # Event handlers (directly in tab creation)
    ref_audio.upload(
        fn=handle_ref_upload,
        inputs=[ref_audio],
        outputs=[voice_status, generate_btn],
        js=""
    )

    generate_btn.click(
        fn=generate_audio_ui_handler,
        inputs=[text_input, ref_audio],
        outputs=[audio_output, generate_status],
        js="",
        show_progress=True,
        concurrency_limit=1
    )

    return audio_output, generate_status


def handle_ref_upload(ref_wav):

    """
    Handles reference audio upload event.

    Args:
        ref_wav: Gradio Audio output

    Returns:
        tuple: (voice_status, generate_btn_state)
    """
    if not ref_wav:
        return "🔇 Upload a reference audio file to begin", gr.update(interactive=False)

    ref_path = get_valid_ref_path(ref_wav)
    if not ref_path or not os.path.exists(ref_path):
        return "⚠️ Invalid reference audio file. Upload a valid WAV file.", gr.update(interactive=False)

    try:
        # Extract basic info for user feedback
        voice_name = Path(ref_path).stem[:30]  # Shorten for display
        logger.debug(f"Valid reference uploaded: {voice_name}")
        return (
            f"✅ Voice loaded: '{voice_name}' (ready to generate)",
            gr.update(interactive=True)
        )
    except Exception as e:
        logger.error(f"Reference validation issue: {e}")
        return (
            "⚠️ Invalid reference: Unable to process file",
            gr.update(interactive=False)
        )

def get_valid_ref_path(ref_wav):
    """
    Validates and processes reference audio path.

    Args:
        ref_wav: Could be string path, dict, or bytes

    Returns:
        str: Validated path or None
    """
    if not ref_wav:
        return None

    # Handle different Gradio Audio component output types
    if isinstance(ref_wav, str) and os.path.exists(ref_wav):
        return ref_wav
    elif isinstance(ref_wav, dict):
        if ref_wav.get('path') and os.path.exists(ref_wav['path']):
            return ref_wav['path']
        elif ref_wav.get('name') and os.path.exists(ref_wav['name']):
            return ref_wav['name']
        elif ref_wav.get('data'):  # Handle browser-side recording
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(ref_wav['data'])
                    return tmp.name
            except Exception as e:
                logger.error(f"Failed to save browser recording: {e}")
                return None
    elif isinstance(ref_wav, bytes):
        # Handle direct bytes upload
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(ref_wav)
                return tmp.name
        except Exception as e:
            logger.error(f"Failed to save byte stream audio: {e}")
            return None

    logger.debug(f"Invalid reference format: {type(ref_wav)}")
    return None

async def generate_audio_ui_handler(text, ref_wav):
    """
    Proper async handler that interfaces with the pipeline.

    This function:
    1. Validates all inputs
    2. Maps UI parameters to bridge API requirements
    3. Calls generate_audio_ui with 29 parameters
    4. Handles response and updates status
    """
    # Initial validation
    text = text.strip() if text else ""
    if not text:
        yield None, "❌ Error: Please enter text to generate"
        return

    ref_path = get_valid_ref_path(ref_wav)
    if not ref_path or not os.path.exists(ref_path):
        yield None, "❌ Error: Please upload a valid reference audio file"
        return

    try:
        # Generate deterministic UUID based on inputs
        uuid_seed = abs(hash(f"{text}_{ref_path}_{random.randint(0, 1000000)}")) % (2**31)

        # Update status step-by-step
        yield None, "🔄 Preparing generation parameters..."

        # Get config values through proper access
        config = get_config().app_config

        # Clean status message with text preview
        text_preview = text[:40] + ('...' if len(text) > 40 else '')
        voice_name = Path(ref_path).stem[:20]
        status_prefix = f"🔊 Generating '{text_preview}' with '{voice_name}'"
        logger.info(f"Pipeline execution initiated: {text_preview} [{voice_name}]")

        try:
            # Transform to bridge API 29 parameters
            api_args = {
                'model_choice': None,
                'text': text,
                'language': "en",
                'speaker_audio': ref_path,
                'prefix_audio': None,
                'e1': None, 'e2': None, 'e3': None, 'e4': None, 'e5': None,
                'e6': None, 'e7': None, 'e8': None,
                'vq_single': None,
                'fmax': None,
                'pitch_std': None,
                'speaking_rate': None,
                'dnsmos_ovrl': None,
                'speaker_noised': False,
                'cfg_scale': float(config.tts.cfg_weight),
                'top_p_param': float(config.tts.top_p),
                'top_k': None,
                'min_p_param': float(config.tts.min_p),
                'linear_temp': float(config.tts.temperature),
                'confidence_rep': float(config.tts.repetition_penalty),
                'quadratic_exagg': float(config.tts.exaggeration),
                'uuid_seed': uuid_seed,
                'randomize_seed_toggle': False,
                'unconditional_keys_list': None
            }

            yield None, f"{status_prefix} - Processing through pipeline..."

            # Call the bridge-compatible UI function
            result = await generate_audio_ui(**api_args)

            # Process the result
            if not result or not isinstance(result, (tuple, list)) or len(result) < 2:
                error_msg = "Bridge responded with invalid format"
                logger.error(error_msg)
                fallback_path, fallback_status = stub_wav_path()
                yield fallback_path, f"❌ {error_msg} - {fallback_status}"
                return

            audio_path, status_msg = result
            message = f"{status_prefix} - Complete! | {status_msg}"

            # Verify audio file
            if audio_path and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                yield audio_path, message
            else:
                fallback_path, fallback_status = stub_wav_path()
                warning = "Generated audio path invalid" if audio_path else "No audio path returned"
                logger.warning(f"{warning}: {audio_path}")
                yield fallback_path, f"⚠️ {warning} | Using silence fallback: {status_msg}"

        except Exception as pipeline_e:
            logger.exception("Pipeline execution failed")
            fallback_path, fallback_status = stub_wav_path()
            error_msg = str(pipeline_e)[:200]
            yield fallback_path, f"❌ Pipeline error: {error_msg} | {fallback_status}"

    except Exception as e:
        logger.exception("Unexpected error in audio handler")
        fallback_path, fallback_status = stub_wav_path()
        yield fallback_path, f"❌ Critical error: {str(e)[:100]} | {fallback_status}"