"""
src/generate_tab.py

Self-contained Generate Audio tab implementation with proper async generator handling.

This tab follows a simple workflow:
1. User uploads a reference WAV audio file
2. User enters text to be spoken
3. User clicks "Generate Audio"
4. System generates speech using the uploaded voice

The voice is determined solely by the uploaded reference audio - no voice selection dropdown.
"""

import os
import random
import tempfile
from pathlib import Path
import warnings

import gradio as gr
from loguru import logger

# We'll import only what we need directly
from src.config import get_config
from src.tts_model import ModelManager, get_model
from src.ui_helpers import safe_to_str, stub_wav_path
from src.generate_audio import generate_audio


# Suppress known warnings for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", message=r"Reference mel length is not equal to 2 \* reference token length\.", category=UserWarning)
warnings.filterwarnings("ignore", message="torchaudio._backend.utils.info has been deprecated")


def create_generate_tab():
    """
    Creates ONLY the Generate Audio tab as a standalone component.

    Returns:
        tuple: (tab, audio_output, generate_status)
            - tab: The TabItem component (used internally by Gradio)
            - audio_output: The audio output component for API integration
            - generate_status: The status component for API integration
    """
    with gr.TabItem("🎤 Generate Audio", id="generate", elem_id="tab-generate"):
        # Text input - first thing user sees
        text_input = gr.Textbox(
            label="Input Text",
            placeholder="Enter text to generate...",
            lines=3,
            value=""
        )

        # Reference audio - WAV file for voice cloning
        ref_audio = gr.Audio(
            label="Reference Audio (WAV file)",
            sources="upload",
            type="filepath",
            format="wav",
            interactive=True
        )

        # Voice status indicator
        voice_status = gr.Markdown("Upload a reference audio file to begin")

        # Generate button
        generate_btn = gr.Button("Generate Audio", variant="primary", interactive=False)

        # Output components
        audio_output = gr.Audio(label="Generated Audio")
        generate_status = gr.Textbox(
            label="Status",
            interactive=False,
            value="Ready"
        )

        # Event handlers
        ref_audio.upload(
            fn=handle_ref_upload,
            inputs=[ref_audio],
            outputs=[voice_status, generate_btn],
            js=""
        )

        # Button click handler (UI-only, no state)
        generate_btn.click(
            fn=generate_audio_ui_handler,
            inputs=[text_input, ref_audio],
            outputs=[audio_output, generate_status],
            js="",
            show_progress=True,
            concurrency_limit=1
        )

    return (gr.TabItem, audio_output, generate_status)


def handle_ref_upload(ref_wav):
    """
    Handles reference audio upload event.

    Args:
        ref_wav: The uploaded reference audio

    Returns:
        tuple: (voice_status, generate_btn_state)
            - voice_status: User-friendly status message
            - generate_btn_state: Whether generate button should be enabled
    """
    if not ref_wav:
        return "Upload a reference audio file to begin", gr.update(interactive=False)

    try:
        # Process reference audio path
        ref_path = get_valid_ref_path(ref_wav)
        if not ref_path:
            return "⚠️ Invalid reference audio file. Please upload a valid WAV file.", gr.update(interactive=False)

        # Extract voice name for user feedback
        voice_name = Path(ref_path).stem
        return f"✅ Voice loaded: '{voice_name}' (ready to generate)", gr.update(interactive=True)

    except Exception as e:
        logger.error(f"Error processing reference audio: {e}")
        return f"❌ Error processing audio: {str(e)}", gr.update(interactive=False)


def get_valid_ref_path(ref_wav):
    """
    Processes and validates reference audio path from Gradio input.

    Args:
        ref_wav: Gradio Audio component value (can be various types)

    Returns:
        str or None: Validated reference audio path or None if invalid
    """
    if not ref_wav:
        return None

    try:
        # Handle different Gradio Audio component output types
        if isinstance(ref_wav, str) and os.path.exists(ref_wav):
            return ref_wav
        elif isinstance(ref_wav, dict):
            if ref_wav.get('path') and os.path.exists(ref_wav['path']):
                return ref_wav['path']
            elif ref_wav.get('name') and os.path.exists(ref_wav['name']):
                return ref_wav['name']
        elif isinstance(ref_wav, bytes):
            # Handle direct bytes upload
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(ref_wav)
                return tmp.name

        logger.warning(f"Invalid reference audio format: {type(ref_wav)}")
        return None
    except Exception as e:
        logger.error(f"Error validating reference path: {e}")
        return None


def get_generation_params(config):
    """
    Gathers and validates all generation parameters from config.

    Args:
        config: CONFIG instance

    Returns:
        dict: All required generation parameters
    """
    # Get generation parameters with fallbacks
    params = {
        'exaggeration': float(config.get_value('exaggeration', 0.75)),
        'temperature': float(config.get_value('temperature', 0.65)),
        'cfg_weight': float(config.get_value('cfg_weight', 0.45)),
        'min_p': float(config.get_value('min_p', 0.1)),
        'top_p': float(config.get_value('top_p', 1.0)),
        'repetition_penalty': float(config.get_value('repetition_penalty', 1.5)),
        'enable_memory_cache': bool(config.get_value('enable_memory_cache', True)),
        'enable_disk_cache': bool(config.get_value('enable_disk_cache', False))
    }

    # Generate a random seed for this generation
    params['seed_num'] = random.randint(0, 2**31 - 1)
    params['cache_uuid'] = params['seed_num']

    # Log parameters (sanitized for privacy)
    logger.debug(f"Using generation params: temp={params['temperature']:.2f}, "
                 f"cfg={params['cfg_weight']:.2f}, seed={params['seed_num']}")

    return params


async def generate_audio_ui_handler(text, ref_wav):
    """
    UI handler that correctly implements the async generator pattern.

    This function:
    1. Validates inputs
    2. Processes reference audio
    3. Yields intermediate status messages
    4. Performs generation
    5. Yields final result

    CORRECTED: No 'return value' inside generator - uses 'yield' for all outputs
    """
    config = get_config()
    MODEL = get_model()
    temp_ref_path = None

    try:
        # Basic input validation
        text = safe_to_str(text, "").strip()
        if not text:
            yield None, "❌ Error: Please enter text to generate"
            return

        # Process reference audio path
        ref_path = get_valid_ref_path(ref_wav)
        if not ref_path:
            yield None, "❌ Error: Please upload a valid reference audio file"
            return

        # Validate reference audio exists
        if not os.path.exists(ref_path):
            yield None, "❌ Error: Reference audio file not found"
            return

        # Extract voice name for user feedback
        voice_name = Path(ref_path).stem
        logger.info(f"Generating audio: text='{text[:50]}...', voice='{voice_name}', ref_path={ref_path}")

        # Prepare generation parameters
        try:
            params = get_generation_params(config)
        except Exception as e:
            logger.error(f"Failed to get generation parameters: {e}")
            yield None, f"❌ Configuration error: {str(e)}"
            return

        # Stage 1: Initial preparation
        yield None, f"📝 Preparing generation for '{voice_name}'..."

        # Stage 2: Audio generation
        yield None, f"🔊 Generating speech with '{voice_name}'..."

        try:
            # Main generation call
            output_path = await generate_audio(
                model=MODEL,
                text=text,
                audio_prompt_path=ref_path,
                exaggeration=params['exaggeration'],
                cache_uuid=params['cache_uuid'],
                temperature=params['temperature'],
                cfgw=params['cfg_weight'],
                min_p=params['min_p'],
                top_p=params['top_p'],
                repetition_penalty=params['repetition_penalty'],
                seed_num=params['seed_num']
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            yield handle_generation_error(f"Audio generation failed: {str(e)}")
            return

        # Stage 3: Post-processing
        yield None, "✨ Processing and enhancing audio..."

        # Validate output
        if not output_path or not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            logger.warning(f"Invalid output path: {output_path}")
            yield handle_generation_error("Generated audio is empty or invalid")
            return

        # Create success message with text preview
        preview_text = text[:30] + ('...' if len(text) > 30 else '')
        status_msg = f"✅ Generated: '{preview_text}' using {voice_name} | Seed: {params['seed_num']}"

        # Cleanup temporary reference if needed
        if isinstance(ref_wav, bytes) and ref_path and tempfile.gettempdir() in ref_path:
            try:
                os.unlink(ref_path)
                logger.debug(f"Cleaned up temporary reference file: {ref_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp reference {ref_path}: {e}")

        # Final return with audio
        yield output_path, status_msg

    except Exception as e:
        logger.exception("Unexpected error in generate_audio_ui_handler")
        yield handle_generation_error(f"Unexpected error: {str(e)}")


def handle_generation_error(error_msg):
    """
    Handles generation errors with proper fallbacks.

    Returns:
        tuple: (stub_path, user_friendly_msg)

    CORRECTED: Returns values for yielding, not for returning
    """
    logger.error(f"Generation error: {error_msg}")
    stub_path, stub_desc = stub_wav_path()

    # Create user-friendly error messages
    if "CUDA" in error_msg or "GPU" in error_msg:
        user_msg = "❌ GPU error: Insufficient memory. Try shorter text."
    elif "reference" in error_msg.lower() or "voice" in error_msg.lower():
        user_msg = "❌ Voice processing error: Invalid reference audio. Use clean WAV."
    elif "text" in error_msg.lower():
        user_msg = "❌ Text processing error: Try simpler text."
    else:
        user_msg = f"❌ Generation failed: {error_msg[:100]}"
        if len(error_msg) > 100:
            user_msg += "..."

    # Append fallback info
    if stub_desc != "success":
        user_msg += f" | Fallback: {stub_desc}"

    return stub_path, user_msg