# skyrimnet_chatterbox.py (Imports and generate shell)
import asyncio
import functools
import tempfile
import warnings

import torchaudio
from pathlib import Path


from src.config import get_config, get_config_value
# Lazy import inside generate (avoids global Gradio scan/inference)
from src.audio_utils import set_torchaudio_backend
from src.fuzzy_cache import load_fuzzy_cache
from src.ui import create_ui
from src.tts_model import ModelManager

backend = set_torchaudio_backend()

import gradio as gr
from argparse import ArgumentParser
import torch
from src.cache import (
    init_conditional_memory_cache, clear_cache_files, clear_output_directories
)
from loguru import logger

import warnings
warnings.filterwarnings('ignore', message=r'.*torchaudio._backend.utils.info.*')
warnings.filterwarnings('ignore', message=r'.*deprecated.*torchaudio.*')
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._backend")
warnings.filterwarnings("ignore", message="torchaudio._backend.set_audio_backend has been deprecated")


def parse_arguments():
    """Parse command line arguments"""
    parser = ArgumentParser()
    parser.add_argument('--share', action='store_true',
                        help="Create a EXTERNAL facing public link using Gradio's servers")
    parser.add_argument("--server", type=str, default='0.0.0.0', help="Server address to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, required=False, default=7860,
                        help="Port to run the server on (default: 7860)")
    parser.add_argument("--inbrowser", action='store_true', help="Open the UI in a new browser window")
    parser.add_argument("--multilingual", action='store_true', default=False,
                        help="Use the multilingual model (requires more VRAM)")
    parser.add_argument("--clearoutput", action='store_true',
                        help="Remove all folders in audio output directory and exit")
    parser.add_argument("--clearcache", action='store_true', help="Remove all .pt cache files and exit")
    return parser.parse_args()



def main():  # FIXED: Make sync (no async def; Easier for script + handles nested async safely)
    """Main entrypoint: Load model, config, and launch UI."""
    logger.info("Starting SkyrimNet Chatterbox...")

    args = parse_arguments()

    # Handle cleanup arguments that exit immediately
    if args.clearoutput:
        logger.info("Clearing output directories...")
        count = clear_output_directories()
        logger.info(f"Cleared {count} output directories. Exiting.")
        exit(0)

    if args.clearcache:
        logger.info("Clearing cache files...")
        count = clear_cache_files()
        logger.info(f"Cleared {count} cache files. Exiting.")
        exit(0)

    # Load configuration at startup
    logger.info("Loading SkyrimNet configuration...")
    config = get_config() # loads the config

    # Load TTS model (idempotent)
    model_type = 'multilingual' if config.get_value('multilingual') else 'english'
    logger.info(f"Loading {model_type.capitalize()} Model")

    instance = ModelManager.get_instance()
    model = instance.get_model(model_type)  # Loads/caches; returns object or None

    if model is None:
        logger.error(f"Failed to load {model_type} model; TTS disabled")
        config.app_config.globals.model = None  # Explicit null
    else:
        # Set runtime (in-memory; for CONFIG.model access)
        config.app_config.globals.model = model
        logger.info(
            f"✓ Model loaded : {model.__class__.__name__} ({model_type}) on {config.app_config.globals.device} (dtype={config.app_config.globals.dtype})")

    init_conditional_memory_cache(model, get_config_value('globals.device'), get_config_value('globals.dtype'), quiet=False, pre_validate_voices=False)  # Quiet for prod
    load_fuzzy_cache()


    # Launch UI (Gradio demo – sync)
    demo = create_ui()
    demo.queue(
        max_size=12,
        default_concurrency_limit=2,
    ).launch(
        share=False,  # Optional: Public sharing
        server_name = args.server,
        server_port = args.port,
        inbrowser = args.inbrowser
    )

    # Optional: Start API server or CLI mode (uncomment if needed)
    # start_api_server()


if __name__ == "__main__":
    main()  # FIXED: Sync call (main now sync; No "never awaited")