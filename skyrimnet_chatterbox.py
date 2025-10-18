# skyrimnet_chatterbox.py

import os
import sys
from argparse import ArgumentParser

import torch
from loguru import logger

# Core application imports (danger: circular risk; we'll be careful)
from src.config import get_config, get_config_value
from src.audio_utils import set_torchaudio_backend
from src.generate.pipeline import AudioGenerationContext
from src.tts_model import ModelManager, GEN_ACTIVE_LOCK
from src.ui import create_ui

backend = set_torchaudio_backend()

# New pipeline and cache system imports
from src.generate.cache.cache_manager import CacheManager
from src.generate.pipeline.coordinator import GenerationCoordinator


# Platform-specific optimization
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def parse_arguments():
    """Parse command line arguments"""
    parser = ArgumentParser()
    parser.add_argument('--share', action='store_true',
                        help="Create a EXTERNAL facing public link using Gradio's servers")
    parser.add_argument("--server", type=str, default='0.0.0.0', help="Server address to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, required=False, default=7860,
                        help="Port to run the server on (default: 7860)")
    parser.add_argument("--root-path", type=str, required=False, default='',
                        help="The path when using a reverse proxy")
    parser.add_argument("--inbrowser", action='store_true', help="Open the UI in a new browser window")
    parser.add_argument("--multilingual", action='store_true', default=False,
                        help="Use the multilingual model (requires more VRAM)")
    parser.add_argument("--clearoutput", action='store_true',
                        help="Remove all folders in audio output directory and exit")
    parser.add_argument("--clearcache", action='store_true', help="Remove all cache files and exit")
    return parser.parse_args()

def initialize_cache_system():
    """Initialize all cache components through the CacheManager."""
    # This loads the config and creates a unified cache interface
    from src.config import get_config
    config = get_config()

    # Create cache manager (already handles directory structure)
    cache_manager = CacheManager(config)

    logger.info("Cache system initialized successfully with the following stats:")
    stats = cache_manager.cache_stats

    # Log cache statistics for visibility
    for cache_type, cache_stats in stats.items():
        logger.info(f"  - {cache_type.replace('_', ' ').capitalize()}: "
                    f"{cache_stats.get('entries', 0)} entries, "
                    f"{cache_stats.get('memory_entries', 0)} memory, "
                    f"{cache_stats.get('disk_entries', 0)} disk, "
                    f"{cache_stats.get('disk_size', 0)/1024/1024:.1f}MB")

    return cache_manager

def clear_all_caches():
    """Utility function to clear ALL cache systems."""
    cache_manager = initialize_cache_system()
    cache_manager.clear_caches(full=True)
    logger.info("All cache systems have been cleared")

def clear_output_directories():
    """Utility function to clear output directories."""
    from src.config import get_config
    config = get_config()

    # Use the same directory structure as the cache manager
    output_dir = config.app_config.globals.audio_cache_dir / "output"

    if not output_dir.exists():
        logger.info(f"Output directory {output_dir} does not exist")
        return 0

    removed_count = 0
    try:
        import shutil
        for item in output_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
                logger.info(f"Removed {item}")
                removed_count += 1
            elif item.is_file():
                item.unlink()
                logger.info(f"Removed {item}")
                removed_count += 1
        logger.info(f"Cleared {removed_count} items from output directory")
        return removed_count
    except Exception as e:
        logger.error(f"Clear output failed: {e}")
        return 0

def main():
    """Main entrypoint: Load model, config, and launch UI."""
    logger.info("Starting SkyrimNet Chatterbox v2.0...")
    logger.info("Initializing new audio generation pipeline with modular cache system")

    args = parse_arguments()

    # Early exit cleanup tasks
    if args.clearcache:
        clear_all_caches()
        exit(0)

    if args.clearoutput:
        clear_output_directories()
        exit(0)

    # Load configuration at startup
    logger.info("Loading SkyrimNet configuration...")
    config = get_config()

    # Initialize cache system
    logger.info("Initializing cache management subsystem...")
    cache_manager = initialize_cache_system()

    # Load TTS model (idempotent)
    model_type = 'multilingual' if config.app_config.globals.multilingual else 'english'
    logger.info(f"Loading {model_type.capitalize()} Model")

    instance = ModelManager.get_instance()
    model = instance.get_model(model_type)

    if model is None:
        logger.error(f"Failed to load {model_type} model. Please check requirements and configuration.")
        exit(1)

    logger.info(f"✓ Model loaded : {model.__class__.__name__} ({model_type}) on {config.app_config.globals.device} (dtype={config.app_config.globals.dtype})")

    # Initialize pipeline coordinator
    logger.info("Initializing audio generation pipeline coordinator...")
    pipeline = GenerationCoordinator()

    # Validate that the pipeline is fully operational
    logger.info("Validating pipeline components...")
    try:
        test_context = AudioGenerationContext(
            text="This is a test",
            audio_prompt_path=None,
            cache_uuid=0,
            voice_stem="default",
            enable_memory_cache=True,
            enable_disk_cache=True,
            model=model,
            device=torch.device(config.app_config.globals.device),
            dtype=config.app_config.globals.dtype,
            config=config,
            cache_manager=cache_manager,
            voice_params={}
        )

        # Basic pipeline validation (doesn't execute generation)
        pipeline.validate(test_context)
        logger.info("✓ Pipeline validation completed successfully")

    except Exception as validation_err:
        logger.warning(f"! Pipeline validation encountered warnings: {str(validation_err)}")
        # Don't fail startup for validation warnings as they may be non-critical

    # Launch UI (Gradio demo)
    logger.info("Preparing UI interface...")
    demo = create_ui(
        cache_manager=cache_manager,
        pipeline=pipeline,
        config=config
    )

    logger.info("✓ UI prepared successfully. Launching server...")

    # Start background warmup task (improves first response time)
    import threading
    def warmup_background():
        with GEN_ACTIVE_LOCK:
            logger.info("Starting background T3 warmup for improved performance")
            try:
                from src.tts_model import warmup_t3
                warmup_t3(model)
                logger.info("✓ T3 warmup completed in background")
            except Exception as warmup_e:
                logger.warning(f"T3 warmup failed in background: {str(warmup_e)}")

    threading.Thread(
        target=warmup_background,
        daemon=True,
        name="T3Warmup"
    ).start()

    # Launch the Gradio server
    logger.info(f"🚀 Server starting at http://{args.server}:{args.port}/")
    if args.share:
        logger.info("⚠️ Public sharing enabled - your UI will be accessible to anyone with the link")

    try:
        demo.queue(
            max_size=12,
            default_concurrency_limit=4,
        ).launch(
            share=args.share,
            server_name=args.server,
            server_port=args.port,
            root_path=args.root_path,
            inbrowser=args.inbrowser,
            favicon_path="skyrim_icon.ico"
        )
    except Exception as launch_error:
        logger.critical(f"Server launch failed: {str(launch_error)}")
        logger.exception("Full traceback:")
        exit(1)

if __name__ == "__main__":
    main()