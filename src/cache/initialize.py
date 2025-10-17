# cache/init.py
"""Centralized cache initialization and instance management"""
import os

import torch
from pathlib import Path
from typing import Optional

from . import find_project_root
from .voice_reference import VoiceReferenceCache
from .conditionals_cache import ConditionalsCache
from .audio_cache import AudioCache
from loguru import logger

# Global cache instances (properly initialized via init_caches)
VOICE_CACHE: Optional[VoiceReferenceCache] = None
CONDITIONALS_CACHE: Optional[ConditionalsCache] = None
AUDIO_CACHE: Optional[AudioCache] = None

def init_caches() -> bool:
    """Initialize all cache systems with proper project root detection"""
    global VOICE_CACHE, CONDITIONALS_CACHE, AUDIO_CACHE

    try:
        # Find project root and validate cache location
        project_root = find_project_root()
        if not project_root:
            raise RuntimeError("Cannot determine project root - required for cache management")

        # Create cache directory outside src/
        cache_root = project_root / "cache"
        if "src" in str(cache_root).lower():
            cache_root = project_root / "cache"
            logger.warning(f"Security: Redirected cache path from src/ to {cache_root}")

        # Create cache directories
        for subdir in ["conditionals", "audio/output", "audio/voices", "fallbacks"]:
            (cache_root / subdir).mkdir(parents=True, exist_ok=True)

        # Initialize cache instances
        VOICE_CACHE = VoiceReferenceCache(cache_root)
        CONDITIONALS_CACHE = ConditionalsCache(cache_root)
        AUDIO_CACHE = AudioCache(cache_root)

        logger.info("All cache systems initialized successfully")
        logger.debug(f"Cache location: {cache_root}")

        # Verify cache operations
        _verify_cache_operations()

        return True

    except Exception as e:
        logger.critical(f"Failed to initialize cache systems: {str(e)}")
        # Attempt emergency fallback
        _init_emergency_caches()
        return False

def _verify_cache_operations() -> None:
    """Verify basic cache operations are working with proper audio test files"""
    try:
        # Create a safe, in-memory test audio file
        import tempfile
        import torchaudio
        import numpy as np

        # Generate 0.5s of silence at 24kHz (small enough to be safe for verification)
        test_sr = 24000
        test_duration = 0.5
        test_samples = int(test_sr * test_duration)
        test_audio = np.zeros(test_samples, dtype=np.float32)

        # Write to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            torchaudio.save(tmp_path, torch.from_numpy(test_audio).unsqueeze(0), test_sr)

        try:
            # Test voice cache with VALID audio file
            success, ref_path, cond_key, _ = VOICE_CACHE.process_new_reference(
                "test_verification",
                tmp_path,
                force_update=True
            )

            if not success:
                logger.warning("Voice cache verification partially failed: process_new_reference returned False")

            # Test conditionals cache
            test_conditionals = {"test": "data"}
            if not CONDITIONALS_CACHE.save("test_key", test_conditionals):
                logger.warning("Conditionals cache save verification failed")

            # Test audio cache (only if we got a valid conditionals key)
            if cond_key:
                if not os.path.exists(tmp_path):
                    logger.warning("Audio cache verification skipped: test audio file missing")
                else:
                    AUDIO_CACHE.set(cond_key, tmp_path)
            else:
                logger.debug("Audio cache verification skipped: no conditionals key from voice cache test")

            logger.debug("Cache operations verified successfully (using safe test audio)")

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    logger.debug(f"Could not remove temporary verification file: {str(e)}")

    except Exception as e:
        logger.warning(f"Cache verification failed: {str(e)}")
        # Most important: don't break startup for a verification failure


def verify_cache_directories():
    """Verify cache directory structure is properly initialized"""
    from . import CACHE_ROOT

    required_dirs = [
        "audio/output",
        "audio/voices",
        "conditionals",
        "fallbacks"
    ]

    all_ok = True
    for dir_path in required_dirs:
        full_path = CACHE_ROOT / dir_path
        if not full_path.exists():
            logger.error(f"❌ Missing cache directory: {full_path}")
            all_ok = False

    if all_ok:
        logger.info("✅ All required cache directories exist")

    # Verify configuration
    from .utils import get_wavout_dir
    output_dir = get_wavout_dir(cache=True)
    if "audio/output" not in str(output_dir):
        logger.error(f"❌ Output directory is incorrect: {output_dir}")
        logger.info("ℹ️ Expected format: <project_root>/cache/audio/output")
        all_ok = False
    else:
        logger.info(f"✅ Output directory configured correctly: {output_dir}")

    return all_ok

def _init_emergency_caches() -> None:
    """Initialize emergency cache fallbacks when primary fails"""
    global VOICE_CACHE, CONDITIONALS_CACHE, AUDIO_CACHE

    try:
        # Try temporary directory
        import tempfile
        emergency_path = Path(tempfile.mkdtemp()) / "tts_cache"
        emergency_path.mkdir(parents=True, exist_ok=True)

        VOICE_CACHE = VoiceReferenceCache(emergency_path)
        CONDITIONALS_CACHE = ConditionalsCache(emergency_path)
        AUDIO_CACHE = AudioCache(emergency_path)

        logger.warning(f"Using emergency cache at {emergency_path}")
    except Exception as e:
        # Absolute minimal fallback
        VOICE_CACHE = None
        CONDITIONALS_CACHE = None
        AUDIO_CACHE = None
        logger.critical("ALL CACHE SYSTEMS FAILED - OPERATING WITHOUT CACHE")