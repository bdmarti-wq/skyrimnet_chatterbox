"""
Compatibility wrapper for legacy audio utilities.

New code should import from `src.audio` directly. This module re-exports
symbols from the legacy `src.audio_utils` module to keep changes minimal
while the codebase migrates.
"""
from __future__ import annotations

from loguru import logger

try:
    # Re-export everything from the legacy module
    from src.audio_utils import *  # type: ignore  # noqa: F401,F403
    logger.debug("Using legacy src.audio_utils through src.audio.utils wrapper")
except Exception as e:  # pragma: no cover - defensive
    logger.warning(f"audio.utils wrapper failed to import legacy audio_utils: {e}")
