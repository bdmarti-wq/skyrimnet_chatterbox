"""
Shared audio path sanitation/validation utilities.

The sanitizer is intentionally lightweight and safe to call from the UI bridge
and phases. Deeper audio integrity checks (duration/frames) remain in the
pipeline BaseGenerationPhase.validate_path.
"""
from pathlib import Path
from typing import Optional
from loguru import logger


def sanitize_input_path(p: Optional[str]) -> Optional[str]:
    """Return a safe audio file path or None.

    - Treat None/empty as None
    - Drop our placeholder .noop files produced by the Gradio patch
    - Drop non-existent paths, directories, or zero-byte files
    """
    try:
        if not p:
            return None
        s = str(p).strip()
        if not s:
            return None
        # Our gradio patch substitutes temp files with .noop suffix for directories
        if s.lower().endswith('.noop'):
            logger.debug("[SANITIZE] Ignoring placeholder path (noop): {}", s)
            return None
        path = Path(s)
        if not path.exists():
            logger.debug("[SANITIZE] Ignoring non-existent path: {}", s)
            return None
        if path.is_dir():
            logger.debug("[SANITIZE] Ignoring directory path: {}", s)
            return None
        try:
            if path.stat().st_size == 0:
                logger.debug("[SANITIZE] Ignoring zero-byte file: {}", s)
                return None
        except Exception:
            return None
        return s
    except Exception:
        return None
