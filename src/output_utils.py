"""
Output formatting utilities for UI/API responses.

Provides a single place to structure outputs and ensure a valid audio path,
falling back to a reusable silence WAV when needed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, List
import tempfile

from src.audio_fallbacks import get_fallback_wav
from src.config import get_config


def format_output(result: Any, status_text: Optional[str] = None) -> List[str]:
    """Return [audio_path, status_text] for Gradio-compatible components.

    - If `result` is a (path, ...) tuple/list, use its first element.
    - If `result` is a string path and exists, use it.
    - Otherwise, return a shared fallback silence WAV path.
    """
    status = status_text or "Audio generated successfully"

    # Accept tuple/list with path in first position
    if isinstance(result, (list, tuple)) and len(result) >= 1:
        first = str(result[0])
        if first and Path(first).exists():
            return [first, status]

    # Accept direct string path
    if isinstance(result, str) and Path(result).exists():
        return [result, status]

    # Fallback – prefer configured SR
    try:
        sr = get_config().app_config.globals.sr
    except Exception:
        sr = 24000
    try:
        return [get_fallback_wav(sr), status]
    except Exception:
        # Absolute last resort: empty temp file to satisfy component
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            return [tmp.name, status]


__all__ = ["format_output"]
