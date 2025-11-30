"""
Deprecated module retained for backward compatibility.

Moved to `src.audio.fallbacks`. Please update imports to:
    from src.audio import get_fallback_wav
"""
from src.audio.fallbacks import get_fallback_wav  # re-export

__all__ = ["get_fallback_wav"]
