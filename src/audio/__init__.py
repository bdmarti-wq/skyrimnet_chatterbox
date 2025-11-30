"""
src.audio: Centralized audio utilities and transforms.

Re-exports the common entry points so callers can simply import from
`src.audio`.
"""
from __future__ import annotations

# Transforms (stateless NumPy-based post-processing)
from .transforms import (
    trim_trailing_artifacts,
    gate_trailing_phantoms,
    suppress_tail_artifacts,
    reverse_tail_suppress,
    apply_notch,
    apply_eq,
    adjust_speaking_rate,
    apply_fade,
    short_padding_trim_head,
    short_trim_padding,
    apply_post_processing,
)

# Path and file utilities
from .paths import sanitize_input_path, validate_user_audio

# Fallback WAV helpers
from .fallbacks import get_fallback_wav

# General audio utilities (re-export from legacy module for now)
from .utils import *  # noqa: F401,F403 (re-export legacy helpers like is_artifact_laden, pad_short_text, etc.)

__all__ = [
    # transforms
    'trim_trailing_artifacts', 'gate_trailing_phantoms', 'suppress_tail_artifacts',
    'reverse_tail_suppress', 'apply_notch', 'apply_eq', 'adjust_speaking_rate',
    'apply_fade', 'short_padding_trim_head', 'short_trim_padding', 'apply_post_processing',
    # paths
    'sanitize_input_path', 'validate_user_audio',
    # fallbacks
    'get_fallback_wav',
]
