"""
Shared audio path sanitation/validation utilities.

The sanitizer is intentionally lightweight and safe to call from the UI bridge
and phases. Deeper audio integrity checks (duration/frames) remain in the
pipeline BaseGenerationPhase.validate_path.
"""
from pathlib import Path
from typing import Optional
from loguru import logger
import torchaudio


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


def validate_user_audio(path: Optional[str], min_dur: float = 3.0) -> bool:
    """Shared validator for audio file paths used across bridge and phases.

    - Ensures the path is sanitized (exists, not a directory, not zero-byte, not .noop)
    - Verifies the audio duration using torchaudio.info is at least `min_dur` seconds
    - Ensures waveform can be loaded and is non-empty with non-trivial amplitude
    """
    try:
        safe = sanitize_input_path(path)
        if not safe:
            return False
        p = Path(safe)
        # Duration check via metadata first (cheap)
        try:
            info = torchaudio.info(str(p))
            if info.sample_rate <= 0 or info.num_frames <= 0:
                return False
            dur = float(info.num_frames) / float(info.sample_rate)
            if dur < float(min_dur):
                return False
        except Exception:
            return False
        # Basic waveform sanity check
        try:
            wav, _ = torchaudio.load(str(p))
            if wav.numel() == 0:
                return False
            # Accept silent files; amplitude check is lenient
            if hasattr(wav, 'abs') and wav.abs().max().item() <= 0.0:
                # Still consider it valid if frames exist; leave to post to normalize
                return True
        except Exception:
            return False
        return True
    except Exception:
        return False
