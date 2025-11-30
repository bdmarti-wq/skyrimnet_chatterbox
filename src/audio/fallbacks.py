"""
Shared audio fallback utilities.

"""
from pathlib import Path
import tempfile
import numpy as np
from scipy.io import wavfile


def get_fallback_wav(sr: int = 24000, duration_s: float = 0.5) -> str:
    """Return a path to a reusable silence WAV file at the given sample rate.

    The file is created once in the system temp directory and reused across calls.
    """
    temp_dir = Path(tempfile.gettempdir())
    temp_dir.mkdir(parents=True, exist_ok=True)
    silence_path = temp_dir / f"silence_{sr}.wav"

    if not silence_path.exists():
        n = int(sr * float(duration_s))
        if n <= 0:
            n = int(sr * 0.5)
        silence = np.zeros((n,), dtype=np.float32)
        wavfile.write(str(silence_path), int(sr), silence)
    return str(silence_path)
