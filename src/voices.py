"""
Voice listing helpers shared across UI and tooling.
"""
from __future__ import annotations

from pathlib import Path
from typing import List
from loguru import logger


def list_available_voice_wavs(voices_dir: str | Path | None) -> List[str]:
    """List .wav files from voices_dir and its 'resampled' subfolder.

    Returns sorted absolute paths as strings. Missing/misconfigured dirs
    are handled gracefully by returning an empty list.
    """
    try:
        if not voices_dir:
            return []
        vdir = Path(voices_dir)
        if not vdir.exists():
            return []
        paths: list[str] = []
        # Top-level wavs
        for p in vdir.glob('*.wav'):
            try:
                paths.append(str(p))
            except Exception:
                continue
        # Resampled subfolder wavs
        rdir = vdir / 'resampled'
        if rdir.exists():
            for p in rdir.glob('*.wav'):
                try:
                    paths.append(str(p))
                except Exception:
                    continue
        return sorted(paths)
    except Exception as e:
        logger.warning(f"Failed listing voice wavs: {e}")
        return []


__all__ = ["list_available_voice_wavs"]
