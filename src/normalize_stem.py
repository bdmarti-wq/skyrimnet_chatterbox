# New Helper: Shared stem normalization (extracted from multiple places; testable: input path → expected stem)
import re
import loguru
from pathlib import Path


def normalize_stem(audio_path: str, provided_stem: str | None = None, min_len: int = 3) -> str:
    """Derive/clean stem from path or provided; handles temps/UUIDs. Returns str >= min_len or fallback."""
    if provided_stem is not None and isinstance(provided_stem, (int, float)):
        provided_stem = str(provided_stem)  # Handle old int calls

    if provided_stem and len(str(provided_stem)) >= min_len:
        # Clean if provided (remove suffixes)
        stem = str(provided_stem).replace('_fixed', '').replace('_padded', '').replace('_resampled', '').replace(
            '_ui_resampled', '').replace('_export', '')
        if len(stem) >= min_len:
            return stem
        loguru.logger.debug(f"Provided stem '{provided_stem}' too short/invalid → derive from path")

    if not audio_path:
        raise ValueError("No audio_path for stem derivation")

    basename = Path(audio_path).stem
    # Regex extract voice before UUID/temp (e.g., 'vp_11_lilia_123hex' → 'vp_11_lilia')
    match = re.match(r'([a-zA-Z0-9_]+[voice]?)(_?[0-9a-f]{15,})?$', basename)
    stem = match.group(1) if match else basename.replace('_fixed', '').replace('_padded', '').replace('_resampled',
                                                                                                      '').replace(
        '_ui_resampled', '').replace('_temp', '')
    if len(stem) < min_len:
        stem = basename  # Fallback to full
    loguru.logger.trace(f"Normalized stem for '{basename}': '{stem}'")
    return stem