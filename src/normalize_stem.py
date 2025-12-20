# Revised src/normalize_stem.py (simplified; drop unused params if no callers use them)
import re
from pathlib import Path
from loguru import logger  # Use logger instead of loguru (it's an alias)

def normalize_stem(audio_path: str, min_len: int = 3) -> str:
    """Derive/clean stem from path; handles temps/UUIDs. Returns str >= min_len or fallback."""
    if not audio_path:
        raise ValueError("No audio_path for stem derivation")

    basename = Path(audio_path).stem
    # Regex: Extract base before UUID/temp suffix (e.g., 'jjsofiavoicetype_upload_123hex' → 'jjsofiavoicetype_upload')
    # Simplified regex: Assumes UUID is hex >10 chars at end; adjust if needed.
    match = re.match(r'([a-zA-Z0-9_]+(?:_upload_[a-f0-9]+)?)(_?[0-9a-f]{10,})?$', basename)
    stem = match.group(1) if match else basename

    # Clean common suffixes (legacy from old code)
    suffixes = ['_fixed', '_padded', '_resampled', '_ui_resampled', '_export', '_temp']
    for suffix in suffixes:
        stem = stem.replace(suffix, '')

    # Ensure min length (for uploads/short names)
    if len(stem) < min_len:
        stem = basename  # Fallback to full

    logger.trace(f"Normalized stem for '{basename}': '{stem}'")
    return stem