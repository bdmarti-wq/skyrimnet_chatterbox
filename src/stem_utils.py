# ADD THIS CODE NEAR THE TOP OF THE FILE (BEFORE OTHER FUNCTIONS)
from pathlib import Path
from typing import Optional
import loguru


def extract_voice_stem(file_path: Optional[str], strict: bool = False) -> str:
    """
    Extracts a normalized voice stem from a file path by removing common suffixes.

    Handles patterns like:
    - 'filename_fixed.wav' → 'filename'
    - 'filename_padded.wav' → 'filename'
    - 'cs_coralyn_voice.wav' → 'cs_coralyn' (special case)

    Args:
        file_path: Path to voice audio file (can be None)
        strict: If True, validates stem quality; if False, returns safe defaults

    Returns:
        Clean voice stem string

    Examples:
        extract_voice_stem("/path/to/voice_fixed.wav") → "voice"
        extract_voice_stem("/path/to/cs_coralyn_voice.wav") → "cs_coralyn"
        extract_voice_stem(None) → "default"
    """
    if not file_path:
        return 'default'

    try:
        path = Path(file_path)
        stem = path.stem

        # Remove common processing suffixes
        for suffix in ['_fixed', '_padded', '_resampled', '_ui_resampled']:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]

        # Special case: remove '_voice' suffix (6 characters)
        if stem.endswith('_voice'):
            stem = stem[:-6]

        # Ensure valid stem length
        if not stem or len(stem) < 3:
            return 'default' if not strict else stem

        return stem
    except Exception as e:
        loguru.logger.debug(f"Error extracting voice stem from '{file_path}': {e}")
        return 'default'


def get_logging_voice_name(file_path: Optional[str]) -> str:
    """
    Extracts a voice name suitable for logging messages.

    Args:
        file_path: Path to voice audio file (can be None)

    Returns:
        Voice name for logs or "No ref audio" if none provided

    Examples:
        get_logging_voice_name("/path/to/voice.wav") → "voice"
        get_logging_voice_name(None) → "No ref audio"
    """
    if not file_path:
        return "No ref audio"
    return extract_voice_stem(file_path)


def validate_voice_stem(voice_stem: str) -> bool:
    """
    Ensures the voice stem is valid for processing.

    Args:
        voice_stem: Candidate voice stem string

    Returns:
        True if valid, False otherwise
    """
    # Rules for valid voice stems
    if not voice_stem or len(voice_stem) < 3:
        return False
    if any(c in voice_stem for c in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']):
        return False
    return True
