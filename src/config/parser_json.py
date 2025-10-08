"""
Parser for voices.json: Loads voice overrides from JSON file.
Supports multi-path search (root, config/, current). Handles list fallback to dict.
Returns voice_overrides: Dict[str, Dict].
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

from loguru import logger

def _load_voices_json(possible_paths: Optional[list] = None) -> Dict[str, Dict]:
    """Load voices.json from possible paths. Returns {voice: params} or {} fallback."""
    if possible_paths is None:
        possible_paths = [
            Path(__file__).parent.parent.parent / "voices.json",  # Root (project root/voices.json)
            Path(__file__).parent / "voices.json",                # src/config/voices.json
            Path("voices.json"),                                   # Current dir fallback
        ]

    voice_overrides = {}
    loaded_path = None
    for voices_path in possible_paths:
        if voices_path.exists():
            try:
                with open(voices_path, 'r') as f:
                    voices_data = json.load(f)
                # Validate/Flatten if list (optional fallback)
                if isinstance(voices_data, list):
                    voices_data = {item.get('voice', f'voice_{i}'): item for i, item in enumerate(voices_data)}
                voice_overrides = voices_data
                loaded_path = voices_path
                logger.info(f"✓ Loaded voices.json: {loaded_path} ({len(voice_overrides)} voices: {list(voice_overrides.keys())})")
                return voice_overrides
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid voices.json {voices_path}: {e} – skipping")
            except Exception as e:
                logger.error(f"Load voices.json {voices_path} failed: {e} – trying next")
        else:
            logger.trace(f"Tried voices.json path (not found): {voices_path}")

    # Fallback if no file
    logger.warning("voices.json not found in expected paths – no per-voice overrides (add to root for voices)")
    fallback_voices = {'global': {}}  # Minimal fallback
    return fallback_voices