"""
Deprecated module retained for backward compatibility.

Moved to `src.audio.paths`. Please update imports to:
    from src.audio import sanitize_input_path, validate_user_audio
"""
from src.audio.paths import sanitize_input_path, validate_user_audio  # re-export
