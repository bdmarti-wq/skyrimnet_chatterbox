import os
from pathlib import Path
from loguru import logger

def find_project_root(marker_file="skyrimnet_chatterbox.py"):
    current_path = Path(os.getcwd())
    while current_path != current_path.parent:  # Stop at the root directory
        if (current_path / marker_file).exists():
            return current_path
        current_path = current_path.parent

    # Extreme fallback (should rarely happen)
    logger.warning("No project root markers found - using CWD as root")
    return current_path