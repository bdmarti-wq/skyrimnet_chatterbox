"""
Parser for config.txt: Loads globals/flags/modes from INI-style txt file.
Returns (default_config: Dict, config_mode: Dict, global_flags: Dict).
Ignores old voice_overrides block (use voices.json).
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple

from loguru import logger

from .defaults import DEFAULTS  # For fallback bools

def _load_txt_config(config_file_path: str = "skyrimnet_config.txt") -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, bool]]:
    """Parse config.txt for globals/flags/modes. Returns defaults dict, modes dict, flags dict."""
    default_config = {
        'temperature': 0.7,
        'min_p': 0.07,
        'top_p': 1.0,
        'repetition_penalty': 2.0,
        'cfg_weight': 0.45,
        'exaggeration': 0.7
    }
    config_mode = {k: 'default' for k in default_config}
    global_flags = {
        'enable_memory_cache': DEFAULTS.get('enable_memory_cache', True),
        'enable_disk_cache': DEFAULTS.get('enable_disk_cache', True)
    }

    config_path = Path(config_file_path)
    if not config_path.exists():
        logger.warning(f"Config file {config_file_path} not found, using hardcoded defaults")
        return default_config, config_mode, global_flags

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Parser: Globals/flags only (ignore old voice_overrides block)
        in_voice_block = False
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('voice_overrides = {'):
                in_voice_block = True
                logger.warning("Ignoring old voice_overrides block in config.txt – use voices.json")
                continue
            if in_voice_block and line == '}':
                in_voice_block = False
                continue
            if in_voice_block:  # Skip block lines
                continue

            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Global boolean flags
                if key in global_flags:
                    if value.lower() in ['true', 'yes', '1', 'on']:
                        global_flags[key] = True
                    elif value.lower() in ['false', 'no', '0', 'off']:
                        global_flags[key] = False
                    else:
                        logger.warning(f"Invalid boolean value '{value}' for {key}, using default")
                        global_flags[key] = DEFAULTS.get(key, True)

                # Parameter modes/values
                elif key in config_mode:
                    if value.lower() == 'default':
                        config_mode[key] = 'default'
                    elif value.lower() == 'api':
                        config_mode[key] = 'api'
                    else:
                        try:
                            custom_value = float(value)
                            config_mode[key] = 'custom'
                            default_config[key] = custom_value
                            logger.info(f"Using custom {key} value: {custom_value}")
                        except ValueError:
                            logger.warning(f"Invalid value '{value}' for {key}, using default")

        logger.debug(f"TXT config loaded: {len(default_config)} params, {len(global_flags)} flags")
        return default_config, config_mode, global_flags

    except Exception as e:
        logger.error(f"TXT config load failed: {e}, using hardcoded defaults")
        return default_config, config_mode, global_flags