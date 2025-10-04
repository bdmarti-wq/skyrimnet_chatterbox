from pathlib import Path

import torch

import loguru
from loguru import logger

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MODEL = None
MULTILINGUAL = False
# Cache flags - defaults that can be overridden by skyrimnet_config.txt
ENABLE_DISK_CACHE = True
ENABLE_MEMORY_CACHE = True
_CONFIG_CACHE = None
_CONFIG_FILE = "skyrimnet_config.txt"
# Testing flag - when True, bypasses config loading and uses all API values
_USE_API_MODE = False


def load_skyrimnet_config():
    """Load configuration from skyrimnet_config.txt with error handling"""
    global _CONFIG_CACHE, ENABLE_MEMORY_CACHE, ENABLE_DISK_CACHE

    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    # Default configuration
    default_config = {
        'temperature': 0.8,
        'min_p': 0.07,
        'top_p': 1.0,
        'repetition_penalty': 2.0,
        'cfg_weight': 0.0,  # Speed optimized default
        'exaggeration': 0.7
    }

    global_flags = {
        'enable_memory_cache': ENABLE_MEMORY_CACHE,
        'enable_disk_cache': ENABLE_DISK_CACHE
    }

    config_mode = {
        'temperature': 'default',
        'min_p': 'default',
        'top_p': 'default',
        'repetition_penalty': 'default',
        'cfg_weight': 'default',
        'exaggeration': 'default'
    }

    try:
        config_path = Path(__file__).parent / _CONFIG_FILE
        if not config_path.exists():
            logger.warning(f"Config file {_CONFIG_FILE} not found, using hardcoded defaults")
            _CONFIG_CACHE = (default_config, config_mode, global_flags)
            return _CONFIG_CACHE

        with open(config_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Handle global boolean flags
                if key in global_flags:
                    if value.lower() in ['true', 'yes', '1', 'on']:
                        global_flags[key] = True
                        # Update global variables
                        if key == 'enable_memory_cache':
                            ENABLE_MEMORY_CACHE = True
                        elif key == 'enable_disk_cache':
                            ENABLE_DISK_CACHE = True
                        logger.info(f"Setting {key} to True")
                    elif value.lower() in ['false', 'no', '0', 'off']:
                        global_flags[key] = False
                        # Update global variables
                        if key == 'enable_memory_cache':
                            ENABLE_MEMORY_CACHE = False
                        elif key == 'enable_disk_cache':
                            ENABLE_DISK_CACHE = False
                        logger.info(f"Setting {key} to False")
                    else:
                        logger.warning(f"Invalid boolean value '{value}' for {key}, using default")

                # Handle parameter modes
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

        logger.info(f"Loaded config: {config_mode}")
        logger.info(f"Global flags: {global_flags}")
        _CONFIG_CACHE = (default_config, config_mode, global_flags)
        return _CONFIG_CACHE

    except Exception as e:
        logger.error(f"Error reading config file {_CONFIG_FILE}: {e}, using hardcoded defaults")
        _CONFIG_CACHE = (default_config, config_mode, global_flags)
        return _CONFIG_CACHE


def get_config_value(param_name, api_value, defaults, modes, bypass_config=False):
    """Get the appropriate value based on configuration mode"""
    if bypass_config:
        # API mode: use API value with fallback to reasonable defaults
        fallback_defaults = {
            'temperature': 0.9,
            'min_p': 0.05,
            'top_p': 1.0,
            'repetition_penalty': 2.0,
            'cfg_weight': 0.0,
            'exaggeration': 0.55
        }
        return api_value if api_value is not None else fallback_defaults.get(param_name, 0.0)

    mode = modes.get(param_name, 'default')

    if mode == 'api':
        return api_value if api_value is not None else defaults[param_name]
    else:  # 'default' or 'custom'
        return defaults[param_name]


def reload_config():
    """Force reload of configuration file"""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None
    return load_skyrimnet_config()