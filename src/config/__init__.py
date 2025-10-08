"""
src/config Package Init: Lazy CONFIG + facades/globals.
"""

from .config import load_skyrimnet_config, get_config_value, reload_config  # Facades first
from .defaults import DEVICE, DTYPE, MODEL, MULTILINGUAL, ENABLE_DISK_CACHE, ENABLE_MEMORY_CACHE, FUZZY_CACHE_LIMIT, _CONFIG_CACHE, _USE_API_MODE

# Lazy singleton
def get_config():
    load_skyrimnet_config()  # Ensures init + load
    from .config import CONFIG
    return CONFIG

CONFIG = get_config()  # Triggers lazy init on access

__all__ = [
    'CONFIG', 'get_config',
    'load_skyrimnet_config', 'get_config_value', 'reload_config',
    'DEVICE', 'DTYPE', 'MODEL', 'MULTILINGUAL', 'ENABLE_DISK_CACHE', 'ENABLE_MEMORY_CACHE',
    'FUZZY_CACHE_LIMIT', '_CONFIG_CACHE', '_USE_API_MODE'
]