"""
src/config Package Init: Lazy CONFIG + facades.
"""
from typing import Optional
from .utils import find_project_root
from .config import (
    get_config, load_skyrimnet_config, get_config_value, reload_config
)  # All facades; no legacy

# Lazy CONFIG (returns singleton on access; triggers load via get_config())
class _LazyConfig:
    _instance: Optional['Config'] = None  # Type hint for clarity

    def __getattr__(self, name: str):
        """Lazy forward: Load instance on first attr access (e.g., CONFIG.device)."""
        if self._instance is None:
            self._instance = get_config()  # Triggers init + load
        # Forward to real instance (handles methods/attrs like .device, .app_config, .get_value)
        return getattr(self._instance, name)

    def __setattr__(self, name: str, value):
        """Optional: Forward sets (e.g., for testing; rarely needed)."""
        if name in ['_instance']:  # Skip internal
            super().__setattr__(name, value)
        else:
            if self._instance is None:
                self._instance = get_config()
            setattr(self._instance, name, value)

    def __dir__(self):
        """Support tab-completion in REPLs (after load)."""
        if self._instance is None:
            return ['get_config']  # Minimal before load
        return dir(self._instance)

# Export as global (lazy)
CONFIG = _LazyConfig()

__all__ = [
    'CONFIG',
    'get_config',
    'load_skyrimnet_config',
    'get_config_value',
    'reload_config',
    'find_project_root'
]