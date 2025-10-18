"""
Initialization module for cache subsystems.

This module provides a unified entrypoint for accessing all cache functionality
while maintaining proper separation of concerns and avoiding circular dependencies.

Usage:
    from src.generate.cache import CacheManager

    # Initialize early in application startup
    cache_manager = CacheManager(get_config())

    # Then use throughout app:
    cache_manager.get_audio_cache(cache_key)
    cache_manager.index_audio_for_fuzzy(text, audio_path, voice_stem)
"""
from .cache_manager import CacheManager

__all__ = [
    "CacheManager"
]