# import tempfile
# from typing import Optional, Any
# from loguru import logger
# import torch
# from pathlib import Path
#
# from src.config import get_config, find_project_root
# from .utils import get_audio_cache_key, get_content_hash, create_dummy_conds, save_torchaudio_wav
# from .conditionals_cache import ConditionalsCache
# from .voice_reference import VoiceReferenceCache, DEBUG_FORCE_SERVER_AUDIO
# from .audio_cache import AudioCache
# # Import and expose core cache functionality
# from .initialize import (
#     VOICE_CACHE,
#     CONDITIONALS_CACHE,
#     AUDIO_CACHE, init_caches
# )
#
# # Example usage:
# project_root = find_project_root()
# if project_root:
#     print(f"Project root found at: {project_root}")
# else:
#     print("Project root not found.")
#
# # Initialize global cache systems
# config = get_config()
# CACHE_ROOT = cache_root = project_root / 'cache'
#
# VOICE_CACHE = VoiceReferenceCache(cache_root)
# CONDITIONALS_CACHE = ConditionalsCache(cache_root)
# AUDIO_CACHE = AudioCache(cache_root)
#
# def get_voice_conditionals(model, voice_stem: str, voice_path: str, force_update: bool = False) -> tuple[bool, str, str, dict]:
#     """
#     Get conditionals for a voice, processing new references as needed.
#
#     CRITICAL: Ensure this fully completes and returns BEFORE conditionals are needed
#     """
#     if DEBUG_FORCE_SERVER_AUDIO:
#         logger.warning("⚠️⚠️⚠️ DEBUG OVERRIDDEN: Forcing conditionals update (bypassing cache)")
#         force_update = True
#
#     # Process the reference and return immediate results
#     success, ref_path, cond_key, voice_config = VOICE_CACHE.process_new_reference(
#         voice_stem,
#         voice_path,
#         force_update=force_update
#     )
#
#     return success, ref_path, cond_key, voice_config
#
# def load_conditionals(cache_key: str, model, device: str, dtype: torch.dtype) -> Optional[Any]:
#     """Load conditionals from cache by key."""
#     return CONDITIONALS_CACHE.get(cache_key, model, device, dtype)
#
# def save_conditionals(cache_key: str, conditionals: Any, model=None, device: str = "cuda",
#                       dtype: torch.dtype = torch.float32) -> bool:
#     """Save conditionals to cache."""
#     return CONDITIONALS_CACHE.save(cache_key, conditionals, model, device, dtype)
#
#
# def get_audio_cache(key: str) -> Optional[str]:
#     """Get path to cached audio."""
#     return AUDIO_CACHE.get(key)
#
# def clear_cache_files():
#     AUDIO_CACHE.clear()
#     CONDITIONALS_CACHE.clear()
#
# def set_audio_cache(key: str, path: str) -> None:
#     """Cache a path to generated audio."""
#     AUDIO_CACHE.set(key, path)
#
# # Initialize caches at import time
# init_caches()
#
#
#
# __all__ = [
#     # Voice reference functions
#     'get_voice_conditionals',
#
#     # Conditionals cache functions
#     'load_conditionals',
#     'save_conditionals',
#
#     # Audio cache functions
#     'get_audio_cache',
#     'set_audio_cache',
#
#     # Utility functions
#     'create_dummy_conds',
#     'save_torchaudio_wav',
#     'get_content_hash',
#     'get_audio_cache_key',
#
#     # Cache instances
#     'VOICE_CACHE',
#     'CONDITIONALS_CACHE',
#     'AUDIO_CACHE',
#
#     # Initialization
#     'init_caches'
# ]