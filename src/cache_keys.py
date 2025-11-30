"""
Shared cache key generation utilities.

Ensures a single, uniform cache key format is used across the app.

Format: {voice_stem}_{text_hash8}_{exaggeration:.2f}_{uuid_hex8}
"""
from __future__ import annotations

import hashlib


def _hash_text8(text: str) -> str:
    if not text:
        return "empty"
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


def _uuid_hex8(cache_uuid: int | str | None) -> str:
    if isinstance(cache_uuid, int):
        return hex(cache_uuid)[2:][:8]
    if isinstance(cache_uuid, str) and cache_uuid:
        return cache_uuid[:8]
    return "00000000"


def generate_audio_cache_key(voice_stem: str,
                             text: str,
                             exaggeration: float,
                             cache_uuid: int | str | None,
                             stem_only: bool = False) -> str:
    """Generate a stable cache key used by audio/fuzzy caches and context.

    Args:
        voice_stem: normalized voice stem
        text: generation text (used for hash only)
        exaggeration: float, formatted to 2 decimals
        cache_uuid: int/str session id used to vary cache key per-run family
        stem_only: if True, return stem (used by some lookups)
    """
    if stem_only:
        return voice_stem
    text_hash = _hash_text8(text or "")
    uuid_hex = _uuid_hex8(cache_uuid)
    try:
        ex = float(exaggeration)
    except Exception:
        ex = 0.0
    return f"{voice_stem}_{text_hash}_{ex:.2f}_{uuid_hex}"


__all__ = ["generate_audio_cache_key"]
