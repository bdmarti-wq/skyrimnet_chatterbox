"""
Seeding utilities for reproducible TTS generation.
Provides UUID-to-seed mapping and torch seed setting.
"""
import functools
import torch
import time
import random


@functools.cache
def cpp_uuid_to_seed(uuid_64: int) -> int:
    """
    Convert a 64-bit UUID to a valid PyTorch seed (0 to 2^32 - 1).
    Uses hash() for better distribution across the seed space.

    Args:
        uuid_64: Input UUID as int (64-bit).

    Returns:
        int: Seeded value in [0, 2^32 - 1] for torch.manual_seed.

    Notes:
        Cached for repeated UUIDs (fast lookup).
        Ensures consistent voice output for same UUID (reproducible cloning).
    """
    return abs(hash(uuid_64)) % (2 ** 32)


def set_seed(seed: int):
    """
    Set random seeds for reproducible generation.

    Args:
        seed: Integer seed (0 to 2^32 - 1).

    Notes:
        Sets torch and CUDA seeds (global state; call before generation).
        For multiprocessing, use torch.manual_seed_all (if needed, extend here).
        Deterministic for same seed/path in pipeline (e.g., cached gens).
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    # Optional: For numpy (if used in fuzzy/audio_utils)
    # import numpy as np
    # np.random.seed(seed)
    # For full determinism (gradients, etc.): torch.backends.cudnn.deterministic = True


def resolve_seed(uuid_seed: int | None, provided_seed: int | None = None, randomize: bool = False) -> int:
    """Resolve a final integer seed for generation.

    Precedence:
    - If provided_seed is not None → return it as-is (trusted explicit value).
    - Else if randomize is True → return a time/random-derived 32-bit seed.
    - Else → derive from uuid_seed using cpp_uuid_to_seed (stable per UUID).

    Args:
        uuid_seed: Optional UUID-like 64-bit int from UI/session.
        provided_seed: Optional explicit seed value to force.
        randomize: If True and provided_seed is None, generate a random seed.

    Returns:
        int: Final seed in [0, 2^32 - 1].
    """
    if isinstance(provided_seed, int) and provided_seed >= 0:
        return int(provided_seed) % (2 ** 32)
    if randomize:
        # Mix monotonic time and random to get a wide range; mask to 32 bits
        t = int(time.time_ns() & 0xFFFFFFFF)
        r = random.getrandbits(32)
        return (t ^ r) & 0xFFFFFFFF
    # Fallback to uuid-based deterministic seed (handles None too)
    try:
        return cpp_uuid_to_seed(int(uuid_seed or 0))
    except Exception:
        return cpp_uuid_to_seed(0)