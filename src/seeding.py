"""
Seeding utilities for reproducible TTS generation.
Provides UUID-to-seed mapping and torch seed setting.
"""
import functools
import torch


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