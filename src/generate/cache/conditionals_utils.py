"""
Shared validators for conditionals objects used across caches and pipeline phases.

This consolidates the logic that decides whether a set of conditionals is
usable (non-empty, structurally valid) and whether it's a known mock/placeholder
object that should never be cached or propagated.
"""
from __future__ import annotations

from typing import Any

import torch

try:
    # Optional T3 types (only present in Chatterbox ML build)
    from src.chatterbox.tts import Conditionals as _T3Conditionals  # type: ignore
    T3_AVAILABLE = True
except Exception:
    _T3Conditionals = object  # type: ignore
    T3_AVAILABLE = False


def is_mock_conditionals(conds: Any) -> bool:
    """Return True if `conds` looks like a mock/placeholder object.

    Heuristics:
    - Explicit None is not mock (it's just absent) → return False here; handled upstream.
    - For T3 Conditionals, consider it mock if it has attribute `is_mock` truthy
      or if it has a `.t3` with `speaker_emb` attribute set but sized 0.
    - For dicts, treat as mock if empty dict or has a sentinel key like 'mock' set True.
    """
    if conds is None:
        return False

    # T3 Conditionals-specific checks
    if T3_AVAILABLE and isinstance(conds, _T3Conditionals):
        if getattr(conds, 'is_mock', False):
            return True
        t3 = getattr(conds, 't3', None)
        if t3 is not None and hasattr(t3, 'speaker_emb'):
            emb = getattr(t3, 'speaker_emb')
            try:
                if isinstance(emb, torch.Tensor) and emb.numel() == 0:
                    return True
            except Exception:
                pass
        return False

    # Dict placeholder
    if isinstance(conds, dict):
        if len(conds) == 0:
            return True
        if conds.get('mock', False) is True:
            return True
        return False

    return False


def is_valid_conditionals(conds: Any) -> bool:
    """Return True if `conds` appears usable for generation.

    Rules:
    - None → False
    - If mock → False
    - torch.Tensor → True when numel > 0 (values may be zeros; still acceptable)
    - T3 Conditionals → True if has a non-empty `t3.speaker_emb` tensor or any
      meaningful attribute present (fallback to True when structure present)
    - dict → True if any value is not None (non-empty structure)
    - other objects → True by default (optimistic, as some backends use custom carriers)
    """
    if conds is None:
        return False
    if is_mock_conditionals(conds):
        return False

    if isinstance(conds, torch.Tensor):
        try:
            return conds.numel() > 0
        except Exception:
            return False

    if T3_AVAILABLE and isinstance(conds, _T3Conditionals):
        t3 = getattr(conds, 't3', None)
        if t3 is not None and hasattr(t3, 'speaker_emb'):
            emb = getattr(t3, 'speaker_emb')
            try:
                return isinstance(emb, torch.Tensor) and emb.numel() > 0
            except Exception:
                return False
        # Fallback: if it quacks like conditionals but no speaker_emb, consider it valid
        return True

    if isinstance(conds, dict):
        try:
            return any(v is not None for v in conds.values())
        except Exception:
            return False

    # Default to True for unknown container types
    return True


__all__ = [
    'is_valid_conditionals',
    'is_mock_conditionals',
]
