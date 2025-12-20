"""
src/gradio_patch.py

Installs a defensive Gradio preprocessing patch to suppress noisy InvalidPathError
when a directory path (or any non-upload absolute path) leaks into Gradio's
preprocess/cache pipeline. The patch also logs concise diagnostics to help
identify the upstream source.

Safe to import multiple times; installation is idempotent.
"""
from __future__ import annotations

import os
import pathlib as _pl
import tempfile as _tmp
import traceback as _tb
import inspect as _inspect
from typing import Any

try:
    from loguru import logger
except Exception:  # pragma: no cover
    class _Stub:
        def debug(self, *a, **k): pass
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
    logger = _Stub()  # type: ignore

try:  # Guard import so test envs without Gradio still work
    import gradio.processing_utils as _pu  # type: ignore
except Exception:  # pragma: no cover
    _pu = None

_P = _pl.Path
_INSTALLED = False


def _is_dirlike_path(p: Any) -> bool:
    try:
        if not p:
            return False
        pp = _P(str(p))
        return pp.exists() and pp.is_dir()
    except Exception:
        return False


def ensure_gradio_patch_installed() -> bool:
    """Install the patch once. Returns True if installed in this call or already installed."""
    global _INSTALLED
    if _INSTALLED:
        return True
    if _pu is None:
        logger.debug("[UI.DIAG] Gradio not present; skipping preprocess patch")
        _INSTALLED = True  # prevent repeated attempts in no-gradio contexts
        return True

    # Preserve originals
    _orig_check_allowed = getattr(_pu, "_check_allowed", None)
    _orig_move_to_cache = getattr(_pu, "_move_to_cache", None)
    _orig_save_file_to_cache = getattr(_pu, "save_file_to_cache", None)
    _orig_hash_file = getattr(_pu, "hash_file", None)

    async def _diag_move_to_cache(payload: Any, check_in_upload_folder: bool = True):
        try:
            p = getattr(payload, "path", None)
            if _is_dirlike_path(p):
                logger.warning(
                    f"[UI.DIAG] Directory path encountered during preprocessing: {p}\n{''.join(_tb.format_stack(limit=6))}"
                )
        except Exception:
            pass
        if callable(_orig_move_to_cache):
            res = _orig_move_to_cache(payload, check_in_upload_folder)
            if _inspect.isawaitable(res):
                return await res
            return res
        return payload

    def _safe_check_allowed(path: str, check_in_upload_folder: bool = True) -> None:
        try:
            if _is_dirlike_path(path):
                logger.debug(f"[UI.DIAG] _check_allowed suppressed for directory: {path}")
                return
            if callable(_orig_check_allowed):
                return _orig_check_allowed(path, check_in_upload_folder)
            return None
        except Exception as e:
            try:
                from gradio.exceptions import InvalidPathError  # type: ignore
            except Exception:
                InvalidPathError = tuple()  # type: ignore
            if isinstance(e, InvalidPathError):
                logger.debug(f"[UI.DIAG] _check_allowed InvalidPathError suppressed for path: {path}")
                return
            raise

    def _safe_hash_file(file_path: str) -> str:
        if _is_dirlike_path(file_path):
            try:
                return "dir_" + str(abs(hash(str(_P(file_path).resolve()))) % (10**8))
            except Exception:
                return "dir_unknown"
        if callable(_orig_hash_file):
            return _orig_hash_file(file_path)
        return "unknown"

    def _safe_save_file_to_cache(file_path: str, cache_dir: str | None = None) -> str:
        if _is_dirlike_path(file_path):
            try:
                base_dir = _P(cache_dir) if cache_dir else _P(_tmp.gettempdir())
                base_dir.mkdir(parents=True, exist_ok=True)
                fd, tmp = _tmp.mkstemp(dir=str(base_dir), suffix=".noop")
                os.close(fd)
                logger.debug(
                    f"[UI.DIAG] save_file_to_cache received directory; substituted temp file: {file_path} -> {tmp}"
                )
                return tmp
            except Exception:
                pass
        if callable(_orig_save_file_to_cache):
            return _orig_save_file_to_cache(file_path, cache_dir)
        return str(file_path)

    # Apply wrappers idempotently
    try:
        if callable(_orig_move_to_cache):
            _pu._move_to_cache = _diag_move_to_cache  # type: ignore[attr-defined]
        if callable(_orig_check_allowed):
            _pu._check_allowed = _safe_check_allowed  # type: ignore[attr-defined]
        if callable(_orig_hash_file):
            _pu.hash_file = _safe_hash_file  # type: ignore[attr-defined]
        if callable(_orig_save_file_to_cache):
            _pu.save_file_to_cache = _safe_save_file_to_cache  # type: ignore[attr-defined]
        logger.info("[UI.DIAG] Gradio cache preprocess patch installed (dir-safe mode)")
        _INSTALLED = True
    except Exception as _e:  # pragma: no cover
        logger.warning(f"[UI.DIAG] Failed to install Gradio preprocess patch: {_e}")
        _INSTALLED = False
    return _INSTALLED


# Install on import for convenience
ensure_gradio_patch_installed()
