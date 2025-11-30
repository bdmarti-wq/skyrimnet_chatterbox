"""
Configuration service helpers.

Provides higher-level operations atop the config layer, such as saving
per-voice overrides with automatic diffing and backups.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import shutil

from loguru import logger

from src.config import get_config
from src.voice_params import get_voice_params as _merge_voice_params


def save_voice_overrides(voice: str,
                         overrides: Dict[str, Any],
                         config=None,
                         create_backup: bool = True,
                         config_path: Optional[Path | str] = None) -> str:
    """Persist per-voice overrides after diffing against effective defaults.

    Merge precedence used for the baseline: models → globals → existing voice.
    Only keys that differ from the baseline are written to the config under
    the voice entry. Optionally creates a timestamped backup of config.json.
    """
    cfg = config or get_config()
    app = getattr(cfg, 'app_config', None)
    if app is None:
        raise ValueError("AppConfig not initialized")

    voice = voice or 'default'
    effective_before = _merge_voice_params(app, voice)

    # Compute minimal set to store (keys actually changed)
    to_store: Dict[str, Any] = {}
    for k, v in (overrides or {}).items():
        if k not in effective_before or effective_before.get(k) != v:
            to_store[k] = v

    # Write only diffs
    for k, v in to_store.items():
        try:
            cfg.set_value(k, v, voice=voice)
        except Exception as e:
            logger.warning(f"Failed setting {k} for voice {voice}: {e}")

    # Determine config path
    cfg_file = Path(config_path) if config_path else Path('config.json')

    # Backup if requested
    if create_backup and cfg_file.exists():
        backups_dir = Path('config_backups')
        backups_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        try:
            shutil.copy2(cfg_file, backups_dir / f"config_{ts}.json")
        except Exception as e:
            logger.warning(f"Backup failed: {e}")

    # Save
    cfg.save_config(create_backup=False, filename=str(cfg_file))
    msg = f"Saved overrides for voice '{voice}'."
    logger.info(msg)
    return msg


__all__ = ["save_voice_overrides"]
