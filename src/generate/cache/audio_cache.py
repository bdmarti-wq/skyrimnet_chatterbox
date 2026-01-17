import os
import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

# Import for fallback values (only used if no passed config or attr missing)
from src.config import get_config_value

class AudioCache:
    _instance = None  # Singleton instance
    _lock = threading.Lock()  # Thread-safe init

    def __new__(cls, cache_dir: Path, config=None):
        """Singleton constructor: Ensures only one instance across the app."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-check for races
                    cls._instance = super(AudioCache, cls).__new__(cls)
                    cls._instance._initialized = False  # Flag: init only once
                    logger.info(f"AudioCache singleton #1 created (ID={id(cls._instance):x})")
        else:
            logger.debug(f"AudioCache singleton reuse (ID={id(cls._instance):x})")
        return cls._instance

    def _get_nested_config(self, section: str, key: str, default=None) -> Any:
        """Helper: Get nested config value from self.config (attr chain); fallback to global get_config_value."""
        if not self.config:
            # No passed config: Direct global fetch
            return get_config_value(f'app_config.globals.{section}.{key}', default=default)

        # Chained attr access on self.config (handles Pydantic/objects)
        try:
            app_config = getattr(self.config, 'app_config')
            globals_obj = getattr(app_config, 'globals')
            section_obj = getattr(globals_obj, section)
            return getattr(section_obj, key)
        except (AttributeError, KeyError):
            # Any missing level: Fallback to global
            return get_config_value(f'app_config.globals.{section}.{key}', default=default)

    def __init__(self, cache_dir: Path, config=None):
        """Initialize the audio cache system (only runs once for the singleton)."""
        # Singleton check—skip if already initialized
        if hasattr(self, '_initialized') and self._initialized:
            # Just return; use existing state
            return

        # Required params
        self.cache_dir = cache_dir
        self.cache_file = self.cache_dir / "audio_cache.json"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Store config for helper use
        self.config = config

        # Config for eviction limits (concise via helper; prioritizes passed config, fallback to globals)
        self.max_entries = self._get_nested_config('audio', 'max_cache_entries', default=1000)
        self.max_total_size_mb = self._get_nested_config('audio', 'max_cache_size_mb', default=1024)

        # In-memory cache of audio paths (shared across all references)
        self.audio_cache: Dict[str, dict] = {}  # Dict of {'path': str, 'size': int, 'added_time': float, 'last_used': float}
        self.total_size = 0  # Total cached size in bytes (for eviction)
        self.cache_lock = threading.RLock()
        self.save_interval = 5.0
        self.last_save_time = 0
        self._save_event = threading.Event()

        # Load existing cache (only once)
        self.load_cache()

        # Start background save worker
        self._start_save_worker()

        # Singleton flag (set after full init)
        self._initialized = True

        # Logging with the resolved values (no dependency on internal config attrs)
        logger.info(f"AudioCache singleton initialized at {self.cache_dir} (max_entries={self.max_entries}, max_size={self.max_total_size_mb}MB)")

    def _start_save_worker(self) -> None:
        """Start background thread for throttled disk persistence."""
        def worker():
            while True:
                # Wait for signal or timeout
                signaled = self._save_event.wait(timeout=self.save_interval)
                
                # Check for shutdown (using _initialized as a proxy, or we could add a signal)
                # For now, we'll just keep it running as a daemon thread.
                
                if signaled or (time.time() - self.last_save_time >= 60.0):
                    self.save_cache()
                    self._save_event.clear()
        
        threading.Thread(
            target=worker,
            daemon=True,
            name="AudioCacheSaver"
        ).start()

    def load_cache(self) -> None:
        """Load audio cache metadata from disk to memory (only once for the singleton)."""
        with self.cache_lock:
            self.total_size = 0
            self.audio_cache = {}
            if self.cache_file.exists():
                try:
                    with open(self.cache_file, 'r') as f:
                        loaded_data = json.load(f)
                    # Handle old-format dict (str paths) vs. new-format (dict entries)
                    if isinstance(loaded_data, dict) and all(isinstance(v, str) for v in loaded_data.values()):
                        # Old format: {'key': 'path'} — convert to new with defaults
                        for key, path in loaded_data.items():
                            if path and os.path.exists(path):
                                size = os.path.getsize(path)
                                self.audio_cache[key] = {
                                    'path': path,
                                    'size': size,
                                    'added_time': time.time(),  # Reset on load
                                    'last_used': time.time()
                                }
                                self.total_size += size
                        logger.info(f"Loaded old-format: {len(self.audio_cache)} valid entries (total size: {self.total_size / (1024*1024):.1f}MB)")
                    elif isinstance(loaded_data, dict) and all(isinstance(v, dict) for v in loaded_data.values()):
                        # New format: {'key': {'path': str, ...}}
                        for key, entry_data in loaded_data.items():
                            path = entry_data.get('path')
                            if path and os.path.exists(path):
                                size = entry_data.get('size', os.path.getsize(path))  # Fallback to getsize
                                added_time = entry_data.get('added_time', time.time())
                                self.audio_cache[key] = {
                                    'path': path,
                                    'size': size,
                                    'added_time': added_time,
                                    'last_used': time.time()  # Update on load
                                }
                                self.total_size += size
                            else:
                                logger.trace(f"Load skip invalid entry for key {key}")
                        logger.info(f"Loaded new-format: {len(self.audio_cache)} valid entries (total size: {self.total_size / (1024*1024):.1f}MB)")
                    else:
                        logger.warning(f"Unexpected cache format: {type(loaded_data)} - starting empty")
                        self.audio_cache = {}
                        self.total_size = 0
                except Exception as e:
                    logger.error(f"Audio cache load failed: {str(e)} - starting with empty cache")
                    self.audio_cache = {}
                    self.total_size = 0
            else:
                logger.debug("No cache file - starting empty")
                self.audio_cache = {}
                self.total_size = 0

    def save_cache(self) -> None:
        """Save audio cache metadata from memory to disk."""
        now = time.time()
        self.last_save_time = now
        with self.cache_lock:
            try:
                # Prepare data for save (only valid paths; new format)
                save_data = {}
                self.total_size = 0  # Recompute to ensure accuracy
                for key, entry in self.audio_cache.items():
                    path = entry['path']
                    if os.path.exists(path):  # Double-check existence
                        save_data[key] = {
                            'path': path,
                            'size': entry['size'],
                            'added_time': entry['added_time'],
                            'last_used': entry.get('last_used', time.time())
                        }
                        self.total_size += entry['size']
                    else:
                        logger.trace(f"Save skip invalid path for key {key}: {path}")
                with open(self.cache_file, 'w') as f:
                    json.dump(save_data, f, indent=2)
                logger.trace(f"Saved {len(save_data)} entries")
            except Exception as e:
                logger.error(f"Failed to save audio cache: {str(e)}")

    def get_key(self, voice_stem: str, text: str, exaggeration: float) -> str:
        """Generate a cache key for audio generation."""
        from hashlib import md5
        text_hash = md5(text.encode('utf-8')).hexdigest()[:8]
        return f"{voice_stem}_{text_hash}_{exaggeration:.2f}"

    def get(self, key: str) -> Optional[str]:
        """Get the path to cached audio for a given key."""
        with self.cache_lock:
            if key in self.audio_cache:
                entry = self.audio_cache[key]
                if os.path.exists(entry['path']):
                    # Update last_used on hit
                    entry['last_used'] = time.time()
                    logger.debug(f"Audio cache hit: {key} → {entry['path']}")
                    return entry['path']
                else:
                    # Path invalid - remove entry
                    del self.audio_cache[key]
                    self.total_size -= entry['size']
                    # Signal background saver instead of sync save
                    self._save_event.set()
                    logger.debug(f"Audio cache miss (stale path): {key}")
            return None

    def set(self, key: str, path: str) -> None:
        """Cache an audio path for a given key with validation and eviction."""
        if not os.path.exists(path):
            logger.warning(f"Cannot cache non-existent path: {path}")
            return

        file_size = os.path.getsize(path)

        with self.cache_lock:
            # Apply dual-limit condition (using pre-fetched limits)
            max_entries = self.max_entries  # From __init__
            max_total_size = self.max_total_size_mb * 1024 * 1024  # MB to bytes

            # Evict until below limit (prioritizes oldest by added_time)
            while (len(self.audio_cache) >= max_entries or self.total_size + file_size > max_total_size) and self.audio_cache:
                # Find oldest entry
                oldest_key = min(
                    self.audio_cache.keys(),
                    key=lambda k: self.audio_cache[k]['added_time']
                )
                self._remove_entry(oldest_key)

            # Remove existing entry if key already present (overwrite, update time/size)
            if key in self.audio_cache:
                old_size = self.audio_cache[key]['size']
                self.total_size -= old_size
                logger.debug(f"Overwrote existing key {key} (old size: {old_size / (1024*1024):.1f}MB)")

            # Add new cache entry
            self.audio_cache[key] = {
                'path': path,
                'size': file_size,
                'added_time': time.time(),
                'last_used': time.time()
            }
            self.total_size += file_size

            # Signal background saver
            self._save_event.set()

            logger.debug(f"Audio cached: {key} → {Path(path).name} ({file_size / (1024*1024):.1f}MB) | Cache stats: {len(self.audio_cache)} entries, {self.total_size / (1024*1024):.1f}MB total")

    def _remove_entry(self, key: str) -> None:
        """Internal helper: Remove a specific entry from the cache."""
        if key in self.audio_cache:
            entry = self.audio_cache[key]
            self.total_size -= entry['size']
            del self.audio_cache[key]
            logger.debug(f"Evicted oldest entry: {key} ({entry['size'] / (1024*1024):.1f}MB freed)")

    def clear(self, voice_stem: Optional[str] = None) -> None:
        """Clear the audio cache (optionally voice-specific)."""
        with self.cache_lock:
            if voice_stem:
                keys_to_remove = [k for k in self.audio_cache if k.startswith(voice_stem + '_')]
                for key in keys_to_remove:
                    entry = self.audio_cache[key]
                    self.total_size -= entry['size']
                    del self.audio_cache[key]
                logger.info(f"Cleared audio cache for voice: {voice_stem} ({len(keys_to_remove)} entries)")
            else:
                self.audio_cache.clear()
                self.total_size = 0
                logger.info("Cleared entire audio cache")
            self.save_cache()

    def __del__(self):
        """Ensure cache is saved on destruction."""
        try:
            if hasattr(self, '_save_event'):
                self.save_cache()
        except:
            pass

    def get_stats(self) -> Dict[str, Any]:
        """Get audio cache statistics with safety fallbacks."""
        with self.cache_lock:
            entries = len(self.audio_cache)
            total_size = self.total_size
            # Recompute disk size safely
            disk_size = 0
            for entry in self.audio_cache.values():
                path = entry['path']
                if os.path.exists(path):
                    disk_size += entry['size']

            return {
                "entries": entries,
                "total_size": total_size,  # In-memory total
                "memory_entries": entries,
                "disk_entries": entries,   # Assuming all valid
                "disk_size": disk_size
            }