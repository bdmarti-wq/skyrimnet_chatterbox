import os
import json
import math
import time
import logging
import hashlib
import threading

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, NamedTuple
from loguru import logger

from src.config import get_config, get_config_value
from src.audio_utils import is_artifact_laden
import torchaudio
import torch

VOICE_CACHE_INSTANCE = None

# ===== TEMPORARY DEBUG OVERRIDES =====
# Set to True to always bypass voice cache and use server-provided audio
DEBUG_FORCE_SERVER_AUDIO = False
# ===== END DEBUG OVERRIDES =====

class VoiceReferenceEntry(NamedTuple):
    """Represents a specific voice reference in our cache system."""
    stem: str
    reference_path: str
    content_hash: str  # MD5 hash of audio content
    conditionals_key: str
    last_updated: float
    voice_config: Dict[str, Any]  # Additional voice-specific config
    custom_path: Optional[str] = None  # If config specifies a path override


class VoiceReferenceCache:
    """Manages voice reference files and their metadata for cloning."""

    def __init__(self, cache_dir: Path = None):
        """Initialize the voice reference cache system."""
        global VOICE_CACHE_INSTANCE
        VOICE_CACHE_INSTANCE = self
        config = get_config()
        cache_base = cache_dir or Path(config.app_config.globals.cache_dir)
        self.cache_dir = cache_base / "audio/voices"
        self.cache_file = self.cache_dir / "voices_metadata.json"
        self.cache_lock = threading.RLock()

        # Create directory structure
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache of voice references
        self.voice_cache: Dict[str, VoiceReferenceEntry] = {}

        # Load existing cache
        self.load_cache()

        logger.info(f"Voice reference cache initialized at {self.cache_dir}")

    def load_cache(self) -> None:
        """Load voice reference metadata from disk to memory with robust recovery."""
        with self.cache_lock:
            if self.cache_file.exists():
                try:
                    with open(self.cache_file, 'r') as f:
                        data = json.load(f)

                    # Create a new clean cache (don't trust corrupted one)
                    new_cache = {}
                    for stem, entry_data in data.items():
                        try:
                            # Skip entries with missing reference files
                            ref_path = entry_data.get('reference_path', '')
                            if not ref_path or not os.path.exists(ref_path):
                                logger.debug(f"Skipping invalid cache entry for {stem}: file missing ({ref_path})")
                                continue

                            # Handle possible JSON corruption issues
                            voice_config = entry_data.get('voice_config', {})
                            if voice_config and isinstance(voice_config, dict):
                                # Create safe copy removing potential JSON issues
                                safe_config = {}
                                for k, v in voice_config.items():
                                    # Skip problematic keys
                                    if isinstance(v, (torch.dtype, torch.device)):
                                        continue
                                    safe_config[k] = v

                            new_cache[stem] = VoiceReferenceEntry(
                                stem=stem,
                                reference_path=ref_path,
                                content_hash=entry_data.get('content_hash', ''),
                                conditionals_key=entry_data.get('conditionals_key', ''),
                                last_updated=entry_data.get('last_updated', time.time()),
                                voice_config=safe_config,
                                custom_path=entry_data.get('custom_path')
                            )
                        except Exception as e:
                            logger.warning(f"Skipping corrupt cache entry for {stem}: {str(e)}")
                            continue

                    self.voice_cache = new_cache
                    logger.info(f"Loaded {len(self.voice_cache)} voice references (recovered from possible corruption)")
                except Exception as e:
                    logger.error(f"Voice cache load failed (corrupted file): {str(e)} - starting with empty cache")
                    # Rename the corrupt file for diagnostics
                    try:
                        corrupt_path = self.cache_file.with_suffix(f".corrupt.{int(time.time())}")
                        self.cache_file.rename(corrupt_path)
                        logger.info(f"Renamed corrupt cache to {corrupt_path} for diagnostics")
                    except Exception as rename_e:
                        logger.debug(f"Could not rename corrupt cache file: {str(rename_e)}")
                    self.voice_cache = {}
            else:
                logger.info("No voice reference cache found - initializing empty")
                self.voice_cache = {}


    def save_cache(self) -> None:
        """Save voice reference metadata from memory to disk with proper serialization."""
        with self.cache_lock:
            try:
                # Convert named tuples to dict for JSON serialization
                cache_data = {}
                for stem, entry in self.voice_cache.items():
                    try:
                        # Make voice_config JSON serializable
                        serializable_config = {}
                        for k, v in entry.voice_config.items():
                            # Convert non-serializable types to strings
                            if isinstance(v, torch.dtype):
                                serializable_config[k] = str(v)
                            elif isinstance(v, torch.device):
                                serializable_config[k] = str(v)
                            else:
                                serializable_config[k] = v

                        cache_data[stem] = {
                            "stem": stem,
                            "reference_path": entry.reference_path,
                            "content_hash": entry.content_hash,
                            "conditionals_key": entry.conditionals_key,
                            "last_updated": entry.last_updated,
                            "voice_config": serializable_config,
                            "custom_path": entry.custom_path
                        }
                    except Exception as e:
                        logger.warning(f"Failed to serialize cache entry for {stem}: {str(e)}")
                        continue

                with open(self.cache_file, 'w') as f:
                    json.dump(cache_data, f, indent=2)

                logger.debug(f"Saved voice reference cache with {len(cache_data)} entries")
            except Exception as e:
                logger.error(f"Failed to save voice cache: {str(e)}")

    def calculate_content_hash(self, audio_path: str) -> str:
        """Calculate stable content hash specifically optimized for voice reference files."""
        try:
            # Load audio with increased precision
            waveform, _ = torchaudio.load(audio_path)

            # Force to mono but preserve all audio information
            if waveform.dim() > 1:
                waveform = waveform.mean(dim=0)

            # Reduced tolerance to capture subtle vocal differences (0.000001 vs 0.00001)
            TOLERANCE = 1e-6

            # Variable chunk size - smaller for voice analysis
            chunk_size = 256
            valid_chunks = []

            for i in range(0, waveform.shape[0], chunk_size):
                chunk = waveform[i:i + chunk_size]

                # Skip empty chunks but with more sensitive threshold
                is_silent = torch.all(torch.abs(chunk) < TOLERANCE)
                if not is_silent:
                    # No padding - keep original chunk size
                    valid_chunks.append(chunk.cpu().numpy())

            # Handle very short voice samples
            if not valid_chunks:
                logger.warning(f"Voice reference {os.path.basename(audio_path)} consists entirely of silence")
                # Return hash of entire waveform for tracking
                return hashlib.md5(waveform.numpy().tobytes()).hexdigest()

            # Convert to consistent format for hashing
            concatenated = np.concatenate(valid_chunks)
            return hashlib.md5(concatenated.tobytes()).hexdigest()

        except Exception as e:
            logger.error(f"Content hash calculation failed: {str(e)}")
            # Fallback: use stronger hash of audio parameters
            try:
                info = torchaudio.info(audio_path)
                fallback_str = f"{info.num_frames}_{info.sample_rate}_{info.num_channels}"
                return hashlib.md5(fallback_str.encode()).hexdigest()
            except:
                # Last resort
                return f"fallback_{int(os.path.getmtime(audio_path))}"




    def get_voice_params(self, voice_stem: str) -> Dict[str, Any]:
        """Get full configuration for a specific voice."""
        # Load voice-specific config from main config
        config = get_config()
        return config.get_voice_params(voice_stem) or {}


    def should_update_reference(self, voice_stem: str, new_path: str) -> Tuple[bool, str, Optional[str]]:
        """
        Determine if a voice reference should be updated.

        Returns:
            (should_update, current_content_hash, new_content_hash)
        """
        current_hash = None
        new_hash = self.calculate_content_hash(new_path)

        # Check if voice exists in cache
        if voice_stem in self.voice_cache:
            current_entry = self.voice_cache[voice_stem]
            current_hash = current_entry.content_hash

            # Check if content differs
            if current_hash != new_hash:
                return True, current_hash, new_hash

        # No cache entry or hash differs
        return True, current_hash, new_hash

    def validate_reference_file(self, audio_path: str, voice_stem: str) -> bool:
        """Validate a voice reference file meets requirements."""
        if not audio_path or not os.path.exists(audio_path):
            return False



        config = get_config()
        required_sr = config.app_config.globals.sr
        min_duration = get_config_value('globals.min_ref_duration', 3.0)

        try:
            # Try new API (2.1+)
            info = torchaudio.info(audio_path)
            duration = info.num_frames / info.sample_rate
            sample_rate = info.sample_rate
            channels = info.num_channels
        except (ImportError, AttributeError):
            # Fallback to old API
            info = torchaudio.info(audio_path)
            if isinstance(info, tuple):
                sample_rate, channels, num_frames = info[0], info[1], info[2]
            else:
                sample_rate, channels = info.sample_rate, info.num_channels
                num_frames = info.num_frames if hasattr(info, 'num_frames') else 0
            duration = num_frames / sample_rate

        # Sample rate check
        if sample_rate != required_sr:
            logger.warning(f"Voice {voice_stem}: Sample rate {sample_rate}Hz != required {required_sr}Hz")
            return False

        # Channel check
        if channels != 1:
            logger.warning(f"Voice {voice_stem}: Channels {channels} != required 1")
            return False

        # Duration check
        if duration < min_duration:
            logger.warning(f"Voice {voice_stem}: Duration {duration:.2f}s < required {min_duration}s")
            return False

        # Artifact check (optional)
        if get_config_value('globals.check_artifacts', True):
            if is_artifact_laden(audio_path):
                logger.warning(f"Voice {voice_stem}: Detected audio artifacts - invalid for cloning")
                return False

        return True


    def process_new_reference(self, voice_stem: str, new_path: str, force_update: bool = False) -> Tuple[
        bool, str, str, Dict[str, Any]]:
        """
        Process a new voice reference file and determine if conditionals need regeneration.

        TEMPORARY DEBUG: When DEBUG_FORCE_SERVER_AUDIO is True, always use received audio file
        """
        if DEBUG_FORCE_SERVER_AUDIO:
            logger.warning("⚠️⚠️⚠️ DEBUG OVERRIDDEN: Always processing server audio (bypassing cache)")
            force_update = True
            # Ensure we don't get stuck using config reference_path
            voice_params = self.get_voice_params(voice_stem)
            voice_params["reference_path"] = None

        # Step 0: Special handling for uploaded files (new voice references)
        is_upload = False
        if "Temp" in new_path or "gradio" in new_path or "tmp" in new_path:
            logger.debug(f"Detected uploaded voice reference: {new_path}")
            is_upload = True
            # Generate a unique stem for this upload to prevent collisions
            unique_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
            voice_stem = f"{voice_stem}_upload_{unique_id}"
            logger.info(f"Created unique voice stem for upload: {voice_stem}")

        # Step 1: Basic validation
        if not self.validate_reference_file(new_path, voice_stem):
            logger.error(f"Voice reference validation failed for {voice_stem}")
            return False, new_path, "", {}

        # Step 2: Check if we should update
        should_update, current_hash, new_hash = self.should_update_reference(voice_stem, new_path)

        # CRITICAL FIX: Always update for new uploads even if hash matches
        if is_upload:
            should_update = True
            logger.info("Force-updating conditionals for uploaded voice reference")

        config = get_config()
        voice_params = self.get_voice_params(voice_stem)

        # Step 3: Should we use a config-specified path instead?
        config_path = voice_params.get("reference_path")
        if config_path and os.path.exists(config_path) and not is_upload:
            # Only use config path if NOT an upload
            final_path = config_path
            use_config_path = True
            logger.info(f"Voice {voice_stem}: Using config-specified reference path {config_path}")
        else:
            final_path = new_path
            use_config_path = False

        # Step 4: Check if this path is already current entry
        if voice_stem in self.voice_cache and not force_update and not is_upload:
            current_entry = self.voice_cache[voice_stem]
            if current_entry.reference_path == final_path:
                # Same reference path - check if we're using config path
                if use_config_path:
                    logger.info(f"Voice {voice_stem}: Using config reference path, skipping conditionals check")
                    return True, final_path, current_entry.conditionals_key, voice_params

                # If hash matches and not an upload, reuse
                if current_entry.content_hash == new_hash:
                    logger.info(f"Voice {voice_stem}: No content change detected, reusing conditionals")
                    return True, final_path, current_entry.conditionals_key, voice_params

        # Step 5: Conditionals need regeneration
        conditionals_key = self._generate_conditionals_key(voice_stem, final_path, voice_params)
        logger.info(
            f"Voice {voice_stem}: Generating new conditionals (config_path={use_config_path}, "
            f"force={force_update}, hash_prev={current_hash[:8] if current_hash else None}, "
            f"hash_new={new_hash[:8]})"
        )

        # Update cache entry
        with self.cache_lock:
            self.voice_cache[voice_stem] = VoiceReferenceEntry(
                stem=voice_stem,
                reference_path=final_path,
                content_hash=new_hash,
                conditionals_key=conditionals_key,
                last_updated=time.time(),
                voice_config=voice_params,
                custom_path=config_path if config_path else None
            )
            self.save_cache()

        logger.debug(
            f"Voice reference processing complete - stem='{voice_stem}', "
            f"final_path='{final_path}', conditionals_key='{conditionals_key}', "
            f"config_path='{config_path}', is_upload={is_upload}"
        )

        return True, final_path, conditionals_key, voice_params


    def _generate_conditionals_key(self, voice_stem: str, audio_path: str, voice_config: Dict[str, Any]) -> str:
        """Generate a unique conditionals cache key focused on voice characteristics."""
        # Get sensitive content hash
        content_hash = self.calculate_content_hash(audio_path)

        # Include ALL relevant voice parameters that affect cloning
        # Not just exaggeration but others that impact voice characteristics
        temperature = voice_config.get("temperature", 0.8)
        top_p = voice_config.get("top_p", 1.0)
        min_p = voice_config.get("min_p", 0.05)

        # Create versioned key to handle future algorithm changes
        return f"v2_{voice_stem}_ref_{content_hash[:12]}_exag{voice_config.get('exaggeration', 0.5):.2f}_temp{temperature:.2f}_topp{top_p:.2f}"


    def get_conditionals_key(self, voice_stem: str) -> Optional[str]:
        """Get the conditionals key for a voice, or None if not cached."""
        with self.cache_lock:
            if voice_stem in self.voice_cache:
                return self.voice_cache[voice_stem].conditionals_key
        return None

    def get_reference_path(self, voice_stem: str) -> Optional[str]:
        """Get the reference path for a voice, or None if not available."""
        with self.cache_lock:
            if voice_stem in self.voice_cache:
                return self.voice_cache[voice_stem].reference_path
        return None

    # ADD THIS NEW METHOD HERE
    def get_entry(self, voice_stem: str) -> Optional[VoiceReferenceEntry]:
        with self.cache_lock:
            return self.voice_cache.get(voice_stem)

def verify_voice_content_integrity():
    """Verify that voice references and conditionals are properly aligned"""

    if not VOICE_CACHE_INSTANCE.voice_cache:
        logger.warning("⚠ No voice cache entries to verify")
        return False

    all_ok = True
    for stem, entry in VOICE_CACHE_INSTANCE.voice_cache.items():
        if not os.path.exists(entry.reference_path):
            logger.error(f"❌ Voice reference missing: {entry.reference_path} (stem='{stem}')")
            all_ok = False
            continue

        # Calculate actual content hash
        actual_hash = VOICE_CACHE_INSTANCE.calculate_content_hash(entry.reference_path)

        # Extract expected hash from conditionals key
        key_parts = entry.conditionals_key.split('_')
        expected_hash = next(
            (part for part in key_parts if len(part) == 12 and all(c in '0123456789abcdef' for c in part)), None)

        if not expected_hash:
            logger.error(f"❌ Invalid conditionals key format: {entry.conditionals_key} (stem='{stem}')")
            all_ok = False
        elif expected_hash != actual_hash[:12]:
            logger.error(f"❌ HASH MISMATCH for stem '{stem}':\n"
                         f"Expected: {expected_hash}\n"
                         f"Actual:   {actual_hash[:12]}\n"
                         f"Reference: {entry.reference_path}")
            all_ok = False

    if all_ok:
        logger.info("✅ Voice reference and conditionals hashes verified")

    return all_ok