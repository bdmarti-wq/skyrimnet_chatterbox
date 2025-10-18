from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import hashlib
from src.config.models import AppConfig
import logging

logger = logging.getLogger(__name__)

@dataclass
class AudioGenerationContext:
    """Encapsulates all state throughout the audio generation pipeline.

    All fields are explicitly defined with defaults to ensure accessibility.
    Critical fields like model, cache_manager must be set at creation via init kwargs.
    Derived fields (device, dtype, sr, multilingual) can be passed or set in __post_init__ from config.
    """

    # Core inputs (required for all generations)
    text: str = field(default="", init=True, repr=True)
    audio_prompt_path: Optional[str] = field(default=None, init=True, repr=True)
    cache_uuid: int = field(default=0, init=True, repr=True)

    # Cache flags
    enable_memory_cache: bool = field(default=True, init=True, repr=False)
    enable_disk_cache: bool = field(default=True, init=True, repr=False)

    # Generation parameters (UI-driven)
    exaggeration: float = field(default=0.5, init=True, repr=True)
    temperature: float = field(default=0.7, init=True, repr=True)
    cfg_weight: float = field(default=0.45, init=True, repr=True)
    cfgw: float = field(default=0.45, init=True, repr=False)  # Legacy
    min_p: float = field(default=0.05, init=True, repr=True)
    top_p: float = field(default=1.0, init=True, repr=True)
    repetition_penalty: float = field(default=1.2, init=True, repr=True)
    language_id: str = field(default="en", init=True, repr=True)
    seed: Optional[int] = field(default=None, init=True, repr=True)
    seed_num: int = field(default=42, init=True, repr=False)  # Legacy

    # Voice metadata
    voice_stem: str = field(default="default", init=True, repr=True)
    voice_params: Dict[str, Any] = field(default_factory=dict, init=True, repr=False)
    t3_params: Dict[str, Any] = field(default_factory=lambda: {
        "generate_token_backend": "cudagraphs-manual",
        "stride_length": 4,
        "skip_when_1": True
    }, init=True, repr=False)

    # System configuration (allow init with defaults; override in __post_init__ if config provided)
    device: torch.device = field(default=torch.device("cpu"), init=True, repr=False)
    dtype: torch.dtype = field(default=torch.bfloat16, init=True, repr=False)
    model: Optional[Any] = field(default=None, init=True, repr=False)
    config: Optional[AppConfig] = field(default=None, init=True, repr=False)
    cache_manager: Optional[Any] = field(default=None, init=True, repr=False)

    # Derived runtime values (allow init with defaults; set from config in __post_init__ if available)
    sr: int = field(default=24000, init=True, repr=False)
    multilingual: bool = field(default=False, init=True, repr=False)

    # Pipeline state (updated during phases)
    original_text: str = field(default="", init=True, repr=False)
    cached_path: Optional[str] = field(default=None, init=True, repr=True)
    cache_hit_type: Optional[str] = field(default=None, init=True, repr=True)
    processed_voice_path: Optional[str] = field(default=None, init=True, repr=True)
    processed_ref_path: Optional[str] = field(default=None, init=True, repr=True)
    voice_content_hash: Optional[str] = field(default=None, init=True, repr=False)
    conditionals_key: Optional[str] = field(default=None, init=True, repr=True)
    cache_key: str = field(default="", init=True, repr=True)
    generated_wav: Optional[torch.Tensor] = field(default=None, init=True, repr=False)
    processed_wav: Optional[torch.Tensor] = field(default=None, init=True, repr=False)
    output_path: Optional[str] = field(default=None, init=True, repr=True)

    # Flags (phase-updated)
    is_cached: bool = field(default=False, init=True, repr=False)
    voice_ref_processed: bool = field(default=False, init=True, repr=False)
    process_after_cache: bool = field(default=True, init=True, repr=False)

    # Cache save flag (computed from enable flags)
    save_cache: bool = field(default=False, init=False, repr=False)

    # Private field for audio_duration (with getter/setter)
    _audio_duration: float = field(default=0.0, init=False, repr=False)

    # Timing metrics (runtime; always dicts)
    timing: Dict[str, float] = field(default_factory=dict, init=True, repr=False)
    step_times: Dict[str, float] = field(default_factory=dict, init=True, repr=False)

    def __post_init__(self):
        """Initialize derived fields safely; ensure timing/step_times are always dicts."""
        # Ensure timing and step_times are initialized as dicts (guard against missing or None)
        if self.timing is None:
            object.__setattr__(self, 'timing', {})
        if self.step_times is None:
            object.__setattr__(self, 'step_times', {})

        # Compute save_cache
        object.__setattr__(self, 'save_cache', self.enable_memory_cache or self.enable_disk_cache)

        # Legacy compatibility
        if self.seed is None and self.seed_num != 42:
            object.__setattr__(self, 'seed', self.seed_num)
        if self.cfgw != 0.45 and self.cfg_weight == 0.45:
            object.__setattr__(self, 'cfg_weight', self.cfgw)

        # FIXED: Override defaults with config values if config is provided (no crash on None)
        if self.config is not None and hasattr(self.config, 'globals'):
            try:
                # Only override if not already explicitly passed (avoids clobbering)
                if hasattr(self.config.globals, 'device') and self.device.type == 'cpu':  # Preserve if set
                    object.__setattr__(self, 'device', torch.device(self.config.globals.device))
                if hasattr(self.config.globals, 'dtype') and self.dtype == torch.bfloat16:  # Default check
                    object.__setattr__(self, 'dtype', self.config.globals.dtype)
                if hasattr(self.config.globals, 'multilingual'):
                    object.__setattr__(self, 'multilingual', self.config.globals.multilingual)
                if hasattr(self.config.globals, 'sr'):
                    object.__setattr__(self, 'sr', self.config.globals.sr)
                logger.debug(f"Context derived from config: device={self.device}, dtype={self.dtype}, sr={self.sr}, multilingual={self.multilingual}")
            except (AttributeError, ValueError, KeyError) as e:
                logger.warning(f"Failed to derive fields from config: {e}; keeping passed defaults")

        # Reset _audio_duration to 0.0 (start fresh)
        object.__setattr__(self, '_audio_duration', 0.0)

        logger.debug(f"AudioGenerationContext post-init: save_cache={self.save_cache}, timing keys={list(self.timing.keys())}")

    @property
    def audio_duration(self) -> float:
        """Get cached or compute audio duration; caches result in _audio_duration."""
        if self._audio_duration > 0:
            return self._audio_duration
        if self.processed_wav is not None and self.processed_wav.numel() > 0:
            self._audio_duration = len(self.processed_wav.squeeze(0)) / self.sr
            return self._audio_duration
        if self.generated_wav is not None and self.generated_wav.numel() > 0:
            self._audio_duration = len(self.generated_wav.squeeze(0)) / self.sr
            return self._audio_duration
        return 0.0

    @audio_duration.setter
    def audio_duration(self, value: float) -> None:
        """Store explicit audio duration."""
        if value >= 0:
            self._audio_duration = value

    def generate_cache_key(self, voice_stem: str = "", text: str = "", exaggeration: float = 0.0, cache_uuid: int = 0) -> str:
        """Generate a cache key using cache_manager if available, or fallback hash."""
        if self.cache_manager and hasattr(self.cache_manager, 'generate_audio_cache_key'):
            try:
                return self.cache_manager.generate_audio_cache_key(
                    voice_stem=voice_stem or self.voice_stem,
                    text=text or self.text,
                    exaggeration=exaggeration or self.exaggeration,
                    cache_uuid=cache_uuid or self.cache_uuid
                )
            except Exception as e:
                logger.warning(f"Cache manager key gen failed: {e}; using fallback")

        # Fallback hash-based key
        hash_str = f"{voice_stem or self.voice_stem}_{hash(text or self.text)}_{exaggeration or self.exaggeration}_{cache_uuid or self.cache_uuid}"
        return hashlib.md5(hash_str.encode()).hexdigest()[:16]