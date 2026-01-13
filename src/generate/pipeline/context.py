import time
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import hashlib
from src.config.models import AppConfig
from src.cache_keys import generate_audio_cache_key as _gen_cache_key
from src.tts_model import get_model  # For fallback if needed
from src.normalize_stem import normalize_stem  # For derive_stem
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
    enable_audio_cache: bool = field(default=True, init=True, repr=False)
    enable_fuzzy_cache: bool = field(default=True, init=True, repr=False)

    # Generation parameters (UI-driven)
    exaggeration: float = field(default=0.5, init=True, repr=True)
    temperature: float = field(default=0.7, init=True, repr=True)
    cfg_weight: float = field(default=0.45, init=True, repr=True)
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

    # Private field for audio_duration (with getter/setter)
    _audio_duration: float = field(default=0.0, init=False, repr=False)

    # Cached globals (lazy, DRY)
    _globals_cache: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    # Timing metrics (runtime; always dicts)
    timing: Dict[str, float] = field(default_factory=dict, init=True, repr=False)
    step_times: Dict[str, float] = field(default_factory=dict, init=True, repr=False)

    def get_globals(self) -> Dict[str, Any]:
        """REFACTORED: Shared lazy access to config globals (sr, device, dtype, etc.). Cache result; fallback defaults. DRY across all files."""
        if self._globals_cache:
            return self._globals_cache

        globals_dict = {}
        if self.config is not None and hasattr(self.config, 'app_config') and hasattr(self.config.app_config, 'globals'):
            g = self.config.app_config.globals
            globals_dict = {
                'sr': getattr(g, 'sr', 24000),
                'device': getattr(g, 'device', 'cuda' if torch.cuda.is_available() else 'cpu'),
                'dtype': getattr(g, 'dtype', torch.bfloat16),
                'multilingual': getattr(g, 'multilingual', False),
                'cache_dir': getattr(g, 'cache_dir', Path('./cache'))
            }
        else:
            # Fallback defaults
            globals_dict = {
                'sr': 24000,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'dtype': torch.bfloat16,
                'multilingual': False,
                'cache_dir': Path('./cache')
            }
            logger.warning("Using fallback globals (no valid config)")

        self._globals_cache = globals_dict
        return globals_dict

    @property
    def audio_duration(self) -> float:
        """Get cached or compute audio duration; caches result in _audio_duration."""
        if self._audio_duration > 0:
            return self._audio_duration
        if self.processed_wav is not None and hasattr(self.processed_wav, 'numel') and self.processed_wav.numel() > 0:
            self._audio_duration = len(self.processed_wav.squeeze(0)) / self.sr
            return self._audio_duration
        if self.generated_wav is not None and hasattr(self.generated_wav, 'numel') and self.generated_wav.numel() > 0:
            self._audio_duration = len(self.generated_wav.squeeze(0)) / self.sr
            return self._audio_duration
        return 0.0

    @audio_duration.setter
    def audio_duration(self, value: float) -> None:
        """Store explicit audio duration."""
        if value >= 0:
            self._audio_duration = value

    def generate_cache_key(self, voice_stem: str = "", text: str = "", exaggeration: float = 0.0, cache_uuid: int = 0) -> str:
        """Generate a cache key using the shared generator (single source of truth)."""
        vs = voice_stem or self.voice_stem
        tx = text or self.text
        ex = exaggeration or self.exaggeration
        cu = cache_uuid or self.cache_uuid
        try:
            return _gen_cache_key(vs, tx, ex, cu)
        except Exception:
            # Very defensive fallback
            raw = f"{vs}_{tx[:16]}_{ex}_{cu}"
            return hashlib.md5(raw.encode()).hexdigest()[:16]


    def __post_init__(self):
        """Initialize derived fields safely; ensure timing/step_times are always dicts. REFACTORED: Call ensure_attrs() for defaults; compute save_cache."""
        # Ensure timing and step_times are initialized as dicts
        if self.timing is None:
            object.__setattr__(self, 'timing', {})
        if self.step_times is None:
            object.__setattr__(self, 'step_times', {})

        # REFACTORED: Ensure common attrs (paths, seeds, etc.) – DRY across phases
        self.ensure_attrs()

        # NEW: Set save_cache flag (default True; override from config if provided)
        if not hasattr(self, 'save_cache'):
            object.__setattr__(self, 'save_cache', True)  # Bool flag for phases (e.g., voice_processing guard)
        if self.config is not None:
            from src.config.models import AppConfig  # Lazy if needed
            if hasattr(self.config, 'save_caches') and not self.save_cache:
                object.__setattr__(self, 'save_cache', self.config.save_caches)  # Config override

        # Legacy compatibility
        if self.seed is None and self.seed_num != 42:
            object.__setattr__(self, 'seed', self.seed_num)

        # REFACTORED: Override defaults with config if provided (lazy; use get_globals below)
        if self.config is not None:
            globals_dict = self.get_globals()
            if globals_dict:
                self.sr = globals_dict.get('sr', 24000)
                self.device = torch.device(globals_dict.get('device', 'cpu'))
                # Standardize dtype/device (dtype may be a string in config.json)
                dtype_val = globals_dict.get('dtype', torch.bfloat16)
                if isinstance(dtype_val, str):
                    try:
                        dtype_val = getattr(torch, dtype_val)
                    except Exception:
                        dtype_val = torch.bfloat16
                self.dtype = dtype_val
                dev_val = globals_dict.get('device', 'cpu')
                try:
                    self.device = torch.device(dev_val)
                except Exception:
                    self.device = torch.device('cpu')
                self.multilingual = globals_dict.get('multilingual', False)
                logger.debug(f"Context derived from config: sr={self.sr}, device={self.device}, dtype={self.dtype}")

        # Reset _audio_duration
        object.__setattr__(self, '_audio_duration', 0.0)


    def ensure_attrs(self):
        """FIXED: Ensure voice_params is dict (prevents str from unpack misalign)."""
        # Paths
        if self.audio_prompt_path is None:
            object.__setattr__(self, 'audio_prompt_path', "")
        if self.processed_voice_path is None:
            object.__setattr__(self, 'processed_voice_path', "")
        if self.processed_ref_path is None:
            object.__setattr__(self, 'processed_ref_path', self.processed_voice_path or "")
        if self.output_path is None:
            object.__setattr__(self, 'output_path', "")

        # Voice stem
        if not self.voice_stem or self.voice_stem == "default":
            if self.audio_prompt_path:
                try:
                    from src.normalize_stem import normalize_stem
                    object.__setattr__(self, 'voice_stem', normalize_stem(self.audio_prompt_path) or "default")
                except ImportError:
                    from pathlib import Path
                    object.__setattr__(self, 'voice_stem', Path(self.audio_prompt_path).stem or "default")
            else:
                object.__setattr__(self, 'voice_stem', "default")

        # FIXED: Ensure voice_params is dict (handles str from mis-set key/params)
        if self.voice_params is None or not isinstance(self.voice_params, dict):
            object.__setattr__(self, 'voice_params', {})

        # Seeds
        if self.seed is None:
            object.__setattr__(self, 'seed', self.seed_num or 42)

        # Cache keys
        if not self.cache_key:
            object.__setattr__(self, 'cache_key', self.generate_cache_key())
        if not self.conditionals_key:
            object.__setattr__(self, 'conditionals_key', f"stub_{self.voice_stem}_{int(time.time() % 10000)}")

        # Flags
        if self.is_cached is None:
            object.__setattr__(self, 'is_cached', False)
        if self.voice_ref_processed is None:
            object.__setattr__(self, 'voice_ref_processed', False)

        # NEW: Ensure save_cache flag (bool; default True)
        if not hasattr(self, 'save_cache') or self.save_cache is None:
            object.__setattr__(self, 'save_cache', True)

    # ... (get_globals, audio_duration property/setter, generate_cache_key unchanged)

    def save_cache(self, cache_type: str = 'conditionals') -> bool:
        """
        NEW: Save current conds (or other) to cache_manager's relevant cache.
        Called post-prep in phases (e.g., VoiceProcessing) or manually (e.g., OutputPhase for audio).
        Args: cache_type='conditionals' (default; extend for 'audio', etc.).
        Returns: bool (success).
        Handles: If no cache_manager/key/conds, skip silently (no crash). Uses existing cache.save sig.
        """
        if not hasattr(self, 'cache_manager') or self.cache_manager is None:
            logger.debug("save_cache skipped: no cache_manager")
            return False

        if cache_type == 'conditionals' and self.conditionals_key and hasattr(self, 'conds') and self.conds is not None:
            try:
                # Use existing conditionals_cache.save(key, conds, model, device_str, dtype)
                success = self.cache_manager.conditionals_cache.save(
                    self.conditionals_key, self.conds, self.model, str(self.device), self.dtype
                )
                if success:
                    logger.info(f"Conds saved to cache: {self.conditionals_key[:20]} (type: {type(self.conds).__name__})")
                return success
            except Exception as save_e:
                logger.error(f"Failed to save conds {self.conditionals_key}: {save_e} – continue without cache")
                return False
        elif cache_type == 'audio' and self.cache_key and hasattr(self, 'generated_wav') and self.generated_wav is not None:
            # Example extension: Save audio if type='audio' (implement cache's audio_save if exists)
            try:
                if hasattr(self.cache_manager, 'audio_cache') and hasattr(self.cache_manager.audio_cache, 'set'):
                    self.cache_manager.audio_cache.set(self.cache_key, self.generated_wav)
                    self.cache_manager.audio_cache.save()  # Persist if method exists
                    logger.info(f"Audio saved to cache: {self.cache_key[:20]}")
                    return True
            except Exception as save_e:
                logger.error(f"Failed to save audio {self.cache_key}: {save_e}")
                return False
        else:
            logger.debug(f"save_cache skipped for {cache_type}: no key/conds (key={bool(self.conditionals_key)}, conds={self.conds is not None})")
            return False