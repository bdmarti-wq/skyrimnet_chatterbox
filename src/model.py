import gc  # NEW: For warmup cleanup
import threading
import functools
from typing import Optional, Any
from loguru import logger
import torch

# Lazy globals (defaults; overridden by CONFIG in methods)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MULTILINGUAL = False  # Default; pulled from CONFIG in methods

from src.cache import clear_cache_files  # Clear conds/audio on unload


def chatterbox_tts_to(model: Any, device: torch.device, dtype: torch.dtype):
    """Granular to() for Chatterbox: bfloat16 compute, fp32 audio (avoids cuFFT/STFT errors). DRY."""
    logger.debug(f"Moving ChatterboxTTS to {device}, {dtype} (granular)")
    if hasattr(model, 've'):
        model.ve.to(device=device)
    if hasattr(model, 't3'):
        model.t3.to(device=device, dtype=dtype)  # Core sampler in bfloat16
    if hasattr(model, 'conds') and hasattr(model.conds, 't3'):  # Conds T3
        model.conds.t3.to(device=device, dtype=dtype)
    if hasattr(model, 's3gen'):
        # fp32 for stable audio/STFT (old code insight)
        model.s3gen.to(device=device, dtype=torch.float32)
        if hasattr(model.s3gen, 'flow') and hasattr(model.s3gen.flow, 'fp16'):
            model.s3gen.flow.fp16 = (dtype == torch.float16)
        if hasattr(model, 's3gen') and hasattr(model.s3gen, 'tokenizer'):
            model.s3gen.tokenizer.to(device=device, dtype=torch.float32)
        if hasattr(model, 's3gen') and hasattr(model.s3gen, 'speaker_encoder'):
            model.s3gen.speaker_encoder.to(device=device, dtype=torch.float32)
        if hasattr(model, 's3gen') and hasattr(model.s3gen, 'mel2wav'):
            model.s3gen.mel2wav.to(device=device, dtype=torch.float32)
    if hasattr(model, 'conds'):
        model.conds.to(device=device)  # Conds to dtype (safe)
    model.device = str(device)  # Str for logs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.debug("ChatterboxTTS dtype/device applied (granular overrides applied)")
    return model


def compile_t3_step(model: Any, compile_mode: str = 'max-autotune'):
    """Compile T3 step with graphs (from old code; togglable). DRY; fallback safe."""
    if not torch.cuda.is_available():
        logger.debug("Skipping t3 compile (no CUDA)")
        return model
    from .config import get_config_value  # Lazy
    if not get_config_value('tts_config.compile_t3', True):  # FIXED: Flat config key (assume globals; adjust if nested)
        logger.debug("Skipping t3 compile (disabled)")
        return model
    # NEW: Debug why skip (attrs post-load/to())
    if not hasattr(model, 't3'):
        logger.debug("Skipping t3 compile (no 't3' attr)")
        return model
    if not hasattr(model.t3, '_step_compilation_target'):
        logger.debug(f"Skipping t3 compile (no '_step_compilation_target'; T3 attrs: {dir(model.t3)})")
        return model
    try:
        original_step = model.t3._step_compilation_target
        if not hasattr(model.t3, '_original_step'):
            model.t3._original_step = original_step
        model.t3._step_compilation_target = torch.compile(
            model.t3._step_compilation_target,
            mode=compile_mode,       # max-autotune for fusion + graphs
            fullgraph=True,
            backend='cudagraphs',    # Unified: Compile traces to graphs
            dynamic=True            # Handles varying token lengths
        )
        logger.info(f"T3 step compiled (mode={compile_mode}, backend=cudagraphs)")
        return model
    except Exception as compile_e:
        logger.warning(f"T3 compile failed (fallback to uncompiled): {compile_e}")
        if hasattr(model.t3, '_original_step'):
            model.t3._step_compilation_target = model.t3._original_step
        return model


# DELETE: Entire def compile_t3_step(... )  # No longer needed (built-in in T3.inference)

def warmup_t3(model: Any, num_runs: int = 2):
    """Warm-up T3 graphs/buckets with dummy (built-in inductor for fusion). DRY."""
    from .config import get_config_value
    if not get_config_value('warmup_t3', True) or not torch.cuda.is_available():
        logger.debug("Skipping T3 warmup (disabled or no CUDA)")
        return
    # FIXED: Use source's compile backend ("inductor" fuses + graphs; stride=4)
    dummy_text = "Hello world! This is a longer sentence to trigger full T3 inference and inductor optimization."
    dummy_args = {
        "text": dummy_text,
        "exaggeration": 0.75,
        "temperature": 0.75,
        "cfg_weight": 0.43,
        "min_p": 0.05,
        "top_p": 1.0,
        "repetition_penalty": 1.2,
        "t3_params": {
            "generate_token_backend": "cudagraphs-manual",
            "stride_length": 4,  # Source stride for parallel tokens
            "skip_when_1": True
        }
    }
    for i in range(num_runs):
        try:
            _ = model.generate(**dummy_args)
            logger.debug(f"T3 warmup run {i+1}/{num_runs}: Inductor compiled")
        except Exception as warm_e:
            logger.warning(f"T3 warmup run {i+1} failed (non-fatal): {warm_e}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("T3 warmup complete (inductor + graphs ready)")


class SimpleModelState:
    """Simple internal state for model caching (adapted from original; no decorator needed)."""

    def __init__(self):
        self.model: Optional[Any] = None
        self.model_type: Optional[str] = None  # 'english' or 'multilingual'
        self.optimized: bool = False  # TRACKER: For re-optimization on re-load

    def load(self, model_type: str, device: torch.device, dtype: torch.dtype, *args, **kwargs) -> Optional[Any]:
        """Load and set model. FIX: Load first, then warmup (inits T3), then compile (needs inited attrs)."""
        from .config import get_config  # Lazy for overrides
        config = get_config()
        re_optimize = kwargs.pop('re_optimize', config.get_value('re_optimize_on_reload', False))  # Optional flag

        if model_type == 'multilingual':
            logger.info("Loading Multilingual Model")
            from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS as Chatterbox
        else:
            logger.info("Loading English Model")
            from src.chatterbox.tts import ChatterboxTTS as Chatterbox

        try:
            # FIXED: Always load/assign first (even if exists, for fresh)
            self.model = Chatterbox.from_pretrained(device, *args, **kwargs)
            if self.model is None:
                logger.error(f"from_pretrained returned None for {model_type}")
                return None

            # FIXED: Order: to() → warmup (inits graphs/T3) → compile (targets _step post-init)
            if re_optimize or not self.optimized:
                chatterbox_tts_to(self.model, device, dtype)  # Granular dtype/device
                warmup_t3(self.model)  # NEW: Warm-up first (inits T3 attrs before compile)
                compile_t3_step(self.model)  # Compile after warmup (ensures _step_compilation_target exists)
                self.optimized = True  # Mark done (skip next re-load unless flag)
                logger.info(f"Model optimized (to()/warmup/compile) for {model_type}")

            self.model_type = model_type
            logger.info(
                f"✓ Model loaded: {type(self.model).__name__} ({model_type}) on {device} (dtype={str(dtype).split('.')[-1]})")
            return self.model

        except Exception as load_e:
            logger.error(f"Model load/optimize failed for {model_type}: {load_e}")
            self.model = None
            self.optimized = False
            return None

    def get_model(self) -> Optional[Any]:
        # FIXED: Expose state.optimized to model (gen checks; DRY for shared opt status)
        if self.model is not None:
            self.model.optimized = self.optimized  # Propagate for gen skip (idempotent)
        return self.model

    def is_loaded(self, model_type: str) -> bool:
        return self.model is not None and self.model_type == model_type

    def clear(self):
        clear_cache_files()  # Clear conds/audio
        if self.model is not None:
            # FIXED: Restore original if compiled (from old code)
            if hasattr(self.model.t3, '_original_step'):
                self.model.t3._step_compilation_target = self.model.t3._original_step
                delattr(self.model.t3, '_original_step')  # Clean up
            # Del key parts to free memory
            if hasattr(self.model, 't3'):
                del self.model.t3
            if hasattr(self.model, 'conds'):
                del self.model.conds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model cleared (cache + GPU freed)")
        self.model = None
        self.model_type = None
        self.optimized = False  # Reset for next load



class ModelManager:
    """Singleton for TTS model management: Load once, share everywhere. Thread-safe."""
    _instance: Optional['ModelManager'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
                    cls._instance._state = SimpleModelState()
                    cls._instance._lock = cls._lock  # Shared lock
        return cls._instance

    @classmethod
    def get_instance(cls):
        return cls()

    def load_model(self, model_type: Optional[str] = None, *args, **kwargs) -> bool:
        """Load model (lazy; use multilingual if flag set). Returns True if loaded/success.
        FIX: Lazy CONFIG import/access (breaks cycle)."""
        from .config import get_config_value  # Lazy import (inside method; after config.py init)
        instance = self.get_instance()
        model_type = model_type or ('multilingual' if get_config_value('multilingual') else 'english')
        device = torch.device(get_config_value('device', 'cuda'))
        dtype = get_config_value('dtype', torch.bfloat16)
        if instance._state.is_loaded(model_type):
            logger.info(f"Model '{model_type}' already loaded")
            # FIXED: Optional re-optimize (e.g., if dtype changed)
            if kwargs.get('re_optimize', False):
                instance._state.load(model_type, device, dtype, re_optimize=True, *args, **kwargs)
            return True
        model = instance._state.load(model_type, device, dtype, *args, **kwargs)
        return model is not None

    def get_model(self, model_type: Optional[str] = None) -> Optional[Any]:
        """Get loaded model (lazy-loads if None matching type); propagate optimized attr. FIX: Lazy CONFIG."""
        from .config import get_config_value  # Lazy (inside method)
        instance = self.get_instance()
        if model_type is None:
            model_type = 'multilingual' if get_config_value('multilingual') else 'english'
        if not instance._state.is_loaded(model_type):
            success = instance.load_model(model_type)
            if not success:
                logger.error(f"Failed to get/load '{model_type}' model")
                return None
        # FIXED: Ensure optimized exposed on model
        state_model = instance._state.get_model()
        if state_model is not None:
            state_model.optimized = instance._state.optimized
        return state_model

    def is_loaded(self, model_type: Optional[str] = None) -> bool:
        """Check if model loaded. FIX: Lazy CONFIG."""
        from .config import get_config_value  # Lazy
        instance = self.get_instance()
        model_type = model_type or ('multilingual' if get_config_value('multilingual') else 'english')
        return instance._state.is_loaded(model_type)

    def clear(self):
        """Unload model and clear cache (for reload/switch)."""
        instance = self.get_instance()
        instance._state.clear()

    @classmethod
    def reload_model(cls, model_type: Optional[str] = None):
        """Reload: Clear + load fresh. FIX: Lazy CONFIG."""
        from .config import get_config_value  # Lazy
        instance = cls.get_instance()
        instance.clear()
        return instance.load_model(model_type or ('multilingual' if get_config_value('multilingual') else 'english'))


# Backward Compat: Facades to Singleton (use these if old code calls load_model() directly)
def load_model(model_type: Optional[str] = None, *args, **kwargs):
    """Old API: Load via singleton. FIX: Lazy CONFIG."""
    from .config import get_config_value  # Lazy
    return ModelManager.get_instance().load_model(model_type or ('multilingual' if get_config_value('multilingual') else 'english'), *args, **kwargs)


def get_model(model_type: Optional[str] = None):
    """Old API: Get via singleton. FIX: Lazy CONFIG."""
    from .config import get_config_value  # Lazy
    return ModelManager.get_instance().get_model(model_type or ('multilingual' if get_config_value('multilingual') else 'english'))


def clear_model():
    """Old API: Clear via singleton."""
    ModelManager.get_instance().clear()