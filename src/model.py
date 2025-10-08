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


class SimpleModelState:
    """Simple internal state for model caching (adapted from original; no decorator needed)."""

    def __init__(self):
        self.model: Optional[Any] = None
        self.model_type: Optional[str] = None  # 'english' or 'multilingual'

    def load(self, model_type: str, device: torch.device, dtype: torch.dtype, *args, **kwargs) -> Optional[Any]:
        """Load and set model. FIX: No dtype in from_pretrained; apply .to(dtype) post-load (original pattern)."""
        if model_type == 'multilingual':
            logger.info("Loading Multilingual Model")
            from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS as Chatterbox
        else:
            logger.info("Loading English Model")
            from src.chatterbox.tts import ChatterboxTTS as Chatterbox

        # FIX: Load without dtype in constructor (device only); apply dtype post-load
        try:
            self.model = Chatterbox.from_pretrained(device, *args, **kwargs)  # Original: device, no dtype
            if self.model is not None:
                # Post-load: Convert key components to dtype (as in original; safe if hasattr)
                self.model.t3.to(device=device, dtype=dtype)
                if hasattr(self.model, 'conds') and hasattr(self.model.conds, 't3'):
                    self.model.conds.t3.to(device=device, dtype=dtype)
                else:
                    logger.warning("No 'conds.t3' found – skipping dtype conversion for conds")
                torch.cuda.empty_cache()
                self.model_type = model_type
                logger.info(
                    f"✓ Model loaded: {type(self.model).__name__} ({model_type}) on {device} (dtype={str(dtype).split('.')[-1]})")
                return self.model
            else:
                logger.error(f"from_pretrained returned None for {model_type}")
                return None
        except Exception as load_e:
            logger.error(f"Model load failed in from_pretrained for {model_type}: {load_e}")
            self.model = None
            return None

    def get_model(self) -> Optional[Any]:
        return self.model

    def is_loaded(self, model_type: str) -> bool:
        return self.model is not None and self.model_type == model_type

    def clear(self):
        clear_cache_files()  # Clear conds/audio
        if self.model is not None:
            # Optional: Del model tensors to free GPU (if supports)
            if hasattr(self.model, 't3'):
                del self.model.t3
            if hasattr(self.model, 'conds'):
                del self.model.conds
            torch.cuda.empty_cache()
            logger.info("Model cleared (cache + GPU freed)")
        self.model = None
        self.model_type = None


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
        from config import CONFIG  # Lazy import (inside method; after config.py init)
        instance = self.get_instance()
        model_type = model_type or ('multilingual' if CONFIG.multilingual else 'english')
        device = CONFIG.device
        dtype = CONFIG.dtype
        if instance._state.is_loaded(model_type):
            logger.info(f"Model '{model_type}' already loaded")
            return True
        model = instance._state.load(model_type, device, dtype, *args, **kwargs)
        return model is not None

    def get_model(self, model_type: Optional[str] = None) -> Optional[Any]:
        """Get loaded model (lazy-loads if None matching type).
        FIX: Lazy CONFIG import/access."""
        from config import CONFIG  # Lazy (inside method)
        instance = self.get_instance()
        if model_type is None:
            model_type = 'multilingual' if CONFIG.multilingual else 'english'
        if not instance._state.is_loaded(model_type):
            success = instance.load_model(model_type)
            if not success:
                logger.error(f"Failed to get/load '{model_type}' model")
                return None
        return instance._state.get_model()

    def is_loaded(self, model_type: Optional[str] = None) -> bool:
        """Check if model loaded. FIX: Lazy CONFIG."""
        from config import CONFIG  # Lazy
        instance = self.get_instance()
        model_type = model_type or ('multilingual' if CONFIG.multilingual else 'english')
        return instance._state.is_loaded(model_type)

    def clear(self):
        """Unload model and clear cache (for reload/switch)."""
        instance = self.get_instance()
        instance._state.clear()

    @classmethod
    def reload_model(cls, model_type: Optional[str] = None):
        """Reload: Clear + load fresh. FIX: Lazy CONFIG."""
        from config import CONFIG  # Lazy
        instance = cls.get_instance()
        instance.clear()
        return instance.load_model(model_type or ('multilingual' if CONFIG.multilingual else 'english'))


# Backward Compat: Facades to Singleton (use these if old code calls load_model() directly)
def load_model(model_type: Optional[str] = None, *args, **kwargs):
    """Old API: Load via singleton. FIX: Lazy CONFIG."""
    from config import CONFIG  # Lazy
    return ModelManager.get_instance().load_model(model_type or ('multilingual' if CONFIG.multilingual else 'english'), *args, **kwargs)


def get_model(model_type: Optional[str] = None):
    """Old API: Get via singleton. FIX: Lazy CONFIG."""
    from config import CONFIG  # Lazy
    return ModelManager.get_instance().get_model(model_type or ('multilingual' if CONFIG.multilingual else 'english'))


def clear_model():
    """Old API: Clear via singleton."""
    ModelManager.get_instance().clear()