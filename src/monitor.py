# src/monitor.py
import time
import torch
from functools import wraps
from loguru import logger



def monitor_resources(enable: bool = True, log_level: str = "INFO"):
    """Decorator: Log VRAM delta/peak and GPU util for NVIDIA. Skips if no CUDA."""

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not torch.cuda.is_available() or not enable:
                logger.debug("No GPU; skipping resource monitor for {}".format(fn.__name__))
                return fn(*args, **kwargs)

            # Init pynvml (once)
            try:
                import pynvml  # pip install nvidia-ml-py
                pynvml.nvmlInit()
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(pynvml.nvmlDeviceGetHandleByIndex(0)).gpu
                logger.debug(f"GPU util start: {gpu_util}% for {fn.__name__}")
            except Exception:
                logger.warning("pynvml unavailable; VRAM only")
                gpu_util = None

            # VRAM start
            torch.cuda.synchronize()  # Ensure complete
            vram_start = torch.cuda.memory_allocated(0) / (1024 ** 3)  # GB
            vram_peak_start = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
            torch.cuda.reset_peak_memory_stats()  # Reset for delta

            start_time = time.perf_counter()
            result = fn(*args, **kwargs)
            end_time = time.perf_counter()

            # VRAM end/peak
            torch.cuda.synchronize()
            vram_end = torch.cuda.memory_allocated(0) / (1024 ** 3)
            vram_delta = vram_end - vram_start
            vram_peak = torch.cuda.max_memory_allocated(0) / (1024 ** 3)

            # GPU end (avg util rough; sample mid)
            if gpu_util is not None:
                mid_util = pynvml.nvmlDeviceGetUtilizationRates(pynvml.nvmlDeviceGetHandleByIndex(0)).gpu
                end_util = pynvml.nvmlDeviceGetUtilizationRates(pynvml.nvmlDeviceGetHandleByIndex(0)).gpu
                avg_util = (gpu_util + mid_util + end_util) / 3
            else:
                avg_util = "N/A"

            # Log
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.log(log_level,
                       f"{fn.__name__}: VRAM delta {vram_delta:.2f}GB (start {vram_start:.2f}GB, peak {vram_peak:.2f}GB/{total_vram:.1f}GB); "
                       f"GPU avg {avg_util}%; time {end_time - start_time:.2f}s")

            pynvml.nvmlShutdown()  # Cleanup (if init succeeded)
            return result

        return wrapper

    return decorator


# Usage: In generate_audio.py
# @monitor_resources(enable=True, log_level="INFO")
# async def generate_audio(...):  # Your fn
#    # Unchanged body
#    pass