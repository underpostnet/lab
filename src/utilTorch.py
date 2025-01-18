import time
import gc
import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()
    # torch.cuda.memory_summary(device=None, abbreviated=False)
    # del variables


def wait_until_enough_gpu_memory(min_memory_available, max_retries=10, sleep_time=5):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    for _ in range(max_retries):
        info = nvmlDeviceGetMemoryInfo(handle)
        if info.free >= min_memory_available:
            break
        print(
            f"Waiting for {min_memory_available} bytes of free GPU memory. Retrying in {sleep_time} seconds..."
        )
        time.sleep(sleep_time)
    else:
        raise RuntimeError(
            f"Failed to acquire {min_memory_available} bytes of free GPU memory after {max_retries} retries."
        )
