import os
from dotenv import load_dotenv

load_dotenv()

min_memory_available = 3.5 * 1024 * 1024 * 1024  # 2GB

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:" + str(
    min_memory_available / (1024 * 1024)
)

print("PYTORCH_CUDA_ALLOC_CONF:", os.environ["PYTORCH_CUDA_ALLOC_CONF"])

import torch
import time
import gc
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()
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


clear_gpu_memory()
wait_until_enough_gpu_memory(min_memory_available)

# sd 3 medium

from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# num_inference_steps: between 20 and 50 (detailed)
# guidance_scale: between 7 and 8.5 (literal prompt)

image = pipe(
    "top view plain game asset pixel art, retro, 8-bit, pokemon gba rom image, of a cyber cowboy sprite",
    negative_prompt="",  # without this concepts
    num_inference_steps=28,
    guidance_scale=5.0,
).images[0]

image.save("test.jpeg")
