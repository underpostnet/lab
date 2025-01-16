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


from diffusers import FluxPipeline
from huggingface_hub import login

# pip install -U "huggingface_hub[cli]"
# pip install git+https://github.com/huggingface/diffusers.git
# pip install transformers[sentencepiece]
# https://huggingface.co/settings/tokens

# https://huggingface.co/black-forest-labs/FLUX.1-dev


login(token=os.getenv("HUGGINGFACE_API_KEY"))

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
)

pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

# pipe = torch.tensor([1, 2, 3])
pipe = pipe.to("cuda")

prompt = "top view plain game asset pixel art, retro, 8-bit, pokemon gba rom image, of a cyber cowboy sprite"
image = pipe(
    prompt,
    height=512,
    width=512,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]

image.save("flux-dev.png")
