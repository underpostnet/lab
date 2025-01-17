import os
from dotenv import load_dotenv

min_memory_available = 3.7 * 1024 * 1024 * 1024
load_dotenv()

# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# os.environ["UNSLOTH_IS_PRESENT"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = (
    "0, 1, 2, 3, 4, 5, 6"  # nvidia-smi -> for list CUDA_VISIBLE_DEVICES id's
)
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

print("cuda device_count:", torch.cuda.device_count())
# exit()

from huggingface_hub import login

login(token=os.getenv("HUGGINGFACE_API_KEY"))

from diffusers import FluxPipeline

# pip install -U "huggingface_hub[cli]"
# pip install -U git+https://github.com/huggingface/diffusers.git
# pip install -U transformers[sentencepiece]
# pip install -U numpy==1.23.4
# pip install -U protobuf==3.20
# pip install -U accelerate
# pip install -U sentencepiece
# https://huggingface.co/settings/tokens
# https://stackoverflow.com/questions/78899566/issue-loading-fluxpipeline-components
# https://huggingface.co/black-forest-labs/FLUX.1-dev

# num_inference_steps: between 20 and 50 (detailed)
# guidance_scale: between 7 and 8.5 (literal prompt)
# pytorch -> model framework
# tensorflow -> deployment tools

pipe = FluxPipeline.from_pretrained(
    # "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float32,
)

# .to(torch.device("cuda"))

pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

# pipe = torch.tensor([1, 2, 3])
pipe = pipe.to("cuda")

prompt = "top view plain game asset pixel art, retro, 8-bit, pokemon gba rom image, of a cyber cowboy sprite"
image = pipe(
    prompt=prompt,
    height=256,
    width=256,
    guidance_scale=3.5,
    # num_inference_steps=5,
    # max_sequence_length=256,
    # output_type="pil",
    num_images_per_prompt=1,
    # generator=torch.Generator("cpu").manual_seed(0),
).images[0]

image.save("flux-dev.png")
