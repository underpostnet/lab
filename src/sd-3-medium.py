import os
from dotenv import load_dotenv

load_dotenv()

min_memory_available = 7.5 * 1024 * 1024 * 1024

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["UNSLOTH_IS_PRESENT"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = (
    "0, 1, 2, 3, 4, 5, 6"  # nvidia-smi -> for list CUDA_VISIBLE_DEVICES id's
)
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:" + str(
#     min_memory_available / (1024 * 1024)
# )
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
#     "garbage_collection_threshold:0.9,max_split_size_mb:512"
# )


print("PYTORCH_CUDA_ALLOC_CONF:", os.environ["PYTORCH_CUDA_ALLOC_CONF"])

import torch
from utilTorch import clear_gpu_memory, wait_until_enough_gpu_memory


clear_gpu_memory()
wait_until_enough_gpu_memory(min_memory_available)

# sd 3 medium

from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)


# pipe = torch.tensor([0, 1, 2, 3], device=device)
pipe = pipe.to(device)

image = pipe(
    "top view plain game asset pixel art, retro, 8-bit, pokemon gba rom image, of a cyber cowboy sprite",
    # negative_prompt="",  # without this concepts
    height=128,
    width=128,
    guidance_scale=3.5,
    num_inference_steps=5,
    num_images_per_prompt=1,
    # generator=torch.Generator("cuda").manual_seed(0),
).images[0]

image.save("test.jpeg")
