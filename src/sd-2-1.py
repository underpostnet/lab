import os
from dotenv import load_dotenv

load_dotenv()

min_memory_available = 3.85 * 1024 * 1024 * 1024

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:" + str(
    min_memory_available / (1024 * 1024)
)

print("PYTORCH_CUDA_ALLOC_CONF:", os.environ["PYTORCH_CUDA_ALLOC_CONF"])

import torch
from utilTorch import clear_gpu_memory, wait_until_enough_gpu_memory


clear_gpu_memory()
wait_until_enough_gpu_memory(min_memory_available)

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe = pipe.to("cuda")

prompt = "top view plain game asset pixel art, retro, 8-bit, pokemon gba rom image, of a cyber cowboy sprite"
image = pipe(
    prompt=prompt,
    height=256,
    width=256,
    guidance_scale=50,
    num_inference_steps=7,
    # max_sequence_length=256,
    # output_type="pil",
    num_images_per_prompt=1,
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]

image.save("test.png")
