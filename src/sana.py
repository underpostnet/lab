import os
from dotenv import load_dotenv

min_memory_available = 3.75 * 1024 * 1024 * 1024
load_dotenv()

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["UNSLOTH_IS_PRESENT"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = (
    "0, 1, 2, 3, 4, 5, 6"  # nvidia-smi -> for list CUDA_VISIBLE_DEVICES id's
)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:" + str(
    min_memory_available / (1024 * 1024)
)

print("PYTORCH_CUDA_ALLOC_CONF:", os.environ["PYTORCH_CUDA_ALLOC_CONF"])

import torch
from utilTorch import clear_gpu_memory, wait_until_enough_gpu_memory


clear_gpu_memory()
wait_until_enough_gpu_memory(min_memory_available)

print("cuda device_count:", torch.cuda.device_count())
# exit()

from diffusers import SanaPipeline

pipe = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
    variant="bf16",
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
pipe.to("cuda")

pipe.vae.to(torch.bfloat16)
pipe.text_encoder.to(torch.bfloat16)

prompt = "generate top view plain, pokemon gba, cartoon anime with big head, of a cyber cowboy pixel, retro, 8bit, front sprite image"

image = pipe(
    prompt=prompt,
    height=512,
    width=512,
    guidance_scale=30,
    num_inference_steps=6,
    # max_sequence_length=256,
    # output_type="pil",
    num_images_per_prompt=1,
    generator=torch.Generator("cuda").manual_seed(0),
)[0]

image[0].save("sana.png")
