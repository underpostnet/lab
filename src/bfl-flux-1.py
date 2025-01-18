import os
from dotenv import load_dotenv

min_memory_available = 7.5 * 1024 * 1024 * 1024
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
model_id = "black-forest-labs/FLUX.1-dev"

print("load model pretrained", model_id)

pipe = FluxPipeline.from_pretrained(
    # "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    model_id,
    torch_dtype=torch.float32,
    # "black-forest-labs/FLUX.1-dev",
    # torch_dtype=torch.float16,
)

# .to(torch.device("cuda"))

pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

# pipe = torch.tensor([1, 2, 3])
pipe = pipe.to("cuda")

prompt = "top view plain game asset pixel art, retro, 8-bit, pokemon gba rom image, of a cyber cowboy sprite"

print("exec pipe prompt:", prompt)

image = pipe(
    prompt=prompt,
    height=256,
    width=256,
    guidance_scale=3.5,
    num_inference_steps=5,
    # max_sequence_length=256,
    # output_type="pil",
    num_images_per_prompt=1,
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]

image.save("flux-dev.png")
