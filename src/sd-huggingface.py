from diffusers import StableDiffusionPipeline
import torch

# https://huggingface.co/docs/diffusers/v0.14.0/en/stable_diffusion

model_id = "runwayml/stable-diffusion-v1-5"


prompt = "generates top view plain game asset image, cyber cowboy, as a pixel art, retro, 8-bit, clasic gba pokemon"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

pipe = pipe.to("cuda")

# guidance scale, inference steps, noise schedule.

generator = torch.Generator("cuda").manual_seed(0)

image = pipe(prompt, generator=generator).images[0]

path = "/dd/lab/tmp"

image.save(path + "/ouput.png")
