from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"

prompt = "portrait photo of a old warrior chief"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

pipe = pipe.to("cuda")

generator = torch.Generator("cuda").manual_seed(0)

image = pipe(prompt, generator=generator).images[0]

path = "/dd/lab/tmp"

image.save(path + "/ouput.png")
