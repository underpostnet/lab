# stable-diffusion
# https://machinelearningmastery.com/running-stable-diffusion-with-python/

from diffusers import StableDiffusionPipeline, DDPMScheduler
import torch
import os
import sys
from util import index_exists
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


path = "/dd/lab/tmp"

os.makedirs(path, exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", variant="fp16", torch_dtype=torch.float16
)

# TODO: Torch not compiled with CUDA enabled
pipe.to("cuda")

prompt = (
    "A cat took a fish and running in a market"
    if not index_exists(sys.argv, 1)
    else sys.argv[1]
)

# scheduler = DDPMScheduler(
#     beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
# )
image = pipe(
    prompt,
    #    scheduler=scheduler,
    num_inference_steps=30,
    guidance_scale=7.5,
).images[0]
image.save(path + "/ouput.png")

if index_exists(sys.argv, 2) and sys.argv[2] == "show":
    img = mpimg.imread(path + "/ouput.png")

    imgplot = plt.imshow(img)

    plt.show()
