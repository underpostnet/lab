# https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/overview
# https://machinelearningmastery.com/further-stable-diffusion-pipeline-with-diffusers/

from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
from io import BytesIO
import requests

pipe = StableDiffusionImageVariationPipeline.from_pretrained(
    "lambdalabs/sd-image-variations-diffusers", revision="v2.0"
)

pipe = pipe.to("cuda")

url = "https://www.cyberiaonline.com/assets/skin/paranoia/08/0.png"

response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert("RGB")


out = pipe(image, num_images_per_prompt=1, guidance_scale=5)
out["images"][0].save("result.jpg")
