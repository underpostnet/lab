import requests
import os
from dotenv import load_dotenv

load_dotenv()


API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
headers = {"Authorization": "Bearer " + os.getenv("HUGGINGFACE_API_KEY")}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content


image_bytes = query(
    {
        "inputs": "top view plain game asset pixel art, retro, 8-bit, pokemon gba rom image, of a cyber cowboy sprite",
    }
)

# You can access the image with PIL.Image for example
import io
from PIL import Image

print("image_bytes", image_bytes)

image = Image.open(io.BytesIO(image_bytes))

image.save("result.png")
