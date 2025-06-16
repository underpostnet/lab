import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ./my-env/bin/pip install -U git+https://github.com/google-gemini/generative-ai-python@imagen
# ./my-env/bin/pip install -U python-dotenv
# ./my-env/bin/python ./src/gemini-img3.py

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

imagen = genai.ImageGenerationModel("models/gemini-1.5-flash")

result = imagen.generate_images(
    prompt="Fuzzy bunnies in my kitchen",
    number_of_images=1,
    safety_filter_level="block_only_high",
    person_generation="allow_adult",
    aspect_ratio="3:4",
    negative_prompt="Outside",
)

for image in result.images:
    print(image)

# Open and display the image using your local operating system.
for image in result.images:
    image._pil_image.show()
