from dotenv import load_dotenv
import google.generativeai as genai
import os


load_dotenv()


genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


def read_text_from_file(file_path):
    with open(file_path, "r") as file:
        text = file.read()
    return text


# The stable version of Python SDK for the Gemini API does not contain Imagen support. Instead of installing the
# google-generativeai package from pypi you need to install it from the imagen GitHub branch: pip install -U
# git+https://github.com/google-gemini/generative-ai-python@imagen

# pip install -U git+https://github.com/google-gemini/generative-ai-python@imagen
# keys https://aistudio.google.com/
# docs https://ai.google.dev/gemini-api/docs/imagen


imagen = genai.ImageGenerationModel("imagen-3.0-generate-001")

result = imagen.generate_images(
    prompt="Fuzzy bunnies in my kitchen",
    number_of_images=1,
    safety_filter_level="block_only_high",
    person_generation="allow_adult",
    aspect_ratio="1:1",
    negative_prompt="Outside",
)

for index, image in enumerate(result.images):
    # Open and display the image using your local operating system.
    image._pil_image.show()
    # Save the image using the PIL library's save function
    image._pil_image.save(f"image_{index}.jpg")
