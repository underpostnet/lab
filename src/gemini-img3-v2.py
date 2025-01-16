import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

response1 = client.models.generate_image(
    model="imagen-3.0-generate-001",
    prompt="Fuzzy bunnies in my kitchen",
    config=types.GenerateImageConfig(
        negative_prompt="people",
        number_of_images=1,
        include_rai_reason=True,
        output_mime_type="image/jpeg",
    ),
)

response1.generated_images[0].image.save("result.jpeg")
