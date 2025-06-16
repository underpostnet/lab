# ./my-env/bin/python ./src/gemini-list-models.py

import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


for m in genai.list_models():
    print(
        "name:", m.name, "supported_generation_methods:", m.supported_generation_methods
    )
