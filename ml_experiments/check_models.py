import os
from dotenv import load_dotenv
import google.generativeai as genai


# Load API key
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")


# Connect to Gemini
genai.configure(api_key=API_KEY)


# List available models
models = genai.list_models()


for model in models:
    print(model.name)
    print(model.supported_generation_methods)
    print("------------------------")