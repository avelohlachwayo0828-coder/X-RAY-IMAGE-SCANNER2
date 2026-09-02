import os
from dotenv import load_dotenv
import google.generativeai as genai


print("Starting Gemini test...")


load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")


print("API Key loaded:")
print(API_KEY[:10] + "********")


genai.configure(
    api_key=API_KEY
)


print("Creating model...")


model = genai.GenerativeModel(
    "gemini-flash-latest"
)


print("Sending request...")


response = model.generate_content(
    "Explain tuberculosis in one sentence"
)


print("Response received!")


print(response.text)