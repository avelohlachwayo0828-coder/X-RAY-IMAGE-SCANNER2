import os
import io
import base64
import concurrent.futures

from dotenv import load_dotenv
load_dotenv()

from groq import Groq

API_KEY = os.environ.get("GROQ_API_KEY")
VISION_MODEL_NAME = "qwen/qwen3.6-27b"

SAFE_FALLBACK = """
Image explanation is temporarily unavailable.

Please have the chest image reviewed by a qualified healthcare professional.

The automated image analysis should not be used by itself
to make a diagnosis or treatment decision.
""".strip()

client = None
if API_KEY:
    try:
        client = Groq(api_key=API_KEY)
        print("Groq client initialized.")
    except Exception as e:
        print("Groq client initialization error:", e)
else:
    print("API key not found. Set GROQ_API_KEY in your .env / Codespace secrets.")

SYSTEM_PROMPT = """
You are an assistant helping a healthcare professional
understand a chest X-ray image in simple language.

Describe visible image features only.
Do NOT provide a diagnosis.
Do NOT mention AI, models, or technical terms.

Use this format:

IMAGE EXPLANATION
(plain description)

LOCATION
(where the noticeable area is)

WHAT THIS MEANS
(what the difference might indicate)

IMPORTANT
(that a professional must review the image)
""".strip()

def clean_response(text):
    if not text:
        return SAFE_FALLBACK
    forbidden = ["classifier", "confidence score", "probability score", "as an ai model", "i am an ai"]
    lowered = text.lower()
    if any(phrase in lowered for phrase in forbidden):
        print("Technical content detected - using fallback. Matched phrase check.")
        return SAFE_FALLBACK
    return text.strip()

def image_to_data_url(image):
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"

def interpret_chest_xray(image):
    if client is None or image is None:
        return SAFE_FALLBACK

    try:
        working_image = image.copy()
        working_image.thumbnail((512, 512))
        data_url = image_to_data_url(working_image)

        def call_groq():
            return client.chat.completions.create(
                model=VISION_MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": SYSTEM_PROMPT},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                temperature=0.5,
                max_tokens=800,
            )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(call_groq)
            try:
                completion = future.result(timeout=45)
                text = completion.choices[0].message.content
                return clean_response(text) if text else SAFE_FALLBACK
            except concurrent.futures.TimeoutError:
                print("Groq API call timed out (45s).")
                return SAFE_FALLBACK

    except Exception as e:
        print("Groq error:", e)
        return SAFE_FALLBACK
