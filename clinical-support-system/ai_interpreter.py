import os
import io
import re
import base64
import concurrent.futures

from dotenv import load_dotenv
load_dotenv()

from groq import Groq

API_KEY = os.environ.get("GROQ_API_KEY")
VISION_MODEL_NAME = "qwen/qwen3.8-27b"  # fixed: was "qwen/qwen3.6-27b" (doesn't exist)

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
You are an expert clinical radiologist dictating a formal chest X-ray report for other healthcare professionals. Interpret the provided chest radiograph using standard, structured medical nomenclature.

Strictly adhere to the following rules:

1. STRUCTURE: Organize the output into three distinct clinical sections:
   - TECHNICAL ASSESSMENT: Note the projection (PA or AP), inspiration quality (rib count), rotation, and penetration.
   - FINDINGS: Break this down systematically (Airway/Hilum, Lungs/Pleura, Cardiac/Mediastinum, Diaphragm/Angles, Osseous/Soft Tissues).
   - IMPRESSION: A concise, actionable final clinical conclusion or diagnosis summary.

2. TONE & VOCABULARY: Use precise medical terminology.
   - Avoid casual phrases like "the lung is clear," "bones look good," or "normal positions."
   - Use professional equivalents: "no focal consolidation," "no acute osseous abnormalities," "cardiomediastinal silhouette is within normal limits," "costophrenic angles are sharp."

3. NEGATIVE FINDINGS: Always explicitly state the absence of critical acute pathologies (e.g., "no pneumothorax," "no pleural effusion," "no free air under the diaphragm") to ensure safety and completeness.

Respond with ONLY the final report. Do not include reasoning, thinking, planning, or references to AI/models.
""".strip()

def strip_thinking(text):
    # Remove <think>...</think> reasoning blocks some models include
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Handle an unclosed <think> tag (model cut off before closing it)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()

def clean_response(text):
    if not text:
        print("Groq returned empty text.")
        return SAFE_FALLBACK

    text = strip_thinking(text)

    if not text:
        print("Response was empty after stripping <think> block (model likely ran out of tokens while reasoning).")
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
                temperature=0.2,
                max_tokens=2000,
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
        print("Groq error:", e)  # <-- watch this in your terminal/console when you run it
        return SAFE_FALLBACK