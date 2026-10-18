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
You are an expert clinical radiologist dictating a formal chest X-ray report. Interpret the provided radiograph using the RIPE and ABCDE frameworks based on standard clinical guidelines.

Strictly adhere to the following structure and clinical criteria:

1. RIPE — Image Quality Assessment
   - Rotation: Medial ends of the clavicles should be equidistant from the spinous processes. Spinous processes should be vertically aligned.
   - Inspiration: 5-6 anterior ribs should be visible, along with lung apices, both costophrenic angles, and lateral rib edges.
   - Projection: State if PA or AP. (If scapulae are not projected within the chest, it is PA).
   - Exposure: Left hemidiaphragm should be visible to the spine; vertebrae should be visible behind the heart.

2. ABCDE — Systematic Interpretation
   - A — Airway: Trachea should be central/slightly right. Note true deviation (pushing: effusion/pneumothorax; pulling: consolidation/collapse) or apparent deviation (rotation). Assess carina, bronchi, and hila.
   - B — Breathing (Lungs & Pleura): Divide lungs into upper, mid, and lower zones. Note airspace shadowing (consolidation), absence of lung markings (pneumothorax), or pleural thickening. Assess for effusions. Explicitly state absence of pneumothorax or large effusion.
   - C — Cardiac & Mediastinum: On PA films, cardiothoracic ratio should be < 50%. Check for loss of right heart border (RML consolidation) or left heart border (lingular consolidation). Assess aortic knuckle and aortopulmonary window.
   - D — Diaphragm: Right hemidiaphragm is normally higher. Identify gastric bubble. Look for free subdiaphragmatic gas (pneumoperitoneum). Costophrenic angles should be acute and well-defined (note blunting if present).
   - E — Everything Else: Bones, soft tissues, tubes/lines/pacemakers. CRITICAL: Evaluate the "review areas" (lung apices, retrocardiac region, behind diaphragm, peripheral lungs, hilar regions).

3. IMPRESSION
   - A concise, actionable final clinical conclusion or differential diagnosis summary.

TONE & VOCABULARY: Use precise medical terminology. Do NOT use casual phrases. Do NOT provide reasoning, thinking, or planning text. 

FORMATTING: Do NOT use Markdown. Do not use asterisks (*) for bold or bullet points. Use plain text, ALL CAPS for headings, and standard hyphens (-) for bullet points.

Respond with ONLY the final report.
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

    # Fallback cleanup: Force remove markdown asterisks if the AI ignores the prompt
    text = text.replace("**", "")
    text = text.replace("* ", "- ")
    
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