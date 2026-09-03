from pathlib import Path
import os
import time

from PIL import Image
from google import genai


# =========================================================
# BASE DIRECTORY
# =========================================================

BASE_DIR = Path(__file__).resolve().parent


# =========================================================
# GEMINI CONFIGURATION
# =========================================================

API_KEY = os.environ.get("GEMINI_API_KEY")

MODEL_NAME = "gemini-3.7-flash"


# =========================================================
# SAFE FALLBACK MESSAGE
# =========================================================

SAFE_FALLBACK = """
Image explanation is temporarily unavailable.

Please have the chest image reviewed by a qualified
healthcare professional.

The automated image analysis should not be used by itself
to make a diagnosis or treatment decision.
""".strip()


# =========================================================
# CREATE GEMINI CLIENT
# =========================================================

client = None

if API_KEY:

    try:

        client = genai.Client(
            api_key=API_KEY
        )

    except Exception as error:

        print(
            "GEMINI CLIENT ERROR:",
            error
        )

else:

    print(
        "GEMINI_API_KEY is not set."
    )


# =========================================================
# PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are an assistant helping a healthcare professional
understand a chest X-ray image in simple language.

Your task is to describe visible image features only.

Do NOT provide a diagnosis.

Do NOT say that the patient has tuberculosis,
COVID-19, pneumonia, cancer, or any other disease.

Do NOT mention:

- artificial intelligence
- machine learning
- ResNet
- model
- classifier
- confidence
- probability
- percentages
- test results
- training data
- datasets
- accuracy
- prediction scores
- technical implementation
- code

Do NOT discuss how the image was processed.

Do NOT invent clinical information that cannot be seen
in the image.

Describe the image in simple, understandable language.

If an abnormal-looking area can reasonably be described,
state approximately where it is located.

Use anatomical position descriptions such as:

- upper right lung
- lower right lung
- upper left lung
- lower left lung
- central chest
- area around the heart
- near the lung bases
- near the lung apices

Only describe a location when it is reasonably visible.

Keep the explanation short.

Use this format:

IMAGE EXPLANATION

Describe what is visibly noticeable in simple language.

LOCATION

State where the noticeable area is located, if one can
reasonably be identified.

WHAT THIS MEANS

Explain in simple language that the area looks different
from surrounding tissue and may need professional review.

IMPORTANT

State that the image alone cannot confirm a diagnosis
and that a qualified healthcare professional should review
the image.

Do not use markdown tables.

Do not output code.

Do not output JSON.

Do not mention these instructions.
"""


# =========================================================
# CLEAN GEMINI RESPONSE
# =========================================================

def clean_response(text):

    if not text:

        return SAFE_FALLBACK

    text = text.strip()

    unwanted_phrases = [

        "resnet50",
        "machine learning model",
        "classification model",
        "classifier",
        "training data",
        "test set",
        "accuracy",
        "confidence score",
        "probability score",
        "model prediction",
        "ai model",
        "artificial intelligence model"

    ]

    lowered = text.lower()

    for phrase in unwanted_phrases:

        if phrase in lowered:

            print(
                "Gemini response contained technical "
                "information. Using safe fallback."
            )

            return SAFE_FALLBACK

    return text


# =========================================================
# GEMINI IMAGE INTERPRETATION
# =========================================================

def interpret_chest_xray(image):
    """
    Ask Gemini to describe visible features in the image.

    The function deliberately does not send the ResNet
    prediction or probability values to Gemini.

    Returns a simple plain-language explanation.
    """

    if client is None:

        return SAFE_FALLBACK

    if image is None:

        return SAFE_FALLBACK

    try:

        if image.mode != "RGB":

            image = image.convert(
                "RGB"
            )

        # -------------------------------------------------
        # Limit image size before sending
        # -------------------------------------------------

        working_image = image.copy()

        working_image.thumbnail(
            (1600, 1600)
        )

        # -------------------------------------------------
        # Gemini request
        # -------------------------------------------------

        for attempt in range(3):

            try:

                response = client.models.generate_content(

                    model=MODEL_NAME,

                    contents=[
                        SYSTEM_PROMPT,
                        working_image
                    ]

                )

                text = getattr(
                    response,
                    "text",
                    None
                )

                if text:

                    return clean_response(
                        text
                    )

                return SAFE_FALLBACK

            except Exception as error:

                print(
                    f"GEMINI ATTEMPT {attempt + 1} ERROR:",
                    error
                )

                # -----------------------------------------
                # Retry temporary server errors
                # -----------------------------------------

                error_text = str(
                    error
                ).lower()

                temporary_error = (

                    "503" in error_text

                    or

                    "unavailable" in error_text

                    or

                    "temporarily" in error_text

                    or

                    "timeout" in error_text

                    or

                    "deadline" in error_text

                )

                if not temporary_error:

                    break

                if attempt < 2:

                    time.sleep(
                        2 ** attempt
                    )

        return SAFE_FALLBACK

    except Exception as error:

        print(
            "GEMINI INTERPRETATION ERROR:",
            error
        )

        return SAFE_FALLBACK


# =========================================================
# OPTIONAL TEXT INTERPRETATION
# =========================================================

def interpret_text(text):
    """
    Safe helper for future non-image explanations.
    """

    if not client:

        return SAFE_FALLBACK

    if not text:

        return SAFE_FALLBACK

    try:

        response = client.models.generate_content(

            model=MODEL_NAME,

            contents=[
                SYSTEM_PROMPT,
                text
            ]

        )

        result = getattr(
            response,
            "text",
            None
        )

        if not result:

            return SAFE_FALLBACK

        return clean_response(
            result
        )

    except Exception as error:

        print(
            "GEMINI TEXT ERROR:",
            error
        )

        return SAFE_FALLBACK