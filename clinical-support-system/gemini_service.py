import os

from google import genai
from google.genai import types


# =========================================================
# GEMINI CONFIGURATION
# =========================================================

GEMINI_MODEL = "gemini-3.6-flash"


# =========================================================
# CREATE GEMINI CLIENT
# =========================================================

def get_gemini_client():

    api_key = os.environ.get(
        "GEMINI_API_KEY"
    )

    if not api_key:
        return None

    return genai.Client(
        api_key=api_key
    )


# =========================================================
# SAFE FALLBACK MESSAGE
# =========================================================

def gemini_error_message():

    return (
        "AI interpretation is temporarily unavailable. "
        "The displayed result is based only on the trained "
        "chest X-ray classification model. Please interpret "
        "the result using appropriate clinical judgement."
    )


# =========================================================
# INTERPRET RESNET50 RESULT
# =========================================================

def interpret_prediction(result):

    client = get_gemini_client()

    if client is None:

        return {
            "success": False,
            "interpretation": gemini_error_message()
        }


    # -----------------------------------------------------
    # Extract model result
    # -----------------------------------------------------

    predicted_class = result.get(
        "predicted_class",
        "Unknown"
    )

    confidence = float(
        result.get(
            "confidence",
            0
        )
    )

    probabilities = result.get(
        "probabilities",
        {}
    )


    covid_probability = float(
        probabilities.get(
            "COVID19",
            0
        )
    )

    normal_probability = float(
        probabilities.get(
            "NORMAL",
            0
        )
    )

    pneumonia_probability = float(
        probabilities.get(
            "PNEUMONIA",
            0
        )
    )

    tb_probability = float(
        probabilities.get(
            "TB",
            0
        )
    )


    # -----------------------------------------------------
    # Prompt
    # -----------------------------------------------------

    prompt = f"""
You are an AI clinical decision-support assistant.

A trained four-class chest X-ray classification model produced
the following result:

Predicted class: {predicted_class}
Model confidence: {confidence:.2f}%

Class probabilities:

COVID19: {covid_probability:.2f}%
NORMAL: {normal_probability:.2f}%
PNEUMONIA: {pneumonia_probability:.2f}%
TB: {tb_probability:.2f}%

Provide a concise clinical-support interpretation.

Important rules:

1. Do NOT claim that the patient definitely has a disease.
2. Do NOT replace a qualified healthcare professional.
3. Do NOT invent radiological findings that were not provided.
4. Do NOT claim that Grad-CAM proves the presence of disease.
5. Clearly distinguish model prediction from clinical diagnosis.
6. Mention that additional clinical assessment may be required.
7. If the prediction is TB, explain that the model prediction
   should be clinically confirmed.
8. If the prediction is NORMAL, explain that a model prediction
   of normal does not rule out disease.
9. Keep the response professional and concise.
10. Do not provide treatment or medication instructions.

Use exactly these sections:

Model Interpretation
Clinical Significance
Important Limitation
Recommended Next Step
"""


    # -----------------------------------------------------
    # Call Gemini
    # -----------------------------------------------------

    try:

        response = client.models.generate_content(

            model=GEMINI_MODEL,

            contents=prompt,

            config=types.GenerateContentConfig(

                temperature=0.2,

                max_output_tokens=500
            )
        )


        if not response or not response.text:

            return {
                "success": False,
                "interpretation": gemini_error_message()
            }


        return {

            "success": True,

            "interpretation": response.text.strip()
        }


    except Exception as error:

        print(
            "GEMINI ERROR:",
            error
        )

        return {

            "success": False,

            "interpretation": gemini_error_message()
        }