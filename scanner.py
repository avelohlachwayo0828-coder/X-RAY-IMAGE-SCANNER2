import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Get API Key
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# Create Gemini client
client = genai.Client(api_key=API_KEY)


def analyze_xray(image):

    prompt = """
You are an expert radiologist specializing in tuberculosis screening.

Analyze this chest X-ray and produce a professional report.

## 1. Image Information
- Imaging modality
- Image quality

## 2. Lung Findings
Evaluate:
- Upper lung abnormalities
- Cavities
- Consolidation
- Infiltrates
- Nodules
- Pleural changes

## 3. TB Screening Assessment

Classify as ONE of:

🟢 Normal

🟡 Possible TB

🔴 Highly Suggestive of TB

⚪ Unable to Determine

Explain why.

## 4. Patient Explanation

Explain the findings using simple language.

## 5. Recommendations

Suggest appropriate medical follow-up.

IMPORTANT:
This AI is a screening tool only.
It is NOT a medical diagnosis.
"""

    try:

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, image]
        )

        return response.text

    except Exception as e:

        return f"""
# ⚠ AI Analysis Failed

Reason:

{e}
"""