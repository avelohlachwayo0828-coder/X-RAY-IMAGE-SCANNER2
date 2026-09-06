import os
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

# Load environment variables
load_dotenv()

# Get API Key
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# Configure Gemini (old SDK)
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')   # or 'gemini-1.5-pro'


def analyze_xray(image):
    """
    Analyze a chest X-ray image and return a Gemini-generated report.
    """
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
        # Send prompt + image to Gemini (old SDK supports PIL Image)
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"""
# ⚠ AI Analysis Failed

Reason:
{e}
"""