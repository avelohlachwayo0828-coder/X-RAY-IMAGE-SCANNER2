import streamlit as st
from PIL import Image

from predict import predict_tb
from scanner import analyze_xray
from database import create_database, save_scan, get_scans
# Create database table if it doesn't exist
create_database()
# -----------------------------
# Page Settings
# -----------------------------
st.set_page_config(
    page_title="TB Image Scanner",
    page_icon="🫁",
    layout="wide"
)

st.title("🫁 AI Tuberculosis Image Scanner")

st.markdown(
"""
Upload a chest X-ray and let the AI:

✅ Detect Tuberculosis

✅ Estimate confidence

✅ Generate a professional medical report

⚠ This is an AI screening tool and **not** a replacement for a doctor.
"""
)

uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Chest X-ray",
        width="stretch"
    )

    if st.button("Analyze X-ray"):

        with st.spinner("Analyzing..."):

            # -----------------------------
            # CNN Prediction
            # -----------------------------
            prediction = predict_tb(image)

            st.divider()

            st.subheader("🧠 CNN Prediction")

            diagnosis = prediction["diagnosis"]

            if "TB" in diagnosis:
                st.error(diagnosis)
            else:
                st.success(diagnosis)

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Confidence",
                f'{prediction["confidence"]}%'
            )

            col2.metric(
                "TB Probability",
                f'{prediction["tb_probability"]}%'
            )

            col3.metric(
                "Normal Probability",
                f'{prediction["normal_probability"]}%'
            )

            # -----------------------------
            # Gemini Report
            # -----------------------------
            st.divider()

            st.subheader("📋 AI Medical Report")

            report = analyze_xray(image)

            st.markdown(report)

            # -----------------------------
            # Save to Database
                        # -----------------------------
            save_scan(
                uploaded_file.name,
                report
            )
            
            st.success("✔ Report saved to database.")
            # ------------------------------------
# Scan History
# ------------------------------------

st.divider()

st.subheader("📜 Previous Scan History")

scans = get_scans()

if len(scans) == 0:

    st.info("No previous scans found.")

else:

    for scan in scans:

        with st.expander(
            f"📅 {scan[1]} | 🖼 {scan[2]}"
        ):

            st.markdown(scan[3])