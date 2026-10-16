from pathlib import Path
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "best.pt"

st.set_page_config(page_title="PPE Detection", page_icon="🦺", layout="wide")

# Explicit, high-contrast light theme. These rules override inherited/dark theme text.
st.markdown("""
<style>
:root {
  color-scheme: light;
  --text: #172033;
  --muted: #475569;
  --border: #d9e1ec;
  --surface: #ffffff;
  --subtle: #f7f9fc;
  --accent: #2457c5;
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .stApp {
  background: #ffffff !important;
  color: var(--text) !important;
}
html, body, [class*="css"], p, label, li, span, small, div {
  font-family: Arial, Helvetica, sans-serif;
}
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li, [data-testid="stMarkdownContainer"] span,
.stText, .stCaption, label, .stRadio, .stCheckbox, [data-testid="stWidgetLabel"] {
  color: var(--text) !important;
}
h1, h2, h3, h4, h5, h6 {
  color: #111827 !important;
}
.block-container { max-width: 1100px; padding-top: 2rem; padding-bottom: 2rem; }
.subtitle { color: #475569 !important; font-size: 1rem; margin: -.3rem 0 1.5rem; }
hr { border-color: var(--border) !important; }
section[data-testid="stSidebar"] {
  background: #f7f9fc !important;
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * {
  color: #172033 !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] .stCaption {
  color: #475569 !important;
}
[data-testid="stFileUploader"] {
  background: #ffffff !important;
  border: 1px dashed #9aa9bd !important;
  border-radius: 10px;
}
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {
  color: #172033 !important;
}
[data-testid="stFileUploader"] button {
  background: #ffffff !important;
  color: #1d4ed8 !important;
  border: 1px solid #9aa9bd !important;
}
div.stButton > button[kind="primary"] {
  background: #2457c5 !important;
  color: #ffffff !important;
  border: 1px solid #2457c5 !important;
  border-radius: 8px;
  font-weight: 600;
  min-height: 2.6rem;
}
div.stButton > button:not([kind="primary"]),
.stDownloadButton > button {
  background: #ffffff !important;
  color: #1d4ed8 !important;
  border: 1px solid #9aa9bd !important;
  border-radius: 8px;
}
input, textarea, [data-baseweb="select"] > div {
  background: #ffffff !important;
  color: #172033 !important;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
  background: #ffffff !important;
  color: #172033 !important;
}
.small-note { color: #475569 !important; font-size: .85rem; }
@media (max-width: 700px) {
  .block-container { padding: 1rem .8rem 2rem; }
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(path: str):
    return YOLO(path)

st.title("PPE Detection")
st.markdown(
    '<div class="subtitle">Upload an image to detect gloves, safety vests, helmets, and people.</div>',
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error("Model file not found. Place your trained best.pt file beside app.py.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    confidence = st.slider("Confidence threshold", 0.10, 0.90, 0.25, 0.05)
    st.caption("Higher confidence may reduce false detections but can miss objects.")
    st.divider()
    st.markdown("**Model:** YOLOv8n")
    st.markdown("**Training:** 25 epochs")
    st.markdown("**Test mAP@50:** 80.0%")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded:
    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception:
        st.error("Could not read that image. Please upload a valid JPG or PNG.")
        st.stop()

    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Detect PPE", type="primary", use_container_width=True):
        with st.spinner("Detecting objects..."):
            model = load_model(str(MODEL_PATH))
            result = model.predict(source=image, conf=confidence, imgsz=640, verbose=False)[0]

        annotated = result.plot()[:, :, ::-1]
        st.subheader("Results")
        st.image(annotated, caption="Detected objects", use_container_width=True)

        rows = []
        boxes = result.boxes
        if boxes is not None:
            for index, box in enumerate(boxes, start=1):
                rows.append({
                    "Object": index,
                    "Class": result.names[int(box.cls.item())],
                    "Confidence": round(float(box.conf.item()), 3),
                })

        if rows:
            df = pd.DataFrame(rows)
            st.write(f"**{len(rows)} object(s) detected**")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download results as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="ppe_detections.csv",
                mime="text/csv",
            )
            buffer = BytesIO()
            Image.fromarray(annotated).save(buffer, format="PNG")
            st.download_button(
                "Download annotated image",
                data=buffer.getvalue(),
                file_name="ppe_detection_result.png",
                mime="image/png",
            )
        else:
            st.info("No objects detected at this confidence threshold. Try lowering it.")

st.divider()
st.markdown(
    '<div class="small-note">YOLOv8n baseline · 25 training epochs · '
    'Prototype for demonstration; verify safety decisions independently.</div>',
    unsafe_allow_html=True,
)
