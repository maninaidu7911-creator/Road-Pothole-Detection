import os
from io import BytesIO
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from utils import analyze_detections, extract_gps_coordinates

st.set_page_config(
    page_title="RoadGuard AI | Pothole Detection",
    page_icon="🛣️",
    layout="wide"
)

st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; margin-bottom: 0; }
    .sub-title { color: #9ca3af; font-size: 1rem; margin-bottom: 1.5rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🛣️ RoadGuard AI: Automated Road Pothole Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Deep Learning Infrastructure Inspection powered by YOLO26 & Computer Vision</p>', unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)


MODEL_PATH = "models/best.pt"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at `{MODEL_PATH}`. Place your trained `best.pt` inside the `models/` folder.")
    st.stop()

model = load_model(MODEL_PATH)

with st.sidebar:
    st.header("⚙️ Detection Settings")
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Lower values detect more subtle potholes; higher values reduce false alarms."
    )

    st.divider()
    st.subheader("🖼️ Try Sample Road Image")
    sample_dir = "sample_images"
    sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))] if os.path.exists(sample_dir) else []
    selected_sample = st.selectbox("Select a demo image:", ["None"] + sample_files)

uploaded_file = st.file_uploader("Upload a road surface image (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

image_source = None
if uploaded_file is not None:
    image_source = Image.open(uploaded_file)
elif selected_sample != "None":
    image_source = Image.open(os.path.join(sample_dir, selected_sample))

if image_source is not None:
    pil_rgb = image_source.convert("RGB")
    img_np = np.array(pil_rgb)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    with st.spinner("Scanning road surface for potholes..."):
        results = model.predict(source=img_bgr, conf=conf_threshold, verbose=False)
        report = analyze_detections(img_bgr, results)

    m1, m2, m3 = st.columns(3)
    m1.metric("Potholes Detected", report["count"])
    m2.metric("Road Hazard Score", f"{report['hazard_score']} / 10")
    m3.metric("Repair Priority", report["overall_status"])

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Road Image")
        st.image(pil_rgb, use_container_width=True)

    with col2:
        st.subheader("AI Detection Output")
        st.image(report["annotated_image"], use_container_width=True)

    if report["count"] > 0:
        st.subheader("📋 Pothole Inspection Breakdown")
        df = pd.DataFrame(report["detections"])
        st.dataframe(df, use_container_width=True, hide_index=True)

        result_pil = Image.fromarray(report["annotated_image"])
        buf = BytesIO()
        result_pil.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download Annotated Inspection Image",
            data=buf.getvalue(),
            file_name="pothole_inspection_result.png",
            mime="image/png"
        )
    else:
        st.success("✅ No potholes detected above the selected confidence threshold.")

    gps = extract_gps_coordinates(image_source)
    if gps:
        st.subheader("📍 Detected Geotag Location")
        st.map(pd.DataFrame([gps]))
else:
    st.info("👆 Upload a road photo above or pick a sample image from the sidebar to start detection.")