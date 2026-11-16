import os
import time
from io import BytesIO
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from utils import analyze_detections, extract_gps_coordinates

st.set_page_config(
    page_title="RoadWatchX AI | Smart Road Inspection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 15% 15%, #111936 0%, #070b19 55%, #04060e 100%);
    color: #f8fafc;
}

@keyframes fadeSlideUp {
    0% { opacity: 0; transform: translateY(18px); }
    100% { opacity: 1; transform: translateY(0); }
}

@keyframes shakePop {
    0% { transform: scale(0.95); opacity: 0; }
    50% { transform: scale(1.01); }
    100% { transform: scale(1); opacity: 1; }
}

@keyframes gradientFlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 18px rgba(56, 189, 248, 0.22); }
    50% { box-shadow: 0 0 32px rgba(168, 85, 247, 0.40); }
}

@keyframes floatRoad {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-6px); }
}

@keyframes laserScan {
    0% { top: 8%; opacity: 0.2; }
    50% { top: 88%; opacity: 1; }
    100% { top: 8%; opacity: 0.2; }
}

@keyframes tickerScroll {
    0% { transform: translateX(100%); }
    100% { transform: translateX(-100%); }
}

.ticker-wrap {
    width: 100%;
    overflow: hidden;
    background: rgba(15, 23, 42, 0.78);
    border: 1px solid rgba(56, 189, 248, 0.28);
    border-radius: 999px;
    padding: 8px 0;
    margin-bottom: 1.3rem;
    backdrop-filter: blur(10px);
}

.ticker-text {
    display: inline-block;
    white-space: nowrap;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.84rem;
    color: #38bdf8;
    animation: tickerScroll 26s linear infinite;
}

.hero-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.68), rgba(15, 23, 42, 0.88));
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 22px;
    padding: 2rem 2.3rem;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
    animation: fadeSlideUp 0.6s ease-out, pulseGlow 5s infinite ease-in-out;
    backdrop-filter: blur(14px);
}

.hero-badge {
    display: inline-block;
    background: linear-gradient(90deg, rgba(56,189,248,0.18), rgba(168,85,247,0.18));
    border: 1px solid rgba(56, 189, 248, 0.4);
    color: #7dd3fc;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 1.1px;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 999px;
    margin-bottom: 0.8rem;
}

.hero-title {
    font-size: 2.7rem;
    font-weight: 800;
    line-height: 1.15;
    margin: 0 0 0.5rem 0;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #f472b6, #38bdf8);
    background-size: 250% 250%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientFlow 6s ease infinite;
}

.hero-sub {
    color: #cbd5e1;
    font-size: 1.05rem;
    max-width: 780px;
    margin: 0;
    line-height: 1.6;
}

.floating-icon {
    font-size: 3.4rem;
    display: inline-block;
    animation: floatRoad 3s ease-in-out infinite;
}

.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1.6rem;
    animation: fadeSlideUp 0.8s ease-out;
}

.f-card {
    background: rgba(15, 23, 42, 0.68);
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 16px;
    padding: 1.15rem 1.35rem;
    transition: all 0.28s ease;
}

.f-card:hover {
    transform: translateY(-5px);
    border-color: #38bdf8;
    box-shadow: 0 10px 25px rgba(56, 189, 248, 0.16);
}

.f-card h4 {
    margin: 0 0 0.3rem 0;
    color: #f1f5f9;
    font-size: 1.02rem;
}

.f-card p {
    margin: 0;
    color: #94a3b8;
    font-size: 0.88rem;
    line-height: 1.45;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1rem 0 1.5rem 0;
    animation: fadeSlideUp 0.6s ease-out;
}

.metric-glass {
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.78), rgba(15, 23, 42, 0.92));
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 18px;
    padding: 1.25rem 1rem;
    text-align: center;
    transition: transform 0.25s ease;
}

.metric-glass:hover {
    transform: translateY(-4px);
    border-color: #818cf8;
}

.metric-label {
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #94a3b8;
    margin-bottom: 0.35rem;
    white-space: nowrap;
}

.metric-val {
    font-size: 1.95rem;
    font-weight: 800;
    color: #f8fafc;
    line-height: 1.2;
}

.metric-hint {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 0.3rem;
}

.reject-popup {
    background: linear-gradient(135deg, rgba(159, 18, 57, 0.42), rgba(15, 23, 42, 0.95));
    border: 2px solid #f43f5e;
    border-radius: 20px;
    padding: 2rem 2.2rem;
    margin: 1.4rem 0;
    box-shadow: 0 16px 40px rgba(244, 63, 94, 0.25);
    animation: shakePop 0.45s ease-out;
}

.reject-badge {
    display: inline-block;
    background: rgba(244, 63, 94, 0.2);
    border: 1px solid #f43f5e;
    color: #fda4af;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 999px;
    margin-bottom: 0.7rem;
}

.scanner-box {
    position: relative;
    background: linear-gradient(180deg, rgba(15,23,42,0.82), rgba(30,41,59,0.52));
    border: 1.5px dashed rgba(56, 189, 248, 0.45);
    border-radius: 18px;
    padding: 2.3rem 1.5rem;
    text-align: center;
    overflow: hidden;
    margin-top: 0.8rem;
    animation: fadeSlideUp 0.9s ease-out;
}

.scanner-line {
    position: absolute;
    left: 0;
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, transparent, #38bdf8, #a855f7, transparent);
    box-shadow: 0 0 15px #38bdf8;
    animation: laserScan 3.2s ease-in-out infinite;
}

[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 16px;
    padding: 1rem;
}

.custom-footer {
    text-align: center;
    padding: 2rem 0 1rem 0;
    color: #94a3b8;
    font-size: 0.92rem;
}

.custom-footer a {
    color: #38bdf8;
    font-weight: 700;
    text-decoration: none;
    border-bottom: 1px dashed #38bdf8;
    transition: color 0.2s ease;
}

.custom-footer a:hover {
    color: #a855f7;
    border-bottom-color: #a855f7;
}

@media (max-width: 900px) {
    .feature-grid { grid-template-columns: 1fr; }
    .metric-grid { grid-template-columns: repeat(2, 1fr); }
    .hero-title { font-size: 2.1rem; }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="ticker-wrap">
    <div class="ticker-text">
        ⚡ REAL-TIME AI ROAD INSPECTION &nbsp;•&nbsp; 🛡️ SMART SCENE VALIDATION &nbsp;•&nbsp; 🛣️ AUTOMATIC POTHOLE DETECTION & SEVERITY GRADING &nbsp;•&nbsp; 🚨 INSTANT ROAD SAFETY SCORING (0–10) &nbsp;•&nbsp; 📍 GPS LOCATION MAPPING
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-card">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
        <div>
            <div class="hero-badge">✨ AI-Powered Road Safety & Infrastructure Tool</div>
            <h1 class="hero-title">RoadWatchX AI</h1>
            <p class="hero-sub">
                Turn any road photo into an instant safety report. RoadWatchX AI validates the road surface, 
                spots dangerous potholes, and generates municipal repair priority insights in seconds.
            </p>
        </div>
        <div class="floating-icon">🛣️</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="feature-grid">
    <div class="f-card">
        <h4>1️⃣ Smart Scene Guardrail</h4>
        <p>Automatically verifies that the uploaded image is a real outdoor road surface before running AI inference.</p>
    </div>
    <div class="f-card">
        <h4>2️⃣ AI Spots & Rates Damage</h4>
        <p>Our custom-trained YOLO26 vision model highlights every pothole and labels it Low, Moderate, or Severe.</p>
    </div>
    <div class="f-card">
        <h4>3️⃣ Actionable Repair Report</h4>
        <p>View the 0–10 road danger score, check GPS map coordinates, and download the annotated inspection image.</p>
    </div>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)


def validate_road_surface(image_bgr: np.ndarray) -> tuple[bool, str]:
    """
    Strict Domain Guardrail: Rejects screenshots, documents, UI captures, indoor floors,
    tiles, faces, and non-road images so only genuine outdoor road photos are inspected.
    """
    resized = cv2.resize(image_bgr, (320, 320))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 1. Detect Screenshots / Documents / UI Cards (large flat uniform pixel blocks)
    gx = np.abs(np.diff(gray.astype(np.int16), axis=1))
    gy = np.abs(np.diff(gray.astype(np.int16), axis=0))
    exact_flat_ratio = float((np.mean(gx == 0) + np.mean(gy == 0)) / 2.0)

    # Pure white / near-white paper or UI container dominance
    pure_white_ratio = float(np.mean((hsv[:, :, 1] < 18) & (hsv[:, :, 2] > 238)))

    if exact_flat_ratio > 0.36 or pure_white_ratio > 0.32:
        return False, "This appears to be a screenshot, document, or graphic rather than a real camera photo of a road."

    # Focus on lower 65% of frame where road surface lies in street/dashcam shots
    road_roi_hsv = hsv[int(320 * 0.35):, :]
    road_roi_gray = gray[int(320 * 0.35):, :]

    sat = road_roi_hsv[:, :, 1]
    val = road_roi_hsv[:, :, 2]
    hue = road_roi_hsv[:, :, 0]

    # 2. Asphalt/concrete/gravel is predominantly low-to-moderate saturation
    low_sat_ratio = float(np.mean(sat < 75))

    # 3. Micro-texture grain (Laplacian variance): smooth tiles/walls/screens have very low grain
    lap_var = float(cv2.Laplacian(road_roi_gray, cv2.CV_64F).var())

    # 4. Warm indoor beige/skin/wood dominance check
    warm_indoor_ratio = float(np.mean((hue >= 8) & (hue <= 28) & (sat >= 42) & (val >= 95)))

    # 5. Straight horizontal/vertical box/border line dominance (UI frames, rooms, furniture)
    edges = cv2.Canny(gray, 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=85, minLineLength=90, maxLineGap=8)
    hv_lines = 0
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            angle = abs(np.degrees(np.arctan2(int(y2) - int(y1), int(x2) - int(x1))))
            if angle < 3 or angle > 177 or (87 < angle < 93):
                hv_lines += 1

    if hv_lines >= 10:
        return False, "Detected UI borders, text boxes, or indoor architectural lines instead of an open outdoor roadway."

    if warm_indoor_ratio > 0.40:
        return False, "Detected indoor flooring, skin tones, or warm non-road surfaces instead of asphalt or concrete."

    if lap_var < 65.0:
        return False, "Surface texture is too smooth (looks like tiles, a wall, sky, or a digital screen rather than a road)."

    if low_sat_ratio < 0.42:
        return False, "Color profile does not match outdoor road pavement (high non-asphalt color saturation detected)."

    return True, "Valid road surface confirmed."


def enhance_road_contrast(image_bgr: np.ndarray) -> np.ndarray:
    """Enhances low-contrast wet/puddle road rims without distorting dry asphalt."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    merged = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


MODEL_PATH = "models/best.pt"
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at `{MODEL_PATH}`. Place your trained `best.pt` inside `models/`.")
    st.stop()

model = load_model(MODEL_PATH)

with st.sidebar:
    st.markdown("### 🖼️ Sample Road Photos")
    sample_dir = "sample_images"
    sample_files = (
        [f for f in os.listdir(sample_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if os.path.exists(sample_dir) else []
    )
    selected_sample = st.selectbox("Try a demo road image:", ["None"] + sample_files)
    st.divider()
    celebrate_mode = st.toggle("✨ Visual Celebrations", value=True)

uploaded_file = st.file_uploader(
    "📸 Upload a road photo to inspect (.jpg, .jpeg, .png)",
    type=["jpg", "jpeg", "png"]
)

image_source = None
if uploaded_file is not None:
    image_source = Image.open(uploaded_file)
elif selected_sample != "None":
    image_source = Image.open(os.path.join(sample_dir, selected_sample))

if image_source is not None:
    pil_rgb = image_source.convert("RGB")
    img_np = np.array(pil_rgb)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    progress_bar = st.progress(0, text="🛡️ Validating road surface domain...")
    is_road, validation_msg = validate_road_surface(img_bgr)

    if not is_road:
        progress_bar.empty()
        st.toast("⚠️ Wrong Image Uploaded! Please upload a valid road photo.", icon="🚫")

        st.markdown(f"""
        <div class="reject-popup">
            <div class="reject-badge">🚫 Invalid Image Detected</div>
            <h3 style="margin: 0 0 0.5rem 0; color: #ffffff; font-size: 1.45rem;">
                Please Upload a Correct Road Surface Image
            </h3>
            <p style="margin: 0 0 0.75rem 0; color: #fecdd3; font-size: 1rem;">
                <b>Why it was blocked:</b> {validation_msg}
            </p>
            <p style="margin: 0; color: #cbd5e1; font-size: 0.92rem; line-height: 1.5;">
                RoadWatchX AI is strictly calibrated for <b>outdoor asphalt, concrete, and highway surfaces</b>. 
                Screenshots, indoor floors, documents, and non-road photos will not generate an inspection report.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col_prev, _ = st.columns([1, 1])
        with col_prev:
            st.image(pil_rgb, caption="Rejected Upload (Not a Road Surface)", use_container_width=True)
    else:
        progress_bar.progress(45, text="🧠 Running YOLO26 road pothole inspection...")
        t0 = time.time()

        # Pass 1: High-precision inference on natural road image
        results = model.predict(
            source=img_bgr,
            conf=0.22,
            iou=0.45,
            verbose=False
        )

        # Pass 2: Adaptive wet/low-contrast road fallback if Pass 1 finds 0 potholes
        if results[0].boxes is None or len(results[0].boxes) == 0:
            enhanced_bgr = enhance_road_contrast(img_bgr)
            results = model.predict(
                source=enhanced_bgr,
                conf=0.16,
                iou=0.45,
                augment=True,
                verbose=False
            )

        report = analyze_detections(img_bgr, results)
        elapsed_sec = round(time.time() - t0, 2)

        progress_bar.progress(100, text=f"✅ Inspection complete in {elapsed_sec}s!")
        time.sleep(0.2)
        progress_bar.empty()

        if celebrate_mode:
            if report["count"] == 0:
                st.balloons()
            else:
                st.toast(f"Inspection Complete: {report['count']} pothole(s) found!", icon="🛣️")

        score = report["hazard_score"]
        if score >= 7.0:
            badge_color = "#f43f5e"
        elif score >= 3.5:
            badge_color = "#f59e0b"
        elif report["count"] > 0:
            badge_color = "#38bdf8"
        else:
            badge_color = "#10b981"

        cards_html = (
            '<div class="metric-grid">'
            f'<div class="metric-glass"><div class="metric-label">Potholes Found</div>'
            f'<div class="metric-val" style="color:#38bdf8;">{report["count"]}</div>'
            f'<div class="metric-hint">Total craters detected</div></div>'
            f'<div class="metric-glass"><div class="metric-label">Danger Score</div>'
            f'<div class="metric-val" style="color:{badge_color};">{report["hazard_score"]} <span style="font-size:1rem;color:#94a3b8;">/ 10</span></div>'
            f'<div class="metric-hint">Based on count &amp; size</div></div>'
            f'<div class="metric-glass"><div class="metric-label">Repair Urgency</div>'
            f'<div class="metric-val" style="font-size:1.35rem;padding-top:0.35rem;color:{badge_color};">{report["overall_status"]}</div>'
            f'<div class="metric-hint">Maintenance priority</div></div>'
            f'<div class="metric-glass"><div class="metric-label">Scan Time</div>'
            f'<div class="metric-val" style="color:#a855f7;">{elapsed_sec} <span style="font-size:1rem;color:#94a3b8;">sec</span></div>'
            f'<div class="metric-hint">AI processing speed</div></div>'
            '</div>'
        )
        st.markdown(cards_html, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📷 Before: Original Road Photo")
            st.image(pil_rgb, use_container_width=True)

        with col2:
            st.markdown("#### 🎯 After: AI Pothole Detection")
            st.image(report["annotated_image"], use_container_width=True)

        if report["count"] > 0:
            st.markdown("### 📋 Individual Pothole Breakdown")
            df = pd.DataFrame(report["detections"])
            st.dataframe(df, use_container_width=True, hide_index=True)

            result_pil = Image.fromarray(report["annotated_image"])
            buf = BytesIO()
            result_pil.save(buf, format="PNG")
            st.download_button(
                label="⬇️ Download Annotated Inspection Report",
                data=buf.getvalue(),
                file_name="roadwatchx_inspection.png",
                mime="image/png"
            )
        else:
            st.success("🎉 Smooth road ahead! No potholes were detected on this verified road surface.")

        gps = extract_gps_coordinates(image_source)
        if gps:
            if isinstance(gps, dict):
                lat = gps.get("lat") or gps.get("latitude") or gps.get("Latitude")
                lon = gps.get("lon") or gps.get("lng") or gps.get("longitude") or gps.get("Longitude")
            elif isinstance(gps, (list, tuple)) and len(gps) >= 2:
                lat, lon = gps[0], gps[1]
            else:
                lat, lon = None, None

            if lat is not None and lon is not None:
                st.markdown("### 📍 Exact GPS Road Location")
                map_df = pd.DataFrame({"lat": [float(lat)], "lon": [float(lon)]})
                st.map(map_df, zoom=15)
                st.caption(f"🌐 Coordinates: `{float(lat):.6f}, {float(lon):.6f}`")

else:
    st.markdown("""
    <div class="scanner-box">
        <div class="scanner-line"></div>
        <div style="font-size: 2.5rem; margin-bottom: 0.4rem;">📡</div>
        <h3 style="margin: 0 0 0.4rem 0; color: #f1f5f9;">Ready to Inspect Any Road</h3>
        <p style="margin: 0; color: #94a3b8; font-size: 0.95rem;">
            Upload an outdoor road photo above — or open the top-left arrow <b>( &gt; )</b> to test a built-in sample photo!
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="custom-footer">
    Designed by <a href="https://www.linkedin.com/in/dandabattula-manikanta" target="_blank">Manikanta</a>
</div>
""", unsafe_allow_html=True)
