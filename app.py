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

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="RoadWatchX | Smart Vision Road Inspection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM ANIMATIONS & MODERN DARK GLASS UI ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

/* Global Background & Typography */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 15% 15%, #111936 0%, #070b19 55%, #04060e 100%);
    color: #f8fafc;
}

/* Smooth Fade-In Animation */
@keyframes fadeSlideUp {
    0% { opacity: 0; transform: translateY(22px); }
    100% { opacity: 1; transform: translateY(0); }
}

@keyframes gradientFlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 18px rgba(56, 189, 248, 0.25); }
    50% { box-shadow: 0 0 34px rgba(168, 85, 247, 0.45); }
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

/* Live Top Status Ticker */
.ticker-wrap {
    width: 100%;
    overflow: hidden;
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 999px;
    padding: 8px 0;
    margin-bottom: 1.4rem;
    backdrop-filter: blur(10px);
}
.ticker-text {
    display: inline-block;
    white-space: nowrap;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #38bdf8;
    animation: tickerScroll 24s linear infinite;
}

/* Hero Banner */
.hero-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.65), rgba(15, 23, 42, 0.85));
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 22px;
    padding: 2.2rem 2.4rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
    animation: fadeSlideUp 0.7s ease-out, pulseGlow 5s infinite ease-in-out;
    backdrop-filter: blur(14px);
}

.hero-badge {
    display: inline-block;
    background: linear-gradient(90deg, rgba(56,189,248,0.18), rgba(168,85,247,0.18));
    border: 1px solid rgba(56, 189, 248, 0.4);
    color: #7dd3fc;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 999px;
    margin-bottom: 0.9rem;
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1.15;
    margin: 0 0 0.6rem 0;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #f472b6, #38bdf8);
    background-size: 250% 250%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientFlow 6s ease infinite;
}

.hero-sub {
    color: #cbd5e1;
    font-size: 1.08rem;
    max-width: 760px;
    margin: 0;
    line-height: 1.6;
}

.floating-icon {
    font-size: 3.6rem;
    display: inline-block;
    animation: floatRoad 3s ease-in-out infinite;
}

/* Interactive Feature Cards */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1.8rem;
    animation: fadeSlideUp 0.9s ease-out;
}

.f-card {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    transition: all 0.3s cubic-bezier(0.22, 1, 0.36, 1);
}

.f-card:hover {
    transform: translateY(-6px) scale(1.01);
    border-color: #38bdf8;
    box-shadow: 0 12px 28px rgba(56, 189, 248, 0.18);
    background: rgba(30, 41, 59, 0.78);
}

.f-card h4 {
    margin: 0 0 0.35rem 0;
    color: #f1f5f9;
    font-size: 1.05rem;
}

.f-card p {
    margin: 0;
    color: #94a3b8;
    font-size: 0.88rem;
    line-height: 1.45;
}

/* Animated Radar Scanner Placeholder */
.scanner-box {
    position: relative;
    background: linear-gradient(180deg, rgba(15,23,42,0.8), rgba(30,41,59,0.5));
    border: 1.5px dashed rgba(56, 189, 248, 0.45);
    border-radius: 18px;
    padding: 2.4rem 1.5rem;
    text-align: center;
    overflow: hidden;
    margin-top: 0.8rem;
    animation: fadeSlideUp 1s ease-out;
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

/* Metric Cards */
.metric-glass {
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.75), rgba(15, 23, 42, 0.9));
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 18px;
    padding: 1.3rem 1.5rem;
    text-align: center;
    transition: transform 0.25s ease;
    animation: fadeSlideUp 0.6s ease-out;
}

.metric-glass:hover {
    transform: translateY(-4px);
    border-color: #818cf8;
}

.metric-label {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #94a3b8;
    margin-bottom: 0.3rem;
}

.metric-val {
    font-size: 2.1rem;
    font-weight: 800;
    color: #f8fafc;
}

/* File Uploader Styling */
[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 16px;
    padding: 1rem;
    transition: border-color 0.3s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: #a855f7;
}

/* Custom Footer */
.custom-footer {
    text-align: center;
    padding: 2rem 0 1rem 0;
    color: #64748b;
    font-size: 0.88rem;
}
.custom-footer span {
    color: #38bdf8;
    font-weight: 600;
}

@media (max-width: 768px) {
    .feature-grid { grid-template-columns: 1fr; }
    .hero-title { font-size: 2rem; }
}
</style>
""", unsafe_allow_html=True)

# --- LIVE TELEMETRY TICKER ---
st.markdown("""
<div class="ticker-wrap">
    <div class="ticker-text">
        ⚡ YOLO26 NEURAL ENGINE ONLINE &nbsp;•&nbsp; 🛣️ REAL-TIME POTHOLE & SURFACE ANOMALY DETECTION &nbsp;•&nbsp; 🎯 AUTOMATED SEVERITY & HAZARD SCORING (0–10) &nbsp;•&nbsp; 📍 EXIF GPS GEOTAG MAPPING READY &nbsp;•&nbsp; 🚀 BUILT FOR SAFER SMART CITIES
    </div>
</div>
""", unsafe_allow_html=True)

# --- HERO HEADER ---
st.markdown("""
<div class="hero-card">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
        <div>
            <div class="hero-badge">✨ Next-Gen Smart Infrastructure Vision</div>
            <h1 class="hero-title">RoadWatchX AI</h1>
            <p class="hero-sub">
                Making every ride smoother and safer. Upload any road snapshot to instantly detect potholes, 
                grade surface damage severity, and generate municipal repair priority insights in milliseconds.
            </p>
        </div>
        <div class="floating-icon">🛣️</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- FEATURE HIGHLIGHTS ---
st.markdown("""
<div class="feature-grid">
    <div class="f-card">
        <h4>⚡ Instant YOLO26 Vision</h4>
        <p>Pinpoints multi-scale road craters and surface cracks with high-precision bounding boxes.</p>
    </div>
    <div class="f-card">
        <h4>🚦 Smart Severity Grading</h4>
        <p>Automatically classifies every pothole into Low, Moderate, or Severe hazard tiers.</p>
    </div>
    <div class="f-card">
        <h4>📍 GPS & Audit Reports</h4>
        <p>Extracts embedded photo coordinates and exports ready-to-share inspection visuals.</p>
    </div>
</div>
""", unsafe_allow_html=True)


# --- LOAD MODEL ---
@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)


MODEL_PATH = "models/best.pt"
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at `{MODEL_PATH}`. Place your trained `best.pt` inside `models/`.")
    st.stop()

model = load_model(MODEL_PATH)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("### 🎛️ Vision Control Center")
    conf_threshold = st.slider(
        "AI Sensitivity (Confidence)",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Slide lower to catch subtle cracks; slide higher for ultra-strict detections."
    )

    celebrate_mode = st.toggle("✨ Celebration Effects", value=True)

    st.divider()
    st.markdown("### 🖼️ Quick Demo Gallery")
    sample_dir = "sample_images"
    sample_files = (
        [f for f in os.listdir(sample_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if os.path.exists(sample_dir) else []
    )
    selected_sample = st.selectbox("Test with a sample road:", ["None"] + sample_files)

    st.divider()
    st.caption("💡 **Pro Tip:** Smartphone photos with Location enabled automatically unlock the interactive GPS map!")

# --- UPLOAD SECTION ---
uploaded_file = st.file_uploader(
    "📸 Drop a road photo here to launch AI inspection (.jpg, .jpeg, .png)",
    type=["jpg", "jpeg", "png"]
)

image_source = None
if uploaded_file is not None:
    image_source = Image.open(uploaded_file)
elif selected_sample != "None":
    image_source = Image.open(os.path.join(sample_dir, selected_sample))

# --- DETECTION & RESULTS ---
if image_source is not None:
    pil_rgb = image_source.convert("RGB")
    img_np = np.array(pil_rgb)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    progress_bar = st.progress(0, text="🔍 Initializing YOLO26 optical scan...")
    for pct in range(0, 85, 20):
        time.sleep(0.04)
        progress_bar.progress(pct, text="🧠 Scanning asphalt texture & depth contours...")

    t0 = time.time()
    results = model.predict(source=img_bgr, conf=conf_threshold, verbose=False)
    report = analyze_detections(img_bgr, results)
    elapsed_ms = round((time.time() - t0) * 1000)

    progress_bar.progress(100, text=f"✅ Inspection complete in {elapsed_ms} ms!")
    time.sleep(0.25)
    progress_bar.empty()

    if celebrate_mode:
        if report["count"] == 0:
            st.balloons()
        else:
            st.toast(f"🎯 Scan Complete! Found {report['count']} pothole(s) in {elapsed_ms}ms.", icon="🛣️")

    # Determine status color
    score = report["hazard_score"]
    if score >= 7.0:
        badge_color = "#f43f5e"
    elif score >= 3.5:
        badge_color = "#f59e0b"
    elif report["count"] > 0:
        badge_color = "#38bdf8"
    else:
        badge_color = "#10b981"

    # Animated Glass Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-glass">
            <div class="metric-label">Potholes Spotted</div>
            <div class="metric-val" style="color:#38bdf8;">{report['count']}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-glass">
            <div class="metric-label">Hazard Index</div>
            <div class="metric-val" style="color:{badge_color};">{report['hazard_score']} <span style="font-size:1rem;color:#94a3b8;">/ 10</span></div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-glass">
            <div class="metric-label">Action Status</div>
            <div class="metric-val" style="font-size:1.35rem;padding-top:0.45rem;color:{badge_color};">{report['overall_status']}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-glass">
            <div class="metric-label">Inference Speed</div>
            <div class="metric-val" style="color:#a855f7;">{elapsed_ms} <span style="font-size:1rem;color:#94a3b8;">ms</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Side-by-Side Comparison
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📷 Original Road Capture")
        st.image(pil_rgb, use_container_width=True)

    with col2:
        st.markdown("#### 🎯 AI Vision Overlay")
        st.image(report["annotated_image"], use_container_width=True)

    if report["count"] > 0:
        st.markdown("### 📋 Detailed Pothole Telemetry")
        df = pd.DataFrame(report["detections"])
        st.dataframe(df, use_container_width=True, hide_index=True)

        result_pil = Image.fromarray(report["annotated_image"])
        buf = BytesIO()
        result_pil.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download High-Res Inspection Report Image",
            data=buf.getvalue(),
            file_name="roadwatchx_inspection.png",
            mime="image/png"
        )
    else:
        st.success("🎉 Smooth road ahead! Zero potholes detected above your confidence threshold.")

    gps = extract_gps_coordinates(image_source)
    if gps:
        st.markdown("### 📍 Geotagged Road Location")
        st.map(pd.DataFrame([gps]))

else:
    # Animated Radar Empty State
    st.markdown("""
    <div class="scanner-box">
        <div class="scanner-line"></div>
        <div style="font-size: 2.6rem; margin-bottom: 0.5rem;">📡</div>
        <h3 style="margin: 0 0 0.4rem 0; color: #f1f5f9;">AI Optical Scanner Ready</h3>
        <p style="margin: 0; color: #94a3b8; font-size: 0.95rem;">
            Upload any street or highway photo above — or pick a sample image from the sidebar — to watch RoadWatchX analyze the surface live!
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
<div class="custom-footer">
    Engineered with ❤️ by <span>Manikanta Dandabattula</span> &nbsp;|&nbsp; Powered by YOLO26 & Computer Vision
</div>
""", unsafe_allow_html=True)
