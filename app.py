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
    page_title="RoadWatchX AI | Smart Road Inspection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM ANIMATIONS & POLISHED DARK GLASS UI ---
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

/* Top Status Ticker */
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

/* Hero Banner */
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

/* 3-Step How It Works Grid */
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

/* Metric Cards (Fixed Responsive Wrapping) */
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

/* Animated Radar Scanner Placeholder */
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
    color: #64748b;
    font-size: 0.88rem;
}
.custom-footer span {
    color: #38bdf8;
    font-weight: 600;
}

@media (max-width: 900px) {
    .feature-grid { grid-template-columns: 1fr; }
    .metric-grid { grid-template-columns: repeat(2, 1fr); }
    .hero-title { font-size: 2.1rem; }
}
</style>
""", unsafe_allow_html=True)

# --- LIVE TELEMETRY TICKER ---
st.markdown("""
<div class="ticker-wrap">
    <div class="ticker-text">
        ⚡ REAL-TIME AI ROAD INSPECTION &nbsp;•&nbsp; 🛣️ AUTOMATIC POTHOLE DETECTION & SEVERITY GRADING &nbsp;•&nbsp; 🚨 INSTANT ROAD SAFETY SCORING (0–10) &nbsp;•&nbsp; 📍 GPS LOCATION MAPPING &nbsp;•&nbsp; 🚀 BUILT FOR SAFER COMMUTES & SMART CITIES
    </div>
</div>
""", unsafe_allow_html=True)

# --- HERO HEADER ---
st.markdown("""
<div class="hero-card">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
        <div>
            <div class="hero-badge">✨ AI-Powered Road Safety & Infrastructure Tool</div>
            <h1 class="hero-title">RoadWatchX AI</h1>
            <p class="hero-sub">
                Turn any road photo into an instant safety report. RoadWatchX AI spots dangerous potholes, 
                rates road damage severity, and helps maintenance teams prioritize urgent repairs in seconds.
            </p>
        </div>
        <div class="floating-icon">🛣️</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 3 SIMPLE STEPS FOR VISITORS ---
st.markdown("""
<div class="feature-grid">
    <div class="f-card">
        <h4>1️⃣ Upload Any Road Photo</h4>
        <p>Drop a street or highway photo below—or pick a ready-made sample road image to test immediately.</p>
    </div>
    <div class="f-card">
        <h4>2️⃣ AI Spots & Rates Damage</h4>
        <p>Our custom-trained YOLO26 vision model highlights every pothole and labels it Low, Moderate, or Severe.</p>
    </div>
    <div class="f-card">
        <h4>3️⃣ Get Actionable Repair Report</h4>
        <p>View the 0–10 road danger score, check GPS map coordinates, and download the annotated inspection image.</p>
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
    st.markdown("### 🎛️ Inspection Settings")
    conf_threshold = st.slider(
        "Detection Sensitivity",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Lower values catch smaller surface cracks; higher values show only major potholes."
    )

    celebrate_mode = st.toggle("✨ Visual Celebrations", value=True)

    st.divider()
    st.markdown("### 🖼️ Sample Road Photos")
    sample_dir = "sample_images"
    sample_files = (
        [f for f in os.listdir(sample_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if os.path.exists(sample_dir) else []
    )
    selected_sample = st.selectbox("Try a demo road image:", ["None"] + sample_files)

# --- UPLOAD SECTION ---
uploaded_file = st.file_uploader(
    "📸 Upload a road photo to inspect (.jpg, .jpeg, .png)",
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

    progress_bar = st.progress(0, text="🔍 Scanning road surface...")
    for pct in range(0, 85, 25):
        time.sleep(0.03)
        progress_bar.progress(pct, text="🧠 Detecting potholes and measuring damage area...")

    t0 = time.time()
    results = model.predict(source=img_bgr, conf=conf_threshold, verbose=False)
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

    # Clean, Responsive 4-Card Summary Grid
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-glass">
            <div class="metric-label">Potholes Found</div>
            <div class="metric-val" style="color:#38bdf8;">{report['count']}</div>
            <div class="metric-hint">Total craters detected</div>
        </div>
        <div class="metric-glass">
            <div class="metric-label">Danger Score</div>
            <div class="metric-val" style="color:{badge_color};">{report['hazard_score']} <span style="font-size:1rem;color:#94a3b8;">/ 10</span></div>
            <div class="metric-hint">Based on count & size</div>
        </div>
        <div class="metric-glass">
            <div class="metric-label">Repair Urgency</div>
            <div class="metric-val" style="font-size:1.35rem;padding-top:0.35rem;color:{badge_color};">{report['overall_status']}</div>
            <div class="metric-hint">Maintenance priority</div>
