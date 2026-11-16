# 🛣️ RoadGuard AI — Automated Road Holes Detection System

An end-to-end Deep Learning computer vision web application built with **YOLO26**, **OpenCV**, and **Streamlit** to detect road potholes, classify damage severity, and compute automated repair priority scores.

🔗 **Live Demo:** (https://road-holes-detection.streamlit.app/).

---

### ✨ Key Features
* **Real-Time Holes Detection:** Custom-trained `YOLO26n` object detection model trained on annotated road surface imagery.
* **Automated Severity Grading:** Classifies each detected pothole as **Low**, **Moderate**, or **Severe** based on relative bounding-box frame coverage.
* **Road Hazard Scoring (0–10):** Aggregates pothole density and surface area coverage into an actionable repair priority (`Safe`, `Low Priority`, `Moderate Priority`, `Critical Priority`).
* **EXIF GPS Geotag Extraction:** Extracts embedded latitude/longitude metadata from geotagged photos and plots coordinates on an interactive map.
* **Inspection Reporting:** Generates downloadable annotated road inspection images and structured tabular breakdown reports.

---

### 🛠️ Tech Stack
* **Deep Learning & Vision:** Ultralytics YOLO26, OpenCV, PyTorch, Pillow
* **Web Framework & UI:** Streamlit, Pandas, NumPy
* **Deployment:** GitHub, Streamlit Community Cloud

---

### 🚀 Run Locally
```bash
git clone [https://github.com/maninaidu7911-creator/Road-Pothole-Detection.git](https://github.com/maninaidu7911-creator/Road-Pothole-Detection.git)
cd Road-Pothole-Detection
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
