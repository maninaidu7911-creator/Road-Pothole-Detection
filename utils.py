import cv2
import numpy as np
from PIL import Image, ExifTags


def analyze_detections(image_bgr: np.ndarray, results) -> dict:
    annotated = image_bgr.copy()
    img_h, img_w = annotated.shape[:2]
    total_img_area = img_h * img_w

    detections = []
    total_pothole_area = 0.0

    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])

            box_area = max(0, (x2 - x1) * (y2 - y1))
            area_ratio = (box_area / total_img_area) * 100
            total_pothole_area += box_area

            if area_ratio >= 4.0:
                severity = "Severe"
                color = (0, 0, 255)
            elif area_ratio >= 1.2:
                severity = "Moderate"
                color = (0, 165, 255)
            else:
                severity = "Low"
                color = (0, 215, 255)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            label = f"#{idx+1} {severity} ({conf:.0%})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 10)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                annotated, label, (x1 + 3, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

            detections.append({
                "Pothole ID": f"#{idx+1}",
                "Severity": severity,
                "Confidence": f"{conf * 100:.1f}%",
                "Frame Coverage": f"{area_ratio:.2f}%",
                "Coordinates (x1,y1,x2,y2)": f"({x1}, {y1}, {x2}, {y2})"
            })

    count = len(detections)
    coverage_pct = (total_pothole_area / total_img_area) * 100

    if count == 0:
        hazard_score = 0.0
        overall_status = "Safe"
    else:
        raw_score = (count * 1.5) + (coverage_pct * 0.6)
        hazard_score = round(min(10.0, raw_score), 1)
        if hazard_score >= 7.0:
            overall_status = "Critical Priority"
        elif hazard_score >= 3.5:
            overall_status = "Moderate Priority"
        else:
            overall_status = "Low Priority"

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return {
        "annotated_image": annotated_rgb,
        "count": count,
        "hazard_score": hazard_score,
        "overall_status": overall_status,
        "detections": detections
    }


def extract_gps_coordinates(pil_image: Image.Image):
    try:
        exif_data = pil_image._getexif()
        if not exif_data:
            return None

        gps_info = {}
        for tag_id, value in exif_data.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            if tag == "GPSInfo":
                for key in value:
                    sub_tag = ExifTags.GPSTAGS.get(key, key)
                    gps_info[sub_tag] = value[key]

        if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
            def to_degrees(val):
                d, m, s = val
                return float(d) + (float(m) / 60.0) + (float(s) / 3600.0)

            lat = to_degrees(gps_info["GPSLatitude"])
            if gps_info.get("GPSLatitudeRef") != "N":
                lat = -lat

            lon = to_degrees(gps_info["GPSLongitude"])
            if gps_info.get("GPSLongitudeRef") != "E":
                lon = -lon

            return {"lat": lat, "lon": lon}
    except Exception:
        return None
    return None