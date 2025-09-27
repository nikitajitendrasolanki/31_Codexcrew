import streamlit as st
import cv2, time, tempfile, os, sys
import pandas as pd

from detector import TrafficDetector
from db import insert_violation, get_all_violations, insert_report

# path fix
sys.path.append(os.path.abspath("cv-module"))
from audit import generate_report

# ---------------------------
# Traffic classes whitelist
# ---------------------------
TRAFFIC_CLASSES = {
    "car", "motorbike", "bus", "truck", "bicycle",
    "person", "traffic light", "stop sign", "train"
}

def classify_content(detected_classes):
    """Classify frame content"""
    if not detected_classes:
        return "unclear"
    if any(cls in TRAFFIC_CLASSES for cls in detected_classes):
        return "traffic"
    return "invalid"


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Traffic Violation Dashboard", layout="wide")
st.title("🚦 AI Traffic Violation Detection Dashboard")

@st.cache_resource
def get_detector():
    return TrafficDetector("models/yolov8n.pt")

detector = get_detector()

# Sidebar controls
st.sidebar.header("⚙️ Controls")
source = st.sidebar.radio("Select Source", ["Webcam", "Video File"])
video_file = None
if source == "Video File":
    video_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

start = st.sidebar.button("▶ Start Detection")
stop = st.sidebar.button("⏹ Stop Detection")

if "running" not in st.session_state:
    st.session_state.running = False
if "violations_log" not in st.session_state:
    st.session_state.violations_log = []

if start:
    st.session_state.running = True
if stop:
    st.session_state.running = False

violation_container = st.sidebar.container()
violation_container.header("⚠ Detected Violations")

filter_type = st.sidebar.selectbox(
    "Filter Violations",
    ["All", "no_helmet", "signal_break", "overspeed", "wrong_way", "triple_riding", "track_crossing"]
)

# Placeholders
stframe = st.empty()
table_placeholder = st.empty()

cap, tmp_file = None, None
try:
    if st.session_state.running:
        if source == "Webcam":
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Couldn't open webcam.")
                st.session_state.running = False
        else:
            if video_file is None:
                st.error("Please upload a video file to start detection.")
                st.session_state.running = False
            else:
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                with open(tmp_file, "wb") as f:
                    f.write(video_file.read())
                cap = cv2.VideoCapture(tmp_file)
                if not cap.isOpened():
                    st.error("Couldn't open uploaded video.")
                    st.session_state.running = False

    # Detection Loop
    while st.session_state.running and cap and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Video finished or invalid frame.")
            st.session_state.running = False
            break

        # Run detector
        annotated, dets, tracked_list, violations, tl_state = detector.detect_frame(frame)

        # Content filter
        detected_classes = {d.get("cls_name", "unknown") for d in dets}
        content_type = classify_content(detected_classes)
        if content_type == "invalid":
            st.warning("⚠️ Non-traffic content detected. Skipping frame.")
            continue
        if content_type == "unclear":
            st.info("⚠️ Frame unclear. Skipping frame.")
            continue

        # Show video
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        stframe.image(annotated_rgb, channels="RGB", use_column_width=True)

        # Process Violations
        if violations:
            for v in violations:
                vtype = v.get("type", "unknown")
                plate = v.get("plate", "NOT_DETECTED")
                tid = v.get("track_id", -1)

                # Save to MongoDB
                record = {
                    "vehicle_no": plate,
                    "violation_type": vtype,
                    "track_id": tid,
                    "cls": v.get("cls_name", "unknown"),
                    "conf": float(v.get("conf", 0.0)),
                    "speed_kmph": v.get("speed_kmph"),
                    "reason": v.get("reason", ""),
                    "tl_state": v.get("tl_state", tl_state),
                    "snapshot_path": None,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                try:
                    insert_violation(record)
                except Exception as e:
                    st.sidebar.error(f"DB Insert Error: {e}")

                # Append to session log
                st.session_state.violations_log.append({
                    "Time": record["timestamp"],
                    "Violation": vtype,
                    "Plate": plate,
                    "Track ID": tid,
                    "Confidence": round(record["conf"], 2),
                    "Speed": record["speed_kmph"],
                })

                # Show violation alert in sidebar
                violation_container.markdown(
                    f"**{vtype.upper()}** — {v.get('cls_name','?')} "
                    f"(track {tid}, plate {plate})"
                )

            violation_container.image(
                annotated_rgb,
                caption=f"Detected Violations (frame at {time.strftime('%H:%M:%S')})",
                use_container_width=True
            )

        # Traffic Light Status
        if tl_state:
            if tl_state.upper() == "RED":
                violation_container.markdown("🚦 **Traffic Light:** 🟥 RED")
            elif tl_state.upper() == "GREEN":
                violation_container.markdown("🚦 **Traffic Light:** 🟩 GREEN")
            elif tl_state.upper() == "YELLOW":
                violation_container.markdown("🚦 **Traffic Light:** 🟨 YELLOW")
            else:
                violation_container.markdown("🚦 **Traffic Light:** ⚫ UNKNOWN")

        # Table update
        if st.session_state.violations_log:
            df = pd.DataFrame(st.session_state.violations_log)
            if filter_type != "All":
                df = df[df["Violation"] == filter_type]
            table_placeholder.dataframe(df, use_container_width=True)

        time.sleep(0.1)

finally:
    if cap:
        cap.release()
    if tmp_file and os.path.exists(tmp_file):
        os.remove(tmp_file)

# ---------------------------
# 📡 Live MongoDB Viewer
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📡 MongoDB Live Data")

if st.sidebar.button("🔄 Refresh MongoDB"):
    st.cache_data.clear()

try:
    mongo_data = get_all_violations(limit=10)  # latest 10 violations
    if mongo_data:
        df_mongo = pd.DataFrame(mongo_data)
        if "_id" in df_mongo.columns:
            df_mongo = df_mongo.drop(columns=["_id"])
        st.sidebar.dataframe(df_mongo, use_container_width=True)
    else:
        st.sidebar.info("No records in MongoDB yet.")
except Exception as e:
    st.sidebar.error(f"Mongo fetch error: {e}")

# ---------------------------
# 📊 Audit Report Button
# ---------------------------
st.sidebar.markdown("---")
if st.sidebar.button("📊 Generate Audit Report"):
    df_metrics = pd.DataFrame([
        {"class": "car", "precision": 0.91, "recall": 0.87, "f1": 0.89},
        {"class": "truck", "precision": 0.85, "recall": 0.81, "f1": 0.83},
    ])

    df_violations = pd.DataFrame(st.session_state.violations_log)

    examples = [
        {"title": "Violation Example 1", "img": "snapshots/example1.png"},
        {"title": "Violation Example 2", "img": "snapshots/example2.png"},
    ]

    try:
        generate_report.generate(
            model_name="yolov8-traffic",
            metrics_df=df_metrics,
            violations_df=df_violations,
            adv_json=None,
            examples=examples,
            out_html="audit_report.html"
        )
        insert_report({
            "model": "yolov8-traffic",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": df_metrics.to_dict(orient="records"),
            "violations": df_violations.to_dict(orient="records"),
            "examples": examples
        })
        st.sidebar.success("Audit report generated & saved in MongoDB ✅")
    except Exception as e:
        st.sidebar.error(f"Report save error: {e}")
