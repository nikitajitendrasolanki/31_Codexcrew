import streamlit as st
import cv2, time
import pandas as pd
from detector import TrafficDetector
from utils import save_snapshot, ensure_dirs, init_db, log_violation_db

# ✅ Ensure folders + DB setup
ensure_dirs()
db_conn = init_db()

st.set_page_config(page_title="Traffic Violation Dashboard", layout="wide")
st.title("🚦 AI Traffic Violation Detection Dashboard")

# Initialize detector
detector = TrafficDetector("models/yolov8n.pt")

# Sidebar Controls
st.sidebar.header("⚙️ Controls")
source = st.sidebar.radio("Select Source", ["Webcam", "Video File"])
if source == "Video File":
    video_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

start = st.sidebar.button("▶ Start Detection")

# Sidebar Violation Panel
violation_container = st.sidebar.container()
violation_container.title("⚠ Detected Violations")

# Sidebar Filter
filter_type = st.sidebar.selectbox(
    "Filter Violations", ["All", "no_helmet", "red_light_jump", "overspeed", "wrong_way"]
)

# Main Video Panel
frame_window = st.empty()
table_placeholder = st.empty()

# Store violations for session
violations_log = []

if start:
    if source == "Webcam":
        cap = cv2.VideoCapture(0)
    else:
        if video_file is None:
            st.error("Please upload a video file!")
            st.stop()
        tmp_file = f"temp_{video_file.name}"
        with open(tmp_file, "wb") as f:
            f.write(video_file.read())
        cap = cv2.VideoCapture(tmp_file)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.info("Video finished or invalid input.")
            break

        # ✅ detector returns 5 outputs
        annotated, dets, tracked_list, violations, tl_state = detector.detect_frame(frame)

        # Show live video
        frame_window.image(annotated, channels="BGR")

        # Process violations
        if violations:
            for v in violations:
                vtype = v.get("type", "unknown")
                plate = v.get("plate", "N/A")

                # Save snapshot (injects into dict also)
                file = save_snapshot(frame, vtype, v)

                # Log to DB
                log_violation_db(db_conn, v)

                # Add to local log
                violations_log.append({
                    "Time": time.strftime("%H:%M:%S"),
                    "Violation": vtype,
                    "Plate": plate,
                    "Track ID": v.get("track_id", -1),
                    "Confidence": round(v.get("conf", 0.0), 2),
                    "Snapshot": file
                })

                # Sidebar quick view
                violation_container.markdown(
                    f"**{vtype.upper()}** — {v.get('cls_name','?')} "
                    f"(track {v.get('track_id','-')}, plate {plate})"
                )
                violation_container.image(file, caption=vtype, use_container_width=True)

        # ✅ Traffic Light State
        if tl_state:
            if tl_state.lower() == "red":
                violation_container.markdown("🚦 **Traffic Light:** 🟥 RED")
            elif tl_state.lower() == "green":
                violation_container.markdown("🚦 **Traffic Light:** 🟩 GREEN")
            else:
                violation_container.markdown("🚦 **Traffic Light:** 🟨 UNKNOWN")

        # Show violations table
        if violations_log:
            df = pd.DataFrame(violations_log)
            if filter_type != "All":
                df = df[df["Violation"] == filter_type]
            table_placeholder.dataframe(df.drop(columns=["Snapshot"]), use_container_width=True)

        time.sleep(0.03)  # smooth playback (~30 FPS)

    cap.release()
