import time, os, re, cv2, numpy as np
from ultralytics import YOLO
from datetime import datetime

from db import insert_violation, init_db
from rule_engine import RuleEngine
from utils import ensure_dirs
from tools.auto_calibrate import AutoCalibrator

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False


# ---------------- Helpers ----------------
def clean_plate_text(text: str) -> str:
    if not text:
        return "NOT_DETECTED"
    t = text.upper()
    t = re.sub(r'[^A-Z0-9]', '', t)
    return t if t else "NOT_DETECTED"


def detect_red_light_violations(tracked_list, tl_state, stop_line_y, min_frames_cross=1):
    """Detect signal break when crossing stop line on RED light."""
    vios = []
    if tl_state is None or tl_state.upper() != "RED":
        return vios

    for t in tracked_list:
        hist = t.get("history", [])
        if not hist or len(hist) < 2:
            continue
        cur_cent = hist[-1]
        prev_idx = max(0, len(hist) - 1 - min_frames_cross)
        prev_cent = hist[prev_idx]
        if prev_cent[1] < stop_line_y and cur_cent[1] >= stop_line_y:
            vios.append({
                "track_id": t["track_id"],
                "type": "signal_break",
                "bbox": t.get("bbox"),
                "plate": t.get("plate"),
                "speed_kmph": t.get("velocity_kmph"),
                "conf": t.get("conf", 0.0)
            })
    return vios


def detect_wrong_way_violations(tracked_list, allowed_direction,
                                wrong_dot_thresh=-0.5, min_speed_kmph=6.0):
    """Detect vehicles moving opposite to allowed traffic direction."""
    vios = []
    if allowed_direction is None:
        return vios

    try:
        adx, ady = allowed_direction
        mag = (adx**2 + ady**2) ** 0.5
        if mag == 0:
            return vios
        adx /= mag
        ady /= mag
    except Exception:
        return vios

    for t in tracked_list:
        hist = t.get("history", [])
        if not hist or len(hist) < 2:
            continue
        x_prev, y_prev = hist[-2]
        x_cur, y_cur = hist[-1]
        vx, vy = x_cur - x_prev, y_cur - y_prev
        vmag = (vx**2 + vy**2) ** 0.5
        if vmag < 1e-6:
            continue

        dot = (vx * adx + vy * ady) / vmag
        speed = t.get("velocity_kmph") or 0.0
        if dot < wrong_dot_thresh and speed >= min_speed_kmph:
            vios.append({
                "track_id": t["track_id"],
                "type": "wrong_way",
                "bbox": t.get("bbox"),
                "plate": t.get("plate"),
                "speed_kmph": speed,
                "conf": t.get("conf", 0.0)
            })
    return vios


# ---------------- Simple Tracker ----------------
class SimpleTracker:
    def __init__(self, max_lost=30, vel_hist_len=5):
        self.next_id = 0
        self.objects = {}
        self.max_lost = max_lost
        self.vel_hist_len = vel_hist_len

    def update(self, detections, timestamp):
        new_objs, used = {}, set()
        centroids = [((d['bbox'][0] + d['bbox'][2]) // 2,
                      (d['bbox'][1] + d['bbox'][3]) // 2) for d in detections]

        for oid, data in list(self.objects.items()):
            best_i, best_d = None, 1e9
            for i, c in enumerate(centroids):
                if i in used:
                    continue
                d2 = (c[0] - data['centroid'][0]) ** 2 + (c[1] - data['centroid'][1]) ** 2
                if d2 < best_d:
                    best_i, best_d = i, d2
            if best_i is not None and best_d < 140 ** 2:
                i = best_i
                used.add(i)
                det = detections[i]
                hist = data.get('history', []) + [centroids[i]]
                velocity = None
                vel_hist = data.get('vel_hist', [])
                if data.get('last_update_time') is not None:
                    dt = max(1e-4, timestamp - data['last_update_time'])
                    dx = centroids[i][0] - data['centroid'][0]
                    dy = centroids[i][1] - data['centroid'][1]
                    step_vel = ((dx ** 2 + dy ** 2) ** 0.5) / dt
                    vel_hist = (vel_hist + [step_vel])[-self.vel_hist_len:]
                    velocity = float(np.mean(vel_hist)) if vel_hist else None
                new_objs[oid] = {
                    'centroid': centroids[i],
                    'bbox': det['bbox'],
                    'conf': det['conf'],
                    'cls': det['cls'],
                    'cls_name': det.get('cls_name'),
                    'lost': 0,
                    'history': hist,
                    'last_update_time': timestamp,
                    'velocity': velocity,
                    'vel_hist': vel_hist
                }
            else:
                data['lost'] = data.get('lost', 0) + 1
                if data['lost'] < self.max_lost:
                    new_objs[oid] = data

        for i, det in enumerate(detections):
            if i in used:
                continue
            oid = self.next_id
            self.next_id += 1
            cent = centroids[i]
            new_objs[oid] = {
                'centroid': cent,
                'bbox': det['bbox'],
                'conf': det['conf'],
                'cls': det['cls'],
                'cls_name': det.get('cls_name'),
                'lost': 0,
                'history': [cent],
                'last_update_time': timestamp,
                'velocity': None,
                'vel_hist': []
            }

        self.objects = new_objs
        return self.objects


# ---------------- TrafficDetector ----------------
class TrafficDetector:
    def __init__(self,
                 model_path="models/yolov8n.pt",
                 plate_model_path="models/plate_best.pt",   # 🔥 apna model
                 conf=0.6,
                 easyocr_gpu=False):
        ensure_dirs()
        self.model = YOLO(model_path)
        self.conf = conf
        self.db_coll = init_db()
        self.rule_engine = RuleEngine(db_conn=self.db_coll)
        self.simple_tracker = SimpleTracker(max_lost=30, vel_hist_len=5)

        # ✅ Plate model load
        if plate_model_path and os.path.exists(plate_model_path):
            self.plate_model = YOLO(plate_model_path)
            print(f"[INFO] Plate model loaded: {plate_model_path}")
        else:
            self.plate_model = None
            print("[WARN] No plate model found")

        self.ocr = easyocr.Reader(['en'], gpu=easyocr_gpu) if EASYOCR_AVAILABLE else None
        self.calibrator = AutoCalibrator(target_seconds=15, fps=25, min_vehicles=6, debug=False)
        self.pixsec_to_kmph = None
        self._assumed_car_width_m = 1.8
        self.valid_classes = {"car", "truck", "bus", "motorbike"}

    def _classify_tl_color(self, crop):
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask_red = cv2.inRange(hsv, (0,70,50), (10,255,255)) + cv2.inRange(hsv, (170,70,50), (180,255,255))
        mask_green = cv2.inRange(hsv, (40,40,40), (90,255,255))
        mask_yellow = cv2.inRange(hsv, (15,40,40), (35,255,255))

        if cv2.countNonZero(mask_red) > 50: return "RED"
        if cv2.countNonZero(mask_green) > 50: return "GREEN"
        if cv2.countNonZero(mask_yellow) > 50: return "YELLOW"
        return "UNKNOWN"

    def detect_frame(self, frame):
        timestamp = time.time()
        results = self.model(frame)
        dets = []

        for r in results:
            for box in getattr(r, "boxes", []):
                coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, coords)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0])
                name = self.model.names[cls_id]
                if conf < self.conf or name not in self.valid_classes and name != "traffic light":
                    continue
                dets.append({"cls": cls_id, "conf": conf,
                             "bbox": (x1, y1, x2, y2),
                             "cls_name": name})

        # tracking
        tracked_list = []
        tracked_obj_map = self.simple_tracker.update(dets, timestamp)
        for oid, obj in tracked_obj_map.items():
            vel_pix = obj.get('velocity')
            vel_kmph = vel_pix * self.pixsec_to_kmph if (vel_pix and self.pixsec_to_kmph) else 0.0
            tracked_list.append({
                'track_id': oid,
                **obj,
                'velocity_kmph': vel_kmph,
                'plate': "NOT_DETECTED"
            })

        # ✅ number plate detection (full frame + match with vehicles)
        if self.plate_model and self.ocr:
            plate_results = self.plate_model(frame, conf=0.25)
            plate_boxes = []
            for r in plate_results:
                for b in getattr(r, "boxes", []):
                    px1, py1, px2, py2 = map(int, b.xyxy[0].cpu().numpy())
                    plate_boxes.append((px1, py1, px2, py2))

            print(f"[DEBUG] Plates detected: {len(plate_boxes)}")  # debug

            # match plates with tracked vehicles
            for t in tracked_list:
                vx1, vy1, vx2, vy2 = t['bbox']
                for (px1, py1, px2, py2) in plate_boxes:
                    if px1 > vx1 and py1 > vy1 and px2 < vx2 and py2 < vy2:
                        plate_crop = frame[py1:py2, px1:px2]
                        text = self.ocr.readtext(plate_crop, detail=0)
                        if text:
                            t['plate'] = clean_plate_text("".join(text))

        # calibration
        cal_out = self.calibrator.update(tracked_list, frame=frame) or {}
        if cal_out.get('meters_per_pixel'):
            self.pixsec_to_kmph = cal_out['meters_per_pixel'] * 3.6
        if not self.pixsec_to_kmph:  # fallback
            widths = [o['bbox'][2]-o['bbox'][0] for o in tracked_list if o['cls_name'] in self.valid_classes]
            if widths:
                avg_w = np.mean(widths)
                self.pixsec_to_kmph = (self._assumed_car_width_m / avg_w) * 3.6

        if self.rule_engine.stop_line_y is None:
            self.rule_engine.stop_line_y = int(frame.shape[0] * 0.8)
        allowed_dir = getattr(self.rule_engine, "allowed_direction", (0, 1))
        tl_state = self.rule_engine._get_traffic_light_state(frame, tracked_list)

        if tl_state is None or tl_state == "UNKNOWN":
            for d in dets:
                if d["cls_name"] == "traffic light":
                    x1,y1,x2,y2 = d["bbox"]
                    crop = frame[y1:y2, x1:x2]
                    tl_state = self._classify_tl_color(crop)
                    break

        # violations
        signal_vios = detect_red_light_violations(tracked_list, tl_state, self.rule_engine.stop_line_y)
        wrong_vios = detect_wrong_way_violations(tracked_list, allowed_dir)
        other_vios = self.rule_engine.check(frame, tracked_list) or []
        violations = signal_vios + wrong_vios + other_vios

        # --- Annotation ---
        annotated = frame.copy()

        color = (0, 255, 0) if tl_state != "RED" else (0, 0, 255)
        cv2.line(annotated, (0, self.rule_engine.stop_line_y),
                 (frame.shape[1], self.rule_engine.stop_line_y), color, 2)

        ax, ay = allowed_dir
        center = (frame.shape[1]//2, frame.shape[0]-30)
        cv2.arrowedLine(annotated, center,
                        (center[0]+int(ax*80), center[1]-int(ay*80)),
                        (0,255,0), 3, tipLength=0.4)

        for t in tracked_list:
            x1,y1,x2,y2 = t['bbox']
            is_violation = any(v["track_id"] == t["track_id"] for v in violations)
            color = (0,0,255) if is_violation else (0,255,0)
            thickness = 3 if is_violation else 2

            cv2.rectangle(annotated, (x1,y1), (x2,y2), color, thickness)
            label = f"{t['cls_name']} {t['velocity_kmph']:.1f}km/h"
            if t.get("plate") and t["plate"] != "NOT_DETECTED":
                label += f" [{t['plate']}]"
            cv2.putText(annotated, label, (x1, max(20,y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for v in violations:
            x1,y1,x2,y2 = v['bbox']
            cv2.putText(annotated, v['type'].upper(), (x1, max(20,y1-35)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        return annotated, dets, tracked_list, violations, tl_state
