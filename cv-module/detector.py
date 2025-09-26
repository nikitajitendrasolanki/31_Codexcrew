import time
import os
import cv2
import numpy as np
from ultralytics import YOLO

from rule_engine import RuleEngine
from utils import ensure_dirs, init_db
from tools.auto_calibrate import AutoCalibrator   # Auto calibration import

# optional libs
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except Exception:
    DEEPSORT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False


# ------------------- Helper utils -------------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(1, (boxA[2] - boxA[0])) * max(1, (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0])) * max(1, (boxB[3] - boxB[1]))
    return interArea / (boxAArea + boxBArea - interArea + 1e-9)


# ------------------- SimpleTracker (fallback) -------------------
class SimpleTracker:
    def __init__(self, max_lost=30):  # Increased max_lost for better persistence
        self.next_id = 0
        self.objects = {}
        self.max_lost = max_lost

    def update(self, detections, timestamp):
        new_objs = {}
        used = set()
        centroids = [((d['bbox'][0] + d['bbox'][2]) // 2,
                      (d['bbox'][1] + d['bbox'][3]) // 2) for d in detections]

        for oid, data in self.objects.items():
            best_i, best_d = None, 1e9
            for i, c in enumerate(centroids):
                if i in used:
                    continue
                d = (c[0] - data['centroid'][0]) ** 2 + (c[1] - data['centroid'][1]) ** 2
                if d < best_d:
                    best_i, best_d = i, d
            if best_i is not None and best_d < 140 ** 2:
                i = best_i
                used.add(i)
                det = detections[i]
                hist = data.get('history', []) + [centroids[i]]
                velocity = None
                if data.get('last_update_time') is not None:
                    dt = max(1e-4, timestamp - data['last_update_time'])
                    dx = centroids[i][0] - data['centroid'][0]
                    dy = centroids[i][1] - data['centroid'][1]
                    velocity = ((dx ** 2 + dy ** 2) ** 0.5) / dt
                new_objs[oid] = {
                    'centroid': centroids[i],
                    'bbox': det['bbox'],
                    'conf': det['conf'],
                    'cls': det['cls'],
                    'cls_name': det.get('cls_name', None),
                    'lost': 0,
                    'history': hist,
                    'last_update_time': timestamp,
                    'velocity': velocity
                }
            else:
                data['lost'] = data.get('lost', 0) + 1
                if data['lost'] < self.max_lost:
                    new_objs[oid] = data  # Preserves conf, cls_name, etc.

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
                'cls_name': det.get('cls_name', None),
                'lost': 0,
                'history': [cent],
                'last_update_time': timestamp,
                'velocity': None
            }

        self.objects = new_objs
        return self.objects


# ------------------- Main Detector -------------------
class TrafficDetector:
    def __init__(self, model_path="models/yolov8n.pt",
                 helmet_model_path="models/helmet_best.pt",
                 plate_model_path="models/plate_best.pt",
                 conf=0.35,
                 plate_ocr_cooldown_s=60):
        ensure_dirs()
        self.model = YOLO(model_path)
        self.conf = conf
        self.rule_engine = RuleEngine()
        self.db_conn = init_db()

        if DEEPSORT_AVAILABLE:
            try:
                self.deepsort = DeepSort(max_age=30)
                self.use_deepsort = True
            except Exception:
                self.deepsort = None
                self.use_deepsort = False
        else:
            self.deepsort = None
            self.use_deepsort = False

        self.simple_tracker = SimpleTracker(max_lost=30)  # Increased for stability

        self.helmet_model = YOLO(helmet_model_path) if os.path.exists(helmet_model_path) else None
        self.plate_model = YOLO(plate_model_path) if os.path.exists(plate_model_path) else None

        self.ocr = None
        if EASYOCR_AVAILABLE:
            try:
                self.ocr = easyocr.Reader(['en'], gpu=False)
            except Exception:
                self.ocr = None

        self.plate_cache = {}
        self.plate_cooldown = plate_ocr_cooldown_s

        self.calibrator = AutoCalibrator(target_seconds=15, fps=25, min_vehicles=6, debug=False)
        self.fps = 25.0
        self.pixsec_to_kmph = None

        self.VALID_CLASSES = {"car", "motorbike", "bus", "truck", "bicycle",
                              "traffic light", "person", "motorcycle", "van"}

        # Traffic light persistence
        self._last_tl_state = "UNKNOWN"
        self._last_tl_conf = 0.0
        self._tl_persist_frames = 0
        self._max_tl_persist = 10

    # --- OCR helper ---
    def _read_plate_easyocr(self, img):
        if self.ocr is None:
            return ""
        try:
            res = self.ocr.readtext(img, detail=1)
            if not res:
                return ""
            res_sorted = sorted(res, key=lambda x: x[2], reverse=True)
            return res_sorted[0][1].strip()
        except Exception:
            return ""

    def _try_plate_detector(self, frame, vehicle_bbox):
        x1, y1, x2, y2 = vehicle_bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return ""
        if self.plate_model is not None:
            try:
                r = self.plate_model.predict(crop, imgsz=320, conf=0.35, verbose=False)[0]
                if r.boxes is not None and len(r.boxes) > 0:
                    b = r.boxes[0]
                    bx1, by1, bx2, by2 = [int(v) for v in b.xyxy[0].cpu().numpy()]
                    plate_crop = crop[by1:by2, bx1:bx2]
                    if plate_crop.size == 0:
                        plate_crop = crop
                    return self._read_plate_easyocr(plate_crop) if self.ocr else ""
            except Exception:
                pass
        return self._read_plate_easyocr(crop) if self.ocr else ""

    # --- Main detection per frame ---
    def detect_frame(self, frame, timestamp=None):
        if timestamp is None:
            timestamp = time.time()

        res = self.model.predict(frame, imgsz=640, conf=self.conf, verbose=False)[0]

        dets = []
        if res.boxes is not None:
            for b in res.boxes:
                xy = b.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = [int(v) for v in xy]
                conf = float(b.conf[0].cpu().numpy())
                cls = int(b.cls[0].cpu().numpy())
                name = res.names[cls] if res.names and cls in res.names else str(cls)
                if name not in self.VALID_CLASSES:
                    continue
                dets.append({'bbox': (x1, y1, x2, y2), 'cls': cls, 'cls_name': name, 'conf': conf})

        # --- tracking ---
        tracked_list = []
        if self.use_deepsort and self.deepsort is not None:
            try:
                boxes_for_ds = [
                    [[d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3]], d['conf'], d['cls_name']]
                    for d in dets
                ]
                tracks = self.deepsort.update_tracks(boxes_for_ds, frame=frame)
                
                # Improved: Match tracks to detections for class/conf propagation
                track_to_det = {}
                for t in tracks:
                    if hasattr(t, "is_confirmed") and not t.is_confirmed():
                        continue
                    l, t_, r, b = t.to_ltrb()
                    bbox = (int(l), int(t_), int(r), int(b))
                    best_det = None
                    best_iou = 0
                    for d in dets:
                        iou_val = iou(bbox, d['bbox'])
                        if iou_val > best_iou:
                            best_iou = iou_val
                            best_det = d
                    track_to_det[int(t.track_id)] = best_det

                for t in tracks:
                    if hasattr(t, "is_confirmed") and not t.is_confirmed():
                        continue
                    l, t_, r, b = t.to_ltrb()
                    bbox = (int(l), int(t_), int(r), int(b))
                    tid = int(t.track_id)
                    best_det = track_to_det.get(tid)
                    cls_name = best_det['cls_name'] if best_det else 'unknown'
                    conf = best_det['conf'] if best_det else 0.0
                    cls = best_det['cls'] if best_det else None
                    tracked_list.append({
                        'track_id': tid,
                        'bbox': bbox,
                        'centroid': ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2),
                        'cls': cls,
                        'cls_name': cls_name,
                        'conf': conf,
                        'velocity': None,
                        'history': []
                    })
            except Exception as e:
                print(f"[WARN] DeepSORT failed: {e}. Falling back to SimpleTracker.")
                self.use_deepsort = False

        if not tracked_list:
            ts_input = [{'bbox': d['bbox'], 'cls': d['cls'],
                         'conf': d['conf'], 'cls_name': d.get('cls_name')} for d in dets]
            tracked_obj_map = self.simple_tracker.update(ts_input, timestamp)
            for oid, obj in tracked_obj_map.items():
                pix_v = obj.get('velocity')
                vel_kmph = None
                if pix_v is not None and getattr(self, "pixsec_to_kmph", None):
                    vel_kmph = pix_v * self.pixsec_to_kmph
                tracked_list.append({
                    'track_id': oid,
                    'bbox': obj['bbox'],
                    'centroid': obj['centroid'],
                    'cls': obj.get('cls'),
                    'cls_name': obj.get('cls_name', None),
                    'conf': obj.get('conf', 0.0),
                    'velocity': pix_v,
                    'velocity_kmph': vel_kmph,
                    'history': obj.get('history', [])
                })

        # --- Improved IoU refresh for every tracked object (looser matching for movement) ---
        for obj in tracked_list:
            if obj.get("cls_name") in [None, "unknown"] or obj.get("conf", 0.0) < 0.1:
                best_match, best_iou, best_dist = None, 0, float('inf')
                obj_centroid = obj['centroid']
                for d in dets:
                    iou_val = iou(obj['bbox'], d['bbox'])
                    # Fallback to centroid distance if IoU low (for movement)
                    d_centroid = ((d['bbox'][0] + d['bbox'][2]) // 2, (d['bbox'][1] + d['bbox'][3]) // 2)
                    cent_dist = ((d_centroid[0] - obj_centroid[0]) ** 2 + (d_centroid[1] - obj_centroid[1]) ** 2) ** 0.5
                    if iou_val > best_iou or (iou_val > 0.05 and cent_dist < best_dist):
                        best_iou = iou_val
                        best_dist = cent_dist
                        best_match = d
                if best_match and (best_iou > 0.1 or best_dist < 100):  # Looser thresholds: IoU 0.1 or dist <100px
                    obj["cls_name"] = best_match["cls_name"]
                    obj["conf"] = best_match["conf"]
                    obj["cls"] = best_match["cls"]

        # --- calibration ---
        cal_out = self.calibrator.update(tracked_list, frame=frame)
        if cal_out.get('allowed_direction') is not None:
            self.rule_engine.allowed_direction = tuple(cal_out['allowed_direction'])
        if cal_out.get('meters_per_pixel') is not None:
            self.rule_engine.meters_per_pixel = cal_out['meters_per_pixel']
            self.pixsec_to_kmph = cal_out['meters_per_pixel'] * 3.6
        else:
            self.pixsec_to_kmph = getattr(self, "pixsec_to_kmph", None)

        # --- violations ---
        violations = self.rule_engine.check(frame, tracked_list, self.db_conn) or []

        # --- plate attach ---
        for v in violations:
            tid = v.get('track_id')
            if tid is None:
                continue
            if not v.get('plate'):
                cache_entry = self.plate_cache.get(tid)
                if cache_entry and (time.time() - cache_entry[1]) < self.plate_cooldown:
                    v['plate'] = cache_entry[0]
                else:
                    bbox = None
                    for t in tracked_list:
                        if t['track_id'] == tid:
                            bbox = t['bbox']; break
                    if bbox is not None:
                        plate_text = self._try_plate_detector(frame, bbox)
                        if plate_text:
                            v['plate'] = plate_text
                            self.plate_cache[tid] = (plate_text, time.time())

        # --- annotation ---
        annotated = frame.copy()
        vio_map = {}
        for v in violations:
            tid = v.get('track_id')
            if tid is not None:
                vio_map.setdefault(tid, []).append(v)

        for obj in tracked_list:
            x1, y1, x2, y2 = obj['bbox']
            tid = obj['track_id']

            cls_name = obj.get('cls_name', 'vehicle')
            conf_val = obj.get('conf', 0.0)
            vel_kmph = obj.get('velocity_kmph')
            vel_str = f" {vel_kmph:.1f}km/h" if vel_kmph else ""

            if tid in vio_map:
                types = [vv.get('type', 'violation') for vv in vio_map[tid]]
                main_label = " | ".join([t.upper() for t in types])
                plate_txt = next((vv.get("plate") for vv in vio_map[tid] if vv.get("plate")), None)
                if plate_txt:
                    label = f"{main_label} | Plate: {plate_txt}"
                else:
                    label = main_label
                box_color = (0, 0, 255)
            else:
                label = f"ID:{tid} {cls_name} {conf_val:.2f}{vel_str}"
                box_color = (0, 255, 0)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(annotated, label, (x1, max(15, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # --- Improved traffic light detection with persistence ---
        traffic_lights = [d for d in dets if d['cls_name'].lower() == "traffic light" and d['conf'] > 0.4]  # Higher conf for stability
        if traffic_lights:
            best = max(traffic_lights, key=lambda x: x['conf'])
            self._last_tl_conf = best['conf']  # Update conf
            tl_state = self.rule_engine._get_traffic_light_state(frame, tracked_list)
            if tl_state == 'unknown' and best['conf'] > 0.5:
                tl_state = f"Detected (conf={best['conf']:.2f})"  # Force if high conf
            self._last_tl_state = tl_state
            self._tl_persist_frames = 0  # Reset counter
        else:
            self._tl_persist_frames += 1
            if self._tl_persist_frames < self._max_tl_persist and self._last_tl_conf > 0.5:
                tl_state = self._last_tl_state  # Persist longer
            else:
                tl_state = "UNKNOWN (Lost)"
                self._last_tl_state = tl_state

        txt = f"Traffic Light: {tl_state} (last conf={self._last_tl_conf:.2f})"
        cv2.rectangle(annotated, (10, annotated.shape[0] - 40),
                      (10 + 300, annotated.shape[0] - 10), (255, 255, 255), -1)
        cv2.putText(annotated, txt, (15, annotated.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        return annotated, dets, tracked_list, violations, tl_state