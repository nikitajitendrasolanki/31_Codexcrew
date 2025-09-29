# rule_engine.py (FINAL MERGED FIXED)
import time
import cv2
import numpy as np


class RuleEngine:
    def __init__(self, db_conn=None):
        # --- Config defaults ---
        self.speed_limit = 50.0
        self.allowed_direction = (0, 1)   # default downward direction
        self.meters_per_pixel = 0.05
        self.stop_line_y = 400
        self.MIN_DETECTIONS = 1

        # --- Cooldown ---
        self.last_violation_time = {}
        self.violation_cooldown_s = 6

        # --- Train/track ---
        self.train_barrier_down = True
        self.train_zone = (250, 600)
        self.platform_zones = []   # zebra-cross ignore zone
        self.db_conn = db_conn

        # --- Traffic light ---
        self._last_tl_state = "UNKNOWN"
        self._ema_red, self._ema_green, self._ema_yellow = 0.0, 0.0, 0.0
        self._alpha = 0.35

    # ----------------------
    # Utils
    # ----------------------
    def _cooldown_ok(self, track_id, vtype):
        key = (track_id, vtype)
        last = self.last_violation_time.get(key, 0)
        if time.time() - last < self.violation_cooldown_s:
            return False
        self.last_violation_time[key] = time.time()
        return True

    def check_invalid_input(self, frame, tracked_objects):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if np.mean(gray) < 18:
                return True, "frame_too_dark"
        except Exception:
            return False, None
        if len(tracked_objects) < self.MIN_DETECTIONS:
            return True, "too_few_detections"
        return False, None

    # ----------------------
    # Traffic Light Detection
    # ----------------------
    def _get_traffic_light_state(self, frame, tracked_objects):
        tl_dets = [o for o in tracked_objects
                   if (o.get("cls_name") or "").lower() == "traffic light"
                   and o.get("conf", 0) > 0.3]
        if not tl_dets:
            return self._last_tl_state

        tl_det = max(tl_dets, key=lambda x: x.get("conf", 0))
        x1, y1, x2, y2 = map(int, tl_det.get("bbox", (0, 0, 0, 0)))
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return self._last_tl_state

        hsv = cv2.cvtColor(cv2.GaussianBlur(crop, (5, 5), 0), cv2.COLOR_BGR2HSV)
        total = crop.shape[0] * crop.shape[1] + 1e-9

        r1 = cv2.inRange(hsv, (0, 80, 70), (10, 255, 255))
        r2 = cv2.inRange(hsv, (170, 80, 70), (180, 255, 255))
        g = cv2.inRange(hsv, (36, 50, 50), (90, 255, 255))
        y = cv2.inRange(hsv, (18, 80, 80), (35, 255, 255))

        red_ratio = (np.count_nonzero(r1) + np.count_nonzero(r2)) / total
        green_ratio = np.count_nonzero(g) / total
        yellow_ratio = np.count_nonzero(y) / total

        self._ema_red = self._alpha * red_ratio + (1 - self._alpha) * self._ema_red
        self._ema_green = self._alpha * green_ratio + (1 - self._alpha) * self._ema_green
        self._ema_yellow = self._alpha * yellow_ratio + (1 - self._alpha) * self._ema_yellow

        state = "UNKNOWN"
        if self._ema_red > 0.06 and self._ema_red > self._ema_green and self._ema_red > self._ema_yellow:
            state = "RED"
        elif self._ema_green > 0.06 and self._ema_green > self._ema_red:
            state = "GREEN"
        elif self._ema_yellow > 0.04 and self._ema_yellow > self._ema_red:
            state = "YELLOW"

        self._last_tl_state = state
        return state

    def _inside_platform_zones(self, x, y):
        for (x1, y1, x2, y2) in self.platform_zones:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False

    # ----------------------
    # Helmet Detection
    # ----------------------
    def _helmet_check_for_bike(self, frame, bike_obj, all_tracked):
        x1, y1, x2, y2 = bike_obj.get("bbox", (0, 0, 0, 0))
        pad_w, pad_h = int((x2 - x1) * 0.25), int((y2 - y1) * 0.4)
        ex1, ey1, ex2, ey2 = x1 - pad_w, y1 - pad_h, x2 + pad_w, y2 + pad_h

        candidates = [o for o in all_tracked
                      if (o.get("cls_name") or "").lower() == "person"
                      and o.get("centroid", (9999, 9999))[0] is not None
                      and ex1 <= o["centroid"][0] <= ex2
                      and ey1 <= o["centroid"][1] <= ey2]
        if not candidates:
            return "unknown"

        rider = candidates[0]
        rx1, ry1, rx2, ry2 = rider.get("bbox", (0, 0, 0, 0))
        rh = ry2 - ry1
        if rh <= 0:
            return "unknown"

        head_y1, head_y2 = ry1, ry1 + max(8, int(0.25 * rh))
        hcrop = frame[max(0, head_y1):max(0, head_y2), max(0, rx1):max(0, rx2)]
        if hcrop.size == 0:
            return "unknown"

        hsv = cv2.cvtColor(hcrop, cv2.COLOR_BGR2HSV)
        masks = [
            cv2.inRange(hsv, (0, 0, 170), (180, 40, 255)),
            cv2.inRange(hsv, (15, 90, 90), (40, 255, 255)),
            cv2.inRange(hsv, (90, 50, 50), (140, 255, 255)),
        ]
        detected = sum(np.count_nonzero(m) for m in masks)
        pct = detected / (hcrop.shape[0] * hcrop.shape[1] + 1e-9)

        if pct < 0.02:
            return "no_helmet"
        elif pct > 0.04:
            return "helmet"
        return "unknown"

    # ----------------------
    # Main check()
    # ----------------------
    def check(self, frame, tracked_objects):
        violations = []

        invalid, reason = self.check_invalid_input(frame, tracked_objects)
        if invalid:
            return [{"type": "invalid_input", "reason": reason}]

        tl_state = self._get_traffic_light_state(frame, tracked_objects)

        for obj in tracked_objects:
            tid = obj.get("track_id")
            cls_name = str(obj.get("cls_name") or obj.get("cls") or "vehicle").lower()
            bbox = obj.get("bbox")
            centroid = obj.get("centroid", (0, 0))
            speed = obj.get("velocity_kmph") or obj.get("speed_kmph") or 0.0
            conf = round(float(obj.get("conf", 0.0)), 2)

            # overspeed
            if speed and speed > self.speed_limit and self._cooldown_ok(tid, "overspeed"):
                violations.append({
                    "type": "overspeed",
                    "reason": f"Speed {speed:.1f} > {self.speed_limit}",
                    "track_id": tid, "cls": cls_name, "conf": conf,
                    "speed_kmph": float(speed), "bbox": bbox, "tl_state": tl_state
                })

            # signal jump
            if self.stop_line_y and tl_state == "RED":
                cy = centroid[1]
                if cy and cy < self.stop_line_y and self._cooldown_ok(tid, "signal_jump"):
                    violations.append({
                        "type": "signal_jump",
                        "reason": f"Crossed stop line at y={cy}",
                        "track_id": tid, "cls": cls_name,
                        "bbox": bbox, "tl_state": "RED"
                    })

            # helmet
            if cls_name in ("motorbike", "motorcycle"):
                helmet_state = self._helmet_check_for_bike(frame, obj, tracked_objects)
                if helmet_state == "no_helmet" and self._cooldown_ok(tid, "no_helmet"):
                    violations.append({
                        "type": "no_helmet",
                        "reason": "Rider without helmet",
                        "track_id": tid, "cls": cls_name,
                        "bbox": bbox, "tl_state": tl_state
                    })

            # wrong way
            vel = obj.get("velocity")
            if vel and self.allowed_direction:
                try:
                    vx, vy = vel if isinstance(vel, (list, tuple)) else (float(vel), 0.0)
                    adx, ady = self.allowed_direction
                    if adx * vx + ady * vy < 0 and self._cooldown_ok(tid, "wrong_direction"):
                        violations.append({
                            "type": "wrong_direction",
                            "reason": f"Opposite direction (vel: {vx:.1f},{vy:.1f})",
                            "track_id": tid, "cls": cls_name,
                            "bbox": bbox, "tl_state": tl_state
                        })
                except Exception:
                    pass

        # train crossing
        if self.train_barrier_down and self.train_zone:
            y1, y2 = self.train_zone
            for obj in tracked_objects:
                if (obj.get("cls_name") or "").lower() == "person":
                    cx, cy = obj.get("centroid", (None, None))
                    if cx and y1 <= cy <= y2 and not self._inside_platform_zones(cx, cy):
                        if self._cooldown_ok(obj.get("track_id"), "track_crossing"):
                            violations.append({
                                "type": "track_crossing",
                                "reason": "Person in train zone while barrier down",
                                "track_id": obj.get("track_id"), "cls": "person",
                                "bbox": obj.get("bbox"), "tl_state": "TRAIN_ZONE"
                            })

        return violations  