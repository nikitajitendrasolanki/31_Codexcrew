# rule_engine.py
"""
RuleEngine: traffic rule checks

- Invalid input checks (dark/static/too few detections)
- Red-light jump (uses traffic light detection or stop_line_y)
- Wrong-way (uses allowed_direction vector; auto-calibrator should set it)
- Overspeed (uses velocity_kmph if provided; otherwise converts pix/sec -> km/h if meters_per_pixel present)
- No-helmet (prefers detector-provided `obj['helmet']`, falls back to safer heuristic)
- Each violation dict includes: type, track_id, bbox, cls, conf, optional: speed_kmph, plate, reason, tl_state
"""

import time
import cv2
import numpy as np
from utils import save_snapshot

class RuleEngine:
    def __init__(self):
        # calibration / config (can be set from detector/autocalibrator)
        self.speed_limit = 50.0          # km/h default (override per-camera)
        self.allowed_direction = None    # (dx, dy) vector; calibrator should set this
        self.meters_per_pixel = None     # set by calibrator (meters / pixel)
        self.stop_line_y = None          # optional stop line pixel y-coordinate

        # input validation helpers
        self.invalid_frames_buffer = []
        self.prev_frame = None
        self.static_frame_count = 0
        self.MIN_DETECTIONS = 1

        # cooldown to avoid spamming same track
        self.last_violation_time = {}

    def _is_frame_dark(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray) < 20

    def _is_static(self, frame):
        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            return False
        prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, cur_gray)
        non_zero = np.count_nonzero(diff)
        self.prev_frame = frame.copy()
        return non_zero < 1000

    def check_invalid_input(self, frame, tracked_objects):
        # dark
        if self._is_frame_dark(frame):
            return True, "frame_too_dark"
        # static
        if self._is_static(frame):
            self.static_frame_count += 1
            if self.static_frame_count > 30:
                return True, "video_static"
        else:
            self.static_frame_count = 0
        # too few detections over short time
        if len(tracked_objects) < self.MIN_DETECTIONS:
            self.invalid_frames_buffer.append(time.time())
            self.invalid_frames_buffer = [t for t in self.invalid_frames_buffer if time.time()-t < 5.0]
            if len(self.invalid_frames_buffer) > 8:
                return True, "too_few_detections"
        else:
            self.invalid_frames_buffer = []
        return False, None

    def _get_traffic_light_state(self, frame, tracked_objects):
        # Keep original traffic-light HSV heuristic
        for obj in tracked_objects:
            name = obj.get('cls_name','').lower() if obj.get('cls_name') else ''
            if 'traffic' in name and 'light' in name:
                x1,y1,x2,y2 = obj['bbox']
                crop = frame[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
                if crop.size == 0: 
                    continue
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                lower_red1 = np.array([0, 120, 70]); upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([170,120,70]); upper_red2 = np.array([180,255,255])
                mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                red_percent = (np.count_nonzero(mask1)+np.count_nonzero(mask2)) / (crop.shape[0]*crop.shape[1] + 1e-6)
                lower_green = np.array([36, 50, 50]); upper_green = np.array([89, 255, 255])
                maskg = cv2.inRange(hsv, lower_green, upper_green)
                green_percent = np.count_nonzero(maskg)/(crop.shape[0]*crop.shape[1] + 1e-6)
                if red_percent > 0.02 and red_percent > green_percent:
                    return 'red'
                if green_percent > 0.02 and green_percent > red_percent:
                    return 'green'
        return 'unknown'

    def _helmet_heuristic(self, frame, motorbike_obj, all_tracked):
        """
        Safer fallback helmet heuristic — returns (helmet_ok:bool, reason:str)
        Important: DO NOT apply this to plain 'person' objects; only to motorbike objects
        """
        x1,y1,x2,y2 = motorbike_obj.get('bbox', (0,0,0,0))
        # find person candidate overlapping the bike bbox
        candidates = []
        for obj in all_tracked:
            if obj.get('cls_name') and obj['cls_name'].lower() == 'person':
                cx,cy = obj.get('centroid', (None,None))
                if cx is None: continue
                # require centroid strictly inside or slightly above the vehicle bbox (reduces false positives)
                if (x1 - 5) <= cx <= (x2 + 5) and (y1 - 40) <= cy <= (y2 + 20):
                    candidates.append(obj)
        if not candidates:
            return True, "no_rider_detected"  # assume no rider, so not a helmet violation
        rider = candidates[0]
        rx1,ry1,rx2,ry2 = rider.get('bbox', (0,0,0,0))
        h = ry2 - ry1
        head_y1 = ry1
        head_y2 = ry1 + max(8, int(0.25 * h))
        head = frame[max(0,head_y1):max(0,head_y2), max(0,rx1):max(0,rx2)]
        if head.size == 0:
            return False, "head_crop_empty"
        hsv = cv2.cvtColor(head, cv2.COLOR_BGR2HSV)
        # helmet-like colors heuristic
        mask_yellow = cv2.inRange(hsv, np.array([15,80,80]), np.array([40,255,255]))
        mask_white = cv2.inRange(hsv, np.array([0,0,200]), np.array([180,40,255]))
        mask_blue = cv2.inRange(hsv, np.array([90,50,50]), np.array([140,255,255]))
        mask_red1 = cv2.inRange(hsv, np.array([0,120,70]), np.array([10,255,255]))
        mask_red2 = cv2.inRange(hsv, np.array([170,120,70]), np.array([180,255,255]))
        detected_pct = (np.count_nonzero(mask_yellow) + np.count_nonzero(mask_white) +
                        np.count_nonzero(mask_blue) + np.count_nonzero(mask_red1) + np.count_nonzero(mask_red2)) / (head.shape[0]*head.shape[1] + 1e-6)
        if detected_pct > 0.04:
            return True, "helmet_likely"
        else:
            return False, "helmet_unlikely"

    def _too_old_violation(self, track_id, cooldown=5.0):
        t = self.last_violation_time.get(track_id, 0)
        if time.time() - t < cooldown:
            return True
        self.last_violation_time[track_id] = time.time()
        return False

    def _obj_speed_kmph(self, obj):
        """
        Compute object's speed in km/h using available fields:
         - prefer obj['velocity_kmph'] (detector may supply)
         - else if obj['velocity'] (pix/sec) and self.meters_per_pixel known -> convert
         - else return None
        """
        if obj is None:
            return None
        if 'velocity_kmph' in obj and obj['velocity_kmph'] is not None:
            try:
                return float(obj['velocity_kmph'])
            except Exception:
                pass
        v_pix = obj.get('velocity')
        if v_pix is None:
            return None
        if self.meters_per_pixel is not None:
            try:
                # pix/sec -> meters/sec -> km/h
                kmph = float(v_pix) * float(self.meters_per_pixel) * 3.6
                return kmph
            except Exception:
                return None
        return None

    def _is_wrong_way(self, obj):
        """
        Determine wrong-way:
         - Requires allowed_direction to be set (vector)
         - Uses object's centroid history to compute movement vector
         - If movement vector is sufficiently opposite to allowed_direction, mark wrong-way
        """
        if self.allowed_direction is None:
            return False
        history = obj.get('history', [])
        if not history or len(history) < 3:
            return False
        # take coarse vector (first -> last of history window)
        x0,y0 = history[0]
        x1,y1 = history[-1]
        dx = x1 - x0
        dy = y1 - y0
        mag = (dx*dx + dy*dy)**0.5
        if mag < 2.0:
            return False
        adx, ady = self.allowed_direction
        amag = (adx*adx + ady*ady)**0.5 + 1e-9
        cosang = (dx*adx + dy*ady) / (mag * amag)
        # cosang near -1 means opposite; threshold tuned to -0.5
        return cosang < -0.5

    def check(self, frame, tracked_objects, db_conn=None):
        """
        Main entrypoint. Returns list of violation dicts.
        Each violation dict includes at least:
          {'type','track_id','bbox','cls','conf', ...}
        """
        violations = []

        # invalid input guard
        invalid, reason = self.check_invalid_input(frame, tracked_objects)
        if invalid:
            return [{'type': 'invalid_input', 'reason': reason}]

        # traffic light state (detector or this engine can detect)
        tl_state = self._get_traffic_light_state(frame, tracked_objects)

        for obj in tracked_objects:
            tid = obj.get('track_id', -1)
            cls_name = (obj.get('cls_name') or '').lower()
            bbox = obj.get('bbox')
            conf = obj.get('conf', 0.0)
            plate = obj.get('plate') or None

            # compute speed in km/h (if possible)
            speed_kmph = self._obj_speed_kmph(obj)

            # 1) RED LIGHT JUMP: only if traffic light is red
            if tl_state == 'red':
                # prefer configured stop_line_y if provided
                if self.stop_line_y is not None:
                    cy = obj.get('centroid', (0,0))[1]
                    # crossing logic: if centroid is past stop_line (y smaller or larger depending camera)
                    # assume stop_line_y is y-coordinate below light; use simple check: centroid above line -> crossed
                    if cy is not None and cy < self.stop_line_y:
                        if not self._too_old_violation(tid):
                            v = {
                                'type': 'red_light_jump',
                                'track_id': tid,
                                'bbox': bbox,
                                'cls': cls_name,
                                'conf': conf,
                                'tl_state': tl_state,
                                'plate': plate
                            }
                            violations.append(v)
                            save_snapshot(frame, f"red_light_{cls_name}_{tid}")
                else:
                    # heuristic: if traffic light detected in frame, we consider objects that moved into the intersection area near light
                    tl_boxes = [t for t in tracked_objects if t.get('cls_name') and 'traffic' in t['cls_name'].lower()]
                    if len(tl_boxes) > 0:
                        tlx1, tly1, tlx2, tly2 = tl_boxes[0].get('bbox', (0,0,0,0))
                        stop_y = tly2 + 30
                        cy = obj.get('centroid', (0,0))[1]
                        if cy is not None and cy < stop_y:
                            if not self._too_old_violation(tid):
                                v = {
                                    'type': 'red_light_jump',
                                    'track_id': tid,
                                    'bbox': bbox,
                                    'cls': cls_name,
                                    'conf': conf,
                                    'tl_state': tl_state,
                                    'plate': plate
                                }
                                violations.append(v)
                                save_snapshot(frame, f"red_light_{cls_name}_{tid}")

            # 2) WRONG WAY: compare trajectory to allowed_direction (if set)
            try:
                if self._is_wrong_way(obj):
                    if not self._too_old_violation(tid):
                        v = {
                            'type': 'wrong_way',
                            'track_id': tid,
                            'bbox': bbox,
                            'cls': cls_name,
                            'conf': conf,
                            'plate': plate
                        }
                        violations.append(v)
                        save_snapshot(frame, f"wrong_way_{cls_name}_{tid}")
            except Exception:
                # safety: ignore errors in wrong-way calc
                pass

            # 3) OVERSPEED: use speed_kmph if available (recommended)
            try:
                if speed_kmph is not None:
                    if speed_kmph > float(self.speed_limit):
                        if not self._too_old_violation(tid):
                            v = {
                                'type': 'overspeed',
                                'track_id': tid,
                                'bbox': bbox,
                                'cls': cls_name,
                                'conf': conf,
                                'speed_kmph': round(speed_kmph, 1),
                                'plate': plate
                            }
                            violations.append(v)
                            save_snapshot(frame, f"overspeed_{cls_name}_{tid}")
                else:
                    # fallback: detector may provide pixel/sec speed and rule_engine may have overspeed_pix_per_sec
                    pix_v = obj.get('velocity')
                    if pix_v is not None and hasattr(self, 'overspeed_pix_per_sec') and self.overspeed_pix_per_sec:
                        if pix_v > self.overspeed_pix_per_sec:
                            if not self._too_old_violation(tid):
                                v = {
                                    'type': 'overspeed',
                                    'track_id': tid,
                                    'bbox': bbox,
                                    'cls': cls_name,
                                    'conf': conf,
                                    'speed_px_s': pix_v,
                                    'plate': plate
                                }
                                violations.append(v)
                                save_snapshot(frame, f"overspeed_px_{cls_name}_{tid}")
            except Exception:
                pass

            # 4) NO HELMET: only for motorbike/motorcycle class — prefer detector-provided obj['helmet']
            try:
                if 'motor' in cls_name:
                    helmet_flag = obj.get('helmet', None)
                    if helmet_flag is False:
                        # explicit no helmet detected by detector or heuristic
                        if not self._too_old_violation(tid):
                            v = {
                                'type': 'no_helmet',
                                'track_id': tid,
                                'bbox': bbox,
                                'cls': cls_name,
                                'conf': conf,
                                'reason': 'helmet_detector',
                                'plate': plate
                            }
                            violations.append(v)
                            save_snapshot(frame, f"nohelmet_{cls_name}_{tid}")
                    elif helmet_flag is None:
                        # fallback heuristic (safer): run rule_engine's helmet heuristic which may return (True/False)
                        ok, reason = self._helmet_heuristic(frame, obj, tracked_objects)
                        # only if definitely no helmet we mark violation
                        if ok is False:
                            if not self._too_old_violation(tid):
                                v = {
                                    'type': 'no_helmet',
                                    'track_id': tid,
                                    'bbox': bbox,
                                    'cls': cls_name,
                                    'conf': conf,
                                    'reason': reason,
                                    'plate': plate
                                }
                                violations.append(v)
                                save_snapshot(frame, f"nohelmet_{cls_name}_{tid}")
            except Exception:
                pass

        return violations
