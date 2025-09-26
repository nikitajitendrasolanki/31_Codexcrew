# tools/auto_calibrate.py
import numpy as np
import cv2
from collections import deque
import time

class AutoCalibrator:
    """
    AutoCalibrator:
    - accumulates short tracks (centroid trajectories) and infers majority direction
    - estimates meters_per_pixel using vehicle-size heuristic (average bounding-box length -> use assumed real lengths)
    - optional zebra/line based scale (simple detection)
    Usage:
      ac = AutoCalibrator(target_seconds=15, fps=20)
      ac.set_fps(20)
      each frame: ac.update(tracked_list, frame)
      if ac.ready: read ac.allowed_direction, ac.meters_per_pixel, ac.status
    """
    def __init__(self, target_seconds=15, fps=25, min_vehicles=8, debug=False):
        self.target_seconds = target_seconds
        self.fps = fps
        self.min_vehicles = min_vehicles
        self.debug = debug

        self.max_frames = int(self.target_seconds * max(1, self.fps))
        self.frame_count = 0

        # store short track histories (deque per track_id) of centroid moves
        self.track_histories = {}  # track_id -> deque of centroids (maxlen ~ fps*5)
        self.vehicle_box_lengths = []  # list of pixel lengths (long dimension of bbox)
        self.timestamps = []

        # results
        self.allowed_direction = None  # (dx,dy) normalized
        self.meters_per_pixel = None   # meters per pixel
        self.status = "not_ready"

    def set_fps(self, fps):
        self.fps = fps
        self.max_frames = int(self.target_seconds * max(1, self.fps))

    def _normalize(self, v):
        v = np.array(v, dtype=float)
        n = np.linalg.norm(v)
        if n < 1e-6:
            return np.array([0.0, 0.0])
        return (v / n).tolist()

    def update(self, tracked_list, frame=None):
        """
        tracked_list: list of dicts with keys: track_id, centroid (x,y), bbox=(x1,y1,x2,y2)
        frame: optional frame to run zebra/line detection fallback
        """
        self.frame_count += 1
        self.timestamps.append(time.time())

        # update histories
        for t in tracked_list:
            tid = t['track_id']
            c = tuple(t['centroid'])
            if tid not in self.track_histories:
                self.track_histories[tid] = deque(maxlen=max(6, int(self.fps*3)))
            self.track_histories[tid].append(c)

            # store bounding box long dimension for vehicles detection
            # heuristic: treat objects with width>30 px as vehicle candidates
            x1,y1,x2,y2 = t['bbox']
            w = x2 - x1
            h = y2 - y1
            long_dim = max(w,h)
            if long_dim > 40 and (t.get('cls_name') or '').lower() in ('car','truck','bus','motorcycle','motorbike','van','truck','bicycle'):
                self.vehicle_box_lengths.append(long_dim)

        # compute allowed_direction if enough trajectories
        # build per-track vectors: last - first
        vecs = []
        for tid, dq in list(self.track_histories.items()):
            if len(dq) >= max(3, int(self.fps*0.5)):
                start = np.array(dq[0], dtype=float)
                end = np.array(dq[-1], dtype=float)
                v = end - start
                mag = np.linalg.norm(v)
                if mag > 5:  # ignore tiny jitter
                    vecs.append(v)
        if len(vecs) >= max(3, self.min_vehicles):
            # compute mean unit vector (account for sign)
            unit = np.array([0.0, 0.0])
            for v in vecs:
                unit += (v / (np.linalg.norm(v)+1e-9))
            unit = unit / (np.linalg.norm(unit)+1e-9)
            self.allowed_direction = (float(unit[0]), float(unit[1]))
            self.status = "direction_ok"
            if self.debug:
                print(f"[AutoCalib] inferred allowed_direction={self.allowed_direction} from {len(vecs)} vectors")
        else:
            # not enough vecs yet
            self.status = "collecting_direction"

        # compute meters_per_pixel from vehicle-size heuristic
        if len(self.vehicle_box_lengths) >= max(4, self.min_vehicles):
            avg_pix_len = float(np.mean(self.vehicle_box_lengths))
            # choose assumedRealLength based on mode detection (cars vs bikes)
            # simple heuristic: if avg_pix_len > 120 -> likely car, else bike
            assumed_real_m = 4.2 if avg_pix_len > 120 else 2.0
            meters_per_pixel = assumed_real_m / max(1e-6, avg_pix_len)
            self.meters_per_pixel = float(meters_per_pixel)
            self.status = "scale_ok" if self.allowed_direction is not None else self.status
            if self.debug:
                print(f"[AutoCalib] avg_pix_len={avg_pix_len:.1f}, assumed_m={assumed_real_m}, m_per_px={self.meters_per_pixel:.6f}")

        # fallback: if zebra/road-line detection desired and frame provided, try to detect stripes (optional)
        # not implemented heavy here; keep as future improvement

        # ready if both direction and scale are available OR enough time passed
        if self.allowed_direction is not None and self.meters_per_pixel is not None:
            self.status = "ready"
        else:
            # if we collected enough frames but missing scale, still set "partial"
            if self.frame_count >= self.max_frames:
                self.status = "partial_ready"

        return {
            "status": self.status,
            "allowed_direction": self.allowed_direction,
            "meters_per_pixel": self.meters_per_pixel,
            "frame_count": self.frame_count
        }

    def reset(self):
        self.track_histories.clear()
        self.vehicle_box_lengths.clear()
        self.frame_count = 0
        self.allowed_direction = None
        self.meters_per_pixel = None
        self.status = "not_ready"
