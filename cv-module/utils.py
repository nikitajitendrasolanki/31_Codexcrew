# utils.py
import os
import cv2
from datetime import datetime

SNAPSHOT_DIR = "snapshots"
MODEL_DIR = "models"

DB_PATH = "violations.db"  # left for legacy if someone still expects a file path (not used now)

def ensure_dirs():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

def save_snapshot(frame, vtype, vdict=None):
    """
    Save the full frame (or annotated crop if caller wants) and return filepath.
    vdict optional (used to construct filename).
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tid = vdict.get("track_id") if isinstance(vdict, dict) else "NA"
    fname = f"{SNAPSHOT_DIR}/{ts}_{vtype}_tid{tid}.jpg"
    try:
        cv2.imwrite(fname, frame)
        return fname
    except Exception as e:
        print("[SAVE SNAPSHOT ERROR]", e)
        return None

def init_db():
    """
    helper to return the mongo collection if db.py provides it.
    call this at startup: coll = init_db()
    """
    try:
        import db
        return db.init_db()
    except Exception as e:
        print("[UTILS:init_db] cannot init mongo collection:", e)
        return None

def log_violation_db(conn, record):
    """
    Backwards-compatible helper used in streamlit code.
    If conn is a pymongo Collection -> insert_one
    If conn is None -> fallback to db.insert_violation
    """
    try:
        if conn is not None and hasattr(conn, "insert_one"):
            conn.insert_one(record)
            return
    except Exception as e:
        print("[log_violation_db] insert via conn failed:", e)

    # fallback - try to use db.insert_violation
    try:
        import db
        db.insert_violation(record)
    except Exception as e:
        print("[log_violation_db] fallback db.insert_violation failed:", e)
