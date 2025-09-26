# utils.py
import os, time, sqlite3, json
import cv2
from datetime import datetime

SNAPSHOT_DIR = "snapshots"
DB_PATH = "violations.db"


def ensure_dirs():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(".streamlit", exist_ok=True)  # harmless if exists


def init_db(db_path=None):
    db_path = db_path or os.path.join("cv-module", DB_PATH) if os.path.exists("cv-module") else DB_PATH
    conn = sqlite3.connect(db_path, check_same_thread=False)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS violations (
        id INTEGER PRIMARY KEY,
        ts TEXT,
        type TEXT,
        file TEXT,
        conf REAL,
        track_id INTEGER,
        extra TEXT
    )
    ''')
    conn.commit()
    return conn


def save_snapshot(frame, violation_text, vdict=None):
    """
    Save annotated snapshot for a violation and return filename.
    If vdict provided, injects 'file' key.
    """
    ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SNAPSHOT_DIR}/{violation_text}_{ts}.jpg"

    ann = frame.copy()
    cv2.putText(ann, violation_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(filename, ann)

    if vdict is not None:
        vdict['file'] = filename
    return filename


def log_violation_db(conn, vdict):
    """
    Store violation in SQLite DB.
    """
    if conn is None:
        return
    c = conn.cursor()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    vtype = vdict.get('type', 'unknown')
    file = vdict.get('file', '')
    conf = float(vdict.get('conf', 0.0))
    track_id = int(vdict.get('track_id', -1))
    extra = json.dumps(vdict)
    c.execute(
        "INSERT INTO violations(ts,type,file,conf,track_id,extra) VALUES(?,?,?,?,?,?)",
        (ts, vtype, file, conf, track_id, extra),
    )
    conn.commit()
