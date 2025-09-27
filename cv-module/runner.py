# runner.py
import cv2
import numpy as np
from detector import TrafficDetector
import sys
from datetime import datetime

def draw_table(frame, violations, max_rows=6):
    h, w = frame.shape[:2]
    row_h = 36
    header_h = 40
    table_h = header_h + row_h * max_rows
    table = np.zeros((table_h, w, 3), dtype=np.uint8)

    header_bg = (50,50,50)
    row_bg = (30,30,30)
    text_color = (255,255,255)

    cv2.rectangle(table, (0,0), (w, header_h), header_bg, -1)
    cv2.putText(table, "Track ID", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    cv2.putText(table, "Violation", (120, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    cv2.putText(table, "Timestamp", (420, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    last = violations[-max_rows:]
    # pad if fewer
    rows = list(last)
    rows = ([(None, "-", "-")] * (max_rows - len(rows))) + rows
    for i, item in enumerate(rows[-max_rows:]):
        y1 = header_h + i*row_h
        y2 = y1 + row_h
        cv2.rectangle(table, (0,y1), (w,y2), row_bg, -1)
        tid, vio, ts = item
        cv2.putText(table, str(tid), (12, y1+24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        cv2.putText(table, str(vio), (120, y1+24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        cv2.putText(table, str(ts), (420, y1+24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    combined = np.vstack((frame, table))
    return combined

def main(video_path=0):
    td = TrafficDetector()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open:", video_path)
        return

    violations_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, dets, tracked, violations = td.detect_frame(frame)

        # pick last_violation from rule_engine if set
        lv = getattr(td.rule_engine, "last_violation", None)
        if lv:
            violations_history.append(lv)
            # to avoid re-adding same timestamped item repeatedly, clear after reading
            td.rule_engine.last_violation = None

        # draw below-frame table
        display = draw_table(annotated, violations_history, max_rows=6)

        cv2.imshow("Traffic Monitor - Press q to quit", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    path = 0
    if len(sys.argv) > 1:
        path = sys.argv[1]
    main(path)
