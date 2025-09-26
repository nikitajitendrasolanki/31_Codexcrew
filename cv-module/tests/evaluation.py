import json, os
from ultralytics import YOLO
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse

# ---------------------------
# Traffic-relevant classes
# ---------------------------
TRAFFIC_CLASSES = {"car", "motorbike", "bus", "truck", "bicycle", "traffic light", "person"}

def classify_content(detected_classes):
    """
    Classify content type:
    - "traffic": contains valid traffic classes
    - "invalid": no traffic classes, but some other detections
    - "unclear": nothing detected (likely blur/dark/low quality)
    """
    if not detected_classes:
        return "unclear"
    if any(cls in TRAFFIC_CLASSES for cls in detected_classes):
        return "traffic"
    return "invalid"

def load_groundtruth(json_path):
    """
    Groundtruth JSON format:
    {
      "test1.jpg": [ {"bbox":[x1,y1,x2,y2], "class":"car"}, ... ],
      "test2.jpg": [ {"bbox":[x1,y1,x2,y2], "class":"truck"}, ... ]
    }
    """
    return json.load(open(json_path))

def evaluate_simple(model_path, images_dir, gt_json):
    model = YOLO(model_path)
    gts = load_groundtruth(gt_json)
    images = list(Path(images_dir).glob("*.*"))
    per_image = []

    skipped = []  # <-- store skipped images
    for img_path in tqdm(images, desc="Evaluating"):
        img = str(img_path)
        res = model.predict(img, imgsz=640, conf=0.25, verbose=False)[0]

        detected = []
        if res.boxes is not None:
            for b in res.boxes:
                xy = b.xyxy[0].cpu().numpy()
                cls = int(b.cls[0].cpu().numpy())
                detected.append({'bbox':[float(x) for x in xy], 'cls': res.names[cls]})

        # classify content type
        detected_classes = [d['cls'] for d in detected]
        content_type = classify_content(detected_classes)

        if content_type != "traffic":
            skipped.append({
                "img": img_path.name,
                "status": content_type,
                "message": "⚠️ Unclear image" if content_type=="unclear" else "⚠️ Non-traffic content"
            })
            continue  # skip metrics for this image

        gt = gts.get(img_path.name, [])
        per_image.append({'img':img_path.name, 'detected': detected, 'gt': gt})

    # ---- Compute metrics ----
    metrics = {}
    for rec in per_image:
        gt_classes = [g['class'] for g in rec['gt']]
        det_classes = [d['cls'] for d in rec['detected']]

        for c in set(gt_classes + det_classes):
            metrics.setdefault(c, {'gt':0,'det':0,'tp':0})

        for g in rec['gt']:
            metrics[g['class']]['gt'] += 1

        for d in rec['detected']:
            metrics[d['cls']]['det'] += 1
            if d['cls'] in gt_classes:
                metrics[d['cls']]['tp'] += 1

    rows = []
    for k,v in metrics.items():
        tp = min(v['tp'], v['gt'])  # safe tp ≤ gt
        prec = tp/v['det'] if v['det']>0 else 0.0
        rec = tp/v['gt'] if v['gt']>0 else 0.0
        f1 = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0

        rows.append({
            'class': k,
            'precision': round(prec,3),
            'recall': round(rec,3),
            'f1_score': round(f1,3),
            'gt': v['gt'],
            'det': v['det'],
            'tp': tp
        })

    df = pd.DataFrame(rows)

    return df, skipped  # <-- also return skipped list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to YOLO model")
    parser.add_argument("--images", required=True, help="Directory with validation images")
    parser.add_argument("--gt", required=True, help="Ground truth JSON file")
    parser.add_argument("--out", default="metrics.csv", help="Output CSV file")
    args = parser.parse_args()

    df, skipped = evaluate_simple(args.model, args.images, args.gt)
    print(df)  # show results in terminal
    df.to_csv(args.out, index=False)
    print(f"[OK] Metrics saved to {args.out}")

    if skipped:
        print("\n⚠️ Skipped images:")
        for s in skipped:
            print(f"- {s['img']}: {s['message']}")
