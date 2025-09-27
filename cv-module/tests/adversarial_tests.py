import cv2, os, json, numpy as np
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

# ---------------------------
# Traffic-relevant classes
# ---------------------------
TRAFFIC_CLASSES = {
    # Vehicles
    "car", "motorbike", "bus", "truck", "bicycle",

    # Human
    "person",

    # Signals / Signs
    "traffic light", "stop sign",

    # Rail transport
    "train"
}

def classify_content(detected_classes):
    """
    Classify content type:
    - "traffic": contains valid traffic classes
    - "invalid": only irrelevant detections (animals, objects, etc.)
    - "unclear": nothing detected (likely blur/dark/low quality)
    """
    if not detected_classes:
        return "unclear"
    if any(cls in TRAFFIC_CLASSES for cls in detected_classes):
        return "traffic"
    return "invalid"

# ---------------------------
# Perturbations
# ---------------------------
def blur(img, k=7):
    return cv2.GaussianBlur(img, (k, k), 0)

def darken(img, factor=0.4):
    return (img * factor).astype(np.uint8)

def add_noise(img, sigma=25):
    noise = np.random.randn(*img.shape) * sigma
    img2 = img.astype(np.float32) + noise
    return np.clip(img2, 0, 255).astype(np.uint8)

def jpeg_compress(img, quality=20):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode(".jpg", img, encode_param)
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec

def occlude_center(img, frac=0.25):
    h, w = img.shape[:2]
    hh, ww = int(h * frac), int(w * frac)
    x1 = w // 2 - ww // 2
    y1 = h // 2 - hh // 2
    img2 = img.copy()
    img2[y1:y1+hh, x1:x1+ww] = (0, 0, 0)
    return img2

def adversarial_patch(img, size_frac=0.15):
    h, w = img.shape[:2]
    ph, pw = int(h * size_frac), int(w * size_frac)
    x = np.random.randint(0, w - pw)
    y = np.random.randint(0, h - ph)
    patch = np.random.randint(0, 255, (ph, pw, 3), dtype=np.uint8)
    img2 = img.copy()
    img2[y:y+ph, x:x+pw] = patch
    return img2

PERTURBS = {
    "blur": lambda img: blur(img, k=9),
    "dark": lambda img: darken(img, 0.3),
    "noise": lambda img: add_noise(img, sigma=30),
    "jpeg": lambda img: jpeg_compress(img, quality=15),
    "occlude": lambda img: occlude_center(img, frac=0.3),
    "patch": lambda img: adversarial_patch(img, size_frac=0.18),
}

# ---------------------------
# Detection utilities
# ---------------------------
def detect_counts(model, img):
    res = model.predict(img, imgsz=640, conf=0.25, verbose=False)[0]
    if res.boxes is None:
        return {}
    names = res.names
    counts = {}
    for b in res.boxes:
        try:
            cls = int(b.cls[0].cpu().numpy())
        except Exception:
            cls = int(b.cls[0])
        name = names.get(cls, str(cls))
        counts[name] = counts.get(name, 0) + 1
    return counts

def compare_counts(base_counts, pert_counts):
    all_keys = set(list(base_counts.keys()) + list(pert_counts.keys()))
    summary = {}
    for k in all_keys:
        summary[k] = {"base": base_counts.get(k, 0), "pert": pert_counts.get(k, 0)}
    return summary

# ---------------------------
# Main adversarial tests
# ---------------------------
def run_tests(model_path, images_dir, out_json="adv_results.json"):
    model = YOLO(model_path)
    images = list(Path(images_dir).glob("*.*"))
    results = {}

    for img_path in tqdm(images):
        img = cv2.imread(str(img_path))
        base = detect_counts(model, img)
        detected_classes = list(base.keys())
        content_type = classify_content(detected_classes)

        if content_type == "invalid":
            results[str(img_path.name)] = {
                "status": "invalid",
                "message": "⚠️ Non-traffic content detected (animals / irrelevant objects). Skipping."
            }
            continue

        if content_type == "unclear":
            results[str(img_path.name)] = {
                "status": "unclear",
                "message": "⚠️ Image too unclear (no detections). Please provide clearer traffic footage."
            }
            continue

        # else traffic = valid → run perturbation tests
        results[str(img_path.name)] = {"status": "valid", "base": base, "perturbations": {}}
        for name, fn in PERTURBS.items():
            img2 = fn(img)
            pert = detect_counts(model, img2)
            results[str(img_path.name)]["perturbations"][name] = compare_counts(base, pert)

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved:", out_json)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--images", required=True)
    p.add_argument("--out", default="adv_results.json")
    args = p.parse_args()
    run_tests(args.model, args.images, args.out)
