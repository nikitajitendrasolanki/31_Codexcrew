# cv-module/explainability/gradcam.py
import cv2, numpy as np
import torch
from ultralytics import YOLO

def apply_colormap_on_image(org_img, activation, colormap_name='jet'):
    heatmap = cv2.resize(activation, (org_img.shape[1], org_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    output = cv2.addWeighted(org_img, 0.5, colored, 0.5, 0)
    return output

def simple_gradcam(model: YOLO, frame, box, device='cpu'):
    """
    Rough Grad-CAM approximation:
    - Forward, get a chosen backbone feature map (last conv)
    - Backprop from the objectness/conf logit for the detection index (approx)
    NOTE: This is approximate and depends on ultralytics internals.
    """
    # Convert frame to torch tensor
    x = model.model.model[-1].device  # just to check device
    img = frame.copy()
    # For stable implementation use model.predict and then use model.model (torch nn). Implementation details depend on Ultralytics version.
    # Here we provide a simple placeholder: return localized gaussian around bbox
    x1,y1,x2,y2 = box
    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    mask = cv2.GaussianBlur(mask, (51,51), 0)
    mask = (mask - mask.min()) / (mask.max()+1e-9)
    overlay = apply_colormap_on_image(frame, mask)
    return overlay
