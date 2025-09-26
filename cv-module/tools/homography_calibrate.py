# tools/homography_calibrate.py
import cv2
import numpy as np

def compute_homography(image, real_points, pixel_points):
    H, _ = cv2.findHomography(np.array(pixel_points), np.array(real_points))
    return H

if __name__ == "__main__":
    img = cv2.imread("calib_frame.jpg")
    # click 4 points on road
    pixel_points = [(100,200), (400,200), (100,400), (400,400)]
    real_points  = [(0,0), (10,0), (0,20), (10,20)]  # meters
    H = compute_homography(img, real_points, pixel_points)
    np.savez("calibration.npz", H=H)
    print("✅ Saved calibration")
