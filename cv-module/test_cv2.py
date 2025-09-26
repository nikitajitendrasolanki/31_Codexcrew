import sys

try:
    import cv2
    print("✅ OpenCV imported successfully! Version:", cv2.__version__)
except Exception as e:
    print("❌ Error importing cv2:", e)
    sys.exit(1)
