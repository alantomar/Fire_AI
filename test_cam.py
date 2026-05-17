import cv2
import time
import sys

def test_camera(backend_id, name):
    print(f"Testing {name}...", flush=True)
    start = time.time()
    cap = cv2.VideoCapture(0, backend_id)
    if not cap.isOpened():
        print(f"  [{name}] Failed to open (took {time.time()-start:.1f}s)", flush=True)
        return False
        
    print(f"  [{name}] Opened successfully! Testing read...", flush=True)
    ret, frame = cap.read()
    if ret and frame is not None:
        print(f"  [{name}] Read successful! Frame shape: {frame.shape}", flush=True)
        cap.release()
        return True
    else:
        print(f"  [{name}] Failed to read frame.", flush=True)
        cap.release()
        return False

print("Starting tests...")
# Try DSHOW first
if not test_camera(cv2.CAP_DSHOW, "DirectShow"):
    # Try ANY
    test_camera(cv2.CAP_ANY, "Default")
