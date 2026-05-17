"""
AI Fire Detection System - Camera Manager
==========================================
Thread-safe camera management using OpenCV.
Handles webcam capture, frame buffering, and graceful shutdown.
"""

import cv2
import threading
import time
import logging

logger = logging.getLogger(__name__)


class CameraManager:
    """Manages webcam capture with thread-safe frame access."""

    def __init__(self, camera_index=0, width=1280, height=720, fps=30):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps

        self._cap = None
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._frame_count = 0
        self._start_time = None

    def start(self):
        """Start the camera capture thread."""
        if self._running:
            logger.warning("Camera is already running.")
            return True

        # Try multiple backends for best Windows compatibility
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_ANY, "Default"),
            (cv2.CAP_MSMF, "MSMF"),
        ]
        opened = False
        for backend_id, backend_name in backends:
            self._cap = cv2.VideoCapture(self.camera_index, backend_id)
            if self._cap.isOpened():
                logger.info(f"Camera opened with {backend_name} backend")
                opened = True
                break
            self._cap.release()

        if not opened:
            logger.error(f"Failed to open camera at index {self.camera_index}")
            return False

        # Set camera properties
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Read actual properties
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self._cap.get(cv2.CAP_PROP_FPS))
        logger.info(f"Camera opened: {actual_w}x{actual_h} @ {actual_fps}fps")

        self._running = True
        self._start_time = time.time()
        self._frame_count = 0
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        return True

    def stop(self):
        """Stop the camera capture."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("Camera stopped.")

    def get_frame(self):
        """Get the latest frame (thread-safe)."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def is_running(self):
        """Check if the camera is running."""
        return self._running

    def get_fps(self):
        """Get the actual capture FPS."""
        if self._start_time and self._frame_count > 0:
            elapsed = time.time() - self._start_time
            if elapsed > 0:
                return round(self._frame_count / elapsed, 1)
        return 0

    def get_available_cameras(self, max_check=5):
        """Detect available camera indices."""
        available = []
        for i in range(max_check):
            cap = cv2.VideoCapture(i, cv2.CAP_ANY)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available

    def switch_camera(self, camera_index):
        """Switch to a different camera."""
        was_running = self._running
        if was_running:
            self.stop()
        self.camera_index = camera_index
        if was_running:
            return self.start()
        return True

    def _capture_loop(self):
        """Internal capture loop running in a separate thread."""
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera.")
                time.sleep(0.1)
                continue

            with self._lock:
                self._frame = frame
                self._frame_count += 1

            # Small sleep to prevent CPU overload
            time.sleep(0.01)

        logger.info("Capture loop ended.")
