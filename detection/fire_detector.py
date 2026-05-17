"""
AI Fire Detection System - Main Fire Detector
===============================================
Orchestrates detection using YOLOv8 (primary) and color-based (fallback).
Manages frame processing, drawing overlays, and detection results.
"""

import cv2
import numpy as np
import time
import logging
import os

from detection.yolo_detector import YOLOFireDetector
from detection.color_detector import ColorFireDetector

logger = logging.getLogger(__name__)


class FireDetector:
    """
    Main fire detection engine.
    Uses YOLOv8 as primary detector if model is available,
    falls back to color-based detection otherwise.
    """

    def __init__(self, config):
        """
        Args:
            config: Configuration module with detection settings
        """
        self.config = config
        self._mode = "initializing"
        self._detection_count = 0
        self._last_detection_time = None
        self._fps = 0
        self._frame_times = []
        self._color_positive_streak = 0
        self._color_negative_streak = 0
        self._color_confirm_frames = 3
        self._color_clear_frames = 4
        self._color_last_detections = []

        # Initialize YOLOv8 detector
        self.yolo_detector = YOLOFireDetector(
            model_path=config.YOLO_MODEL_PATH,
            confidence=config.YOLO_CONFIDENCE_THRESHOLD,
            iou_threshold=config.YOLO_IOU_THRESHOLD,
            img_size=config.YOLO_IMG_SIZE,
        )

        # Initialize color-based fallback detector
        self.color_detector = ColorFireDetector(
            hsv_ranges=config.FIRE_HSV_RANGES,
            min_area=config.COLOR_MIN_AREA,
            sensitivity=config.COLOR_SENSITIVITY,
        )

        # Try to load YOLOv8 model
        if os.path.exists(config.YOLO_MODEL_PATH):
            if self.yolo_detector.load():
                self._mode = "yolo"
                logger.info("[FIRE DETECTOR] Using YOLOv8 model (primary)")
            else:
                self._mode = "color"
                logger.warning("[WARNING] YOLOv8 model failed to load. Using color-based detection.")
        else:
            self._mode = "color"
            logger.info("[INFO] No YOLOv8 model found. Using color-based detection (fallback).")
            logger.info(f"  To train a model, run: python train_model.py")

    def detect(self, frame):
        """
        Run fire detection on a frame.

        Args:
            frame: BGR image (numpy array)

        Returns:
            dict with keys:
                - detections: list of detection dicts
                - fire_detected: bool
                - mode: str ('yolo' or 'color')
                - fps: float
                - frame: processed frame with overlays
        """
        start_time = time.time()

        if frame is None:
            return {
                "detections": [],
                "fire_detected": False,
                "mode": self._mode,
                "fps": 0,
                "frame": None,
            }

        # Run detection based on active mode
        if self._mode == "yolo":
            detections = self.yolo_detector.detect(frame)
        else:
            raw_detections = self.color_detector.detect(frame)
            detections = self._apply_color_temporal_filter(raw_detections)

        fire_detected = len(detections) > 0

        if fire_detected:
            self._detection_count += 1
            self._last_detection_time = time.time()

        # Draw overlays on the frame
        processed_frame = self._draw_overlays(frame, detections, fire_detected)

        # Calculate FPS
        elapsed = time.time() - start_time
        self._frame_times.append(elapsed)
        if len(self._frame_times) > 30:
            self._frame_times.pop(0)
        avg_time = sum(self._frame_times) / len(self._frame_times)
        self._fps = round(1.0 / max(avg_time, 0.001), 1)

        return {
            "detections": detections,
            "fire_detected": fire_detected,
            "mode": self._mode,
            "fps": self._fps,
            "frame": processed_frame,
        }

    def _apply_color_temporal_filter(self, detections):
        """
        Stabilize color-mode detections across multiple frames.
        Reduces false positives from single-frame warm light flashes.
        """
        has_detection = len(detections) > 0
        if has_detection:
            self._color_positive_streak += 1
            self._color_negative_streak = 0
            self._color_last_detections = detections
        else:
            self._color_negative_streak += 1
            self._color_positive_streak = 0

        # Require N consecutive positive frames before raising fire_detected.
        if self._color_positive_streak >= self._color_confirm_frames:
            return self._color_last_detections

        # Keep previous detections for a short grace window to avoid rapid flicker.
        if self._color_negative_streak < self._color_clear_frames:
            return self._color_last_detections

        self._color_last_detections = []
        return []

    def _draw_overlays(self, frame, detections, fire_detected):
        """Draw detection overlays on the frame."""
        display = frame.copy()
        h, w = display.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            label = det["label"]

            # Choose color based on label
            if "smoke" in label:
                color = (200, 200, 50)  # Cyan-ish for smoke
            else:
                color = (0, 0, 255)  # Red for fire

            # Draw bounding box with thick border
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)

            # Draw filled label background
            label_text = f"{label.upper()} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(display, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(
                display, label_text,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

            # Draw corner accents
            corner_len = 20
            cv2.line(display, (x1, y1), (x1 + corner_len, y1), color, 4)
            cv2.line(display, (x1, y1), (x1, y1 + corner_len), color, 4)
            cv2.line(display, (x2, y1), (x2 - corner_len, y1), color, 4)
            cv2.line(display, (x2, y1), (x2, y1 + corner_len), color, 4)
            cv2.line(display, (x1, y2), (x1 + corner_len, y2), color, 4)
            cv2.line(display, (x1, y2), (x1, y2 - corner_len), color, 4)
            cv2.line(display, (x2, y2), (x2 - corner_len, y2), color, 4)
            cv2.line(display, (x2, y2), (x2, y2 - corner_len), color, 4)

        # Draw status bar at top
        status_h = 40
        overlay = display.copy()
        if fire_detected:
            # Pulsing red bar
            alpha = 0.6 + 0.2 * np.sin(time.time() * 5)
            cv2.rectangle(overlay, (0, 0), (w, status_h), (0, 0, 200), -1)
            cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, display)
            status_text = f"🔥 FIRE DETECTED | Mode: {self._mode.upper()} | FPS: {self._fps}"
            cv2.putText(
                display, status_text,
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )
        else:
            cv2.rectangle(overlay, (0, 0), (w, status_h), (50, 50, 50), -1)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
            status_text = f"MONITORING | Mode: {self._mode.upper()} | FPS: {self._fps}"
            cv2.putText(
                display, status_text,
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2,
            )

        # Draw border flash when fire detected
        if fire_detected:
            border = 4
            cv2.rectangle(display, (0, 0), (w - 1, h - 1), (0, 0, 255), border * 2)

        return display

    def get_mode(self):
        """Get current detection mode."""
        return self._mode

    def get_stats(self):
        """Get detection statistics."""
        return {
            "mode": self._mode,
            "total_detections": self._detection_count,
            "last_detection": self._last_detection_time,
            "fps": self._fps,
            "yolo_loaded": self.yolo_detector.is_loaded(),
            "yolo_info": self.yolo_detector.get_model_info() if self.yolo_detector.is_loaded() else None,
        }

    def set_confidence(self, confidence):
        """Update confidence threshold for both detectors."""
        self.yolo_detector.set_confidence(confidence)
        self.color_detector.set_sensitivity(confidence)

    def reload_model(self):
        """Attempt to reload the YOLOv8 model."""
        if os.path.exists(self.config.YOLO_MODEL_PATH):
            if self.yolo_detector.load():
                self._mode = "yolo"
                logger.info("YOLOv8 model reloaded successfully.")
                return True
        return False
