"""
AI Fire Detection System - YOLOv8 Fire Detector
=================================================
Primary detection engine using YOLOv8 deep learning model.
Loads a trained fire detection model and performs real-time inference.
"""

import cv2
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)


class YOLOFireDetector:
    """
    Fire detection using YOLOv8 (Ultralytics).
    Requires a trained model file (best.pt) for fire/smoke detection.
    """

    def __init__(self, model_path, confidence=0.45, iou_threshold=0.50, img_size=640):
        """
        Args:
            model_path: Path to the trained YOLOv8 model (.pt file)
            confidence: Minimum confidence threshold
            iou_threshold: NMS IoU threshold
            img_size: Input image size for the model
        """
        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.model = None
        self._loaded = False
        self._device = "cpu"

    def load(self):
        """Load the YOLOv8 model."""
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found: {self.model_path}")
            return False

        try:
            from ultralytics import YOLO
            import torch

            self.model = YOLO(self.model_path)

            # Auto-detect device
            if torch.cuda.is_available():
                self._device = "cuda"
                logger.info("Using NVIDIA GPU (CUDA) for inference.")
            else:
                self._device = "cpu"
                logger.info("Using CPU for inference.")

            self._loaded = True
            logger.info(f"YOLOv8 model loaded from: {self.model_path}")
            logger.info(f"Model classes: {self.model.names}")
            return True

        except ImportError:
            logger.error("ultralytics package not installed. Run: pip install ultralytics")
            return False
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            return False

    def is_loaded(self):
        """Check if the model is loaded."""
        return self._loaded

    def detect(self, frame):
        """
        Detect fire/smoke in a frame using YOLOv8.

        Args:
            frame: BGR image (numpy array)

        Returns:
            list of detection dicts with keys:
                - bbox: (x1, y1, x2, y2)
                - confidence: float (0-1)
                - label: str
        """
        if not self._loaded or frame is None:
            return []

        try:
            # Run inference
            results = self.model.predict(
                source=frame,
                conf=self.confidence,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                device=self._device,
                verbose=False,
                half=(self._device == "cuda")
            )

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    label = self.model.names.get(cls_id, f"class_{cls_id}")

                    detections.append({
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "confidence": round(conf, 3),
                        "label": label.lower(),
                    })

            return detections

        except Exception as e:
            logger.error(f"YOLOv8 inference error: {e}")
            return []

    def set_confidence(self, confidence):
        """Update confidence threshold."""
        self.confidence = max(0.1, min(1.0, confidence))

    def get_device(self):
        """Get the current inference device."""
        return self._device

    def get_model_info(self):
        """Get model information."""
        if not self._loaded:
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "path": self.model_path,
            "device": self._device,
            "confidence": self.confidence,
            "classes": self.model.names if self.model else {},
            "img_size": self.img_size,
        }
