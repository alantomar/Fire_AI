"""
AI Fire Detection System - HSV Color-Based Fire Detector
=========================================================
Fallback fire detection using HSV color space analysis,
contour detection, and motion/flicker analysis.
Works without any trained model — provides immediate functionality.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ColorFireDetector:
    """
    Detects fire using HSV color space analysis and contour detection.
    This serves as a fallback when no YOLOv8 model is available.
    """

    def __init__(self, hsv_ranges=None, min_area=5000, sensitivity=0.5):
        """
        Args:
            hsv_ranges: List of dicts with 'lower' and 'upper' HSV tuples
            min_area: Minimum contour area to be considered fire
            sensitivity: Detection sensitivity (0.0 to 1.0)
        """
        self.hsv_ranges = hsv_ranges or [
            {"lower": (0, 80, 150), "upper": (20, 255, 255)},
            {"lower": (160, 80, 150), "upper": (180, 255, 255)},
            {"lower": (15, 40, 200), "upper": (40, 255, 255)},
        ]
        self.base_min_area = min_area
        self.sensitivity = sensitivity
        self._prev_gray = None
        self._motion_weight = 0.3
        self._max_bbox_fill_ratio = 0.80
        self._min_fire_ratio = 0.20

    @property
    def min_area(self):
        """Dynamic min_area based on sensitivity."""
        # Higher sensitivity = lower area threshold
        factor = 1.0 - (self.sensitivity * 0.8)
        return int(self.base_min_area * factor)

    def detect(self, frame):
        """
        Detect fire in a frame using color analysis.

        Args:
            frame: BGR image (numpy array)

        Returns:
            list of detection dicts with keys:
                - bbox: (x1, y1, x2, y2)
                - confidence: float (0-1)
                - label: str
        """
        if frame is None:
            return []

        detections = []
        h, w = frame.shape[:2]

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create combined mask from all HSV ranges
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for hrange in self.hsv_ranges:
            lower = np.array(hrange["lower"])
            upper = np.array(hrange["upper"])
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Apply morphological operations to clean up noise and merge hollow flame cores
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, close_kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, open_kernel)

        # Motion analysis reduces static false positives from wallpaper/skin-like colors.
        motion_mask = self._get_motion_mask(frame)

        # Apply Gaussian blur to smooth the mask
        combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)
        _, combined_mask = cv2.threshold(combined_mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue

            # Get bounding box
            x, y, bw, bh = cv2.boundingRect(contour)
            bbox_area = max(bw * bh, 1)
            fill_ratio = area / bbox_area
            if fill_ratio > self._max_bbox_fill_ratio:
                # Large smooth blobs are often warm lights/screens, not flame contours.
                continue
            x2 = min(x + bw, w)
            y2 = min(y + bh, h)

            # Calculate aspect ratio — fire tends to be taller than wide
            aspect_ratio = bh / max(bw, 1)

            motion_ratio = 0.0
            if motion_mask is not None and x2 > x and y2 > y:
                roi_motion = motion_mask[y:y2, x:x2]
                if roi_motion.size > 0:
                    motion_ratio = float(np.count_nonzero(roi_motion)) / float(roi_motion.size)

            # Calculate confidence based on multiple factors
            confidence = self._calculate_confidence(
                frame, hsv, contour, area, aspect_ratio, w, h, x, y, bw, bh, motion_ratio
            )

            # Apply sensitivity-based threshold
            threshold = 0.3 + (1.0 - self.sensitivity) * 0.3
            if confidence < threshold:
                continue

            detections.append({
                "bbox": (x, y, x + bw, y + bh),
                "confidence": round(confidence, 3),
                "label": "fire",
            })

        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d["confidence"], reverse=True)

        return detections

    def _calculate_confidence(self, frame, hsv, contour, area, aspect_ratio, img_w, img_h, x, y, bw, bh, motion_ratio):
        """Calculate detection confidence based on multiple fire characteristics."""
        scores = []

        # 1. Area score — larger fire regions = more confident
        max_area = img_w * img_h * 0.3
        area_score = min(area / max_area, 1.0) * 0.5 + 0.3
        scores.append(area_score)

        # 2. Color intensity score — brighter = more fire-like
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_val = cv2.mean(frame, mask=mask)
        brightness = (mean_val[0] + mean_val[1] + mean_val[2]) / (3 * 255)
        scores.append(min(brightness * 1.5, 1.0))

        # 3. Saturation score — fire has high saturation
        mean_hsv = cv2.mean(hsv, mask=mask)
        sat_score = mean_hsv[1] / 255
        # If the area is very bright (like a fire core), it may have low saturation. Don't penalize.
        if brightness > 0.8:
            sat_score = max(sat_score, 0.8)
        scores.append(sat_score)

        # 4. Red-channel dominance & progression — fire is red/orange dominant (R > G > B)
        b, g, r = mean_val[0], mean_val[1], mean_val[2]
        
        # Fire usually follows R >= G >= B (added tolerance for white/bright cores)
        color_prog_score = 1.0 if (r + 10 >= g and g + 10 >= b) else (0.7 if r + 10 >= g else 0.3)
        
        red_ratio = r / max(r + g + b, 1)
        red_score = min(red_ratio * 2, 1.0)
        
        # Combine classic RGB heuristic with red ratio
        scores.append((red_score + color_prog_score) / 2)

        # 5. Contour solidity — fire has irregular shapes (lower solidity)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            # Fire typically has solidity 0.3-0.8
            if 0.2 < solidity < 0.9:
                scores.append(0.7)
            else:
                scores.append(0.3)
        else:
            scores.append(0.3)

        # 6. Color consistency score inside detected region.
        # Requires a meaningful fraction of pixels in fire HSV ranges to reduce false positives.
        x2 = min(x + bw, img_w)
        y2 = min(y + bh, img_h)
        roi_hsv = hsv[y:y2, x:x2]
        if roi_hsv.size > 0:
            roi_mask = np.zeros((roi_hsv.shape[0], roi_hsv.shape[1]), dtype=np.uint8)
            for hrange in self.hsv_ranges:
                lower = np.array(hrange["lower"])
                upper = np.array(hrange["upper"])
                roi_mask = cv2.bitwise_or(roi_mask, cv2.inRange(roi_hsv, lower, upper))
            fire_ratio = float(np.count_nonzero(roi_mask)) / float(roi_mask.size)
            if fire_ratio < self._min_fire_ratio:
                return 0.0
            scores.append(min(fire_ratio * 1.5, 1.0))
        else:
            scores.append(0.0)

        # 7. Motion/flicker score.
        # Keep this as a soft score so very steady flames can still pass when color evidence is strong.
        motion_score = min(motion_ratio * 6.0, 1.0)
        scores.append(0.35 + (0.65 * motion_score))

        return sum(scores) / len(scores)

    def _get_motion_mask(self, frame):
        """Detect motion between frames (fire flickers)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return None

        # Frame difference
        diff = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray

        _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.dilate(motion_mask, None, iterations=2)

        return motion_mask

    def set_sensitivity(self, sensitivity):
        """Update detection sensitivity (0.0 to 1.0)."""
        self.sensitivity = max(0.0, min(1.0, sensitivity))
