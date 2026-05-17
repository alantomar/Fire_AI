"""
AI Fire Detection System - Alert Manager
==========================================
Manages fire detection alerts: sound alarms, desktop notifications,
screenshot capture, and cooldown logic to prevent alert spam.
"""

import os
import cv2
import time
import json
import threading
import logging
import winsound
from datetime import datetime

from alerts.hardware import ArduinoController

logger = logging.getLogger(__name__)

class AlertManager:
    """
    Manages alerts when fire is detected.
    Supports sound alarms, screenshot capture, and event logging.
    """

    def __init__(self, config, arduino_port=None):
        self.config = config
        self.sound_enabled = config.ALERT_SOUND_ENABLED
        self.cooldown = config.ALERT_COOLDOWN_SECONDS
        self.snapshot_enabled = config.ALERT_SNAPSHOT_ENABLED

        self.arduino = ArduinoController(arduino_port, config.ARDUINO_BAUDRATE)

        self._last_alert_time = 0
        self._last_event_time = 0
        self._alert_active = False
        self._alert_count = 0
        self._lock = threading.Lock()
        self._sound_thread = None
        self._last_hardware_ping_time = 0

        # Ensure log file exists
        if not os.path.exists(config.LOG_FILE):
            with open(config.LOG_FILE, "w") as f:
                json.dump([], f)

    def trigger_alert(self, frame, detections):
        """
        Trigger/maintain fire alert while fire is present.

        Args:
            frame: Current camera frame (for snapshot)
            detections: List of detection dicts

        Returns:
            bool: True if a new alert event was logged, False otherwise
        """
        current_time = time.time()

        with self._lock:
            # Keep alarm active as long as fire frames keep arriving.
            self._alert_active = True
            self._last_alert_time = current_time

            # Re-send alarm command periodically while fire persists.
            # This keeps hardware latched ON even if one serial write is missed.
            if (
                self.arduino
                and current_time - self._last_hardware_ping_time >= 0.5
            ):
                self.arduino.trigger_alarm()
                self._last_hardware_ping_time = current_time

            # Cooldown now controls log/snapshot/sound frequency only.
            should_log_event = (
                (current_time - self._last_event_time) >= self.cooldown
                or self._alert_count == 0
            )

        if not should_log_event:
            return False

        with self._lock:
            self._alert_count += 1
            self._last_event_time = current_time

        # Log the detection event
        self._log_detection(detections)

        # Save snapshot
        if self.snapshot_enabled and frame is not None:
            self._save_snapshot(frame)

        # Play alarm sound (non-blocking)
        if self.sound_enabled:
            self._play_alarm()

        logger.warning(f"[FIRE ALERT] #{self._alert_count} triggered!")
        return True

    def clear_alert(self):
        """Clear the active alert state and turn off hardware alarm."""
        with self._lock:
            if self._alert_active:
                self._alert_active = False
                if self.arduino:
                    self.arduino.clear_alarm()
                self._last_hardware_ping_time = 0
                self._last_alert_time = 0

    def is_alert_active(self):
        """Check if an alert is currently active."""
        with self._lock:
            return self._alert_active

    def get_alert_count(self):
        """Get total number of alerts triggered."""
        return self._alert_count

    def _play_alarm(self):
        """Play alarm sound in a separate thread."""
        def _sound_worker():
            try:
                # Use Windows system beep as alarm
                for _ in range(5):
                    winsound.Beep(2500, 300)
                    time.sleep(0.1)
                    winsound.Beep(1800, 300)
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Sound playback error: {e}")

        if self._sound_thread and self._sound_thread.is_alive():
            return

        self._sound_thread = threading.Thread(target=_sound_worker, daemon=True)
        self._sound_thread.start()

    def _save_snapshot(self, frame):
        """Save a snapshot of the fire detection frame."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fire_snapshot_{timestamp}.jpg"
            filepath = os.path.join(self.config.SNAPSHOT_DIR, filename)
            cv2.imwrite(filepath, frame)
            logger.info(f"[SNAPSHOT] Saved: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Snapshot save error: {e}")
            return None

    def _log_detection(self, detections):
        """Log detection event to JSON file."""
        try:
            event = {
                "timestamp": datetime.now().isoformat(),
                "alert_number": self._alert_count,
                "detections": [
                    {
                        "label": d["label"],
                        "confidence": d["confidence"],
                        "bbox": list(d["bbox"]),
                    }
                    for d in detections
                ],
            }

            # Read existing log
            log_data = []
            if os.path.exists(self.config.LOG_FILE):
                try:
                    with open(self.config.LOG_FILE, "r") as f:
                        log_data = json.load(f)
                except (json.JSONDecodeError, IOError):
                    log_data = []

            # Append new event
            log_data.append(event)

            # Trim to max entries
            if len(log_data) > self.config.MAX_LOG_ENTRIES:
                log_data = log_data[-self.config.MAX_LOG_ENTRIES:]

            # Write updated log
            with open(self.config.LOG_FILE, "w") as f:
                json.dump(log_data, f, indent=2)

        except Exception as e:
            logger.error(f"Detection logging error: {e}")

    def get_detection_history(self, limit=50):
        """Get recent detection history from the log file."""
        try:
            if not os.path.exists(self.config.LOG_FILE):
                return []
            with open(self.config.LOG_FILE, "r") as f:
                log_data = json.load(f)
            return log_data[-limit:]
        except Exception:
            return []

    def set_sound_enabled(self, enabled):
        """Enable or disable sound alerts."""
        self.sound_enabled = enabled

    def set_cooldown(self, seconds):
        """Set alert cooldown period."""
        self.cooldown = max(1, seconds)

    def test_alarm(self):
        """Test local sound and Arduino buzzer (if connected)."""
        self._play_alarm()

        # Also pulse Arduino buzzer for hardware verification.
        if self.arduino and self.arduino.is_connected:
            self.arduino.trigger_alarm()

            def _auto_clear():
                time.sleep(2)
                if self.arduino and self.arduino.is_connected:
                    self.arduino.clear_alarm()

            threading.Thread(target=_auto_clear, daemon=True).start()
        return True
