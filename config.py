"""
AI Fire Detection System - Configuration
==========================================
Centralized configuration for all system components.
"""

import os

# ─── Paths ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
SNAPSHOT_DIR = os.path.join(LOG_DIR, "snapshots")
SOUND_DIR = os.path.join(BASE_DIR, "static", "sounds")

# Create directories if they don't exist
for d in [MODEL_DIR, LOG_DIR, SNAPSHOT_DIR, SOUND_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Camera Settings ───────────────────────────────────
CAMERA_INDEX = 0                  # Default camera (0 = built-in webcam)
CAMERA_WIDTH = 1280               # Capture width
CAMERA_HEIGHT = 720               # Capture height
CAMERA_FPS = 30                   # Target FPS

# ─── YOLOv8 Detection Settings ─────────────────────────
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
YOLO_CONFIDENCE_THRESHOLD = 0.45  # Minimum confidence to trigger detection
YOLO_IOU_THRESHOLD = 0.50        # NMS IoU threshold
YOLO_IMG_SIZE = 320               # Lower resolution for much faster inference

# ─── Color-Based Detection Settings ────────────────────
COLOR_MIN_AREA = 1400             # Raise area floor to reject patterned background noise
COLOR_SENSITIVITY = 0.35          # Stricter default for color-mode false-positive control

# ─── HSV Ranges for Fire Detection ─────────────────────
# These ranges capture the typical color spectrum of fire/flames
FIRE_HSV_RANGES = [
    # Lower red-orange flames (higher saturation/value to avoid skin/wallpaper tones)
    {"lower": (0, 140, 170), "upper": (18, 255, 255)},
    # Upper red flames
    {"lower": (165, 120, 170), "upper": (180, 255, 255)},
    # Bright orange-yellow flame cores (kept narrower than before)
    {"lower": (12, 110, 210), "upper": (35, 255, 255)},
]

# ─── Alert Settings ────────────────────────────────────
ALERT_SOUND_ENABLED = True
ALERT_SOUND_FILE = os.path.join(SOUND_DIR, "alarm.wav")
ALERT_COOLDOWN_SECONDS = 10       # Minimum seconds between alerts
ALERT_SNAPSHOT_ENABLED = True     # Save screenshot on detection

# ─── Flask Server Settings ─────────────────────────────
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5001
FLASK_DEBUG = False
SECRET_KEY = "fire-detection-system-2025-secret"

# ─── Logging ───────────────────────────────────────────
LOG_FILE = os.path.join(LOG_DIR, "detections.json")
MAX_LOG_ENTRIES = 10000           # Maximum log entries to keep

# ─── Hardware Integration ──────────────────────────────
ARDUINO_BAUDRATE = 9600           # Serial baud rate for Arduino
ARDUINO_ENABLED = True            # Master toggle for hardware features
