# 🔥 AI Fire Detection System

**ECE341 - Programming IoT | Final Project**

A real-time AI-powered fire and smoke detection system with a live web dashboard, dual-mode detection (YOLOv8 + HSV color fallback), and Arduino hardware integration for physical alarm output.

---

## 📌 Project Overview

This system monitors a live camera feed and detects fire or smoke in real time using a custom-trained YOLOv8 deep learning model. When fire is detected, it triggers a sound alarm, saves a snapshot, logs the event, and activates a buzzer connected to an Arduino via serial communication. Everything is viewable through a browser-based dashboard with live video streaming and WebSocket status updates.

---

## 🧠 How It Works

### Detection Modes
- **Primary — YOLOv8 (Deep Learning):** A YOLOv8n model fine-tuned on a custom fire/smoke dataset. Detects 2 classes: `fire` and `smoke`.
- **Fallback — HSV Color Detection:** If no model file is found, the system automatically switches to HSV color-space analysis with morphological filtering and a temporal stability filter (requires 3 consecutive positive frames before raising an alert).

### Alert Pipeline
When fire is detected:
1. Sound alarm plays (`winsound`)
2. Snapshot saved to `logs/snapshots/`
3. Event logged to `logs/detections.json`
4. Arduino buzzer activated via serial (`'1'` → Pin 13 HIGH)
5. WebSocket event pushed to all connected browser clients

---

## 🗂️ Project Structure

```
Ai fire - 1.0/
│
├── app.py                  # Main Flask application & entry point
├── config.py               # All system configuration (camera, YOLO, alerts, Flask)
├── requirements.txt        # Python dependencies
├── train_model.py          # YOLOv8 training script
│
├── detection/
│   ├── camera.py           # Camera capture manager (OpenCV)
│   ├── fire_detector.py    # Main detector orchestrator (YOLO + color)
│   ├── yolo_detector.py    # YOLOv8 inference engine
│   └── color_detector.py   # HSV color-based fallback detector
│
├── alerts/
│   ├── alert_manager.py    # Alert logic: sound, snapshot, logging, cooldown
│   └── hardware.py         # Arduino serial controller
│
├── arduino_alarm/
│   └── arduino_alarm.ino   # Arduino sketch (buzzer on Pin 13)
│
├── models/
│   └── best.pt             # Trained YOLOv8 model weights (place here)
│
├── datasets/
│   └── fire_smoke/         # Training dataset (YOLO format)
│       ├── train/images/
│       ├── train/labels/
│       ├── valid/images/
│       ├── valid/labels/
│       └── data.yaml
│
├── runs/
│   └── fire_detection/     # Training outputs (results.csv, weights, plots)
│
├── logs/
│   ├── detections.json     # Detection event log
│   └── snapshots/          # Saved fire detection screenshots
│
├── static/
│   ├── css/dashboard.css
│   └── js/dashboard.js
│
└── templates/
    └── dashboard.html      # Web dashboard UI
```

---

## ⚙️ Requirements

- Python 3.10+
- Webcam (USB or built-in)
- Arduino Uno (optional, for hardware buzzer)
- Windows OS (for `winsound` alarm; Linux/Mac users can disable sound in `config.py`)

---

## 🚀 Setup & Running

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/ai-fire-detection.git
cd ai-fire-detection
```

### 2. Install Dependencies
Dependencies auto-install on first run, but you can also install manually:
```bash
pip install -r requirements.txt
```

### 3. Add the Trained Model
Place your trained `best.pt` file in the `models/` folder:
```
models/best.pt
```
If no model is found, the system runs in HSV color-detection mode automatically.

### 4. Upload Arduino Sketch (Optional)
Open `arduino_alarm/arduino_alarm.ino` in the Arduino IDE and upload it to your Arduino Uno. Connect a buzzer to **Pin 13**.

### 5. Run the System
```bash
python app.py
```
A dialog will appear asking you to select your Arduino COM port (or skip it). Then open your browser at:
```
http://localhost:5001
```

---

## 🌐 Web Dashboard Features

| Feature | Description |
|---|---|
| Live Video Feed | Annotated MJPEG stream at ~30 FPS |
| Real-time Alerts | WebSocket fire/smoke notifications |
| Detection Mode | Shows YOLO or Color mode |
| FPS Counter | Live processing speed |
| Detection History | Timestamped log of all events |
| Settings Panel | Adjust confidence, cooldown, camera source |
| Hardware Status | Arduino connection status |
| Test Alarm | Manually test buzzer and sound |

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web dashboard |
| `/video_feed` | GET | MJPEG video stream |
| `/api/status` | GET | Current system status (JSON) |
| `/api/detections` | GET | Detection history log |
| `/api/settings` | POST | Update detection settings |
| `/api/test_alarm` | POST | Test the alarm system |
| `/api/hardware_status` | GET | Arduino connection info |
| `/api/cameras` | GET | Available camera list |
| `/api/reload_model` | POST | Reload the YOLOv8 model |

---

## 🤖 Model Training

The YOLOv8n model was trained on a custom dataset of 200 images (fire, smoke, and negative samples) in YOLO format.

To retrain the model:
```bash
python train_model.py
```

**Training Configuration:**
- Model: YOLOv8n (nano — optimized for edge/IoT devices)
- Classes: `fire`, `smoke`
- Image size: 320px
- Epochs: 12
- Best mAP50: ~44.4% (epoch 12)
- Best Precision: ~88%

---

## 🔧 Configuration

All settings are in `config.py`:

```python
CAMERA_INDEX = 0              # Webcam index
CAMERA_WIDTH = 1280           # Resolution
CAMERA_FPS = 30               # Target FPS

YOLO_CONFIDENCE_THRESHOLD = 0.45   # Detection confidence
YOLO_IMG_SIZE = 320                # Inference resolution

ALERT_COOLDOWN_SECONDS = 10   # Seconds between logged alerts
ARDUINO_BAUDRATE = 9600       # Serial baud rate

FLASK_PORT = 5001             # Web server port
```

---

## 🛠️ Hardware Wiring

```
Arduino Uno
  Pin 13  ──────────────►  Buzzer (+)
  GND     ──────────────►  Buzzer (-)
  USB     ──────────────►  PC (Serial Communication)
```

Arduino receives:
- `'1'` → Buzzer ON (fire detected)
- `'0'` → Buzzer OFF (all clear)

---
