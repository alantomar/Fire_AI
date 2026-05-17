"""
AI Fire Detection System - Main Application
==============================================
Flask web server with real-time video streaming,
fire detection processing, and REST API endpoints.

Usage:
    python app.py
    Then open http://localhost:5000 in your browser.
"""

import os
import sys
import subprocess
import importlib

# ─── Auto-Dependency Installer ─────────────────────────────
# Checks all required packages on startup and installs any
# that are missing, so the app works on a fresh machine.
# ────────────────────────────────────────────────────────────

# Mapping: pip package name -> Python import name
REQUIRED_PACKAGES = {
    "flask":              "flask",
    "flask-socketio":     "flask_socketio",
    "ultralytics":        "ultralytics",
    "opencv-python":      "cv2",
    "numpy":              "numpy",
    "Pillow":             "PIL",
    "playsound":          "playsound",
    "python-engineio":    "engineio",
    "python-socketio":    "socketio",
    "gevent":             "gevent",
    "gevent-websocket":   "geventwebsocket",
    "pyserial":           "serial",
}

def check_and_install_dependencies():
    """Check all required packages and install missing ones automatically."""
    missing = []

    for pip_name, import_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pip_name)

    if not missing:
        return  # Everything is already installed

    print("\n" + "=" * 60)
    print("  AUTO-INSTALLING MISSING DEPENDENCIES")
    print("=" * 60)
    print(f"\n  Missing packages: {', '.join(missing)}\n")

    # Try installing all at once first (faster)
    req_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    if os.path.exists(req_file):
        print("  Installing from requirements.txt ...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", req_file],
            capture_output=False,
        )
        if result.returncode == 0:
            print("\n  [OK] All dependencies installed successfully!\n")
            print("=" * 60 + "\n")
            return

    # Fallback: install missing packages one by one
    print("  Installing packages individually ...\n")
    failed = []
    for pkg in missing:
        print(f"  Installing {pkg} ...", end=" ", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("[OK]")
        else:
            print("[FAILED]")
            failed.append(pkg)

    if failed:
        print(f"\n  [WARNING] Could not install: {', '.join(failed)}")
        print("  The system may still work in color-detection mode.\n")
    else:
        print("\n  [OK] All dependencies installed successfully!\n")

    print("=" * 60 + "\n")

# Run the check before importing anything else
check_and_install_dependencies()

import cv2
import json
import time
import threading
import logging
from datetime import datetime

from flask import Flask, Response, render_template, jsonify, request
from flask_socketio import SocketIO

import config
from detection.camera import CameraManager
from detection.fire_detector import FireDetector
from alerts.alert_manager import AlertManager
from alerts.hardware import ArduinoController

# ─── Logging Setup ─────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("FireDetection")

# ─── Flask App Setup ───────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ─── Initialize Components ─────────────────────────────
camera = CameraManager(
    camera_index=config.CAMERA_INDEX,
    width=config.CAMERA_WIDTH,
    height=config.CAMERA_HEIGHT,
    fps=config.CAMERA_FPS,
)
detector = FireDetector(config)

# AlertManager will be initialized after Arduino COM port is selected
alert_manager = None

# ─── Global State ──────────────────────────────────────
system_state = {
    "running": False,
    "fire_detected": False,
    "detection_mode": detector.get_mode(),
    "fps": 0,
    "total_detections": 0,
    "last_detection": None,
    "start_time": None,
    "camera_active": False,
}

detection_lock = threading.Lock()
latest_result = {"detections": [], "fire_detected": False, "fps": 0}


# ─── Video Processing Thread ──────────────────────────
def detection_loop():
    """Main detection processing loop."""
    global latest_result

    logger.info("[STARTED] Detection loop started.")
    system_state["start_time"] = datetime.now().isoformat()

    while system_state["running"]:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        # Run fire detection
        result = detector.detect(frame)

        with detection_lock:
            latest_result = result

        # Update system state
        system_state["fps"] = result["fps"]
        system_state["detection_mode"] = result["mode"]

        if result["fire_detected"]:
            system_state["fire_detected"] = True
            system_state["total_detections"] += 1
            system_state["last_detection"] = datetime.now().isoformat()

            # Trigger alert
            alert_manager.trigger_alert(frame, result["detections"])

            # Emit fire event via SocketIO
            socketio.emit("fire_detected", {
                "timestamp": datetime.now().isoformat(),
                "detections": result["detections"],
                "alert_count": alert_manager.get_alert_count(),
            })
        else:
            system_state["fire_detected"] = False
            if alert_manager:
                alert_manager.clear_alert()

        # Emit status update
        socketio.emit("status_update", {
            "fire_detected": result["fire_detected"],
            "fps": result["fps"],
            "mode": result["mode"],
            "total_detections": system_state["total_detections"],
        })

        time.sleep(0.01)

    logger.info("Detection loop stopped.")


def generate_frames():
    """Generator for MJPEG video stream."""
    while system_state["running"]:
        with detection_lock:
            result = latest_result

        frame = result.get("frame")
        if frame is None:
            # Send a blank frame if no camera feed
            frame = create_placeholder_frame()

        # Encode frame to JPEG
        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

        time.sleep(0.033)  # ~30 FPS cap for streaming


def create_placeholder_frame():
    """Create a placeholder frame when camera is not active."""
    import numpy as np
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        frame, "Camera Initializing...",
        (140, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2,
    )
    return frame


# ─── Routes ────────────────────────────────────────────
@app.route("/")
def dashboard():
    """Serve the main dashboard page."""
    return render_template("dashboard.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG video stream endpoint."""
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/status")
def api_status():
    """Get current system status."""
    return jsonify({
        "running": system_state["running"],
        "fire_detected": system_state["fire_detected"],
        "detection_mode": system_state["detection_mode"],
        "fps": system_state["fps"],
        "total_detections": system_state["total_detections"],
        "last_detection": system_state["last_detection"],
        "start_time": system_state["start_time"],
        "camera_active": camera.is_running(),
        "alert_count": alert_manager.get_alert_count(),
        "detector_stats": detector.get_stats(),
    })


@app.route("/api/detections")
def api_detections():
    """Get detection history."""
    limit = request.args.get("limit", 50, type=int)
    history = alert_manager.get_detection_history(limit=limit)
    return jsonify(history)


@app.route("/api/settings", methods=["POST"])
def api_settings():
    """Update detection settings."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    if "confidence" in data:
        detector.set_confidence(float(data["confidence"]))

    if "sound_enabled" in data:
        alert_manager.set_sound_enabled(bool(data["sound_enabled"]))

    if "cooldown" in data:
        alert_manager.set_cooldown(int(data["cooldown"]))

    if "camera_index" in data:
        new_index = int(data["camera_index"])
        camera.switch_camera(new_index)

    return jsonify({"status": "ok", "message": "Settings updated"})


@app.route("/api/test_alarm", methods=["POST"])
def api_test_alarm():
    """Test local alarm sound and Arduino buzzer."""
    alert_manager.test_alarm()
    arduino_connected = bool(
        getattr(alert_manager, "arduino", None)
        and alert_manager.arduino.is_connected
    )
    return jsonify({
        "status": "ok",
        "message": "Alarm test triggered",
        "arduino_connected": arduino_connected,
    })


@app.route("/api/hardware_status")
def api_hardware_status():
    """Get Arduino link status and available ports."""
    ports = ArduinoController.get_available_ports()
    status = {"enabled": bool(config.ARDUINO_ENABLED), "available_ports": ports}
    if alert_manager and getattr(alert_manager, "arduino", None):
        status.update(alert_manager.arduino.get_status())
    return jsonify(status)


@app.route("/api/cameras")
def api_cameras():
    """Get list of available cameras."""
    cameras = camera.get_available_cameras()
    return jsonify({"cameras": cameras, "current": camera.camera_index})


@app.route("/api/reload_model", methods=["POST"])
def api_reload_model():
    """Reload the YOLOv8 model."""
    success = detector.reload_model()
    if success:
        system_state["detection_mode"] = detector.get_mode()
        return jsonify({"status": "ok", "message": "Model reloaded", "mode": detector.get_mode()})
    return jsonify({"status": "error", "message": "No model found to reload"}), 404


# ─── SocketIO Events ──────────────────────────────────
@socketio.on("connect")
def handle_connect():
    logger.info("Client connected to WebSocket.")
    socketio.emit("status_update", {
        "fire_detected": system_state["fire_detected"],
        "fps": system_state["fps"],
        "mode": system_state["detection_mode"],
        "total_detections": system_state["total_detections"],
    })


@socketio.on("disconnect")
def handle_disconnect():
    logger.info("Client disconnected from WebSocket.")


# ─── Application Lifecycle ─────────────────────────────
def start_system():
    """Start the camera and detection system."""
    logger.info("=" * 60)
    logger.info("AI FIRE DETECTION SYSTEM")
    logger.info("=" * 60)
    logger.info(f"  Detection Mode: {detector.get_mode().upper()}")
    logger.info(f"  Camera Index:   {config.CAMERA_INDEX}")
    logger.info(f"  Resolution:     {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}")
    logger.info(f"  Dashboard:      http://localhost:{config.FLASK_PORT}")
    logger.info("=" * 60)

    # Start camera in background (don't block Flask startup)
    def _init_camera():
        if not camera.start():
            logger.error("Failed to start camera! Check your webcam connection.")
            logger.info("The dashboard will still work, but with no video feed.")
        system_state["camera_active"] = camera.is_running()

    system_state["running"] = True

    cam_thread = threading.Thread(target=_init_camera, daemon=True)
    cam_thread.start()

    # Start detection thread
    detection_thread = threading.Thread(target=detection_loop, daemon=True)
    detection_thread.start()


def shutdown_system():
    """Clean shutdown of all components."""
    logger.info("Shutting down...")
    system_state["running"] = False
    camera.stop()
    logger.info("System shutdown complete.")


def prompt_arduino_port():
    """Show a Tkinter dialog to select the Arduino COM port."""
    import tkinter as tk
    from tkinter import simpledialog, ttk
    
    ports = ArduinoController.get_available_ports()
    ports.insert(0, "None")  # Option to skip Arduino integration
    
    root = tk.Tk()
    root.title("Hardware Configuration")
    root.geometry("350x150")
    root.attributes("-topmost", True)
    root.eval('tk::PlaceWindow . center')
    
    selected_port = tk.StringVar(root)
    selected_port.set(ports[0])
    
    tk.Label(root, text="Select Arduino COM Port:", font=("Arial", 12)).pack(pady=10)
    dropdown = ttk.Combobox(root, textvariable=selected_port, values=ports, state="readonly", font=("Arial", 11))
    dropdown.pack(pady=5)
    
    def on_ok():
        root.quit()
        
    tk.Button(root, text="Start System", command=on_ok, font=("Arial", 11, "bold"), bg="#4CAF50", fg="white", padx=15).pack(pady=15)
    
    root.mainloop()
    port = selected_port.get()
    root.destroy()
    
    return port if port != "None" else None

# ─── Main Entry Point ─────────────────────────────────
if __name__ == "__main__":
    # Prompt for Arduino Port
    arduino_port = prompt_arduino_port()
    alert_manager = AlertManager(config, arduino_port=arduino_port)

    try:
        start_system()
        socketio.run(
            app,
            host=config.FLASK_HOST,
            port=config.FLASK_PORT,
            debug=config.FLASK_DEBUG,
            use_reloader=False,
            allow_unsafe_werkzeug=True,
        )
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        shutdown_system()
