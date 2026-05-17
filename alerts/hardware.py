"""
AI Fire Detection System - Hardware Integration
==============================================
Manages serial connection to Arduino for triggering
hardware buzzers and alarms.
"""

import serial
import serial.tools.list_ports
import time
import threading
import logging

logger = logging.getLogger(__name__)

class ArduinoController:
    """Manages serial communication with an Arduino."""

    def __init__(self, port=None, baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self._lock = threading.Lock()
        self.is_connected = False
        self._last_signal = None
        self._last_rx_line = None
        self._rx_thread = None
        self._stop_event = threading.Event()
        
        if port and port.upper() != "NONE":
            self.connect()

    @staticmethod
    def get_available_ports():
        """Return a list of available COM ports."""
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]

    def connect(self):
        """Establish serial connection to Arduino."""
        if not self.port or self.port.upper() == "NONE":
            logger.info("Arduino hardware integration disabled.")
            return False

        try:
            # Connect with a short timeout
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Allow Arduino to reset after serial connect
            self.is_connected = True
            logger.info(f"Connected to Arduino on {self.port} at {self.baudrate} baud.")

            # Start RX reader thread
            self._stop_event.clear()
            self._rx_thread = threading.Thread(target=self._rx_worker, daemon=True)
            self._rx_thread.start()
            
            # Send initial 'safe' signal to ensure buzzer is off
            self.send_signal('0')
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Arduino on {self.port}: {e}")
            self.is_connected = False
            self.serial_conn = None
            return False

    def disconnect(self):
        """Close the serial connection."""
        if self.serial_conn and self.serial_conn.is_open:
            self.send_signal('0')  # Turn off before disconnecting
            self._stop_event.set()
            self.serial_conn.close()
            self.is_connected = False
            logger.info("Disconnected from Arduino.")

    def _rx_worker(self):
        """Continuously read lines from Arduino (ACK/READY/debug)."""
        while not self._stop_event.is_set():
            try:
                if not self.serial_conn or not self.serial_conn.is_open:
                    time.sleep(0.2)
                    continue
                line = self.serial_conn.readline()
                if not line:
                    continue
                decoded = line.decode("utf-8", errors="ignore").strip()
                if decoded:
                    self._last_rx_line = decoded
                    logger.info(f"[ARDUINO RX] {decoded}")
            except Exception as e:
                logger.error(f"Arduino RX error: {e}")
                self.is_connected = False
                return

    def send_signal(self, signal):
        """Send a command signal to the Arduino with newline framing."""
        if not self.is_connected or not self.serial_conn:
            return False
            
        with self._lock:
            try:
                # Send line-delimited commands to avoid partial-read issues.
                data = f"{signal}\n".encode("ascii")
                # Retry a few times to survive occasional serial hiccups.
                for _ in range(3):
                    self.serial_conn.write(data)
                    self.serial_conn.flush()
                    time.sleep(0.02)
                self._last_signal = str(signal)
                return True
            except Exception as e:
                logger.error(f"Error sending signal to Arduino: {e}")
                self.is_connected = False
                return False

    def get_status(self):
        """Return current hardware link status for debugging."""
        return {
            "port": self.port,
            "baudrate": self.baudrate,
            "connected": bool(self.is_connected and self.serial_conn and self.serial_conn.is_open),
            "last_tx": self._last_signal,
            "last_rx": self._last_rx_line,
        }

    def trigger_alarm(self):
        """Send '1' to turn ON the Arduino buzzer."""
        if self.send_signal('1'):
            logger.info("Sent ALARM_ON signal to Arduino.")
            
    def clear_alarm(self):
        """Send '0' to turn OFF the Arduino buzzer."""
        if self.send_signal('0'):
            logger.info("Sent ALARM_OFF signal to Arduino.")
