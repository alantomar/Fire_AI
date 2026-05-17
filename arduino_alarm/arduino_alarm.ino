// AI Fire Detection System - Arduino Buzzer Controller
// ====================================================
// Listens to the serial port (9600 baud) for commands.
// Receives '1' -> Turns buzzer fully ON (constant HIGH)
// Receives '0' -> Immediately turns buzzer OFF

const int BUZZER_PIN = 13;  // Connect your buzzer to Pin 13
bool isAlarmActive = false;
String serialBuffer = "";

void stopBuzzer() {
  digitalWrite(BUZZER_PIN, LOW);
}

void setup() {
  Serial.begin(9600);
  pinMode(BUZZER_PIN, OUTPUT);
  stopBuzzer(); // Ensure buzzer is off on startup
  Serial.println("ARDUINO_READY");
}

void loop() {
  // Read complete line-delimited commands from serial.
  if (Serial.available() > 0) {
    char ch = Serial.read();
    if (ch == '\n' || ch == '\r') {
      if (serialBuffer.length() > 0) {
        char command = serialBuffer.charAt(0);
        serialBuffer = "";

        // Command '1' = Fire Detected -> Turn buzzer fully ON
        if (command == '1') {
          isAlarmActive = true;
          digitalWrite(BUZZER_PIN, HIGH);
          Serial.println("ACK:ALARM_ON");
        }
        // Command '0' = Clear / Safe -> Stop alarm immediately
        else if (command == '0') {
          isAlarmActive = false;
          stopBuzzer();
          Serial.println("ACK:ALARM_OFF");
        }
      }
    } else {
      serialBuffer += ch;
    }
  }

  // Handle the alarm pulsing and timeout autonomously
  if (isAlarmActive) {
    // Keep full power ON while alarm is active.
    digitalWrite(BUZZER_PIN, HIGH);
  }
}
