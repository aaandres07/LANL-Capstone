/*
  Arduino Motor Controller for 2-Channel Driver (Full Speed Digital Control)
  The Arduino reads a command from the Raspberry Pi:
    "255" for full-speed forward,
    "0" to stop.
*/
// Motor 1
const int IN1 = 22; // IN1 = D28
const int IN2 = 23; // IN2 = D28
const int ENA = 6;  // ENA = D6 PWM

// Motor 2
const int IN3 = 24; // IN3 = D28
const int IN4 = 25; // IN4 = D28
const int ENB = 7;  // ENB = D7 PWM

String inputString = "";   // Buffer for incoming serial data
bool stringComplete = false;

void setup() {
  // Set motor control pins as outputs
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  // Set PWM pins as outputs
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);

  // Initialize all pins to LOW and 0 (motors off)
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);

  Serial.begin(115200);
  Serial.println("Motor controller ready (full speed).");
}

void loop() {
  // Read serial input until newline is received
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n') {
      stringComplete = true;
    } else {
      inputString += inChar;
    }
  }

  // If a complete command is received, process it
  if (stringComplete) {
    int command = inputString.toInt();
    inputString = "";
    stringComplete = false;

    if (command > 0) { //setting 10 instead of 0 provides deadzone
      // Full forward
      digitalWrite(IN1, HIGH);
      digitalWrite(IN2, LOW);
      digitalWrite(IN3, HIGH);
      digitalWrite(IN4, LOW);

      // Set enable pins 255 for full power
      // Can edit between 0 to 255 to calibrate
      analogWrite(ENA, 255);
      analogWrite(ENB, 255);
    } else if (command < 0) { //setting 10 instead of 0 provides deadzone
      // Full reverse
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, HIGH);
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, HIGH);

      // Set enable pins 255 for full power
      // Can edit between 0 to 255 to calibrate
      analogWrite(ENA, 255);
      analogWrite(ENB, 255);
    } else {
      // Stop: disable motor outputs
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, LOW);
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, LOW);
      analogWrite(ENA, 0);
      analogWrite(ENB, 0);
    }

    // Optionally, send feedback over Serial
    Serial.print("Command received: ");
    Serial.println(command);
  }
}
