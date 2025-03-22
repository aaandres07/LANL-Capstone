/*
  Arduino Motor Controller for 2-Channel Driver (Full Speed Digital Control)
  Pin assignments:
    - Motor 1: IN1 = D2, IN2 = D3, ENA (digital) = D6
    - Motor 2: IN3 = D4, IN4 = D5, ENB (digital) = D7
  
  The Arduino reads a command from the Raspberry Pi:
    "255" for full-speed forward,
    "-255" for full-speed reverse,
    "0" to stop.
  
  Full-speed is achieved by simply setting the enable pins HIGH (and LOW to stop).
*/

const int IN1 = 2;
const int IN2 = 3;
const int IN3 = 4;
const int IN4 = 5;
const int ENA = 6;  // Digital control for Motor 1
const int ENB = 7;  // Digital control for Motor 2

String inputString = "";   // Buffer for incoming serial data
bool stringComplete = false;

void setup() {
  // Set motor control pins as outputs
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);

  // Initialize all pins to LOW (motors off)
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  digitalWrite(ENA, LOW);
  digitalWrite(ENB, LOW);

  Serial.begin(115200);
  Serial.println("Motor controller ready (full-speed digital control).");
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

    if (command > 0) {
      // Full forward
      digitalWrite(IN1, HIGH);
      digitalWrite(IN2, LOW);
      digitalWrite(IN3, HIGH);
      digitalWrite(IN4, LOW);
      // Set enable pins HIGH for full power
      digitalWrite(ENA, HIGH);
      digitalWrite(ENB, HIGH);
    } else if (command < 0) {
      // Full reverse
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, HIGH);
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, HIGH);
      digitalWrite(ENA, HIGH);
      digitalWrite(ENB, HIGH);
    } else {
      // Stop: disable motor outputs
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, LOW);
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, LOW);
      digitalWrite(ENA, LOW);
      digitalWrite(ENB, LOW);
    }

    // Optionally, send feedback over Serial
    Serial.print("Command received: ");
    Serial.println(command);
  }
}
