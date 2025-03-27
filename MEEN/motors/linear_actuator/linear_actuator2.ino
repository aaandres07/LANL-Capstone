/*
  Arduino Motor Controller for 2-Channel Driver (Full Speed Digital Control)
  Updated Pin assignments:
    - Motor 1: IN1 = D22, IN2 = D23, ENA (digital) = D6
    - Motor 2: IN3 = D24, IN4 = D25, ENB (digital) = D7
  
  The Arduino reads a command from the Raspberry Pi:
    "255" for full-speed forward,
    "-255" for full-speed reverse,
    "0" to stop.
  
  Full-speed is achieved by simply setting the enable pins HIGH (and LOW to stop).
*/

const int IN1 = 28;
const int IN2 = 29;
const int IN3 = 30;
const int IN4 = 31;
const int ENA = 4;  // Digital control for Motor 1
const int ENB = 5;  // Digital control for Motor 2

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

  // Initialize all pins to LOW (motors off)
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);

  Serial.begin(115200);
  Serial.println("Motor controller ready.");
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

    if (command > 10) {
      // Full forward
      digitalWrite(IN1, HIGH);
      digitalWrite(IN2, LOW);
      digitalWrite(IN3, HIGH);
      digitalWrite(IN4, LOW);
      // Set enable pins 255 for full power
      analogWrite(ENA, 255);
      analogWrite(ENB, 255);
    } else if (command < 10) {
      // Full reverse
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, HIGH);
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, HIGH);
      // Set enable pins 255 for full power
      analogWrite(ENA, 255);
      analogWrite(ENB, 255);
    } else {
      // Stop: disable motor outputs
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, LOW);
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, LOW);
      digitalWrite(ENA, 0);
      digitalWrite(ENB, 0);
    }

    // Optionally, send feedback over Serial
    Serial.print("Command received: ");
    Serial.println(command);
  }
}
