/*
  Arduino Motor Controller for 2-Channel Driver
  Adapted for your current motor driver:
    - Motor 1: IN1 (D2), IN2 (D3), ENA (D6, PWM)
    - Motor 2: IN3 (D4), IN4 (D5), ENB (D7, PWM)
  
  The Arduino reads a net speed (–255 to 255) from the Raspberry Pi.
  A positive value drives the motors forward;
  a negative value drives them in reverse.
*/

const int IN1 = 2;
const int IN2 = 3;
const int IN3 = 4;
const int IN4 = 5;
const int ENA = 6;  // PWM pin for Motor 1
const int ENB = 7;  // PWM pin for Motor 2

String inputString = "";   // A string to hold incoming data
bool stringComplete = false;

void setup() {
  // Set motor control pins as outputs
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);

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
    int netSpeed = inputString.toInt();
    inputString = "";
    stringComplete = false;

    // Constrain the value to -255 to 255
    if (netSpeed > 255) {
      netSpeed = 255;
    } else if (netSpeed < -255) {
      netSpeed = -255;
    }

    // Set motor directions based on netSpeed
    if (netSpeed > 0) {
      // Forward: Motor 1 and Motor 2
      digitalWrite(IN1, HIGH);
      digitalWrite(IN2, LOW);
      digitalWrite(IN3, HIGH);
      digitalWrite(IN4, LOW);
    } else if (netSpeed < 0) {
      // Reverse: Motor 1 and Motor 2
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, HIGH);
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, HIGH);
    } else {
      // Stop: Disable motor outputs
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, LOW);
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, LOW);
    }

    // Use the absolute value for PWM speed
    int pwmSpeed = abs(netSpeed);
    analogWrite(ENA, pwmSpeed);
    analogWrite(ENB, pwmSpeed);

    // Send feedback over Serial (optional)
    Serial.print("Speed set to: ");
    Serial.println(netSpeed);
  }
}
