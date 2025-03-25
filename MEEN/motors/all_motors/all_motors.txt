
/*
  Combined Arduino Controller for Three 2-Channel Drivers

  Subsystems and their pin assignments:

  1. Linear Actuators:
     Actuator 1:
       IN1 -> Digital Pin 28
       IN2 -> Digital Pin 29
       ENA -> Digital Pin 4
     Actuator 2:
       IN1 -> Digital Pin 30
       IN2 -> Digital Pin 31
       ENA -> Digital Pin 5

     Command protocol (prefixed with "A:"):
       1  -> Actuator 1 forward
       2  -> Actuator 1 reverse
       3  -> Actuator 2 forward
       4  -> Actuator 2 reverse
       Any other value -> Stop actuators

  2. Drive Motors (Servo-controlled ESCs):
     Left Motor attached to Digital Pin 9
     Right Motor attached to Digital Pin 10

     Command protocol (prefixed with "D:"):
       Two comma-separated PWM values (e.g., "1500,1500")
       PWM values are constrained between 500 and 2500 µs.

  3. Net Motors (Digital full-speed control):
     Motor 1:
       IN1 -> Digital Pin 22
       IN2 -> Digital Pin 23
       ENA -> Digital Pin 6
     Motor 2:
       IN1 -> Digital Pin 24
       IN2 -> Digital Pin 25
       ENB -> Digital Pin 7

     Command protocol (prefixed with "N:"):
       A single integer command:
         255  for full forward,
         -255 for full reverse,
         0    to stop.

  The Arduino expects commands terminated by a newline.
  Example command strings:
    A:1\n        --> Linear actuators: actuator1 forward
    D:1500,1500\n --> Drive motors: both at neutral
    N:255\n      --> Net motors: full forward
*/

#include <Servo.h>

// -------- Linear Actuators --------
// Actuator 1
#define ACT1_IN1 28
#define ACT1_IN2 29
#define ACT1_ENA 4
// Actuator 2
#define ACT2_IN1 30
#define ACT2_IN2 31
#define ACT2_ENA 5

// -------- Net Motors --------
const int NET_IN1 = 22;
const int NET_IN2 = 23;
const int NET_IN3 = 24;
const int NET_IN4 = 25;
const int NET_ENA  = 6;
const int NET_ENB  = 7;

// -------- Drive Motors (ESC-controlled) --------
Servo leftMotor;   // Attached to pin 9
Servo rightMotor;  // Attached to pin 10

// Serial input buffer
String inputString = "";
bool stringComplete = false;

void setup() {
  Serial.begin(115200);

  // Initialize Linear Actuator pins
  pinMode(ACT1_IN1, OUTPUT);
  pinMode(ACT1_IN2, OUTPUT);
  pinMode(ACT1_ENA, OUTPUT);
  pinMode(ACT2_IN1, OUTPUT);
  pinMode(ACT2_IN2, OUTPUT);
  pinMode(ACT2_ENA, OUTPUT);
  digitalWrite(ACT1_ENA, LOW);
  digitalWrite(ACT2_ENA, LOW);

  // Initialize Net Motor pins
  pinMode(NET_IN1, OUTPUT);
  pinMode(NET_IN2, OUTPUT);
  pinMode(NET_IN3, OUTPUT);
  pinMode(NET_IN4, OUTPUT);
  pinMode(NET_ENA, OUTPUT);
  pinMode(NET_ENB, OUTPUT);
  digitalWrite(NET_IN1, LOW);
  digitalWrite(NET_IN2, LOW);
  digitalWrite(NET_IN3, LOW);
  digitalWrite(NET_IN4, LOW);
  digitalWrite(NET_ENA, LOW);
  digitalWrite(NET_ENB, LOW);

  // Setup Drive Motors (ESCs)
  leftMotor.attach(9);
  rightMotor.attach(10);
  leftMotor.writeMicroseconds(1500);
  rightMotor.writeMicroseconds(1500);
  delay(1000);

  Serial.println("Combined Arduino controller ready.");
}

void loop() {
  if (stringComplete) {
    processCommand(inputString);
    // Echo the received command for debugging.
    Serial.println(inputString);
    inputString = "";
    stringComplete = false;
  }
}

void processCommand(String cmd) {
  // Command format: <Subsystem>:<Data>
  // e.g., "A:1", "D:1500,1500", "N:255"
  cmd.trim();
  if (cmd.length() < 3 || cmd.charAt(1) != ':') return;
  
  char subsystem = cmd.charAt(0);
  String data = cmd.substring(2);
  
  switch (subsystem) {
    case 'A': // Linear Actuators
      controlActuator(data);
      break;
    case 'D': // Drive Motors (ESC)
      controlDriveMotors(data);
      break;
    case 'N': // Net Motors
      controlNetMotors(data);
      break;
    default:
      Serial.println("Unknown subsystem command");
      break;
  }
}

void controlActuator(String data) {
  int commandValue = data.toInt();
  switch (commandValue) {
    case 1: // Actuator 1 forward
      digitalWrite(ACT1_IN1, HIGH);
      digitalWrite(ACT1_IN2, LOW);
      digitalWrite(ACT1_ENA, HIGH);
      break;
    case 2: // Actuator 1 reverse
      digitalWrite(ACT1_IN1, LOW);
      digitalWrite(ACT1_IN2, HIGH);
      digitalWrite(ACT1_ENA, HIGH);
      break;
    case 3: // Actuator 2 forward
      digitalWrite(ACT2_IN1, HIGH);
      digitalWrite(ACT2_IN2, LOW);
      digitalWrite(ACT2_ENA, HIGH);
      break;
    case 4: // Actuator 2 reverse
      digitalWrite(ACT2_IN1, LOW);
      digitalWrite(ACT2_IN2, HIGH);
      digitalWrite(ACT2_ENA, HIGH);
      break;
    default: // Stop actuators
      digitalWrite(ACT1_ENA, LOW);
      digitalWrite(ACT2_ENA, LOW);
      break;
  }
  Serial.print("Actuator command: ");
  Serial.println(commandValue);
}

void controlDriveMotors(String data) {
  // Expected format: "leftPWM,rightPWM"
  int commaIndex = data.indexOf(',');
  if (commaIndex == -1) return;
  
  int leftPWM = data.substring(0, commaIndex).toInt();
  int rightPWM = data.substring(commaIndex + 1).toInt();
  leftPWM = constrain(leftPWM, 500, 2500);
  rightPWM = constrain(rightPWM, 500, 2500);
  
  leftMotor.writeMicroseconds(leftPWM);
  rightMotor.writeMicroseconds(rightPWM);
  
  Serial.print("Drive Motors PWM: ");
  Serial.print(leftPWM);
  Serial.print(", ");
  Serial.println(rightPWM);
}

void controlNetMotors(String data) {
  int command = data.toInt();
  if (command > 0) {
    // Full forward
    digitalWrite(NET_IN1, HIGH);
    digitalWrite(NET_IN2, LOW);
    digitalWrite(NET_IN3, HIGH);
    digitalWrite(NET_IN4, LOW);
    digitalWrite(NET_ENA, HIGH);
    digitalWrite(NET_ENB, HIGH);
  } else if (command < 0) {
    // Full reverse
    digitalWrite(NET_IN1, LOW);
    digitalWrite(NET_IN2, HIGH);
    digitalWrite(NET_IN3, LOW);
    digitalWrite(NET_IN4, HIGH);
    digitalWrite(NET_ENA, HIGH);
    digitalWrite(NET_ENB, HIGH);
  } else {
    // Stop motors
    digitalWrite(NET_IN1, LOW);
    digitalWrite(NET_IN2, LOW);
    digitalWrite(NET_IN3, LOW);
    digitalWrite(NET_IN4, LOW);
    digitalWrite(NET_ENA, LOW);
    digitalWrite(NET_ENB, LOW);
  }
  Serial.print("Net Motors command: ");
  Serial.println(command);
}

void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n') {
      stringComplete = true;
    } else {
      inputString += inChar;
    }
  }
}
