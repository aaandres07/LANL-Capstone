#include <Servo.h>

// Create Servo objects for left and right channels.
Servo leftMotor;
Servo rightMotor;

// Serial input buffer configuration.
const byte numChars = 32;
char receivedChars[numChars];
bool newData = false;

void setup() {
  Serial.begin(115200);
  
  // Attach the servos to appropriate digital pins
  leftMotor.attach(9);
  rightMotor.attach(10);
  
  // Arm the ESCs: send neutral (1500µs) for at least 1 second.
  leftMotor.writeMicroseconds(1500);
  rightMotor.writeMicroseconds(1500);
  delay(1000);
}

void loop() {
  recvWithEndMarker();
  
  if (newData) {
    int leftPWM, rightPWM;
    if (sscanf(receivedChars, "%d,%d", &leftPWM, &rightPWM) == 2) {
      // Constrain PWM values to the valid range for the ESC
      leftPWM = constrain(leftPWM, 500, 2500);
      rightPWM = constrain(rightPWM, 500, 2500);

      // Print received values for debugging
      Serial.print("Received PWM: ");
      Serial.print(leftPWM);
      Serial.print(", ");
      Serial.println(rightPWM);

      // Output the pulse widths to the ESC channels
      leftMotor.writeMicroseconds(leftPWM);
      rightMotor.writeMicroseconds(rightPWM);
    }
    newData = false;
  }
}

// This helper function accumulates serial data until a newline is received.
void recvWithEndMarker() {
  static byte ndx = 0;
  char endMarker = '\n';
  char rc;
  
  while (Serial.available() > 0 && !newData) {
    rc = Serial.read();
    if (rc != endMarker) {
      if (ndx < numChars - 1) {
        receivedChars[ndx++] = rc;
      }
    } else {
      receivedChars[ndx] = '\0'; // Terminate the string.
      ndx = 0;
      newData = true;
    }
  }
}
