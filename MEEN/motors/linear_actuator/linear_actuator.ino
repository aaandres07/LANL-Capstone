/*
  Updated linear_actuator.ino
  This code collects serial input into a buffer until a newline is received.
  When a complete command is available, it processes the command immediately
  and sends feedback back to the Python controller.
*/

String inputString = "";     // A string to hold incoming data
bool stringComplete = false; // Flag for when a full command is received

void setup() {
  Serial.begin(115200);
  // Add any motor or actuator initialization code here
}

void loop() {
  // Process the command only when a complete string is available
  if (stringComplete) {
    // Convert the received string to an integer command value
    int commandValue = inputString.toInt();

    // Process the command (for example, control the actuator/motor)
    controlActuator(commandValue);

    // Send feedback for the current command back to the Python program
    Serial.println(commandValue);

    // Clear the input buffer for the next command
    inputString = "";
    stringComplete = false;
  }
}

// Example function to control the actuator based on the command value
void controlActuator(int commandValue) {
  // Implement your actuator control logic here.
  // For instance:
  // if (commandValue > 0) {
  //   // Drive motor forward
  // } else if (commandValue < 0) {
  //   // Drive motor in reverse
  // } else {
  //   // Stop motor
  // }
}

// This function is automatically called when new serial data arrives.
void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    // Check for newline character which indicates the end of a command
    if (inChar == '\n') {
      stringComplete = true;
    } else {
      inputString += inChar;
    }
  }
}
