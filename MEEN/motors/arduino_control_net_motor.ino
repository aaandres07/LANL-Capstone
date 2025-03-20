// Define motor driver pin assignments (digital pins 2-7)
#define IN1 2  // Motor 1 direction pin
#define IN2 3  // Motor 1 direction pin
#define IN3 4  // Motor 2 direction pin
#define IN4 5  // Motor 2 direction pin
#define ENA 6  // Motor 1 PWM speed pin
#define ENB 7  // Motor 2 PWM speed pin

void setup() {
  // Set direction pins as outputs
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  
  // Set PWM pins as outputs
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);
  
  // Start serial communication
  Serial.begin(9600);
}

void loop() {
  // Check if there is any serial data available
  if (Serial.available() > 0) {
    // Read a line until newline character
    String input = Serial.readStringUntil('\n');
    
    // Expecting input in the format: "S:<value>"
    if (input.startsWith("S:")) {
      // Extract the speed value
      int speed = input.substring(2).toInt();
      
      if (speed > 0) {
        // Move forward: set Motor 1 and Motor 2 for forward rotation
        digitalWrite(IN1, HIGH);
        digitalWrite(IN2, LOW);
        digitalWrite(IN3, HIGH);
        digitalWrite(IN4, LOW);
        
        // Constrain speed value to PWM limits (0 to 255)
        int pwmVal = constrain(speed, 0, 255);
        analogWrite(ENA, pwmVal);
        analogWrite(ENB, pwmVal);
      } else if (speed < 0) {
        // Move in reverse: set Motor 1 and Motor 2 for reverse rotation
        digitalWrite(IN1, LOW);
        digitalWrite(IN2, HIGH);
        digitalWrite(IN3, LOW);
        digitalWrite(IN4, HIGH);
        
        int pwmVal = constrain(abs(speed), 0, 255);
        analogWrite(ENA, pwmVal);
        analogWrite(ENB, pwmVal);
      } else {
        // Stop the motors
        digitalWrite(IN1, LOW);
        digitalWrite(IN2, LOW);
        digitalWrite(IN3, LOW);
        digitalWrite(IN4, LOW);
        analogWrite(ENA, 0);
        analogWrite(ENB, 0);
      }
    }
  }
}
