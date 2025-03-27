// Define motor driver pin assignments for dual linear actuators
#define IN1 2  // Left actuator direction pin 1
#define IN2 3  // Left actuator direction pin 2
#define IN3 4  // Right actuator direction pin 1
#define IN4 5  // Right actuator direction pin 2
#define ENA 6  // Left actuator PWM (speed)
#define ENB 7  // Right actuator PWM (speed)

// Calibration factors for each actuator for up and down directions
// Adjust these constants based on testing:
// - For up motion, if the right actuator is too fast, reduce its factor (e.g., 0.8).
// - For down motion, if the left actuator is too fast, reduce its factor (e.g., 0.8).
float calibrationFactorLeftUp = 1.0;    
float calibrationFactorRightUp = 1.0;   
float calibrationFactorLeftDown = 1.0;  
float calibrationFactorRightDown = 1.0; 

void setup() {
  Serial.begin(115200);

  // Set direction pins as outputs
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  
  // Set PWM pins as outputs
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);

  // Ensure actuators are stopped initially
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);
  
  Serial.println("Arduino ready.");
}

void loop() {
  // Check if there is serial data available
  if (Serial.available() > 0) {
    // Read input until newline (expecting a single signed number)
    String input = Serial.readStringUntil('\n');
    int netSpeed = input.toInt();
    int absSpeed = abs(netSpeed);
    int pwmLeft, pwmRight;
    
    if (netSpeed > 0) {
      // Up motion
      digitalWrite(IN1, HIGH);
      digitalWrite(IN2, LOW);
      digitalWrite(IN3, HIGH);
      digitalWrite(IN4, LOW);
      
      // Use calibration factors for up motion:
      pwmLeft = constrain(absSpeed * calibrationFactorLeftUp, 0, 255);
      pwmRight = constrain(absSpeed * calibrationFactorRightUp, 0, 255);
      
      analogWrite(ENA, pwmLeft);
      analogWrite(ENB, pwmRight);
      
      Serial.print("Moving up. Left PWM: ");
      Serial.print(pwmLeft);
      Serial.print(", Right PWM: ");
      Serial.println(pwmRight);
}
    else if (netSpeed < 0) {
      // Down motion
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, HIGH);
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, HIGH);
      
      // Use calibration factors for down motion:
      pwmLeft = constrain(absSpeed * calibrationFactorLeftDown, 0, 255);
      pwmRight = constrain(absSpeed * calibrationFactorRightDown, 0, 255);
      
      analogWrite(ENA, pwmLeft);
      analogWrite(ENB, pwmRight);
      
      Serial.print("Moving down. Left PWM: ");
      Serial.print(pwmLeft);
      Serial.print(", Right PWM: ");
      Serial.println(pwmRight);
    }
    else {
      // Stop the actuators
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, LOW);
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, LOW);
      analogWrite(ENA, 0);
      analogWrite(ENB, 0);
      
      Serial.println("Actuators stopped.");
    }
  }
}
