// Define motor driver pin assignments for dual linear actuators
#define IN1 2  // Left actuator direction pin 1
#define IN2 3  // Left actuator direction pin 2
#define IN3 4  // Right actuator direction pin 1
#define IN4 5  // Right actuator direction pin 2
#define ENA 6  // Left actuator PWM (speed)
#define ENB 7  // Right actuator PWM (speed)

// Calibration factors for each actuator for up and down directions
// (These remain as originally set, but their outputs will be swapped.)
float calibrationFactorLeftUp = 1.0;    
float calibrationFactorRightUp = 0.98;   
float calibrationFactorLeftDown = 0.99;  
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
    int pwmLeftOrig, pwmRightOrig;
    int pwmLeftFlipped, pwmRightFlipped;
    
    if (netSpeed > 0) {
      // Up motion originally:
      // Left actuator: IN1 HIGH, IN2 LOW; Right actuator: IN3 HIGH, IN4 LOW.
      // Now, we swap the PWM outputs.
      digitalWrite(IN1, HIGH);
      digitalWrite(IN2, LOW);
      digitalWrite(IN3, HIGH);
      digitalWrite(IN4, LOW);
      
      // Calculate original PWM values for up motion
      pwmLeftOrig = constrain(absSpeed * calibrationFactorLeftUp, 0, 255);
      pwmRightOrig = constrain(absSpeed * calibrationFactorRightUp, 0, 255);
      
      // Flip the channels:
      // Apply the right channel's PWM (pwmRightOrig) to the left actuator (ENA)
      // and the left channel's PWM (pwmLeftOrig) to the right actuator (ENB).
      pwmLeftFlipped = pwmRightOrig;
      pwmRightFlipped = pwmLeftOrig;
      
      analogWrite(ENA, pwmLeftFlipped);
      analogWrite(ENB, pwmRightFlipped);
      
      Serial.print("Moving up (flipped). Left PWM: ");
      Serial.print(pwmLeftFlipped);
      Serial.print(", Right PWM: ");
      Serial.println(pwmRightFlipped);
    }
    else if (netSpeed < 0) {
      // Down motion originally:
      // Left actuator: IN1 LOW, IN2 HIGH; Right actuator: IN3 LOW, IN4 HIGH.
      // Now, we swap the PWM outputs.
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, HIGH);
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, HIGH);
      
      // Calculate original PWM values for down motion
      pwmLeftOrig = constrain(absSpeed * calibrationFactorLeftDown, 0, 255);
      pwmRightOrig = constrain(absSpeed * calibrationFactorRightDown, 0, 255);
      
      // Swap the outputs:
      pwmLeftFlipped = pwmRightOrig;
      pwmRightFlipped = pwmLeftOrig;
      
      analogWrite(ENA, pwmLeftFlipped);
      analogWrite(ENB, pwmRightFlipped);
      
      Serial.print("Moving down (flipped). Left PWM: ");
      Serial.print(pwmLeftFlipped);
      Serial.print(", Right PWM: ");
      Serial.println(pwmRightFlipped);
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
