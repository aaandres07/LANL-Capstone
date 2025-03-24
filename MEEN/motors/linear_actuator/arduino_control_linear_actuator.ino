// Define motor driver pin assignments for dual linear actuators
#define IN1 2  // Actuator 1 direction pin 1
#define IN2 3  // Actuator 1 direction pin 2
#define IN3 4  // Actuator 2 direction pin 1
#define IN4 5  // Actuator 2 direction pin 2
#define ENA 6  // Actuator 1 PWM (speed)
#define ENB 7  // Actuator 2 PWM (speed)

// Calibration factors for each actuator
// Adjust these factors based on testing until both actuators move at the same rate.
float calibrationFactor1 = 0.5;  // Actuator 1 (e.g., left)
float calibrationFactor2 = 1.0;  // Actuator 2 (e.g., right)

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
    
    // Calculate the PWM values using calibration factors
    int pwmVal1 = constrain(abs(netSpeed) * calibrationFactor1, 0, 255);
    int pwmVal2 = constrain(abs(netSpeed) * calibrationFactor2, 0, 255);
    
    if (netSpeed > 0) {
      // Both actuators move in one direction (e.g., up)
      digitalWrite(IN1, HIGH);
      digitalWrite(IN2, LOW);
      digitalWrite(IN3, HIGH);
      digitalWrite(IN4, LOW);
      
      analogWrite(ENA, pwmVal1);
      analogWrite(ENB, pwmVal2);
      
      Serial.print("Moving up. PWM values: ");
      Serial.print(pwmVal1);
      Serial.print(" / ");
      Serial.println(pwmVal2);
    }
    else if (netSpeed < 0) {
      // Both actuators move in the opposite direction (e.g., down)
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, HIGH);
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, HIGH);
      
      analogWrite(ENA, pwmVal1);
      analogWrite(ENB, pwmVal2);
      
      Serial.print("Moving down. PWM values: ");
      Serial.print(pwmVal1);
      Serial.print(" / ");
      Serial.println(pwmVal2);
    }
    else {
      // Stop both actuators
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
