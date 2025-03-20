/* 
   Arduino code to control a 2-channel motor driver
   with the following pin connections:

   Motor 1: 
     IN1 = D2
     IN2 = D3
     ENA1 = D6 (PWM)
   Motor 2:
     IN3 = D4
     IN4 = D5
     ENA2 = D7 (PWM)

   Make sure D6 and D7 are PWM-capable pins on the Mega (they are on most Arduino boards).
*/

int in1 = 2;
int in2 = 3;
int in3 = 4;
int in4 = 5;
int ena1 = 6; // PWM for Motor 1
int ena2 = 7; // PWM for Motor 2

void setup() {
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);
  pinMode(ena1, OUTPUT);
  pinMode(ena2, OUTPUT);

  // Initialize serial communication
  Serial.begin(115200);
  // A short delay to ensure serial is up
  delay(1000);
}

void loop() {
  // We'll read the net speed from the Pi
  static String inputString = "";
  
  // Check if there is data in the serial buffer
  while (Serial.available() > 0) {
    char c = Serial.read();
    
    // If we get a newline, that means we have a full line to parse
    if (c == '\n') {
      int speedVal = inputString.toInt();  // Convert to integer
      inputString = "";                    // Clear the string for next time
      
      // We clamp speedVal to 0..255 for forward or 0..-255 for reverse
      // But let's handle direction logic:
      if (speedVal > 0) {
        // Forward
        digitalWrite(in1, HIGH);
        digitalWrite(in2, LOW);
        digitalWrite(in3, HIGH);
        digitalWrite(in4, LOW);
      } else if (speedVal < 0) {
        // Reverse
        digitalWrite(in1, LOW);
        digitalWrite(in2, HIGH);
        digitalWrite(in3, LOW);
        digitalWrite(in4, HIGH);
      } else {
        // Stop
        digitalWrite(in1, LOW);
        digitalWrite(in2, LOW);
        digitalWrite(in3, LOW);
        digitalWrite(in4, LOW);
      }
      
      // Set the PWM speed (absolute value)
      int pwmVal = abs(speedVal);
      if (pwmVal > 255) {
        pwmVal = 255;
      }
      
      // Write the same PWM to both motor channels
      analogWrite(ena1, pwmVal);
      analogWrite(ena2, pwmVal);
    } 
    else {
      // Build up the string until newline
      inputString += c;
    }
  }
}
