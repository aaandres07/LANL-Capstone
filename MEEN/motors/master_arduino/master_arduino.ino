#include <Servo.h>

Servo leftMotor;
Servo rightMotor;

const int IN1 = 22, IN2 = 23, IN3 = 24, IN4 = 25;  
const int ENA = 6, ENB = 7;

const int ACT1_IN1 = 28, ACT1_IN2 = 29, ACT2_IN1 = 30, ACT2_IN2 = 31;
const int ACT_ENA = 4, ACT_ENB = 5;

String input = "";
int lastLeft = 1500, lastRight = 1500, lastNet = 0, lastActuator = 0;
unsigned long lastHeartbeat = 0;

void controlDrive(int l, int r);
void controlNet(int speed);
void controlActuator(int cmd);
void parseCommand(String cmd);

void setup() {
  Serial.begin(115200);
  leftMotor.attach(9); rightMotor.attach(10);
  leftMotor.writeMicroseconds(1500); rightMotor.writeMicroseconds(1500);

  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT); pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT); pinMode(ENB, OUTPUT);

  pinMode(ACT1_IN1, OUTPUT); pinMode(ACT1_IN2, OUTPUT);
  pinMode(ACT2_IN1, OUTPUT); pinMode(ACT2_IN2, OUTPUT);
  pinMode(ACT_ENA, OUTPUT); pinMode(ACT_ENB, OUTPUT);
}

void loop() {
  if (millis() - lastHeartbeat >= 5000) {
    Serial.println("[Arduino] alive");
    lastHeartbeat = millis();
  }

  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      input.trim();
      if (input.length() > 0) {
        parseCommand(input);
      }
      input = "";
    } else if (isPrintable(c)) {
      input += c;
      if (input.length() > 100) {
        Serial.println("Warning: input too long, resetting");
        input = "";
      }
    }
  }
}

void parseCommand(String cmd) {
  Serial.println("Raw command: " + cmd);
  int d = cmd.indexOf("D:");
  int n = cmd.indexOf("N:");
  int a = cmd.indexOf("A:");

  if (d != -1) {
    int comma = cmd.indexOf(",", d);
    int semi = cmd.indexOf(";", d);
    if (comma != -1 && semi != -1) {
      int left = cmd.substring(d + 2, comma).toInt();
      int right = cmd.substring(comma + 1, semi).toInt();
      controlDrive(left, right);
      lastLeft = left;
      lastRight = right;
    }
  }

  if (n != -1) {
    int semi = cmd.indexOf(";", n);
    if (semi == -1) semi = cmd.length();
    int net = cmd.substring(n + 2, semi).toInt();
    controlNet(net);
    lastNet = net;
  }

  if (a != -1) {
    int actuator = cmd.substring(a + 2).toInt();
    controlActuator(actuator);
    lastActuator = actuator;
  }

  Serial.println("Arduino Command State:");
  Serial.print("  Wheels: "); Serial.print((lastLeft == 1500) ? "deadzone" : String(lastLeft));
  Serial.print(", "); Serial.println((lastRight == 1500) ? "deadzone" : String(lastRight));
  Serial.print("  Net: "); Serial.println((lastNet == 0) ? "deadzone" : String(lastNet));
  Serial.print("  Linear Actuators: "); Serial.println((lastActuator == 0) ? "deadzone" : String(lastActuator));
}

void controlDrive(int l, int r) {
  l = constrain(l, 1000, 2000); r = constrain(r, 1000, 2000);
  leftMotor.writeMicroseconds(l); rightMotor.writeMicroseconds(r);
}

void controlNet(int speed) { //net motors calibrated (4/01/25)
  if (speed > 0) {
    // extend net
    digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
    digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
    analogWrite(ENA, 255); analogWrite(ENB, 230); // Calibration Left (255); Right (240)
  } else if (speed < 0) {
    // retract net
    digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);
    digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);
    analogWrite(ENA, 255); analogWrite(ENB, 241); // Calibration Left (255); Right (240)
  } else {
    // stop
    digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
    digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
    analogWrite(ENA, 0); analogWrite(ENB, 0); 
  }
}

void controlActuator(int cmd) { //linear actuators calibrated (4/01/25)
  if (cmd == 1) {
    // up
    digitalWrite(ACT1_IN1, HIGH); digitalWrite(ACT1_IN2, LOW);
    digitalWrite(ACT2_IN1, HIGH); digitalWrite(ACT2_IN2, LOW);
    analogWrite(ACT_ENA, 255); analogWrite(ACT_ENB, 251); // Calibration Left (255); Right (251)
  } else if (cmd == 2) {
    // down
    digitalWrite(ACT1_IN1, LOW); digitalWrite(ACT1_IN2, HIGH);
    digitalWrite(ACT2_IN1, LOW); digitalWrite(ACT2_IN2, HIGH);
    analogWrite(ACT_ENA, 251); analogWrite(ACT_ENB, 255); // Calibration Left (251); Right (255)
  } else {
    // stop
    digitalWrite(ACT1_IN1, LOW); digitalWrite(ACT1_IN2, LOW);
    digitalWrite(ACT2_IN1, LOW); digitalWrite(ACT2_IN2, LOW);
    analogWrite(ACT_ENA, 0); analogWrite(ACT_ENB, 0);
  }
}
