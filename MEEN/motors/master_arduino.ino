#include <Servo.h>

Servo leftMotor;
Servo rightMotor;

// Net Motors
const int IN1 = 22, IN2 = 23, IN3 = 24, IN4 = 25;  
const int ENA = 6, ENB = 7;

// Linear Actuators
const int ACT1_IN1 = 28, ACT1_IN2 = 29, ACT2_IN1 = 30, ACT2_IN2 = 31;
const int ACT_ENA = 4, ACT_ENB = 5;

String input = "";

void setup() {
  Serial.begin(115200);

  // Drive motors
  leftMotor.attach(9);
  rightMotor.attach(10);
  leftMotor.writeMicroseconds(1500);
  rightMotor.writeMicroseconds(1500);

  // Net motors
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT); pinMode(ENB, OUTPUT);

  // Actuators
  pinMode(ACT1_IN1, OUTPUT); pinMode(ACT1_IN2, OUTPUT);
  pinMode(ACT2_IN1, OUTPUT); pinMode(ACT2_IN2, OUTPUT);
  pinMode(ACT_ENA, OUTPUT); pinMode(ACT_ENB, OUTPUT);
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      parseCommand(input);
      input = "";
    } else {
      input += c;
    }
  }
}

void parseCommand(String cmd) {
  int d = cmd.indexOf("D:");
  int n = cmd.indexOf("N:");
  int a = cmd.indexOf("A:");

  if (d != -1) {
    int comma = cmd.indexOf(",", d);
    int semi = cmd.indexOf(";", d);
    int left = cmd.substring(d + 2, comma).toInt();
    int right = cmd.substring(comma + 1, semi).toInt();
    controlDrive(left, right);
  }

  if (n != -1) {
    int semi = cmd.indexOf(";", n);
    if (semi == -1) semi = cmd.length();
    int net = cmd.substring(n + 2, semi).toInt();
    controlNet(net);
  }

  if (a != -1) {
    int actuator = cmd.substring(a + 2).toInt();
    controlActuator(actuator);
  }

  Serial.println("Parsed command.");
}

void controlDrive(int l, int r) {
  l = constrain(l, 1000, 2000);
  r = constrain(r, 1000, 2000);
  leftMotor.writeMicroseconds(l);
  rightMotor.writeMicroseconds(r);
}

void controlNet(int speed) {
  if (speed > 0) {
    digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
    digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
    analogWrite(ENA, 255); analogWrite(ENB, 255);
  } else if (speed < 0) {
    digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);
    digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);
    analogWrite(ENA, 255); analogWrite(ENB, 255);
  } else {
    digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
    digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
    analogWrite(ENA, 0); analogWrite(ENB, 0);
  }
}

void controlActuator(int cmd) {
  if (cmd == 1) {
    digitalWrite(ACT1_IN1, HIGH); digitalWrite(ACT1_IN2, LOW);
    digitalWrite(ACT2_IN1, HIGH); digitalWrite(ACT2_IN2, LOW);
    analogWrite(ACT_ENA, 255); analogWrite(ACT_ENB, 255);
  } else if (cmd == 2) {
    digitalWrite(ACT1_IN1, LOW); digitalWrite(ACT1_IN2, HIGH);
    digitalWrite(ACT2_IN1, LOW); digitalWrite(ACT2_IN2, HIGH);
    analogWrite(ACT_ENA, 255); analogWrite(ACT_ENB, 255);
  } else {
    digitalWrite(ACT1_IN1, LOW); digitalWrite(ACT1_IN2, LOW);
    digitalWrite(ACT2_IN1, LOW); digitalWrite(ACT2_IN2, LOW);
    analogWrite(ACT_ENA, 0); analogWrite(ACT_ENB, 0);
  }
}
