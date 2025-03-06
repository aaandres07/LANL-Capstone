from gpiozero import Motor
from time import sleep

# Test Motor 1
motor1 = Motor(forward=17, backward=18)  # GPIO 17 and 18
motor2 = Motor(forward=22, backward=23)  # GPIO 22 and 23

try:
    print("Motor 1 forward")
    motor1.forward()
    sleep(2)
    print("Motor 1 backward")
    motor1.backward()
    sleep(2)
    print("Stopping Motor 1")
    motor1.stop()

    print("Motor 2 forward")
    motor2.forward()
    sleep(2)
    print("Motor 2 backward")
    motor2.backward()
    sleep(2)
    print("Stopping Motor 2")
    motor2.stop()

finally:
    motor1.stop()
    motor2.stop()
