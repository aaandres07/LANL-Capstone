import evdev
from evdev import InputDevice, categorize, ecodes
from gpiozero import Motor

motor1 = Motor(forward=17, backward=18)
motor2 = Motor(forward=22, backward=23)

DEADZONE = 10

ps5_controller = InputDevice('/dev/input/event0')
print(f"Connected to {ps5_controller.name} at {ps5_controller.path}")

def control_motor(motor, value):
    print(f"Joystick value: {value}")  # Add this to debug joystick input
    if value < (128 - DEADZONE):  # Forward
        print(f"Motor {motor} forward")
        motor.forward()
    elif value > (128 + DEADZONE):  # Backward
        print(f"Motor {motor} backward")
        motor.backward()
    else:  # Stop
        print(f"Motor {motor} stop")
        motor.stop()

try:
    for event in ps5_controller.read_loop():
        if event.type == ecodes.EV_ABS:
            absevent = categorize(event)

            if absevent.event.code == ecodes.ABS_Y:
                control_motor(motor1, absevent.event.value)

            if absevent.event.code == ecodes.ABS_RY:
                control_motor(motor2, absevent.event.value)

except KeyboardInterrupt:
    motor1.stop()
    motor2.stop()
