import evdev
from evdev import InputDevice, categorize, ecodes
from gpiozero import Motor
from time import sleep

# Initialize the motors using gpiozero
motor1 = Motor(forward=17, backward=18)  # Motor 1 on GPIO pins 17 and 18
motor2 = Motor(forward=22, backward=23)  # Motor 2 on GPIO pins 22 and 23

# Deadzone threshold for analog sticks (to avoid jitter)
DEADZONE = 10

# Controller setup (ensure the correct event device is used)
ps5_controller = InputDevice('/dev/input/event0')  # Find your controller device path
print(f"Connected to {ps5_controller.name} at {ps5_controller.path}")

# Motor control functions
def control_motor(motor, value):
    """Control motor based on joystick value."""
    if value < (128 - DEADZONE):  # Forward
        motor.forward()
    elif value > (128 + DEADZONE):  # Backward
        motor.backward()
    else:  # Stop (value within deadzone)
        motor.stop()

# Main loop to read inputs from the controller
try:
    for event in ps5_controller.read_loop():
        if event.type == ecodes.EV_ABS:
            absevent = categorize(event)

            # Left stick vertical axis (ABS_Y) to control motor 1
            if absevent.event.code == ecodes.ABS_Y:
                control_motor(motor1, absevent.event.value)

            # Right stick vertical axis (ABS_RY) to control motor 2
            if absevent.event.code == ecodes.ABS_RY:
                control_motor(motor2, absevent.event.value)

except KeyboardInterrupt:
    print("Stopping motors and exiting...")
    motor1.stop()
    motor2.stop()
