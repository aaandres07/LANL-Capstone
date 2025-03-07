#!/usr/bin/env python3
import serial
import time
from evdev import InputDevice, categorize, ecodes, list_devices

# --- Find the PS5 Controller ---
devices = [InputDevice(path) for path in list_devices()]
controller = None
for device in devices:
    if 'Wireless Controller' in device.name or 'DualSense' in device.name:
        controller = device
        break

if controller is None:
    print("PS5 controller not found. Please connect your controller.")
    exit(1)

# --- Open Serial Port to Arduino ---
# Adjust '/dev/ttyACM0' as needed for your system.
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
time.sleep(2)  # Wait for Arduino to reset

# --- Helper Function to Map Joystick Value to PWM ---
def joystick_to_pwm(val):
    # Map -128..128 to 500..2500 (1500 is neutral)
    return int(1500 + (val / 128.0) * 1000)

# Initialize joystick values (assumed 0 at start => neutral)
left_val = 0
right_val = 0

print("Starting control loop. Move the joysticks to control the motors.")

# --- Main Loop: Read Controller Events and Send PWM Values ---
for event in controller.read_loop():
    if event.type == ecodes.EV_ABS:
        # Update left/right values based on axis events
        if event.code == ecodes.ABS_Y:
            left_val = event.value
        elif event.code == ecodes.ABS_RY:
            right_val = event.value

        # Map joystick values to PWM pulse widths
        left_pwm = joystick_to_pwm(left_val)
        right_pwm = joystick_to_pwm(right_val)

        # Create command string (e.g., "1500,1500\n")
        command = f"{left_pwm},{right_pwm}\n"
        ser.write(command.encode('utf-8'))
