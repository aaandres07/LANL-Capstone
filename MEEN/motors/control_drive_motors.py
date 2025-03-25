#!/usr/bin/env python3
import serial
import time
from evdev import InputDevice, categorize, ecodes, list_devices

# Find the PS5 Controller --> usually event8 and is a trusted device
# mac address of PS5 - 10:18:49:66:9E:E7
devices = [InputDevice(path) for path in list_devices()]
controller = None
for device in devices:
    if 'Wireless Controller' in device.name or 'DualSense' in device.name:
        controller = device
        break

if controller is None:
    print("PS5 controller not found. Please connect your controller.")
    exit(1)

# Open Serial Port to Arduino 
# Arduino is usually ACMO when we do USBB-USBA connection
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
time.sleep(2)  # Wait for Arduino to reset

# Helper Function to Map Joystick Value to PWM
def joystick_to_pwm(val):
    # PS5 joystick range: -32767 to 32767 (should be -128 to 128)
    normalized_val = val / 32767  # Normalize to -1.0 to 1.0
    return int(1500 + (normalized_val * 1000))  # Map to 500 - 2500µs

# Initialize joystick values (neutral) 
# NEUTRAL is 1500 NOT 2500!!!
left_val = 0
right_val = 0

# Logging --> wait
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

        # Print Debugging Information
        print(f"Raw Joystick: Left={left_val}, Right={right_val}")
        print(f"Mapped PWM: Left={left_pwm}, Right={right_pwm}")

        # Create command string (e.g., "1500,1500\n")
        command = f"{left_pwm},{right_pwm}\n" # this is how arduino expects __ new line between each
        print(f"Sending: {command.strip()}")  # Print the actual command sent
        ser.write(command.encode('utf-8'))

        # Read Arduino response
        response = ser.readline().decode('utf-8').strip()
        if response:
            print(f"Arduino: {response}")  # Print what Arduino sends back
