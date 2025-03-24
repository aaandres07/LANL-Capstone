#!/usr/bin/env python3
import serial
import time
from evdev import InputDevice, ecodes, list_devices

# Find the PS5 Controller (DualSense or Wireless Controller)
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
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
time.sleep(2)  # Wait for Arduino to reset

print("Starting control loop. Use D-pad Up/Down to control the motors.")

# Define a fixed PWM speed value for D-pad commands
FIXED_SPEED = 150

for event in controller.read_loop():
    # Check for D-pad vertical (up/down) events on ABS_HAT0Y
    if event.type == ecodes.EV_ABS and event.code == ecodes.ABS_HAT0Y:
        hat_y = event.value  # Typically: -1 for up, 0 for neutral, 1 for down

        # Map D-pad up to forward and down to reverse
        if hat_y == 1:
            net_speed = FIXED_SPEED   # Forward
        elif hat_y == -1:
            net_speed = -FIXED_SPEED  # Reverse
        else:
            net_speed = 0  # Stop when the D-pad is released

        print(f"D-pad state: {hat_y}, Net Speed: {net_speed}")
        command = f"{net_speed}\n"
        print(f"Sending: {command.strip()}")
        ser.write(command.encode('utf-8'))

        # Optionally read and print any Arduino feedback
        response = ser.readline().decode('utf-8').strip()
        if response:
            print(f"Arduino: {response}")
