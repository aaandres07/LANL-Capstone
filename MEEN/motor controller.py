#!/usr/bin/env python3
import sys
import time
import serial
from evdev import InputDevice, list_devices, ecodes

# Find the Xbox controller device by scanning available input devices
devices = [InputDevice(path) for path in list_devices()]
xbox = None
for device in devices:
    if "Xbox" in device.name:
        xbox = device
        break

if xbox is None:
    print("Xbox controller not found. Please connect your controller and try again.")
    sys.exit(1)

# Configure the serial connection to the Arduino
SERIAL_PORT = '/dev/ttyACM0'  # Adjust as needed (/dev/ttyUSB0, etc.)
BAUDRATE = 9600
try:
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    time.sleep(2)  # Allow time for the Arduino to reset
except Exception as e:
    print(f"Error opening serial port: {e}")
    sys.exit(1)

print(f"Using Xbox controller on {xbox.path} ({xbox.name})")

# Retrieve axis information for the left stick vertical axis (ABS_Y)
abs_info = xbox.absinfo(ecodes.ABS_Y)
min_val = abs_info.min
max_val = abs_info.max
center = (min_val + max_val) // 2
max_disp = max(center - min_val, max_val - center)

print(f"ABS_Y info: min={min_val}, max={max_val}, center={center}, max_disp={max_disp}")

# Main loop: read events from the controller and send a scaled speed value over serial.
try:
    for event in xbox.read_loop():
        if event.type == ecodes.EV_ABS and event.code == ecodes.ABS_Y:
            raw_val = event.value
            # Calculate offset from center and scale it to -255 to 255
            normalized = raw_val - center
            speed = int((normalized / max_disp) * 255)
            # Apply a deadzone to avoid noise around the center
            if abs(speed) < 15:
                speed = 0
            # Send the speed value as a string ending with newline
            cmd = f"{speed}\n"
            ser.write(cmd.encode('utf-8'))
            print(f"Sent command: {speed}")
except KeyboardInterrupt:
    print("Exiting...")
finally:
    ser.close()
