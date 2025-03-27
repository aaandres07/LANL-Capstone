'''Known good working code for making a master python file'''
#!/usr/bin/env python3
import threading
import queue
import serial
import time
from evdev import InputDevice, list_devices, ecodes

# --- Shared Serial Interface ---
try:
    ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
    time.sleep(2)  # Allow time for Arduino to reset
except Exception as e:
    print("Error opening serial port:", e)
    exit(1)

# --- Locate the PS5 Controller ---
devices = [InputDevice(path) for path in list_devices()]
controller = None
for device in devices:
    if 'Wireless Controller' in device.name or 'DualSense' in device.name:
        controller = device
        break

if controller is None:
    print("PS5 controller not found. Please connect your controller.")
    exit(1)

'''
left_bumper is BTN_TL
right_bumper is BTN_TR

D-pad is hat_y

left_joystick is ABS_Y
right_joystick is ABS_RY

'''