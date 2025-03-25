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

# Initialize bumper states
left_bumper = 0  # BTN_TL
right_bumper = 0  # BTN_TR

print("Starting full-speed control loop. Use Left and Right Bumpers for reverse/forward.")

for event in controller.read_loop():
    if event.type == ecodes.EV_KEY:
        if event.code == ecodes.BTN_TL:
            left_bumper = event.value
        elif event.code == ecodes.BTN_TR:
            right_bumper = event.value

        # Determine net command based on bumper states:
        if right_bumper and not left_bumper:
            command = "255\n"  # Full forward
        elif left_bumper and not right_bumper:
            command = "-255\n"  # Full reverse
        else:
            command = "0\n"     # Stop

        print(f"Left Bumper: {left_bumper}, Right Bumper: {right_bumper}")
        print(f"Sending command: {command.strip()}")
        ser.write(command.encode('utf-8'))

        # Optionally, read Arduino feedback
        response = ser.readline().decode('utf-8').strip()
        if response:
            print(f"Arduino: {response}")
