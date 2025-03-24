#!/usr/bin/env python3
import serial
import time
from evdev import InputDevice, categorize, ecodes, list_devices

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

# Define deadzone threshold
DEADZONE = 20  # Any net value below this is treated as zero

# Initialize trigger values
left_trigger = 0   # Typically from ABS_Z (L2)
right_trigger = 0  # Typically from ABS_RZ (R2)

print("Starting full-speed control loop. Use L2 and R2 for reverse/forward.")

for event in controller.read_loop():
    if event.type == ecodes.EV_ABS:
        if event.code == ecodes.ABS_Z:
            left_trigger = event.value
        elif event.code == ecodes.ABS_RZ:
            right_trigger = event.value

        net_speed = right_trigger - left_trigger

        # Apply deadzone: if net_speed is small, treat it as zero
        if abs(net_speed) < DEADZONE:
            net_speed = 0

        # Instead of variable speed, send full speed commands:
        if net_speed > 0:
            command = "255\n"
        elif net_speed < 0:
            command = "-255\n"
        else:
            command = "0\n"

        print(f"Left Trigger: {left_trigger}, Right Trigger: {right_trigger}, Net: {net_speed}")
        print(f"Sending command: {command.strip()}")
        ser.write(command.encode('utf-8'))

        # Optionally, read Arduino feedback
        response = ser.readline().decode('utf-8').strip()
        if response:
            print(f"Arduino: {response}")
