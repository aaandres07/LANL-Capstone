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

# Define baseline offsets and deadzone threshold
RIGHT_TRIGGER_BASELINE = 20   # Adjust this after testing your raw values
DEADZONE = 5

# Initialize trigger values
left_trigger = 0   # Typically from ABS_Z (L2)
right_trigger = 0  # Typically from ABS_RZ (R2)

print("Starting control loop. Use L2 and R2 to control the motors.")

for event in controller.read_loop():
    if event.type == ecodes.EV_ABS:
        if event.code == ecodes.ABS_Z:
            left_trigger = event.value
        elif event.code == ecodes.ABS_RZ:
            right_trigger = event.value

        # Adjust the right trigger value by subtracting its baseline
        effective_right = max(0, right_trigger - RIGHT_TRIGGER_BASELINE)
        net_speed = effective_right - left_trigger

        # Apply deadzone: if the net_speed is very small, treat it as zero
        if abs(net_speed) < DEADZONE:
            net_speed = 0

        # Debug prints for monitoring
        print(f"Left Trigger: {left_trigger}, Right Trigger: {right_trigger} (Effective: {effective_right}), Net Speed: {net_speed}")

        # Send the net speed over serial (as a string with newline)
        command = f"{net_speed}\n"
        print(f"Sending: {command.strip()}")
        ser.write(command.encode('utf-8'))

        # Optionally read Arduino feedback
        response = ser.readline().decode('utf-8').strip()
        if response:
            print(f"Arduino: {response}")
