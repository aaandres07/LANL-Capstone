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

print("Starting combined control loop.")

# -------- Variables for Drive Motors (ESC-controlled) --------
drive_left = 0
drive_right = 0

def joystick_to_pwm(val):
    normalized_val = val / 32767.0  # Joystick range normalized to -1.0 to 1.0
    return int(1500 + (normalized_val * 1000))  # Map to 500 - 2500 µs

# -------- Variables for Linear Actuators --------
# We'll use the D-pad vertical (ABS_HAT0Y) to control actuator commands:
#  - Assume: -1 (up) sends actuator forward (command 1)
#            1 (down) sends actuator reverse (command 2)
actuator_cmd = 0

# -------- Variables for Net Motors --------
left_trigger = 0
right_trigger = 0
net_motor_cmd = 0
DEADZONE = 20

for event in controller.read_loop():
    if event.type == ecodes.EV_ABS:
        # Drive Motors: Use ABS_Y (left stick vertical) and ABS_RY (right stick vertical)
        if event.code == ecodes.ABS_Y:
            drive_left = event.value
        elif event.code == ecodes.ABS_RY:
            drive_right = event.value

        # Linear Actuators: Use D-pad vertical (ABS_HAT0Y)
        elif event.code == ecodes.ABS_HAT0Y:
            hat_y = event.value  # Typically: -1 for up, 1 for down, 0 for neutral
            if hat_y == -1:
                actuator_cmd = 1  # Actuator forward
            elif hat_y == 1:
                actuator_cmd = 2  # Actuator reverse
            else:
                actuator_cmd = 0  # Stop

        # Net Motors: Use triggers (ABS_Z for L2 and ABS_RZ for R2)
        elif event.code == ecodes.ABS_Z:
            left_trigger = event.value
        elif event.code == ecodes.ABS_RZ:
            right_trigger = event.value
            net_speed = right_trigger - left_trigger
            if abs(net_speed) < DEADZONE:
                net_motor_cmd = 0
            elif net_speed > 0:
                net_motor_cmd = 255
            else:
                net_motor_cmd = -255

        # Map joystick values to PWM for drive motors
        drive_left_pwm = joystick_to_pwm(drive_left)
        drive_right_pwm = joystick_to_pwm(drive_right)

        # Send commands for each subsystem:
        drive_command = f"D:{drive_left_pwm},{drive_right_pwm}\n"
        ser.write(drive_command.encode('utf-8'))

        actuator_command_str = f"A:{actuator_cmd}\n"
        ser.write(actuator_command_str.encode('utf-8'))

        net_motor_command_str = f"N:{net_motor_cmd}\n"
        ser.write(net_motor_command_str.encode('utf-8'))

        # Optionally, print debugging info and any Arduino responses
        print(f"Drive PWM: {drive_left_pwm}, {drive_right_pwm}")
        print(f"Actuator command: {actuator_cmd}")
        print(f"Net motor command: {net_motor_cmd}")
        while ser.in_waiting:
            response = ser.readline().decode('utf-8').strip()
            if response:
                print("Arduino:", response)
