#!/usr/bin/env python3
import threading
import serial
import time
from evdev import InputDevice, list_devices, ecodes
from dataclasses import dataclass

@dataclass
class ArduinoCommand:
    drive_left: int = 1500
    drive_right: int = 1500
    net_speed: int = 0
    actuator_cmd: int = 0

    def to_serial(self) -> str:
        return f"D:{self.drive_left},{self.drive_right};N:{self.net_speed};A:{self.actuator_cmd}\n"

# --- Shared Serial Interface ---
try:
    ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
    time.sleep(2)
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

# --- Command State ---
shared_cmd = ArduinoCommand()

# --- Drive Motors Thread ---
def drive_thread():
    def joystick_to_pwm(val):
        normalized = val / 32767.0
        return int(1500 + (normalized * 1000))

    while True:
        event = controller.read_one()
        if event and event.type == ecodes.EV_ABS:
            if event.code == ecodes.ABS_Y:
                shared_cmd.drive_left = joystick_to_pwm(event.value)
            elif event.code == ecodes.ABS_RY:
                shared_cmd.drive_right = joystick_to_pwm(event.value)
        time.sleep(0.01)

# --- Net Motors Thread ---
def net_thread():
    left_bumper = 0
    right_bumper = 0
    while True:
        event = controller.read_one()
        if event and event.type == ecodes.EV_KEY:
            if event.code == ecodes.BTN_TL:
                left_bumper = event.value
            elif event.code == ecodes.BTN_TR:
                right_bumper = event.value

            if right_bumper and not left_bumper:
                shared_cmd.net_speed = 255
            elif left_bumper and not right_bumper:
                shared_cmd.net_speed = -255
            else:
                shared_cmd.net_speed = 0
        time.sleep(0.01)

# --- Actuator Thread ---
def actuator_thread():
    while True:
        event = controller.read_one()
        if event and event.type == ecodes.EV_ABS and event.code == ecodes.ABS_HAT0Y:
            if event.value == -1:
                shared_cmd.actuator_cmd = 1
            elif event.value == 1:
                shared_cmd.actuator_cmd = 2
            else:
                shared_cmd.actuator_cmd = 0
        time.sleep(0.01)

# --- Serial Sender Thread ---
def serial_sender():
    while True:
        cmd = shared_cmd.to_serial()
        ser.write(cmd.encode('utf-8'))
        response = ser.readline().decode('utf-8').strip()
        if response:
            print(f"Arduino: {response}")
        time.sleep(0.05)

# --- Start Threads ---
threading.Thread(target=drive_thread, daemon=True).start()
threading.Thread(target=net_thread, daemon=True).start()
threading.Thread(target=actuator_thread, daemon=True).start()
threading.Thread(target=serial_sender, daemon=True).start()

print("Master control running. Press buttons or joysticks to control motors.")

# Keep alive
while True:
    time.sleep(1)
