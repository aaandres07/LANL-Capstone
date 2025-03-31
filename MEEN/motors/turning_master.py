#!/usr/bin/env python3
import threading
import serial
import time
from evdev import InputDevice, list_devices, ecodes
from dataclasses import dataclass, field

@dataclass
class ArduinoCommand:
    drive_left: int = 1500
    drive_right: int = 1500
    net_speed: int = 0
    actuator_cmd: int = 0
    _last_serial: str = field(default="", init=False, repr=False)

    def to_serial(self) -> str:
        return f"D:{self.drive_left},{self.drive_right};N:{self.net_speed};A:{self.actuator_cmd}\n"

    def changed(self) -> bool:
        curr = self.to_serial()
        if curr != self._last_serial:
            self._last_serial = curr
            return True
        return False

    def describe(self) -> str:
        left = 'deadzone' if self.drive_left == 1500 else str(self.drive_left)
        right = 'deadzone' if self.drive_right == 1500 else str(self.drive_right)
        net = 'deadzone' if self.net_speed == 0 else str(self.net_speed)
        actuator = 'deadzone' if self.actuator_cmd == 0 else str(self.actuator_cmd)
        return (
            f"[{time.strftime('%H:%M:%S')}] Raspberry Pi Command:\n"
            f"  Wheels: {left}, {right}\n"
            f"  Net: {net}\n"
            f"  Linear Actuators: {actuator}"
        )

# --- Serial setup ---
try:
    ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
    time.sleep(2)
    print("[INFO] Serial connection to Arduino established.\n")
except Exception as e:
    print("[ERROR] Serial port failure:", e)
    exit(1)

# --- Controller detection ---
devices = [InputDevice(path) for path in list_devices()]
controller = next((d for d in devices if 'Wireless Controller' in d.name or 'DualSense' in d.name), None)
if not controller:
    print("[ERROR] PS5 controller not found.")
    exit(1)

print(f"[INFO] Connected to controller: {controller.name} ({controller.path})\n")

shared_cmd = ArduinoCommand()
cmd_lock = threading.Lock()
tank_turn_mode = {"L3": False, "R3": False}

# --- Joystick mapping with deadzone ---
def map_joystick_to_pwm(value, center=128, deadzone=15, scale=4):
    offset = value - center
    if abs(offset) < deadzone:
        return 1500
    pwm = int(1500 + offset * scale)
    return max(1000, min(2000, pwm))

# --- Controller thread ---
def controller_event_loop():
    left_bumper = 0
    right_bumper = 0
    while True:
        try:
            events = controller.read()
            for event in events:
                with cmd_lock:
                    if event.type == ecodes.EV_ABS:
                        if event.code == ecodes.ABS_Y:
                            if tank_turn_mode["R3"]:
                                val = map_joystick_to_pwm(event.value)
                                shared_cmd.drive_left = val
                                shared_cmd.drive_right = 3000 - val
                            else:
                                shared_cmd.drive_left = map_joystick_to_pwm(event.value)
                        elif event.code == ecodes.ABS_RY:
                            if tank_turn_mode["L3"]:
                                val = map_joystick_to_pwm(event.value)
                                shared_cmd.drive_left = 3000 - val
                                shared_cmd.drive_right = val
                            else:
                                shared_cmd.drive_right = map_joystick_to_pwm(event.value)
                        elif event.code == ecodes.ABS_HAT0Y:
                            shared_cmd.actuator_cmd = 1 if event.value == -1 else 2 if event.value == 1 else 0

                    elif event.type == ecodes.EV_KEY:
                        if event.code == ecodes.BTN_TL:
                            left_bumper = event.value
                        elif event.code == ecodes.BTN_TR:
                            right_bumper = event.value
                        elif event.code == ecodes.BTN_THUMBL:
                            tank_turn_mode["L3"] = event.value == 1
                        elif event.code == ecodes.BTN_THUMBR:
                            tank_turn_mode["R3"] = event.value == 1

                        shared_cmd.net_speed = (
                            255 if right_bumper and not left_bumper else
                            -255 if left_bumper and not right_bumper else
                            0
                        )
        except BlockingIOError:
            pass
        except Exception as e:
            print(f"[Controller Error] {e}")
        time.sleep(0.01)

# --- Serial thread ---
def serial_sender():
    print("[INFO] Serial sender thread running.\n")
    while True:
        with cmd_lock:
            if shared_cmd.changed():
                cmd = shared_cmd.to_serial()
                print(shared_cmd.describe())
                try:
                    ser.reset_input_buffer()
                    ser.write(cmd.encode('utf-8'))
                    print(f"  ↪ Sent: {cmd.strip()}")
                    start = time.time()
                    response = ''
                    while True:
                        if ser.in_waiting:
                            response = ser.readline().decode('utf-8').strip()
                            break
                        if time.time() - start > 1:
                            response = "[Timeout waiting for Arduino]"
                            break
                        time.sleep(0.01)
                    print(f"  ↩ Arduino: {response}\n")
                except Exception as e:
                    print(f"[ERROR] Serial write failed: {e}")
        time.sleep(0.05)

# --- Launch threads ---
threading.Thread(target=controller_event_loop, daemon=True).start()
threading.Thread(target=serial_sender, daemon=True).start()

print("[INFO] Master control running. Use PS5 controller to operate.\n")

while True:
    time.sleep(1)
