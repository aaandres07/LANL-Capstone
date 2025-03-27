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

# --- Shared Serial Interface ---
try:
    ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
    time.sleep(2)
    print("[INFO] Serial connection to Arduino established.\n")
except Exception as e:
    print("[ERROR] Serial port failure:", e)
    exit(1)

# --- Locate the PS5 Controller ---
devices = [InputDevice(path) for path in list_devices()]
controller = None
for device in devices:
    if 'Wireless Controller' in device.name or 'DualSense' in device.name:
        controller = device
        break

if controller is None:
    print("[ERROR] PS5 controller not found. Please connect your controller.")
    exit(1)

print(f"[INFO] Connected to controller: {controller.name} ({controller.path})\n")

# --- Command State ---
shared_cmd = ArduinoCommand()
cmd_lock = threading.Lock()

# --- Helper Function ---
def apply_deadzone(value, threshold=0.1):
    normalized = value / 32767.0
    if abs(normalized) < threshold:
        return 1500
    return int(1500 + (normalized * 1000))

# --- Unified Input Event Handler Thread ---
def controller_event_loop():
    left_bumper = 0
    right_bumper = 0

    print("[INFO] Controller input thread running.\n")

    while True:
        try:
            events = controller.read()
            for event in events:
                with cmd_lock:
                    if event.type == ecodes.EV_ABS:
                        if event.code == ecodes.ABS_Y:
                            shared_cmd.drive_left = apply_deadzone(event.value)
                        elif event.code == ecodes.ABS_RY:
                            shared_cmd.drive_right = apply_deadzone(event.value)
                        elif event.code == ecodes.ABS_HAT0Y:
                            if event.value == -1:
                                shared_cmd.actuator_cmd = 1
                            elif event.value == 1:
                                shared_cmd.actuator_cmd = 2
                            else:
                                shared_cmd.actuator_cmd = 0

                    elif event.type == ecodes.EV_KEY:
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

        except BlockingIOError:
            pass  # no new events
        except Exception as e:
            print(f"[ERROR] Controller thread crashed: {e}")
        time.sleep(0.01)

# --- Serial Sender Thread ---
def serial_sender():
    print("[INFO] Serial sender thread running.\n")
    while True:
        with cmd_lock:
            if shared_cmd.changed():
                cmd = shared_cmd.to_serial()
                print(shared_cmd.describe())
                try:
                    ser.write(cmd.encode('utf-8'))
                    print(f"  ↪ Sent: {cmd.strip()}")
                    response = ser.readline().decode('utf-8').strip()
                    if response:
                        print(f"  ↩ Arduino: {response}\n")
                    else:
                        print("  ⚠ Warning: No response from Arduino\n")
                except Exception as e:
                    print(f"[ERROR] Serial communication failed: {e}")
        time.sleep(0.05)

# --- Start Threads ---
threading.Thread(target=controller_event_loop, daemon=True).start()
threading.Thread(target=serial_sender, daemon=True).start()

print("[INFO] Master control running. Use the PS5 controller to drive your bot!\n")

# Keep alive
while True:
    time.sleep(1)
