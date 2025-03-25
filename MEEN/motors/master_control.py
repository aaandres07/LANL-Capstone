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

# --- Queues for dispatching events ---
drive_queue = queue.Queue()
actuator_queue = queue.Queue()
net_queue = queue.Queue()

# --- Drive Motors Processing ---
def drive_motors_thread(ser, drive_queue):
    left_val = 0
    right_val = 0

    def joystick_to_pwm(val):
        # Map joystick range (-32768 to 32767) to PWM (500 to 2500 µs)
        normalized_val = val / 32767.0
        return int(1500 + (normalized_val * 1000))
    
    while True:
        event = drive_queue.get()
        # Update left/right joystick values as events come in.
        if event.code == ecodes.ABS_Y:
            left_val = event.value
        elif event.code == ecodes.ABS_RY:
            right_val = event.value

        left_pwm = joystick_to_pwm(left_val)
        right_pwm = joystick_to_pwm(right_val)
        # Prepend "D:" to signal drive motor commands.
        command = f"D:{left_pwm},{right_pwm}\n"
        ser.write(command.encode('utf-8'))
        # Optionally, print Arduino feedback:
        response = ser.readline().decode('utf-8').strip()
        if response:
            print(f"Drive Motors - Arduino: {response}")
        drive_queue.task_done()

# --- Linear Actuator Processing ---
def actuator_thread_func(ser, actuator_queue):
    # For this example, we assume D-pad up/down controls actuator 1.
    # Mapping: D-pad up (hat_y == -1) → actuator forward; down (hat_y == 1) → actuator reverse.
    while True:
        event = actuator_queue.get()
        hat_y = event.value
        if hat_y == -1:
            command = "A:1\n"  # Actuator 1 forward
        elif hat_y == 1:
            command = "A:2\n"  # Actuator 1 reverse
        else:
            command = "A:0\n"  # Stop actuators
        ser.write(command.encode('utf-8'))
        response = ser.readline().decode('utf-8').strip()
        if response:
            print(f"Linear Actuators - Arduino: {response}")
        actuator_queue.task_done()

# --- Net Motors Processing ---
def net_motors_thread(ser, net_queue):
    left_trigger = 0
    right_trigger = 0
    DEADZONE = 20  # Trigger deadzone threshold
    while True:
        event = net_queue.get()
        if event.code == ecodes.ABS_Z:
            left_trigger = event.value
        elif event.code == ecodes.ABS_RZ:
            right_trigger = event.value

        net_speed = right_trigger - left_trigger
        if abs(net_speed) < DEADZONE:
            net_speed = 0

        # Instead of variable speed, send full speed commands:
        if net_speed > 0:
            command = "N:255\n"  # Full forward
        elif net_speed < 0:
            command = "N:-255\n"  # Full reverse
        else:
            command = "N:0\n"   # Stop motors
        ser.write(command.encode('utf-8'))
        response = ser.readline().decode('utf-8').strip()
        if response:
            print(f"Net Motors - Arduino: {response}")
        net_queue.task_done()

# --- Start Threads ---
drive_thread = threading.Thread(target=drive_motors_thread, args=(ser, drive_queue), daemon=True)
actuator_thread_obj = threading.Thread(target=actuator_thread_func, args=(ser, actuator_queue), daemon=True)
net_thread = threading.Thread(target=net_motors_thread, args=(ser, net_queue), daemon=True)

drive_thread.start()
actuator_thread_obj.start()
net_thread.start()

# --- Main Event Loop: Dispatch Controller Events ---
print("Starting master control loop. Listening for controller events...")

for event in controller.read_loop():
    if event.type == ecodes.EV_ABS:
        # Dispatch drive motor events: Left joystick (ABS_Y) and right joystick (ABS_RY)
        if event.code in (ecodes.ABS_Y, ecodes.ABS_RY):
            drive_queue.put(event)
        # Dispatch actuator events: D-pad vertical (ABS_HAT0Y)
        elif event.code == ecodes.ABS_HAT0Y:
            actuator_queue.put(event)
        # Dispatch net motor events: Triggers (ABS_Z for L2, ABS_RZ for R2)
        elif event.code in (ecodes.ABS_Z, ecodes.ABS_RZ):
            net_queue.put(event)
