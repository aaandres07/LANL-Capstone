#!/usr/bin/env python3
import serial
import time
from evdev import InputDevice, ecodes, categorize

# ---- Configuration ----
# Update the serial port to match your Arduino’s connection (e.g., /dev/ttyACM0 or /dev/ttyUSB0)
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200

# Update the event device for your PS5 controller.
# You can list devices with: "ls /dev/input/" and then use evdev to list capabilities.
PS5_DEVICE = '/dev/input/event8'  # <-- Replace X with your device number

# ---- Setup Serial and Controller Device ----
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
device = InputDevice(PS5_DEVICE)

# Give time for the serial port to initialize.
time.sleep(2)

# ---- Mapping Function ----
def map_joystick_to_pwm(val, dead_zone=300):
    """
    Maps a joystick value (assumed range -32768 to 32767) to a PWM pulse width (500 to 2500 µs).
    If the value is within a dead zone around 0, returns neutral (1500 µs).
    """
    if abs(val) < dead_zone:
        return 1500
    # Scale: full negative (-32768) -> 500 µs, full positive (32767) -> 2500 µs
    pwm = 1500 + int((val / 32767.0) * 1000)
    return max(500, min(2500, pwm))

# Initialize joystick values.
left_y = 0
right_y = 0

print("Starting control loop. Use CTRL+C to exit.")
# ---- Main Event Loop ----
for event in device.read_loop():
    # We only care about absolute axis events.
    if event.type == ecodes.EV_ABS:
        # The PS5 controller typically uses:
        #   ABS_Y for the left joystick vertical axis,
        #   ABS_RY for the right joystick vertical axis.
        if event.code == ecodes.ABS_Y:
            left_y = event.value
        elif event.code == ecodes.ABS_RY:
            right_y = event.value

        # Map the raw axis values to PWM pulse widths.
        left_pwm = map_joystick_to_pwm(left_y)
        right_pwm = map_joystick_to_pwm(right_y)

        # Create a command string: "left_pwm,right_pwm\n"
        command = f"{left_pwm},{right_pwm}\n"
        ser.write(command.encode('utf-8'))
        # Optionally, print the values for debugging.
        print(f"Sent: {command.strip()}")
