import time
import serial
from evdev import InputDevice, categorize, ecodes

# --------------------------------------------------
# 1. Configure your serial connection to Arduino:
# --------------------------------------------------
# Replace '/dev/ttyACM0' with the correct port for your Arduino.
arduino_port = '/dev/ttyACM0'
baud_rate = 115200

try:
    ser = serial.Serial(arduino_port, baud_rate, timeout=1)
    print(f"Connected to Arduino on {arduino_port}")
except Exception as e:
    print(f"Error opening serial port: {e}")
    exit(1)

# --------------------------------------------------
# 2. Identify the PS5 controller device path
# --------------------------------------------------
# Update '/dev/input/eventX' to match your system.
controller_path = '/dev/input/event8'
try:
    gamepad = InputDevice(controller_path)
    print(f"Listening to {gamepad.name} at {controller_path}")
except OSError:
    print(f"Could not find a device at {controller_path}. Update the path!")
    exit(1)

# --------------------------------------------------
# 3. Joystick data and scaling
# --------------------------------------------------
# Your PS5 controller now reports -128..128 on each axis.
# We’ll map that to -255..255 for motor speed.

def scale_joystick_value(value, in_min=-128, in_max=128, out_min=-255, out_max=255):
    # Scale from one range to another
    # e.g. -128..128 -> -255..255
    return int((value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

left_y = 0
right_y = 0

# --------------------------------------------------
# 4. Main loop: read events, parse joystick positions, send to Arduino
# --------------------------------------------------
try:
    for event in gamepad.read_loop():
        # We only care about absolute axis events
        if event.type == ecodes.EV_ABS:
            if event.code == ecodes.ABS_Y:    # Left stick Y
                left_y = scale_joystick_value(event.value)
            elif event.code == ecodes.ABS_RY: # Right stick Y
                right_y = scale_joystick_value(event.value)

            # --------------------------------------------------
            # SEND UPDATED VALUES TO ARDUINO
            # --------------------------------------------------
            # Format a simple comma-separated string: "LY:<val>,RY:<val>\n"
            cmd = f"LY:{left_y},RY:{right_y}\n"
            ser.write(cmd.encode('utf-8'))

except KeyboardInterrupt:
    print("Exiting program...")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if ser.is_open:
        ser.close()
