import time
import serial
from evdev import InputDevice, categorize, ecodes

# Setup serial connection
arduino_port = '/dev/ttyACM0'
baud_rate = 115200

try:
    ser = serial.Serial(arduino_port, baud_rate, timeout=1)
    print(f"Connected to Arduino on {arduino_port}")
except Exception as e:
    print(f"Error opening serial port: {e}")
    exit(1)

# Setup PS5 controller
controller_path = '/dev/input/event8'
try:
    gamepad = InputDevice(controller_path)
    print(f"Listening to {gamepad.name} at {controller_path}")
except OSError:
    print(f"Could not find a device at {controller_path}. Update the path!")
    exit(1)

# Function to scale joystick values (-128 to 128) to PWM (1000 to 2000)
def scale_joystick_value(value, in_min=-128, in_max=128, out_min=1000, out_max=2000):
    return int((value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

left_y = 1500
right_y = 1500

# Main loop
try:
    for event in gamepad.read_loop():
        if event.type == ecodes.EV_ABS:
            if event.code == ecodes.ABS_Y:    # Left stick Y
                left_y = scale_joystick_value(event.value)
            elif event.code == ecodes.ABS_RY: # Right stick Y
                right_y = scale_joystick_value(event.value)

            # Send correctly formatted PWM values to Arduino
            cmd = f"{left_y},{right_y}\n"
            ser.write(cmd.encode('utf-8'))
            print(f"Sent: {cmd.strip()}")  # Debugging output

except KeyboardInterrupt:
    print("Exiting program...")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if ser.is_open:
        ser.close()
