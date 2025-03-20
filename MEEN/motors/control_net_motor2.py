#!/usr/bin/env python3

import serial
import time
from evdev import InputDevice, categorize, ecodes

# Replace '/dev/input/eventX' with the correct device for your PS5 controller
GAMEPAD_DEVICE = '/dev/input/event8'

# Replace '/dev/ttyACM0' with whichever port your Arduino shows up on
ARDUINO_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200

def main():
    # Initialize serial to Arduino
    arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Give the serial port a moment to set up

    # Open the controller device
    gamepad = InputDevice(GAMEPAD_DEVICE)

    left_trigger_value = 0
    right_trigger_value = 0

    print("Listening for PS5 controller events...")
    try:
        for event in gamepad.read_loop():
            if event.type == ecodes.EV_ABS:
                # Check if it's the left trigger (L2)
                if event.code == ecodes.ABS_Z:
                    left_trigger_value = event.value
                # Check if it's the right trigger (R2)
                elif event.code == ecodes.ABS_RZ:
                    right_trigger_value = event.value

                # Each trigger typically ranges from 0 to 255
                # We'll treat the right trigger as forward, left trigger as reverse

                # net_speed = (right trigger) - (left trigger)
                net_speed = right_trigger_value - left_trigger_value

                # Clamp net_speed to -255..255 for safety
                if net_speed > 255:
                    net_speed = 255
                elif net_speed < -255:
                    net_speed = -255

                # Send the speed value as text, followed by newline
                # Arduino will parse it
                speed_str = f"{net_speed}\n"
                arduino.write(speed_str.encode())

    except KeyboardInterrupt:
        print("Exiting...")

    finally:
        arduino.close()

if __name__ == "__main__":
    main()
