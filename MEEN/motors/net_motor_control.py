#!/usr/bin/env python3
import serial
import asyncio
from evdev import InputDevice, ecodes

# --- Configuration ---
# Set the serial port where the Arduino is connected (update as needed)
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600

# Set the event device path for your PS5 controller (update this path accordingly)
CONTROLLER_DEVICE = '/dev/input/eventX'  # e.g., '/dev/input/event0'

# Create a serial connection to the Arduino
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

# Open the controller device
controller = InputDevice(CONTROLLER_DEVICE)

# Initialize trigger values (assuming range 0-255)
left_trigger = 0
right_trigger = 0

async def read_controller_events():
    global left_trigger, right_trigger
    async for event in controller.async_read_loop():
        if event.type == ecodes.EV_ABS:
            # Update trigger values based on event code.
            # (Some systems might report different event codes; adjust if necessary.)
            if event.code == ecodes.ABS_Z:
                left_trigger = event.value
            elif event.code == ecodes.ABS_RZ:
                right_trigger = event.value

            # Calculate net speed.
            # Positive value => forward, negative => reverse.
            net_speed = right_trigger - left_trigger

            # Build the command string. Here we use the format: S:<net_speed>
            command_str = f"S:{net_speed}\n"
            ser.write(command_str.encode('utf-8'))
            print("Sent command:", command_str.strip())

# Run the asynchronous event loop
if __name__ == "__main__":
    try:
        asyncio.run(read_controller_events())
    except KeyboardInterrupt:
        print("Exiting controller loop.")
