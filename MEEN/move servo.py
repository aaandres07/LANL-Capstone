import serial
import time
from evdev import InputDevice, list_devices, ecodes, categorize

# Find the Xbox controller device
devices = [InputDevice('/dev/input/event8') for path in list_devices()]
controller = None
for dev in devices:
    if 'Xbox' in dev.name:
        controller = dev
        break

if controller is None:
    print("Xbox controller not found! Ensure it is connected.")
    exit()

print("Using Xbox controller:", controller.name)

# Initialize serial connection to the Arduino (adjust port if needed)
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Allow time for the Arduino to reset
except Exception as e:
    print(f"Failed to open serial port {SERIAL_PORT}: {e}")
    exit()

# Get ABS_X calibration info from the controller
abs_info = controller.absinfo(ecodes.ABS_X)
min_val = abs_info.min
max_val = abs_info.max

print(f"ABS_X range: {min_val} to {max_val}")

try:
    for event in controller.read_loop():
        if event.type == ecodes.EV_ABS:
            absevent = categorize(event)
            if absevent.event.code == ecodes.ABS_X:  # Left stick horizontal axis
                raw_value = absevent.event.value
                # Map raw_value from [min_val, max_val] to [0, 180]
                angle = int((raw_value - min_val) / (max_val - min_val) * 180)
                print(f"Raw ABS_X value: {raw_value} -> Mapped angle: {angle}")
                ser.write(f"{angle}\n".encode('utf-8'))
except KeyboardInterrupt:
    print("Exiting program.")
finally:
    ser.close()
