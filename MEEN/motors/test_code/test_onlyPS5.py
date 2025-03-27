import evdev
from evdev import InputDevice, categorize, ecodes

# Adjust the path based on your device
devices = [InputDevice(path) for path in list_devices()]
controller = next((d for d in devices if 'Wireless Controller' in d.name or 'DualSense' in d.name), None)
if not controller:
    print("[ERROR] PS5 controller not found.")
    exit(1)

# Optional: Define deadzone for ignoring tiny joystick movement
DEADZONE = 10

try:
    for event in ps5_controller.read_loop():
        if event.type == ecodes.EV_ABS:
            absevent = categorize(event)

            # Left stick vertical (Y)
            if absevent.event.code == ecodes.ABS_Y:
                value = absevent.event.value
                if abs(value - 128) > DEADZONE:
                    print(f"Left stick Y: {value}")

            # Right stick vertical (RY)
            elif absevent.event.code == ecodes.ABS_RY:
                value = absevent.event.value
                if abs(value - 128) > DEADZONE:
                    print(f"Right stick Y: {value}")

            # Optional: print other joystick directions
            elif absevent.event.code == ecodes.ABS_X:
                value = absevent.event.value
                if abs(value - 128) > DEADZONE:
                    print(f"Left stick X: {value}")

            elif absevent.event.code == ecodes.ABS_RX:
                value = absevent.event.value
                if abs(value - 128) > DEADZONE:
                    print(f"Right stick X: {value}")

except KeyboardInterrupt:
    print("Exiting joystick monitor")
