# this will print out the devices connected 
import evdev

devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
for device in devices:
    print(device.path, device.name)
