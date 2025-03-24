from gpiozero import PWMOutputDevice
from time import sleep

# Use GPIO pin 17. Set an initial 0 (off) and 1 kHz frequency for PWM
motor = PWMOutputDevice(pin=18, active_high=True, initial_value=0, frequency=1000)

try:
    while True:
        print("Motor ~50% speed")
        motor.value = 0.5  # 50% duty cycle
        sleep(2)

        print("Motor 100% speed")
        motor.value = 1.0  # 100% duty cycle
        sleep(2)

        print("Motor off")
        motor.value = 0.0  # 0% duty cycle (off)
        sleep(2)

except KeyboardInterrupt:
    pass
