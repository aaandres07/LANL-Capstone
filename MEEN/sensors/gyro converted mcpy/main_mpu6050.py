#!/usr/bin/env python3
"""
Example usage of the MPU6050 driver on Raspberry Pi 5.
"""

from imu import MPU6050, I2CAdapter
import time
from gpiozero import LED  # For controlling an LED on a Raspberry Pi.

# Initialize an LED (assumed to be connected to GPIO 17)
led = LED(17)
led.on()

# Create an I2C adapter using bus 1 (common for Raspberry Pi)
i2c = I2CAdapter(bus_number=1)

# Instantiate the MPU6050 using the I2C adapter.
imu = MPU6050(i2c)

while True:
    try:
        # Access accelerometer and gyroscope data via the Vector3d properties.
        ax = round(imu.accel.x, 2)
        ay = round(imu.accel.y, 2)
        az = round(imu.accel.z, 2)
        gx = round(imu.gyro.x)
        gy = round(imu.gyro.y)
        gz = round(imu.gyro.z)
        tem = round(imu.temperature, 2)
    except Exception as e:
        print("Error reading IMU:", e)
        time.sleep(0.2)
        continue

    # Print sensor data.
    print("ax", ax, "\t", "ay", ay, "\t", "az", az, "\t",
          "gx", gx, "\t", "gy", gy, "\t", "gz", gz, "\t",
          "Temperature", tem, "        ", end="\r")
    time.sleep(0.2)
