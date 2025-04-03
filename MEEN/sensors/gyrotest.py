import smbus
import time
import math

# MPU-6050 Registers
MPU6050_ADDR   = 0x68   # Device address
PWR_MGMT_1     = 0x6B   # Power management register
ACCEL_XOUT_H   = 0x3B   # Starting register for accelerometer data

# Initialize I2C bus (bus 1 is typical on Raspberry Pi)
bus = smbus.SMBus(1)

# Wake up MPU-6050 (it starts in sleep mode)
bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0)

def read_raw_data(addr):
    """
    Reads two consecutive bytes from the given register address,
    combines them into a signed 16-bit integer.
    """
    high = bus.read_byte_data(MPU6050_ADDR, addr)
    low = bus.read_byte_data(MPU6050_ADDR, addr + 1)
    value = (high << 8) | low
    # Convert to signed value (2's complement)
    if value > 32767:
        value -= 65536
    return value

def get_angles():
    """
    Reads raw accelerometer data, scales it, and computes the
    roll and pitch angles.
    """
    # Read raw accelerometer data
    acc_x = read_raw_data(ACCEL_XOUT_H)
    acc_y = read_raw_data(ACCEL_XOUT_H + 2)
    acc_z = read_raw_data(ACCEL_XOUT_H + 4)

    # Convert raw data to 'g' values (±2g sensitivity: 16384 LSB/g)
    Ax = acc_x / 16384.0
    Ay = acc_y / 16384.0
    Az = acc_z / 16384.0

    # Calculate roll and pitch angles in degrees
    roll  = math.degrees(math.atan2(Ay, Az))
    pitch = math.degrees(math.atan2(-Ax, math.sqrt(Ay * Ay + Az * Az)))

    return roll, pitch

def calibrate(num_samples=100, delay=0.05):
    """
    Calibrates the sensor by averaging a number of readings while
    the sensor is in a known flat position. Returns roll and pitch offsets.
    """
    roll_sum = 0.0
    pitch_sum = 0.0
    print("Calibrating... keep the sensor flat!")
    for i in range(num_samples):
        roll, pitch = get_angles()
        roll_sum += roll
        pitch_sum += pitch
        time.sleep(delay)

    roll_offset = roll_sum / num_samples
    pitch_offset = pitch_sum / num_samples
    print("Calibration complete.")
    print("Roll offset: {:.2f}°, Pitch offset: {:.2f}°".format(roll_offset, pitch_offset))
    return roll_offset, pitch_offset

if __name__ == '__main__':
    # Perform calibration with the sensor flat
    roll_offset, pitch_offset = calibrate()

    try:
        while True:
            roll, pitch = get_angles()
            # Adjust by subtracting the offset values
            adjusted_roll = roll - roll_offset
            adjusted_pitch = pitch - pitch_offset
            print("Adjusted Roll: {:.2f}°, Adjusted Pitch: {:.2f}°".format(adjusted_roll, adjusted_pitch))
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nProgram stopped by user")                                                                         2,1           Top
