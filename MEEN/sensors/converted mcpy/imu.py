#!/usr/bin/env python3
"""
MPU6050 Python3 driver for InvenSense MPU6050 sensor on Raspberry Pi.
Adapted from the MicroPython driver.
"""

import time
import smbus2

# I2CAdapter to mimic MicroPython's I2C interface using smbus2.
class I2CAdapter:
    def __init__(self, bus_number=1):
        self.bus = smbus2.SMBus(bus_number)

    def scan(self):
        devices = []
        # I2C addresses normally range from 0x03 to 0x77.
        for addr in range(0x03, 0x78):
            try:
                self.bus.read_byte(addr)
                devices.append(addr)
            except Exception:
                continue
        return devices

    def readfrom_mem_into(self, addr, memaddr, buf):
        length = len(buf)
        data = self.bus.read_i2c_block_data(addr, memaddr, length)
        for i in range(length):
            buf[i] = data[i]

    def writeto_mem(self, addr, memaddr, buf):
        # Assumes buf is a bytearray of length 1.
        self.bus.write_byte_data(addr, memaddr, buf[0])

# A simple Vector3d class implementation for Python3.
class Vector3d:
    def __init__(self, transposition=(0, 1, 2), scaling=(1, 1, 1), callback=None):
        self.transposition = transposition
        self.scaling = scaling
        self.callback = callback
        self._vector = [0, 0, 0]
        self._ivector = [0, 0, 0]

    @property
    def x(self):
        if self.callback:
            self.callback()
        return self._vector[self.transposition[0]]

    @property
    def y(self):
        if self.callback:
            self.callback()
        return self._vector[self.transposition[1]]

    @property
    def z(self):
        if self.callback:
            self.callback()
        return self._vector[self.transposition[2]]

    @property
    def xyz(self):
        return (self.x, self.y, self.z)

def bytes_toint(msb, lsb):
    """
    Convert two bytes (big endian) to a signed integer.
    """
    if not msb & 0x80:
        return (msb << 8) | lsb  # positive number
    return - (((msb ^ 0xFF) << 8) | (lsb ^ 0xFF) + 1)

class MPUException(OSError):
    """Exception raised for MPU devices errors."""
    pass

class MPU6050(object):
    _I2Cerror = "I2C failure when communicating with IMU"
    _mpu_addr = (104, 105)  # possible addresses for MPU6050 devices.
    _chip_id = 104

    def __init__(self, i2c_instance, device_addr=None, transposition=(0, 1, 2), scaling=(1, 1, 1)):
        # Create Vector3d objects for accelerometer and gyroscope.
        self._accel = Vector3d(transposition, scaling, self._accel_callback)
        self._gyro = Vector3d(transposition, scaling, self._gyro_callback)
        # Pre-allocated buffers.
        self.buf1 = bytearray(1)
        self.buf2 = bytearray(2)
        self.buf3 = bytearray(3)
        self.buf6 = bytearray(6)

        # Wait for power to settle.
        time.sleep(0.2)
        # Use the provided I2C instance.
        if isinstance(i2c_instance, str):
            raise ValueError("Invalid I2C instance for Python3. Provide an I2CAdapter instance.")
        elif hasattr(i2c_instance, 'readfrom_mem_into'):
            self._mpu_i2c = i2c_instance
        else:
            raise ValueError("Invalid I2C instance")

        # Determine device address.
        if device_addr is None:
            devices = set(self._mpu_i2c.scan())
            mpus = devices.intersection(set(self._mpu_addr))
            if len(mpus) == 0:
                raise MPUException("No MPU's detected")
            elif len(mpus) == 1:
                self.mpu_addr = mpus.pop()
            else:
                raise ValueError("Multiple MPU's detected: specify a device address")
        else:
            if device_addr not in (0, 1):
                raise ValueError("Device address must be 0 or 1")
            self.mpu_addr = self._mpu_addr[device_addr]

        # Verify communication by reading the chip id.
        _ = self.chip_id
        # Wake device and set parameters.
        self.wake()
        self.passthrough = True
        self.accel_range = 0
        self.gyro_range = 0

    def _read(self, buf, memaddr, addr):
        self._mpu_i2c.readfrom_mem_into(addr, memaddr, buf)

    def _write(self, data, memaddr, addr):
        self.buf1[0] = data
        self._mpu_i2c.writeto_mem(addr, memaddr, self.buf1)

    def wake(self):
        try:
            self._write(0x01, 0x6B, self.mpu_addr)  # Use best clock source.
        except Exception:
            raise MPUException(self._I2Cerror)
        return 'awake'

    def sleep(self):
        try:
            self._write(0x40, 0x6B, self.mpu_addr)
        except Exception:
            raise MPUException(self._I2Cerror)
        return 'asleep'

    @property
    def chip_id(self):
        try:
            self._read(self.buf1, 0x75, self.mpu_addr)
        except Exception:
            raise MPUException(self._I2Cerror)
        chip_id = int(self.buf1[0])
        if chip_id != self._chip_id:
            raise ValueError("Bad chip ID retrieved: MPU communication failure")
        return chip_id

    @property
    def sensors(self):
        return self._accel, self._gyro

    @property
    def temperature(self):
        try:
            self._read(self.buf2, 0x41, self.mpu_addr)
        except Exception:
            raise MPUException(self._I2Cerror)
        return bytes_toint(self.buf2[0], self.buf2[1]) / 340 + 35

    @property
    def passthrough(self):
        try:
            self._read(self.buf1, 0x37, self.mpu_addr)
            return (self.buf1[0] & 0x02) > 0
        except Exception:
            raise MPUException(self._I2Cerror)

    @passthrough.setter
    def passthrough(self, mode):
        if isinstance(mode, bool):
            val = 2 if mode else 0
            try:
                self._write(val, 0x37, self.mpu_addr)
                self._write(0x00, 0x6A, self.mpu_addr)
            except Exception:
                raise MPUException(self._I2Cerror)
        else:
            raise ValueError("pass either True or False")

    @property
    def sample_rate(self):
        try:
            self._read(self.buf1, 0x19, self.mpu_addr)
            return self.buf1[0]
        except Exception:
            raise MPUException(self._I2Cerror)

    @sample_rate.setter
    def sample_rate(self, rate):
        if not (0 <= rate <= 255):
            raise ValueError("Rate must be in range 0-255")
        try:
            self._write(rate, 0x19, self.mpu_addr)
        except Exception:
            raise MPUException(self._I2Cerror)

    @property
    def filter_range(self):
        try:
            self._read(self.buf1, 0x1A, self.mpu_addr)
            res = self.buf1[0] & 7
        except Exception:
            raise MPUException(self._I2Cerror)
        return res

    @filter_range.setter
    def filter_range(self, filt):
        if filt in range(7):
            try:
                self._write(filt, 0x1A, self.mpu_addr)
            except Exception:
                raise MPUException(self._I2Cerror)
        else:
            raise ValueError("Filter coefficient must be between 0 and 6")

    @property
    def accel_range(self):
        try:
            self._read(self.buf1, 0x1C, self.mpu_addr)
            ari = self.buf1[0] // 8
        except Exception:
            raise MPUException(self._I2Cerror)
        return ari

    @accel_range.setter
    def accel_range(self, accel_range):
        ar_bytes = (0x00, 0x08, 0x10, 0x18)
        if accel_range in range(len(ar_bytes)):
            try:
                self._write(ar_bytes[accel_range], 0x1C, self.mpu_addr)
            except Exception:
                raise MPUException(self._I2Cerror)
        else:
            raise ValueError("accel_range can only be 0, 1, 2 or 3")

    @property
    def gyro_range(self):
        try:
            self._read(self.buf1, 0x1B, self.mpu_addr)
            gri = self.buf1[0] // 8
        except Exception:
            raise MPUException(self._I2Cerror)
        return gri

    @gyro_range.setter
    def gyro_range(self, gyro_range):
        gr_bytes = (0x00, 0x08, 0x10, 0x18)
        if gyro_range in range(len(gr_bytes)):
            try:
                self._write(gr_bytes[gyro_range], 0x1B, self.mpu_addr)
            except Exception:
                raise MPUException(self._I2Cerror)
        else:
            raise ValueError("gyro_range can only be 0, 1, 2 or 3")

    @property
    def accel(self):
        return self._accel

    def _accel_callback(self):
        try:
            self._read(self.buf6, 0x3B, self.mpu_addr)
        except Exception:
            raise MPUException(self._I2Cerror)
        self._accel._ivector[0] = bytes_toint(self.buf6[0], self.buf6[1])
        self._accel._ivector[1] = bytes_toint(self.buf6[2], self.buf6[3])
        self._accel._ivector[2] = bytes_toint(self.buf6[4], self.buf6[5])
        scale = (16384, 8192, 4096, 2048)
        self._accel._vector[0] = self._accel._ivector[0] / scale[self.accel_range]
        self._accel._vector[1] = self._accel._ivector[1] / scale[self.accel_range]
        self._accel._vector[2] = self._accel._ivector[2] / scale[self.accel_range]

    def get_accel_irq(self):
        self._read(self.buf6, 0x3B, self.mpu_addr)
        self._accel._ivector[0] = bytes_toint(self.buf6[0], self.buf6[1])
        self._accel._ivector[1] = bytes_toint(self.buf6[2], self.buf6[3])
        self._accel._ivector[2] = bytes_toint(self.buf6[4], self.buf6[5])

    @property
    def gyro(self):
        return self._gyro

    def _gyro_callback(self):
        try:
            self._read(self.buf6, 0x43, self.mpu_addr)
        except Exception:
            raise MPUException(self._I2Cerror)
        self._gyro._ivector[0] = bytes_toint(self.buf6[0], self.buf6[1])
        self._gyro._ivector[1] = bytes_toint(self.buf6[2], self.buf6[3])
        self._gyro._ivector[2] = bytes_toint(self.buf6[4], self.buf6[5])
        scale = (131, 65.5, 32.8, 16.4)
        self._gyro._vector[0] = self._gyro._ivector[0] / scale[self.gyro_range]
        self._gyro._vector[1] = self._gyro._ivector[1] / scale[self.gyro_range]
        self._gyro._vector[2] = self._gyro._ivector[2] / scale[self.gyro_range]

    def get_gyro_irq(self):
        self._read(self.buf6, 0x43, self.mpu_addr)
        self._gyro._ivector[0] = bytes_toint(self.buf6[0], self.buf6[1])
        self._gyro._ivector[1] = bytes_toint(self.buf6[2], self.buf6[3])
        self._gyro._ivector[2] = bytes_toint(self.buf6[4], self.buf6[5])
