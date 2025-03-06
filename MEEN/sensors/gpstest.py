import serial

# Beitian TTL GNSS Receiver For Track GPS Module Ninth And Tenth Generation GPS Module Series
# we specifically have BE 180 
serial_port = "/dev/serial0"
baud_rate = 38400 # spec from website

try:
    with serial.Serial(serial_port, baud_rate, timeout=1) as ser:
        while True:
            # Read a line from the GPS
            line = ser.readline().decode('ascii', errors='replace').strip()
            if line:
                print(line)
except serial.SerialException as e:
    print("Error opening serial port: ", e)
