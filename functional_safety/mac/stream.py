import time
import serial
from serial.tools import list_ports

PORT_NAME = "/dev/cu.usbserial-1100" # todo, check to make sure is right.. js use ls /dev/cu.*
BAUD_RATE = 115200

# pretty much use serial usb-a port to receive any communication from the stm32.
# uses a TTL to usb-a, where ttl handles UART from the board's pins
def find_and_read_serial():
    print("Available ports:")
    for port in list_ports.comports():
        print(f"- {port.device}")

    try:
        # init serial connection
        # timeout of 1 means read operations will unblock after 1 sec of no data
        with serial.Serial(PORT_NAME, BAUD_RATE, timeout=1) as ser:
            
            time.sleep(2) # give hw some time

            # flush buffers
            ser.reset_input_buffer()

            print("Listening for incoming data.. Ctrl C to stop")
            while True:
                if ser.in_waiting > 0:
                    raw_data = ser.readline()

                    # decode_bytes into a readable string
                    decoded_string = raw_data = raw_data.decode('utf-8', errors='ignore').strip()

                    if decoded_string:
                        print(f"Received: {decoded_string}")

                        # TODO: should give to some other worked thread to then convert to a view
                        # pretty sure its important to not have a view convertor in the read loop, because it will slow things down/block

    except serial.SerialException as e:
        print(f"\nError opening or communicating with port {PORT_NAME}: {e}")
    except KeyboardInterrupt:
        print("\nExiting Script.")

# gives a visual representation of the Pi's camera, bounding boxes, thermal imaging, etc
def convert_to_view():
    # TODO: implement a program that converts the raw data from the UART serial debugging and then makes
    # Planning on giving tons of visualization functionality
    return