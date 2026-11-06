import cv2
import board
import busio
import adafruit_amg88xx
from gpiozero import InputDevice
from picamera2 import Picamera2
from constants import *
from raspi.constants import *
from raspi.utility import *
from gpiozero import OutputDevice, InputDevice

class Raspi():
    def __init__(self, pin_cfg):
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.ir_sensor = IRSensor(self.i2c)
        self.camera = Camera()
        self.pins = {}

        for dev_type in pin_cfg:
            for pin in dev_type:
                self.pins[f"{pin}"] = OutputDevice(pin) if dev_type == 0 else InputDevice(pin)

    def set_pin(self, pin_num, on=False):
        device = self.pins.get(pin_num)
        if on:
            device.on()
        else: 
            device.off()

    def read_pin(self, pin_num):
        device = self.pins.get(pin_num, pull_up=False)
        return device.is_active
    
# read camera, thermal sensor
class Camera():
    def __init__(self):
        self.picam2 = Picamera2()

        camera_config = self.picam2.create_video_configuration(
            main={"size": (int(STREAM_W), int(STREAM_H)), "format": "RGB888"}
        )

        self.picam2.configure(camera_config)
        try:
            res = self.picam2.start()
            print("Camera start returned with: ", res)
            return 1 if res == None else 0
        except Exception as e:
            print(f"Camera failed to start: {e}")

    def capture(self): return self.picam2.capture_array()

    def stop(self):
        self.picam2.stop()


class IRSensor():
    def __init__(self):
        i2c = busio.I2C(board.SCL, board.SDA)

        # init the AMG8833 sensor
        self.sensor = adafruit_amg88xx.AMG88XX(i2c)
        self.thermal_readings = [] # will get replaced as 8*8 reading.    

    def _update_thermal_frame(self):
        # udpates 8x8 list of temps in C
        self.thermal_readings = self.sensor.pixels

        # if needed you can run some reading manipulation here...

    def get_thermals_frame(self):
        self._update_thermal_frame()
        return self.thermal_readings