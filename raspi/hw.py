import cv2
import board
import busio
import adafruit_amg88xx
from picamera2 import Picamera2
from shared.constants import *
from shared.constants import *
from raspi.utility import *
from gpiozero import OutputDevice, InputDevice, Motor

class Raspi():
    def __init__(self, pin_cfg):
        self._i2c = busio.I2C(board.SCL, board.SDA)
        self._ir_sensor = IRSensor(self.i2c)
        self._camera = Camera()
        self._motor = Motor(forward=4, backward=5)

        # tracks pins for reading/write functionality
        self._pins = {} # 1 = output, 0 = input
        
    def set_pin(self, pin_num, on=False):
        if self._pins.get(pin_num, -1) != 1: # doesn't exist or isn't output
            self._pins[pin_num] = OutputDevice(pin_num)

        device = self.pins.get(pin_num)
        if on:
            device.on()
        else: 
            device.off()

    def read_pin(self, pin_num):
        if self._pins.get(pin_num, -1) != 0: # doesn't exist or isn't output
            self._pins[pin_num] = InputDevice(pin_num)

        device = self.pins.get(pin_num, pull_up=False)
        return device.is_active

    def led_warning(self):
        self.set_pin(WARNING_LED_PIN, True)
        self.set_pin(WORKING_LED_PIN, False)
        return 1

    def led_working(self):
        self.set_pin(WARNING_LED_PIN, False)
        self.set_pin(WORKING_LED_PIN, True)
        return 1

    def ring_buzzer(self):
        self.set_pin(BUZZER_PIN, True)
        return 1

    def stop_buzzer(self):
        self.set_pin(BUZZER_PIN, False)
        return

    def motor_run(self, speed=0, forward=True):
        # TODO: abstract to the nucleo
        if forward:
            self._motor.forward(speed=speed)
        else:
            self._motor.backaward(speed=speed)
        return

    def stop_motor_run(self):
        # TODO: abstract to the nucleo
        self._motor.stop()
        return

    def get_camera_frame(self):
        return self._camera.capture()

    def get_ir_thermals_frame(self):
        return self._ir_sensor.get_thermals_frame()
    
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
        print(f"Thermal Sensor Readings: {self.thermal_readings}")
        return self.thermal_readings