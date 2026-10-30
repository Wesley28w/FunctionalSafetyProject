import cv2
import board
import busio
import adafruit_amg88xx
from gpiozero import InputDevice
from picamera2 import Picamera2
from constants import *

# read camera, thermal sensor

def init_camera() -> int:
    picam2 = Picamera2()

    camera_config = picam2.create_video_configuration(
        main={"size": (int(STREAM_W), int(STREAM_H)), "format": "RGB888"}
    )

    picam2.configure(camera_config)
    try:
        res = picam2.start()
        print("Camera start returned with: ", res)
        return 1 if res == None else 0
    except Exception as e:
        print(f"Camera failed to start: {e}")

