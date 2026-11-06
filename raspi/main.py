import time
import board
import busio
import adafruit_amg88xx
import cv2

import utility
from sensor import *
from model import *

from ultralytics import YOLO
from raspi.constants import *
from raspi.utility import *
from picamera2 import Picamera2
from gpiozero import OutputDevice

general_model = YOLO('yolo11n.pt')
face_model = YOLO('yolov11n-face.pt')

pin_cfg = [[19], []] # index 0 is output, index 1 is input

raspi = Raspi(pin_cfg=pin_cfg)
camera = Camera()
ir = IRSensor()
cv = CV()

while True:
    frame = camera.capture_array()
    if counter % 3 == 0:
         thermal_readings = ir.get_thermals_frame()
         print(f"Thermal Sensor Readings: {thermal_readings}")

    # run inference on frame
    person_results = list(general_model(frame, classes=[0], stream=True))[0] # classes is list of items to track - 0 is person, stream = True makes more effecient
    face_results = list(face_model(frame, classes=[0], stream=True))[0] # detects faces for estimating human frame

    person_xywhn = cv.get_person_wywhn(frame)
    face_xywhn = cv.get_face_wywhn(frame)
    person_detected = cv.person_detected(frame)
    shutdown = crunch(person_xywhn, face_xywhn, thermal_readings, cv.person_detected(frame))

    counter += 1

    if (shutdown): 
        raspi.set_pin(19, True)
    else: 
        raspi.set_pin(19, False)
    
    if (cv2.waitKey(1) & 0xFF) == ord("q"):
      break

    
# clean up
camera.stop()
cv2.destroyAllWindows()
