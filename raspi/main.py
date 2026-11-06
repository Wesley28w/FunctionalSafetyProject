import time
import board
import busio
import adafruit_amg88xx
import cv2

import utility
from sensor import *

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

while True:
    frame = camera.capture_array()
    if counter % 3 == 0:
         thermal_readings = ir.get_thermals_frame()
         print(f"Thermal Sensor Readings: {thermal_readings}")

    # run inference on frame
    person_results = list(general_model(frame, classes=[0], stream=True))[0] # classes is list of items to track - 0 is person, stream = True makes more effecient
    face_results = list(face_model(frame, classes=[0], stream=True))[0] # detects faces for estimating human frame

    person_results = person_results[filter_low_conf(CONFIDENCE_THRESHOLD_PERSON, person_results.boxes.conf)]
    face_results = face_results[filter_low_conf(CONFIDENCE_THRESHOLD_FACE, face_results.boxes.conf)]

    person_xywhn = person_results.boxes.xywhn
    face_xywhn = face_results.boxes.xywhn

    face_close = False
    person_detected = len(person_results.boxes) > 0
    thermal_person = False

    for box in face_xywhn:
        ratio = float(box[2] * box[3])
        print("Ratio:", ratio)
        if ratio > 0.01:
            face_close = True
            break

    for box in person_xywhn:
        heat = bbox_heat_value(box, thermal_readings)

        print("Heat Value:", heat)

        if abs(heat - 25.0) < 2.5: thermal_person = True
 
    shutdown = (thermal_person and person_detected) or (face_close)

    counter += 1

    if (shutdown): 
        raspi.set_pin(19, True)
    else: 
        raspi.set_pin(19, False)
    
    # cv2.putText(annotated_frame, "Shutdown" if shutdown else "Running", org, font, font_scale, (0, 0, 255) if shutdown else (0, 255, 0), thickeness, line_type)

    # cv2.imshow("Camera Stream", annotated_frame)

    # pauses execution for 1 ms delay, checks for q key press (quits)
    # if (cv2.waitKey(1) & 0xFF) == ord("q"):
    #   break

    # if (shutdown): break

    
# clean up
picam2.stop()
cv2.destroyAllWindows()
