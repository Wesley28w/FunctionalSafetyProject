import time
import board
import busio
import adafruit_amg88xx
import cv2
from ultralytics import YOLO
from constants import *
from utility import *
from picamera2 import Picamera2

general_model = YOLO('yolo11n.pt')
face_model = YOLO('yolov11n-face.pt')

i2c = busio.I2C(board.SCL, board.SDA)

# init the AMG8833 sensor
sensor = adafruit_amg88xx.AMG88XX(i2c)

def get_thermal_frame():
    # returns 8x8 list of temps in C
    return sensor.pixels() 

# track whether shutdown or not
shutdown = False

cap = cv2.VideoCapture(0, cv2.CAP_V4L2) # 0 is for video cam

picam2 = Picamera2()

camera_config = picam2.create_video_configuration(
    main={"size": (int(STREAM_W), int(STREAM_H)), "format": "RGB888"}
)

picam2.configure(camera_config)
picam2.start()

print("Camera connected successfully.")

while True:
    frame = picam2.capture_array()
    thermal_readings = get_thermal_frame()
    for row in frame:
        print(" ".join(f"{temp:6.2f}" for temp in row))
    print("-------------------------------")

    # run inference on frame
    person_results = list(general_model(frame, classes=[0], stream=True))[0] # classes is list of items to track - 0 is person, stream = True makes more effecient
    face_results = list(face_model(frame, classes=[0], stream=True))[0] # detects faces for estimating human frame

    person_results = person_results[filter_low_conf(CONFIDENCE_THRESHOLD_PERSON, person_results.boxes.conf)]
    face_results = face_results[filter_low_conf(CONFIDENCE_THRESHOLD_FACE, face_results.boxes.conf)]

    person_xywh = person_results.boxes.xywh
    face_xywh = face_results.boxes.xywh
    
    for box in face_xywh:
        ratio = (box[2] * box[3]) / (STREAM_W * STREAM_H)
        print(ratio)
        if ratio > 0.01:
            shutdown = True
            break
        else:
            shutdown = False

    print(shutdown)

    if not person_results:
       shutdown = False 
    # cv2.putText(annotated_frame, "Shutdown" if shutdown else "Running", org, font, font_scale, (0, 0, 255) if shutdown else (0, 255, 0), thickeness, line_type)

    # cv2.imshow("Camera Stream", annotated_frame)

    # pauses execution for 1 ms delay, checks for q key press (quits)
    # if (cv2.waitKey(1) & 0xFF) == ord("q"):
    #   break

    # if (shutdown): break

    
# clean up
picam2.stop()
cv2.destroyAllWindows()
