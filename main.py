import cv2
from ultralytics import YOLO
import math

# CONSTANTS:
FOV_H = 110.0 # how wide is the camera
RESOLUTION = (1920, 1080) # pixel size resolution
STREAM_W = RESOLUTION[0]
STREAM_H = RESOLUTION[1]
FOCAL_LENGTH = STREAM_W / (2 * math.tan(FOV_H / 2))
FOV_V = 2 * math.atan(STREAM_H / (2 * FOCAL_LENGTH)) # focal length is same for V/H FOV

# TODO: make this work for kids too (maybe use ratio of both?)
# LIMITATION: if someone is too close to camera where you can't see entire body then distance will be off. But should disable regardless.
HUMAN_HEIGHT = 71.0 # inches - 5'11"
HUMAN_WIDTH = 16.1 # inches - Shoulder SPAN

# Calculation Constants
WEIGHT_H = 0.6 # How much to weigh height distance calculation over width
SCREEN_COVER_THRESHOLD = 0.5 # How much of the screen a person covers to be considered too close
DISTANCE_THRESHOLD = 120 # inches. 10 feet
model = YOLO('yolo11n.pt')

# track whether shutdown or not
shutdown = False

cap = cv2.VideoCapture(0) # 0 is for video cam

def past_threshold(distance_w: float, distance_h: float, screen_cover_ratio: float, confidence: float):
    # if someone is too close (where their whole body is not even showing) we want to default to True
    if screen_cover_ratio > SCREEN_COVER_THRESHOLD: return True

    distance = (WEIGHT_H * distance_h) + ((1 - WEIGHT_H) * distance_w) # consolidate distances. Weighted
    # use confidence to reduce distance in order to stay safer on less confident predictions
    distance *= (confidence) # Can make this confidence scaling exponential, or weighted in future
    
    # if distance is too far 
    if distance < DISTANCE_THRESHOLD: return True
    return False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # run inference on frame
    results = model(frame, classes=[0], stream=True) # classes is list of items to track - 0 is person, stream = True makes more effecient

    # plot the bounding box
    for result in results:
        annotated_frame = result.plot()
        xywh = result.boxes.xywh # center x/y, width, height
        width, height = xywh[2], xywh[3] # in pixels
        conf = result.boxes.conf # confidence
        
        d_h = HUMAN_HEIGHT * (height / STREAM_H) # distance using height
        d_w = HUMAN_WIDTH * (width / STREAM_W) # distance using width
        screenCoverRatio = (width * height) / (STREAM_W * STREAM_H)

        shutdown = past_threshold(d_w, d_h, screenCoverRatio, conf)
        if (shutdown): break
        
    if (shutdown): 
        print("SHUTDOWN") 
    else: 
        print("ON")

    cv2.imshow("Camera Stream", annotated_frame)

    # pauses execution for 1 ms delay, checks for q key press (quits)
    if (cv2.waitKey(1) and 0xFF == ord("q")) or shutdown:
        break
    
# clean up
cap.release()
cv2.destroyAllWindows()
