import cv2
from ultralytics import YOLO
import math

# CONSTANTS:
FOV_H = 122.0 # how wide is the camera. (Currently for Mac Air M4)
RESOLUTION = (1920.0, 1080.0) # pixel size resolution
STREAM_W = RESOLUTION[0]
STREAM_H = RESOLUTION[1]
FOCAL_LENGTH = STREAM_W / (2 * math.tan(FOV_H / 2))
FOV_V = 2 * math.atan(STREAM_H / (2 * FOCAL_LENGTH)) # focal length is same for V/H FOV

# TODO: make this work for kids too (maybe use ratio of both?)
# LIMITATION: if someone is too close to camera where you can't see entire body then distance will be off. But should disable regardless.
HUMAN_HEIGHT = 68.0 # inches - 5'8"
HUMAN_WIDTH = 16.1 # inches - Shoulder SPAN

# Calculation Constants
# CURRENTLY NOT USING WIDTH BECAUSE VARIES TO HEAVILY
WEIGHT_H = 1.0 # How much to weigh height distance calculation over width
SCREEN_COVER_THRESHOLD = 0.3 # How much of the screen a person covers to be considered too close
DISTANCE_THRESHOLD = 32 # inches. 10 feet
model = YOLO('yolo11n.pt')

# track whether shutdown or not
shutdown = False

cap = cv2.VideoCapture(0) # 0 is for video cam

# set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, STREAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_H)

# for text display
org = (50, 50)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1 
thickeness = 2
line_type = cv2.LINE_AA

# calculate whether or not a box is TOO close to the camera.
def past_threshold(distance_w: float, distance_h: float, screen_cover_ratio: float, confidence: float):
    # if someone is too close (where their whole body is not even showing) we want to default to True
    if screen_cover_ratio > SCREEN_COVER_THRESHOLD: return True

    distance = (WEIGHT_H * distance_h) + ((1 - WEIGHT_H) * distance_w) # consolidate distances. Weighted
    # use confidence to reduce distance in order to stay safer on less confident predictions
    # distance = distance * (confidence) # Can make this confidence scaling exponential, or weighted in future
    print(f"Distance: {distance}")
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
        xywh = result.boxes.xywh
        if len(xywh) == 0: continue
        
        for index, box in enumerate(xywh):
            print(box)
            width, height = box[2], box[3] # in pixels

            conf = result.boxes.conf[index] # confidence
            
            d_h = HUMAN_HEIGHT * (height / STREAM_H) # distance using height
            d_w = HUMAN_WIDTH * (width / STREAM_W) # distance using width
            screenCoverRatio = (width * height) / (STREAM_W * STREAM_H)

            print(f"[LOG] Width: {width} Height: {height} D_h: {d_h} D_w: {d_w} SCR: {screenCoverRatio}")

            shutdown = past_threshold(d_w, d_h, screenCoverRatio, conf)
            if (shutdown): break
        if (shutdown): break

    cv2.putText(annotated_frame, "Shutdown" if shutdown else "Running", org, font, font_scale, (0, 0, 255) if shutdown else (0, 255, 0), thickeness, line_type)

    cv2.imshow("Camera Stream", annotated_frame)

    # pauses execution for 1 ms delay, checks for q key press (quits)
    if (cv2.waitKey(1) and 0xFF == ord("q")):
        break
    
# clean up
cap.release()
cv2.destroyAllWindows()
