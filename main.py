import cv2
from ultralytics import YOLO
from constants import *
from utility import *

general_model = YOLO('yolo11n.pt')
face_model = YOLO('yolov11n-face.pt')

# track whether shutdown or not
shutdown = False

cap = cv2.VideoCapture(0) # 0 is for video cam

# set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, STREAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_H)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

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
    
    # pass annotated into itself
    if person_results:
        annotated_frame = person_results[0].plot(img=frame)
        if face_results:
            annotated_frame = face_results[0].plot(img=annotated_frame)
    else:
        annotated_frame = frame

    cv2.putText(annotated_frame, "Shutdown" if shutdown else "Running", org, font, font_scale, (0, 0, 255) if shutdown else (0, 255, 0), thickeness, line_type)

    cv2.imshow("Camera Stream", annotated_frame)

    # pauses execution for 1 ms delay, checks for q key press (quits)
    if (cv2.waitKey(1) and 0xFF == ord("q")):
        break

    # if (shutdown): break

    
# clean up
cap.release()
cv2.destroyAllWindows()
