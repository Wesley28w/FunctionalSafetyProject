import cv2
from .model import *
# acts as a redundant camera + streams it own results for a visual representation

print("Camera starting")

cap = cv2.VideoCapture(0)
cv = CV()

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to grab frame.")
        break

    # run inference on frame
    person_results = cv.get_person_results(frame) # classes is list of items to track - 0 is person, stream = True makes more effecient
    face_results = cv.get_face_results(frame) # detects faces for estimating human frame

    # fetch correct info from results
    person_xywhn = cv.get_person_wywhn(frame)
    face_xywhn = cv.get_face_wywhn(frame)
    person_detected = cv.person_detected(frame)

    shutdown = crunch(person_xywhn, face_xywhn, thermal_readings, cv.person_detected(frame))

    cv2.imshow("Mac Camera Stream (Press 'q' to quit)", frame)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()