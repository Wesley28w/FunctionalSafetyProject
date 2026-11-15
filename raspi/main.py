import cv2
from raspi.hw import *
from shared.cv import *

from shared.constants import *

raspi = Raspi()
cv = CV()

while True:
    frame = raspi.get_camera_frame()
    thermal_readings = raspi.get_ir_thermals_frame()

    # run inference on frame
    person_results = cv.get_person_results(frame) # classes is list of items to track - 0 is person, stream = True makes more effecient
    face_results = cv.get_face_results(frame) # detects faces for estimating human frame

    # fetch correct info from results
    person_xywhn = cv.get_person_wywhn(frame)
    face_xywhn = cv.get_face_wywhn(frame)
    person_detected = cv.person_detected(frame)

    # determine if should shutdown..
    shutdown = crunch(person_xywhn, face_xywhn, thermal_readings, cv.person_detected(frame))

    if (shutdown): 
        raspi.led_warning()
        raspi.ring_buzzer()
    else: 
        raspi.led_working()
        raspi.stop_buzzer()
    
    if (cv2.waitKey(1) & 0xFF) == ord("q"):
      break

    
# clean up
raspi.stop()
cv2.destroyAllWindows()
