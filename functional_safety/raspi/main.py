import cv2

from functional_safety.raspi.hw import Raspi
from functional_safety.shared.cv import CV, crunch

raspi = Raspi()
cv = CV()

while True:
    frame = raspi.get_camera_frame()
    thermal_readings = raspi.get_ir_thermals_frame()

    # determine if should shutdown..
    shutdown = cv.check_shutdown(thermal_readings)

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
