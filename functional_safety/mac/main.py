import cv
import cv2
from stream import Stream

from functional_safety.shared.cv import CV, check

# acts as a redundant camera (TX) + streams it own results for a visual representation
# streams raspi camera too (RX)
stream = Stream()

while True:
    frame = stream.grab_frame()

    shutdown = check(frame)

    stream.show_frame(frame)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break