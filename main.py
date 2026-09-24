import cv2
from ultralytics import YOLO

# CONSTANTS:
FOV = 110.0 # how wide is the camera
RESOLUTION = (1920, 1080) # pixel size resolution
FOCAL_LENGTH = FOV / 
model = YOLO('yolo11n.pt')

cap = cv2.VideoCapture(0) # 0 is for video cam

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
        
        # math to convert into relative yaw distance


        conf = result.boxes.conf # confidence

        
        # we want to consider humans CLOSER if confidence is low to be safer

    cv2.imshow("Camera Stream", annotated_frame)

    # pauses execution for 1 ms delay, checks for q key press (quits)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    

# clean up
cap.release()
cv2.destroyAllWindows()
