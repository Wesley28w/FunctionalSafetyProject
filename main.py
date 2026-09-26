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

    # pass annotated into itself
    annotated_frame = person_results[0].plot(img=frame)
    annotated_frame = face_results[0].plot(img=annotated_frame)

    person_xywh = person_results.boxes.xywh
    face_xywh = face_results.boxes.xywh
    if len(person_xywh) == 0: continue

    people_bb = [i for i in person_xywh] # list of the bboxes
    faces_bb= [j for j in person_xywh] # list of the bboxes

    people_options = [id_p for id_p in range(len(people_bb))] # people to match with
    faces_options = [id_f for id_f in range(len(faces_bb))] # faces to match with

    grouped = [] # this is where the body and face bounding boxes are stored after matching

    for person_id in people_options: # for each person
        conf = person_results.boxes.conf[person_id] # get confidence from the person's bbox. Currently, will ignore face confidence...
        eligible_heads = []

        if len(faces_options) != 0: # while there are heads to pick from
            for face_id in faces_options: # loops through faces left and scores the overlapping ones
                is_in, diff = face_in_box_using_boxes(person_box=people_bb[person_id], face_box=faces_bb[face_id]) # passes in box
                if is_in: eligible_heads.append([face_id, diff]) # eligible heads will track each eligible head and its score
            
            best = eligible_heads[0] # set to first eligible head
            for head in eligible_heads: # for each next eligible head
                if head[1] > best[1]: # if the score is higher
                    best = head # set best to that one instead

            people_options.pop(person_id) # remove person from options
            faces_options.pop(best[0]) # remove head from options

            best = faces_bb[best[0]] # reassign to actual value
            grouped.append([people_bb[person_id], best, conf]) # appends a person with its best fitted head
        else:
            grouped.append([people_bb[person_id], None, conf]) # append body with no face

    # grouped is now all the the body detection w/ the corresponding face if found
    for human in grouped:
        # these are bboxes
        person = human[0]
        face = human[1] # can be None
        conf = human[2] # confidence of person, not face bbox
        
        p_height, p_width = person[2], person[3]

        d_h = HUMAN_HEIGHT * (p_height / STREAM_H) # distance using height
        d_w = HUMAN_WIDTH * (p_width / STREAM_W) # distance using width
        screenCoverRatio = (p_width * p_height) / (STREAM_W * STREAM_H)

        print(f"[LOG] Width: {p_width} Height: {p_height} D_h: {d_h} D_w: {d_w} SCR: {screenCoverRatio}")

        shutdown = past_threshold(d_w, d_h, screenCoverRatio, conf)
        if (shutdown): break

    cv2.putText(annotated_frame, "Shutdown" if shutdown else "Running", org, font, font_scale, (0, 0, 255) if shutdown else (0, 255, 0), thickeness, line_type)

    cv2.imshow("Camera Stream", annotated_frame)

    # pauses execution for 1 ms delay, checks for q key press (quits)
    if (cv2.waitKey(1) and 0xFF == ord("q")):
        break

    # if (shutdown): break

    
# clean up
cap.release()
cv2.destroyAllWindows()
