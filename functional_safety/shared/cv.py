import numpy as np
from ultralytics import YOLO

from functional_safety.paths import FACE_MODEL, PERSON_MODEL
from functional_safety.shared.constants import (
    CONFIDENCE_THRESHOLD_FACE,
    CONFIDENCE_THRESHOLD_PERSON,
)

class CV():
    def __init__(self):
        self.person_model = YOLO(PERSON_MODEL)
        self.face_model = YOLO(FACE_MODEL)

    # model fetching utility
    def get_person_results(self, frame):
        all = list(self.person_model(frame, classes=[0], stream=True))[0]
        return filter_low_conf(CONFIDENCE_THRESHOLD_PERSON, all)

    def get_person_wywhn(self, frame):
        return self.get_person_results(frame).boxes.xywhn

    def get_face_results(self, frame):
        all = list(self.face_model(frame, classes=[0], stream=True))[0]
        return filter_low_conf(CONFIDENCE_THRESHOLD_FACE, all)
    
    def get_face_wywhn(self, frame):
        return self.get_face_results(frame).boxes.xywhn

    def person_detected(self, frame):
        return len(self.get_person_results(frame)) > 0

    def face_detected(self, frame):
        return len(self.get_face_results(frame)) > 0

    def get_all(self, frame):
        return (
            self.get_person_results(frame),
            self.get_face_results(frame), 
            self.get_person_wywhn(frame), 
            self.get_face_results(frame), 
            self.person_detected(frame)
        )

    def check_shutdown(self, frame, heatmap):
        return check(
            self.get_person_wywhn(frame), 
            self.get_face_wywhn(frame), 
            heatmap # defaults to none
        )

# TODO: add like confidence scores to returns make cumalitive across the checks.
def check(person_xywhn, face_xywhn, heatmap=None):
    if heatmap is None: # mac case
        for person in person_xywhn:
            if person[2] * person[3] > 0.2: # should make more sensitive than the heatmap
                # Big body part on screen
                return True
    else:
        # if we detect a person frame and a thermal reading in that area
        # and the heat total is over X amount
        for person in person_xywhn:
            if (bbox_heat_value(person, heatmap) > 25.0):
                return True

    # if we detect a face closer than some screen ratio
    for face in face_xywhn:
        if face[2] * face[3] > 0.01:
            return True

    return False

def filter_low_conf(threshold, results):
    result = [i for i in results if i.boxes.confidence > threshold]
    return result

def bbox_heat_value(box, thermals):
    thermals = np.asarray(thermals)
    x, y, w, h, = [float(v) for v in box]
    
    thermal_h, thermal_w = thermals.shape

    x1 = int(np.floor((x - w / 2) *thermal_w))
    x2 = int(np.ceil((x + w / 2) * thermal_w))

    y1 = int(np.floor((y - h / 2) * thermal_h))
    y2 = int(np.ceil((y + h / 2) * thermal_h))

    # clamp
    x1 = max(0, min(thermal_w - 1, x1))
    x2 = max(x1 + 1, min(thermal_w, x2))

    y1 = max(0, min(thermal_h - 1, y1))
    y2 = max(y1 + 1, min(thermal_h, y2))

    region = thermals[y1:y2, x1:x2]

    print("Normalized bbox:", x, y, w, h)
    print("Thermal bbox:", x1, y1, x2, y2)
    print("Thermal region:")
    print(region)

    if region.size == 0:
        return 0.0
    
    return float(region.mean())