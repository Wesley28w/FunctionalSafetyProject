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

# CV Utility

def crunch(person_xywhn, face_xywhn, heatmap, person_detected):
    for box in face_xywhn:
        ratio = float(box[2] * box[3])
        print("Ratio: ", ratio)
        if ratio > 0.01:
            return True

    for box in person_xywhn:
        heat = bbox_heat_value(box, heatmap)

        if abs(heat - 25.0) < 2.5: 
            thermal_person = True

    return (thermal_person and person_detected)

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