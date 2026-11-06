from ultralytics import YOLO
from utility import *

class CV():
    def __init__(self):
        self.person_model = YOLO('yolo11n.pt')
        self.face_model = YOLO('yolov11n-face.pt')

    # model fetching utility
    def get_person_results(self, frame):
        return list(self.person_model(frame, classes=[0], stream=True))[0]

    def get_person_wywhn(self, frame):
        return self.get_person_results(frame).boxes.xywhn

    def get_face_results(self, frame):
        return list(self.face_model(frame, classes=[0], stream=True))[0]

    def get_person_wywhn(self, frame):
        return self.get_face_results(frame).boxes.xywhn

    def person_detected(self, frame):
        return len(self.get_person_results(frame)) > 0

    def face_detected(self, frame):
        return len(self.get_face_results(frame)) > 0

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
