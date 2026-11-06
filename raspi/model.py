from ultralytics import YOLO

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