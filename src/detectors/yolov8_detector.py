class YOLOv8Detector:
    def __init__(self, model_path):
        from ultralytics import YOLO
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, stream=True)
        person_boxes = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if self.model.names[cls] == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_boxes.append((x1, y1, x2, y2))
        return person_boxes