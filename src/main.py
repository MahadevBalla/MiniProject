import cv2
from ultralytics import YOLO
import numpy as np
from auto_frame import draw_bounding_box, create_zoomed_view

# Load YOLOv8 model
model = YOLO("models/yolov8n.pt")

# Initialize webcam feed
cap = cv2.VideoCapture(0)

# Set webcam resolution to a larger size for better full screen experience
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize previous box for smoothing
prev_box = None
alpha = 0.2  # smoothing factor

# Create a named window for full screen
cv2.namedWindow("Person Detection - Press 'q' to quit", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Person Detection - Press 'q' to quit", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    person_boxes = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if model.names[cls] == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_boxes.append((x1, y1, x2, y2))

    if person_boxes:
        # Pick largest person
        person_boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        x1, y1, x2, y2 = person_boxes[0]

        new_box = np.array([x1, y1, x2, y2], dtype=float)

        if prev_box is None:
            prev_box = new_box
        else:
            # Smooth transition
            prev_box = alpha * new_box + (1 - alpha) * prev_box

        x1, y1, x2, y2 = map(int, prev_box)

        # Draw smoothed box
        draw_bounding_box(frame, (x1, y1, x2, y2))

        # Zoomed view
        zoomed = create_zoomed_view(frame, (x1, y1, x2, y2))

    cv2.imshow("Person Detection - Press 'q' to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()