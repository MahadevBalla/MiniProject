import cv2
import numpy as np

def draw_bounding_box(frame, box):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

def create_zoomed_view(frame, box):
    h, w, _ = frame.shape
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 > x1 and y2 > y1:
        zoomed = frame[y1:y2, x1:x2]
        zoomed = cv2.resize(zoomed, (640, 480))
        return zoomed
    return None

def auto_frame(frame, person_box, alpha=0.2, prev_box=None):
    new_box = np.array(person_box, dtype=float)

    if prev_box is None:
        prev_box = new_box
    else:
        prev_box = alpha * new_box + (1 - alpha) * prev_box

    draw_bounding_box(frame, prev_box)
    zoomed_view = create_zoomed_view(frame, prev_box)

    return frame, zoomed_view, prev_box