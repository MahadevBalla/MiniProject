from typing import Tuple
import numpy as np
import cv2
import torch

from .. import config


class HaarCascadeDetector:
    """
    Haar Cascade detector (OpenCV) wrapper implementing the same detect()
    API as YOLODetector / FasterRCNN / SSDDetector.

    This is a classical (non-deep) detector.
    Extremely fast, very low accuracy — useful for baseline comparisons.
    """

    def __init__(
        self,
        cascade_path: str = cv2.data.haarcascades + "haarcascade_frontalface_alt.xml",
        conf_threshold: float = 1.0,  # Haar does not output confidence; keep for compatibility
        device: torch.device = torch.device("cpu"),
    ):
        # Haar cascade always runs on CPU
        self.device = device
        self.conf_threshold = conf_threshold
        self.cascade = cv2.CascadeClassifier(cascade_path)

        if self.cascade.empty():
            raise FileNotFoundError(f"Could not load Haar cascade at {cascade_path}")

        print(f"HaarCascadeDetector initialized with cascade={cascade_path}")

    def detect(self, frame_bgr: np.ndarray):
        """
        Returns:
            bboxes_xyxy: (N,4)
            scores: dummy (all ones)
            class_ids: dummy (all zero)
            filtered_indices: array of kept indices
        """

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        if len(faces) == 0:
            return (
                np.empty((0, 4), np.float32),
                np.empty((0,), np.float32),
                np.empty((0,), np.int32),
                np.empty((0,), np.int32),
            )

        # faces = [x,y,w,h]
        boxes = []
        for x, y, w, h in faces:
            boxes.append([x, y, x + w, y + h])

        boxes_np = np.array(boxes, dtype=np.float32)
        scores_np = np.ones(
            (len(faces),), dtype=np.float32
        )  # Haar = no confidence scores
        class_ids_np = np.zeros(
            (len(faces),), dtype=np.int32
        )  # treat as class 0 ("face")
        kept_indices = np.arange(len(faces))

        return boxes_np, scores_np, class_ids_np, kept_indices
