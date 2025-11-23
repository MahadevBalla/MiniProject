from typing import Tuple
import numpy as np
import cv2
import torch

from .. import config


class MobileNetSSDDetector:
    """
    MobileNet-SSD (Caffe) GPU-accelerated detector using OpenCV DNN.
    Matches the interface of YOLODetector & FasterRCNNDetector.
    """

    def __init__(
        self,
        prototxt_path: str = str(config.MOBILENET_SSD_PROTOTXT),
        weights_path: str = str(config.MOBILENET_SSD_WEIGHTS),
        conf_threshold: float = config.YOLO_CONF_THRESHOLD,
        use_cuda: bool = True,
    ):
        self.conf_threshold = float(conf_threshold)

        # Load network
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)

        # Enable CUDA acceleration
        if use_cuda and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print("MobileNetSSD: Using CUDA acceleration")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        else:
            print("MobileNetSSD: Using CPU (CUDA not available)")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # MobileNet-SSD class names (VOC)
        self.class_names = [
            "background",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]

        # Create mapping from label → class_id that matches your pipeline
        # Only return detections for classes existing in config.CLASSES
        self.coco_mapping = {name: i for i, name in enumerate(config.CLASSES)}

        print("MobileNetSSDDetector initialized")

    def detect(self, frame_bgr: np.ndarray):
        """
        Returns:
            bboxes_xyxy: (N,4)
            scores: (N,)
            class_ids: (N,)  (mapped to config.CLASSES indices)
            filtered_indices: indices after confidence threshold
        """
        h, w = frame_bgr.shape[:2]

        # Prepare blob for MobileNet-SSD (300×300)
        blob = cv2.dnn.blobFromImage(
            frame_bgr,
            scalefactor=0.007843,
            size=(300, 300),
            mean=(127.5, 127.5, 127.5),
            swapRB=False,
            crop=False,
        )
        self.net.setInput(blob)

        detections = self.net.forward()  # shape: (1,1,N,7)

        bboxes = []
        scores = []
        class_ids = []

        # Loop through detections
        for det in detections[0, 0]:
            score = float(det[2])
            if score < self.conf_threshold:
                continue

            class_idx = int(det[1])  # VOC ID 0–20
            class_name = self.class_names[class_idx]

            if class_name not in self.coco_mapping:
                continue  # skip classes not in global pipeline

            # Bounding box scaled to original image
            x1 = int(det[3] * w)
            y1 = int(det[4] * h)
            x2 = int(det[5] * w)
            y2 = int(det[6] * h)

            bboxes.append([x1, y1, x2, y2])
            scores.append(score)
            class_ids.append(self.coco_mapping[class_name])

        if len(bboxes) == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
            )

        bboxes = np.asarray(bboxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        class_ids = np.asarray(class_ids, dtype=np.int32)

        filtered_indices = np.arange(len(bboxes), dtype=np.int32)

        return bboxes, scores, class_ids, filtered_indices
