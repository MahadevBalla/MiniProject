# src/detector/unet_detector.py
"""
UNet / Segmentation-based detector
(Using TorchVision DeepLabv3-ResNet50 pretrained on COCO)

Pipeline:
    1. Run segmentation → get mask for "person" class.
    2. Find connected components.
    3. Convert each component into a bounding box.
    4. Return boxes, scores (mask confidences), class_ids=person(0).

This works well for scenes where people face different directions.
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision import models

from .. import config


class UNetSegDetector:
    def __init__(
        self,
        device: torch.device = None,
        conf_threshold: float = 0.3,  # mask confidence threshold
        min_component_area: int = 400,  # small blobs removed
    ):
        self.device = (
            device
            if device is not None
            else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )

        self.conf_threshold = conf_threshold
        self.min_component_area = min_component_area

        # DeepLabv3-ResNet50 pretrained on COCO (person class is present)
        weights = models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
        self.model = models.segmentation.deeplabv3_resnet50(weights=weights)
        self.model.eval().to(self.device)

        self.transform = weights.transforms()

        print(
            f"UNetSegDetector (DeepLabv3 backbone) initialized on {self.device}, "
            f"conf_thr={self.conf_threshold}, min_area={self.min_component_area}"
        )

    def detect(self, frame_bgr: np.ndarray):
        if frame_bgr is None:
            return (
                np.empty((0, 4), np.float32),
                np.empty((0,), np.float32),
                np.empty((0,), np.int32),
                np.empty((0,), np.int32),
            )

        h, w = frame_bgr.shape[:2]

        # Convert to RGB + preprocess
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(frame_rgb).to(self.device).unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor)["out"]  # (1,21,H,W) COCO classes

        # Softmax over 21 classes
        probs = torch.softmax(output, dim=1)[0]  # (21,H,W)

        # take person class (class index 15 in COCO segmentation)
        PERSON_CLASS = 15
        person_mask = probs[PERSON_CLASS].cpu().numpy()  # (H,W)

        # threshold the mask
        mask_bin = (person_mask >= self.conf_threshold).astype(np.uint8) * 255

        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin)

        boxes = []
        scores = []
        class_ids = []

        for i in range(1, num_labels):  # skip background label 0
            x, y, bw, bh, area = stats[i]

            if area < self.min_component_area:
                continue

            x1, y1 = x, y
            x2, y2 = x + bw, y + bh

            boxes.append([x1, y1, x2, y2])
            # score = average mask probability inside the region
            scores.append(float(np.mean(person_mask[y1:y2, x1:x2])))
            class_ids.append(PERSON_CLASS)

        if len(boxes) == 0:
            return (
                np.empty((0, 4), np.float32),
                np.empty((0,), np.float32),
                np.empty((0,), np.int32),
                np.empty((0,), np.int32),
            )

        boxes_np = np.array(boxes, dtype=np.float32)
        scores_np = np.array(scores, dtype=np.float32)
        class_ids_np = np.array(class_ids, dtype=np.int32)
        kept_indices = np.arange(len(boxes_np))

        # Clip box bounds
        boxes_np[:, [0, 2]] = np.clip(boxes_np[:, [0, 2]], 0, w - 1)
        boxes_np[:, [1, 3]] = np.clip(boxes_np[:, [1, 3]], 0, h - 1)

        return boxes_np, scores_np, class_ids_np, kept_indices
