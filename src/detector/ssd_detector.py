from typing import Tuple
import numpy as np
import torch
import torchvision
import torchvision.transforms as T

from .. import config


class SSDDetector:
    """
    SSD300 Detector (torchvision) wrapper with same detect() API used by YOLO and FasterRCNN wrappers.
    Uses ssd300_vgg16 pretrained on COCO.
    """

    def __init__(
        self,
        conf_threshold: float = config.YOLO_CONF_THRESHOLD,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self.device = device
        self.conf_threshold = conf_threshold

        weights_path = config.SSD300_WEIGHTS_PATH

        # Load model architecture
        self.model = torchvision.models.detection.ssd300_vgg16(
            weights=None,
            weights_backbone=None,  # <<< without this it downloads the VGG file
        )

        # Load local checkpoint
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        # Put into eval mode and move to device
        self.model.to(self.device)
        self.model.eval()

        # SSD expects:
        # - RGB
        # - scaled to [0..1]
        # - img resized to 300x300 (the model does this automatically internally)
        self.transform = T.Compose(
            [
                T.ToTensor(),  # Converts HWC uint8 RGB → CHW float32 in [0..1]
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        print(
            f"SSDDetector initialized on device {self.device} with conf={self.conf_threshold}"
        )

    def _preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        frame_rgb = frame_bgr[:, :, ::-1].copy()  # BGR→RGB
        tensor = self.transform(frame_rgb).to(self.device)
        return tensor.unsqueeze(0)

    def detect(self, frame_bgr: np.ndarray):
        original_h, original_w = frame_bgr.shape[:2]

        batch = self._preprocess(frame_bgr)

        with torch.no_grad():
            output = self.model(batch)[0]  # dict: boxes, labels, scores

        boxes = output.get("boxes", torch.empty((0, 4), device=self.device))
        scores = output.get("scores", torch.empty((0,), device=self.device))
        labels = output.get("labels", torch.empty((0,), device=self.device))

        if boxes.numel() == 0:
            return (
                np.empty((0, 4), np.float32),
                np.empty((0,), np.float32),
                np.empty((0,), np.int32),
                np.empty((0,), np.int32),
            )

        boxes_np = boxes.cpu().numpy().astype(np.float32)
        scores_np = scores.cpu().numpy().astype(np.float32)
        labels_np = labels.cpu().numpy().astype(np.int32)

        # Threshold
        keep = scores_np >= float(self.conf_threshold)
        if keep.sum() == 0:
            return (
                np.empty((0, 4), np.float32),
                np.empty((0,), np.float32),
                np.empty((0,), np.int32),
                np.empty((0,), np.int32),
            )

        boxes_np = boxes_np[keep]
        scores_np = scores_np[keep]
        labels_np = labels_np[keep]
        kept_indices = np.where(keep)[0]

        # Clip coords (safety)
        boxes_np[:, [0, 2]] = np.clip(boxes_np[:, [0, 2]], 0, original_w)
        boxes_np[:, [1, 3]] = np.clip(boxes_np[:, [1, 3]], 0, original_h)

        return boxes_np, scores_np, labels_np, kept_indices
