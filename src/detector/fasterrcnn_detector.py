from typing import Tuple, List

import numpy as np
import torch
import torchvision
import torchvision.transforms as T

from .. import config

# Keep same function signature as YOLODetector:
# detect(frame_bgr) -> (bboxes_xyxy, scores, class_ids, filtered_indices)


class FasterRCNNDetector:
    """
    Faster R-CNN detector wrapper (torchvision) that provides the same detect()
    signature used by the pipeline.

    Notes:
        - Uses torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        - Runs on `device` passed at init (cuda:0 by default if available).
        - Returns boxes in (x1,y1,x2,y2) in original image coordinates.
    """

    def __init__(
        self,
        model_name: str = "fasterrcnn_resnet50_fpn",
        conf_threshold: float = config.YOLO_CONF_THRESHOLD,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self.device = device
        self.conf_threshold = conf_threshold

        # Build model
        if model_name == "fasterrcnn_resnet50_fpn":
            # Build empty model (no downloading, no pretrained flags)
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights=None,
                weights_backbone=None,
                pretrained=False,
                pretrained_backbone=False,
                progress=False,
            )

            # Load your local weights
            weights_path = config.FASTER_RCNN_WEIGHTS_PATH
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"FasterRCNN weights not found at: {weights_path}"
                )

            print(f"Loading FasterRCNN weights from: {weights_path}")
            state = torch.load(weights_path, map_location=device)
            self.model.load_state_dict(state)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        # Put into eval mode and move to device
        self.model.to(self.device)
        self.model.eval()

        # Transform: convert BGR->RGB, to tensor, normalize using ImageNet stats
        self.transform = T.Compose(
            [
                T.ToTensor(),  # converts HWC [0..255] to CHW [0..1] and RGB ordering expected by ToTensor when input is PIL/np.ndarray in RGB
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        print(
            f"FasterRCNNDetector initialized on device {self.device} with conf_threshold={self.conf_threshold}"
        )

    def _preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """
        Convert BGR numpy image to normalized torch tensor (1,C,H,W) in RGB order.
        """
        # Convert BGR -> RGB
        frame_rgb = frame_bgr[:, :, ::-1].copy()
        # transform expects HxW x C numpy or PIL; ToTensor handles numpy arrays
        tensor = self.transform(frame_rgb).to(self.device)
        return tensor.unsqueeze(0)  # add batch

    def detect(
        self, frame_bgr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run Faster R-CNN on `frame_bgr`.

        Returns:
            bboxes_xyxy: (N,4) float32 numpy array in original image coordinates [x1,y1,x2,y2]
            scores: (N,) float32 numpy array
            class_ids: (N,) int32 numpy array (COCO labels)
            filtered_indices: indices (np.ndarray) of detections kept after thresholding
        """
        original_h, original_w = frame_bgr.shape[:2]

        # Preprocess and forward
        input_tensor = self._preprocess(frame_bgr)  # (1,C,H,W)
        with torch.no_grad():
            outputs = self.model(input_tensor)[0]

        # outputs: dict with keys 'boxes', 'labels', 'scores'
        boxes = outputs.get("boxes", torch.empty((0, 4), device=self.device))
        scores = outputs.get("scores", torch.empty((0,), device=self.device))
        labels = outputs.get("labels", torch.empty((0,), device=self.device))

        if boxes.numel() == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
            )

        # Move to cpu numpy
        boxes_np = boxes.cpu().numpy().astype(np.float32)
        scores_np = scores.cpu().numpy().astype(np.float32)
        labels_np = labels.cpu().numpy().astype(np.int32)

        # Filter by conf_threshold
        keep_mask = scores_np >= float(self.conf_threshold)
        if keep_mask.sum() == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
            )

        boxes_np = boxes_np[keep_mask]
        scores_np = scores_np[keep_mask]
        labels_np = labels_np[keep_mask]
        kept_indices = np.where(keep_mask)[0]

        # No need to scale because torchvision returns boxes in original image coordinates
        # But ensure clipping
        boxes_np[:, [0, 2]] = np.clip(boxes_np[:, [0, 2]], 0, original_w)
        boxes_np[:, [1, 3]] = np.clip(boxes_np[:, [1, 3]], 0, original_h)

        return boxes_np, scores_np, labels_np, kept_indices
