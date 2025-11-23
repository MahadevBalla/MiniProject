from typing import Tuple
import torch
import cv2
from ultralytics import YOLO as UltralyticsYOLO
import numpy as np

from .. import config


class YOLOv8Detector:
    """
    YOLOv8 PyTorch detector wrapper using Ultralytics API.
    Loads weights from config (YOLOV8*_PT_WEIGHTS) and exposes `detect(frame_bgr)`.

    Returns:
        boxes: np.ndarray (N,4) in xyxy (x1,y1,x2,y2) with float32
        scores: np.ndarray (N,) float32
        class_ids: np.ndarray (N,) int32
        kept_indices: np.ndarray (N,) int32
    """

    def __init__(
        self,
        variant: str = "n",  # choose 'n', 's', 'm' depending on which weights exist in config
        conf_threshold: float = None,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ):
        if UltralyticsYOLO is None:
            raise ImportError(
                "ultralytics package is required for YOLOv8Detector. "
                "Install with: pip install ultralytics\nOriginal import error: "
                "ModuleNotFoundError: No module named 'ultralytics'"
            )

        self.device = device
        self.conf_threshold = (
            config.YOLO_CONF_THRESHOLD
            if conf_threshold is None
            else float(conf_threshold)
        )

        # map variant -> config weight path
        variant = variant.lower()
        if variant == "n":
            weights_path = getattr(config, "YOLOV8N_PT_WEIGHTS", None)
        elif variant == "s":
            weights_path = getattr(config, "YOLOV8S_PT_WEIGHTS", None)
        elif variant == "m":
            weights_path = getattr(config, "YOLOV8M_PT_WEIGHTS", None)
        else:
            raise ValueError("variant must be one of: 'n', 's', 'm'")

        if weights_path is None or not weights_path.exists():
            raise FileNotFoundError(
                f"YOLOv8 weights for variant '{variant}' not found in config at {weights_path}"
            )

        self.weights_path = str(weights_path)
        print(f"Initializing YOLOv8 ({variant}) using weights: {self.weights_path}")

        # Load model via ultralytics, no auto-download (local path)
        # Note: Ultralytics' YOLO automatically picks device if 'model.to' is used.
        # We'll pass device via model.to(device)
        self.model = UltralyticsYOLO(self.weights_path)
        # Put to device
        try:
            self.model.to(str(self.device))
        except Exception:
            # ultralytics might accept 'cpu'/'gpu' strings; fallback: let it decide
            pass

        # Set model confidence (Ultralytics has .model.params but safest to apply post-filter)
        print(
            f"YOLOv8Detector ready on device {self.device}; conf_threshold={self.conf_threshold}"
        )

    def detect(
        self, frame_bgr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run YOLOv8 on a single BGR frame.

        Returns:
            boxes_xyxy (N,4), scores (N,), class_ids (N,), kept_indices (N,)
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
            )

        # Ultralytics expects RGB input (H,W,C) uint8 or PIL
        img_rgb = frame_bgr[:, :, ::-1]

        # Run inference: use `conf` param to speed-up filtering at model side,
        # but still do an explicit thresholding to ensure consistency.
        # Set verbose=False to avoid printing per-frame info.
        results = self.model.predict(
            source=img_rgb, conf=self.conf_threshold, verbose=False
        )

        # results is a list-like; take first (single image) result
        if len(results) == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
            )

        res = results[0]

        # Ultralytics Box API: res.boxes (has .xyxy, .conf, .cls); may be tensors or lists
        boxes_xyxy = np.empty((0, 4), dtype=np.float32)
        scores = np.empty((0,), dtype=np.float32)
        class_ids = np.empty((0,), dtype=np.int32)

        if hasattr(res, "boxes") and res.boxes is not None:
            # res.boxes.xyxy, res.boxes.conf, res.boxes.cls
            try:
                xyxy = (
                    res.boxes.xyxy.cpu().numpy()
                    if hasattr(res.boxes.xyxy, "cpu")
                    else np.asarray(res.boxes.xyxy)
                )
                confs = (
                    res.boxes.conf.cpu().numpy()
                    if hasattr(res.boxes.conf, "cpu")
                    else np.asarray(res.boxes.conf)
                )
                cls = (
                    res.boxes.cls.cpu().numpy()
                    if hasattr(res.boxes.cls, "cpu")
                    else np.asarray(res.boxes.cls)
                )
            except Exception:
                # Sometimes ultralytics may return lists of floats
                xyxy = np.asarray(res.boxes.xyxy)
                confs = np.asarray(res.boxes.conf)
                cls = np.asarray(res.boxes.cls)

            if xyxy is None or len(xyxy) == 0:
                return (
                    np.empty((0, 4), dtype=np.float32),
                    np.empty((0,), dtype=np.float32),
                    np.empty((0,), dtype=np.int32),
                    np.empty((0,), dtype=np.int32),
                )

            # Ensure shape Nx4
            boxes_xyxy = xyxy[:, :4].astype(np.float32)
            scores = confs.astype(np.float32)
            class_ids = cls.astype(np.int32)

        else:
            # No boxes attribute (defensive)
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
            )

        # Post-filter by conf_threshold for consistency (even though predict() used conf)
        keep_mask = scores >= float(self.conf_threshold)
        if keep_mask.sum() == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
            )

        boxes_xyxy = boxes_xyxy[keep_mask]
        scores = scores[keep_mask]
        class_ids = class_ids[keep_mask].astype(np.int32)
        kept_indices = np.where(keep_mask)[0].astype(np.int32)

        # Clip boxes to image size (safety)
        h, w = frame_bgr.shape[:2]
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, h)

        return boxes_xyxy, scores, class_ids, kept_indices
