import sys
from pathlib import Path
import torch
import numpy as np

from .. import config


# --------------------------------------------------------------------
# Add local YOLOv5 repo to PYTHONPATH (no installation required)
# --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent  # AutoTrackCam/
YOLOV5_DIR = ROOT / "yolov5"
sys.path.insert(0, str(YOLOV5_DIR))

# Now import only the minimal runtime modules (safe, no deps)
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes


class YOLOv5Detector:
    """
    Minimal YOLOv5 detector using local YOLOv5 repo.
    Fully compatible with your CUDA/TensorRT environment.
    """

    def __init__(
        self,
        variant="s",
        conf_threshold=0.3,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        self.conf_threshold = conf_threshold
        self.device = device

        weight_map = {
            "n": config.YOLOV5N_WEIGHTS,
            "s": config.YOLOV5S_WEIGHTS,
            "m": config.YOLOV5M_WEIGHTS,
        }

        if variant not in weight_map:
            raise ValueError(f"Invalid YOLOv5 variant: {variant}")

        self.weights_path = str(weight_map[variant])

        print(f"Loading YOLOv5-{variant} from: {self.weights_path}")

        # This unpickles the original YOLOv5 checkpoint using the local repo.
        self.model = DetectMultiBackend(
            self.weights_path,
            device=self.device,
            fp16=False,
        )

        self.stride = self.model.stride
        self.img_size = (640, 640)

    def detect(self, frame_bgr):
        import cv2

        # Convert to RGB + letterbox
        img_rgb = frame_bgr[:, :, ::-1]
        img, _, _ = self._letterbox(img_rgb, self.img_size)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).float().to(self.device)
        img /= 255.0
        img = img.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(img)

        pred = non_max_suppression(
            pred, conf_thres=self.conf_threshold, iou_thres=0.45, max_det=300
        )[0]

        if pred is None or len(pred) == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
            )

        pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], frame_bgr.shape).round()

        bboxes = pred[:, :4].cpu().numpy().astype(np.float32)
        scores = pred[:, 4].cpu().numpy().astype(np.float32)
        class_ids = pred[:, 5].cpu().numpy().astype(np.int32)

        ids = np.arange(len(bboxes), dtype=np.int32)

        return bboxes, scores, class_ids, ids

    @staticmethod
    def _letterbox(image, new_shape=(640, 640)):
        import cv2

        shape = image.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(shape[1] * r), int(shape[0] * r)
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(dh), int(dh)
        left, right = int(dw), int(dw)
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return image, (r, r), (dw, dh)
