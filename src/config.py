import random
from pathlib import Path

import cv2

# --- Project Root ---
# Assuming this config.py is in src/, so project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Model Configuration ---
# Paths to TensorRT engine files
YOLO_ENGINE_PATH = PROJECT_ROOT / "models/detection/yolov8n.engine"

# Path to local weights for other detectors
FASTER_RCNN_WEIGHTS_PATH = PROJECT_ROOT / "models/detection/fasterrcnn_resnet50.pth"
SSD300_WEIGHTS_PATH = PROJECT_ROOT / "models/detection/ssd300_vgg16.pth"
MOBILENET_SSD_PROTOTXT = PROJECT_ROOT / "models/detection/mobilenet_ssd_v2.prototxt"
MOBILENET_SSD_WEIGHTS = PROJECT_ROOT / "models/detection/mobilenet_ssd_v2.caffemodel"
YOLOV5N_WEIGHTS = PROJECT_ROOT / "models/detection/yolov5n.pt"
YOLOV5S_WEIGHTS = PROJECT_ROOT / "models/detection/yolov5s.pt"
YOLOV5M_WEIGHTS = PROJECT_ROOT / "models/detection/yolov5m.pt"
YOLOV7_WEIGHTS = PROJECT_ROOT / "models/detection/yolov7.pt"
YOLOV7_TINY_WEIGHTS = PROJECT_ROOT / "models/detection/yolov7-tiny.pt"
YOLOV8N_PT_WEIGHTS = PROJECT_ROOT / "models/detection/yolov8n.pt"
YOLOV8S_PT_WEIGHTS = PROJECT_ROOT / "models/detection/yolov8s.pt"
YOLOV8M_PT_WEIGHTS = PROJECT_ROOT / "models/detection/yolov8m.pt"

# YOLOv8 specific
YOLO_INPUT_SHAPE = (640, 640)  # H, W
YOLO_CONF_THRESHOLD = 0.3  # Confidence threshold for YOLO detections
YOLO_NMS_THRESHOLD = 0.5  # NMS threshold for YOLO (if NMS is done post-inference)
# Note: Ultralytics export often includes NMS in the engine.

# ByteTrack Configuration
BYTETRACK_TRACK_THRESH = 0.5
BYTETRACK_TRACK_BUFFER = 30
BYTETRACK_MATCH_THRESH = 0.8

# --- Class Configuration (COCO for YOLOv8) ---
# List of class names corresponding to the YOLOv8 model's output indices
CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

# --- Tracking Configuration ---
# Specify which classes to track (e.g., only 'person')
# Use a set for efficient lookup
# CLASSES_TO_TRACK = {"person", "car", "bus", "truck", "motorcycle"}
CLASSES_TO_TRACK = {"person"}

# --- Visualization Configuration ---
# Seed for consistent random colors, or remove for fully random colors each run
# random.seed(42)

# Generate a unique color for each class for bounding boxes
CLASS_COLORS = {
    cls_name: [random.randint(0, 255) for _ in range(3)] for cls_name in CLASSES
}

# Fallback color for tracks if class-specific color isn't found (should not happen if configured correctly)
DEFAULT_TRACK_COLOR = (0, 255, 0)  # Green

# Font for text overlay
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_ID = 0.7
FONT_SCALE_INFO = 0.9
FONT_THICKNESS = 2

# --- Video I/O ---
DEFAULT_OUTPUT_FPS = 60


# --- Helper function to get color for a track ---
def get_track_color(class_name):
    """Returns a color for a given class name, or a default color."""
    return CLASS_COLORS.get(class_name, DEFAULT_TRACK_COLOR)


def get_class_color(class_name):
    """Returns a color for a given class name for general detections."""
    return CLASS_COLORS.get(class_name, (200, 200, 200))  # Light gray for unknown


# --- Sanity Checks (Optional, but good for development) ---
if not YOLO_ENGINE_PATH.exists():
    print(f"Warning: YOLO Engine not found at {YOLO_ENGINE_PATH}")
