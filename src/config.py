import random
from pathlib import Path

import cv2

# --- Project Root ---
# Assuming this config.py is in src/, so project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Model Configuration ---
# Paths to TensorRT engine files
YOLO_ENGINE_PATH = PROJECT_ROOT / "models/detection/yolov8n.engine"

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
CLASSES_TO_TRACK = {"person", "car", "bus", "truck", "motorcycle"}

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
DEFAULT_OUTPUT_FPS = 30


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


# --- Autoframing Configuration ---
# Enable/disable autoframing feature
AUTOFRAMING_ENABLED = True  # Set to False to disable autoframing

# Dual view layout
AUTOFRAMING_LAYOUT = "horizontal"  # 'horizontal' (side-by-side) or 'vertical' (stacked)

# Smoothing parameters
AUTOFRAMING_SMOOTHING_SPEECH = 0.5  # Faster response during speech
AUTOFRAMING_SMOOTHING_NORMAL = 0.2  # Slower response during silence

# Active speaker tracking timeout (seconds)
AUTOFRAMING_SPEECH_TIMEOUT = 2.0  # Keep tracking speaker for 2s after speech ends

# Zoom padding around active speaker (0.3 = 30% padding)
AUTOFRAMING_ZOOM_PADDING = 0.3

# Audio processing configuration
AUDIO_ENABLED = True  # Set to False to disable audio processing
AUDIO_SAMPLERATE = 16000  # Audio sample rate
AUDIO_CHANNELS = 1  # Number of audio channels (1 for mono)
AUDIO_DEVICE = None  # None for default audio device, or specify device ID
AUDIO_ENABLE_DOA = (
    False  # Direction of Arrival (requires 4+ channels and pyroomacoustics)
)

# Speaker detection confidence thresholds
SPEAKER_DETECTION_MIN_SCORE = (
    0.5  # Minimum detection score to consider as potential speaker
)
