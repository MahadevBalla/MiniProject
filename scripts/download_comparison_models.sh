#!/bin/bash
set -e

echo "  Downloading Comparison Models for MOT  "

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DETECTION_DIR="$BASE_DIR/models/detection"

mkdir -p "$DETECTION_DIR"
cd "$DETECTION_DIR"

echo "Saving weight files to: $DETECTION_DIR"
echo ""

# 1. Clone YOLO repositories (required for loading .pt models)
cd "$BASE_DIR"

echo "[+] Cloning YOLOv5 repository (required for loading yolov5 .pt)"
if [ ! -d "yolov5" ]; then
    git clone https://github.com/ultralytics/yolov5
    echo "✓ YOLOv5 repo cloned."
else
    echo "✓ YOLOv5 repo already exists. Skipping."
fi
echo ""

echo "[+] Cloning YOLOv7 repository (required for loading yolov7 .pt)"
if [ ! -d "yolov7" ]; then
    git clone https://github.com/WongKinYiu/yolov7
    echo "✓ YOLOv7 repo cloned."
else
    echo "✓ YOLOv7 repo already exists. Skipping."
fi
echo ""

# 2. Download model weights
cd "$DETECTION_DIR"

# YOLOv8
echo "[+] Downloading YOLOv8 models"
wget -nc https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt
wget -nc https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt
wget -nc https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt
echo "✓ YOLOv8 models downloaded."
echo ""

# YOLOv5
echo "[+] Downloading YOLOv5 models"
wget -nc https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt
wget -nc https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
wget -nc https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt
echo "✓ YOLOv5 models downloaded."
echo ""

# YOLOv7
echo "[+] Downloading YOLOv7 models"
wget -nc https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
wget -nc https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
echo "✓ YOLOv7 models downloaded."
echo ""

# YOLO-NAS
echo "[+] Downloading YOLO-NAS models"
wget -nc https://storage.googleapis.com/deci-model-repository/yolo-nas-s.pt
wget -nc https://storage.googleapis.com/deci-model-repository/yolo-nas-m.pt
echo "✓ YOLO-NAS models downloaded."
echo ""

# RT-DETR
echo "[+] Downloading RT-DETR models"
wget -nc https://github.com/ultralytics/assets/releases/download/v8.1.0/rtdetr-l.pt
wget -nc https://github.com/ultralytics/assets/releases/download/v8.1.0/rtdetr-x.pt
echo "✓ RT-DETR models downloaded."