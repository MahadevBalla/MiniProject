#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define model URLs
YOLO_ONNX_URL="https://raw.githubusercontent.com/nabang1010/YOLO_Object_Tracking_TensorRT/main/models/onnx/yolov8n.onnx"

# Define output directories
MODELS_DIR="./models" # Relative to the scripts directory
DETECTION_MODEL_DIR="$MODELS_DIR/detection"

# Define output filenames
YOLO_ONNX_FILENAME="yolov8n.onnx"

# Create directories if they don't exist
mkdir -p "$DETECTION_MODEL_DIR"

echo "Starting model download..."

# Download YOLOv8 ONNX model
echo "Downloading YOLOv8 ONNX model from $YOLO_ONNX_URL..."
if curl -L "$YOLO_ONNX_URL" -o "$DETECTION_MODEL_DIR/$YOLO_ONNX_FILENAME"; then
    echo "YOLOv8 ONNX model downloaded successfully to $DETECTION_MODEL_DIR/$YOLO_ONNX_FILENAME"
else
    echo "Failed to download YOLOv8 ONNX model. Please check the URL and your internet connection."
    exit 1
fi