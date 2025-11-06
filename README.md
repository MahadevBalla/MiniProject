# Autonomous Camera Director

A high-performance computer vision system that combines YOLOv8 detection, ByteTrack multi-object tracking, and intelligent auto-framing to automatically focus on active speakers in real-time video.

## What It Does

- **Detects & Tracks People**: Uses YOLOv8 for fast person detection and ByteTrack for smooth tracking across frames
- **Multi-Camera Input**: Supports multiple webcams simultaneously, each tracked independently.
- **Auto-Frames Active Speakers**: Automatically zooms in on the person who's speaking using audio cues + visual tracking
- **Dual View Display**: Displays the active speaker crop in a dedicated window while keeping all camera feeds stitched horizontally in the main preview.
- **GPU Accelerated**: Leverages TensorRT for fast inference on NVIDIA GPUs

## Tech Stack

- **Detection**: YOLOv8 (optimized with TensorRT)
- **Tracking**: ByteTrack (Kalman filter + IoU matching; per camera tracking)
- **Audio**: Silero VAD for speech detection
- **Acceleration**: NVIDIA TensorRT + CUDA

## Requirements

### Hardware

- NVIDIA GPU (GTX 1660 Ti or better recommended; tested on RTX 4060 8GB)
- Microphone (for auto-framing feature)

### Software

- Ubuntu 24.04 LTS (or compatible Linux)
- Python 3.10+
- NVIDIA Driver: 580.65.06
- CUDA Toolkit: 13.0 (Build 13.0.48)
- cuDNN: 9.0+
- TensorRT: 10.13.03

## Installation

1. **Install NVIDIA Stack**

    First, install the NVIDIA driver, CUDA, cuDNN, and TensorRT. Verify your installation:

    ```bash
    nvidia-smi  # Should show your GPU
    nvcc --version  # Should show CUDA version
    trtexec --help  # Should show TensorRT help
    ```

    If not installed, follow [NVIDIA's official documentation](https://docs.nvidia.com/) to set them up.

2. **Clone Repository**

    ```bash
    git clone https://github.com/MahadevBalla/MiniProject
    cd MiniProject
    ```

3. **Set Up Python Environment**

    **Note**: Install PyTorch with CUDA support before installing other dependencies.

- Create and activate a virtual environment:

    ```bash
    python3.10 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    ```

- Install PyTorch (with CUDA) — choose the command for your setup from the [official PyTorch installation page](https://pytorch.org/get-started/locally/)

    Example for CUDA 13.0:

    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
    ```

- Install remaining dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. **Download Models & TensorRT Engines**

    ```bash
    bash scripts/download_models.sh
    bash scripts/export_trt_engines.sh
    ```

    This downloads the YOLOv8n ONNX model to ```models/detection/``` and exports a TensorRT engine (```models/detection/yolov8n.engine```) for optimized inference.

## Usage

- Run on a video file

    ```bash
    python -m src.aicamera_tracker --input assets/test.mp4 --show_display
    ```

- With Webcam

    ```bash
    python -m src.aicamera_tracker --webcam_id 0 --show_display
    ```

- Multi-Camera Setup

    Each camera feed is tracked separately and stitched into a combined horizontal view; the active speaker crop appears in a separate window.

    ```bash
    python -m src.aicamera_tracker --webcam_id "0,2" --show_display
    ```

Auto-framing is enabled by default and displays two synchronized views:

- Left: Normal view with all tracked people
- Right: Zoomed view of the active speaker

## Project Structure

```
MiniProject/
├── models/
│   └── detection/          # YOLO models (.onnx, .engine)
├── scripts/                # Setup scripts
├── src/
│   ├── autoframing/        # Auto-framing logic
│   │   ├── auto_framer.py  # Active speaker tracking
│   │   ├── smoothing.py    # Box smoothing
│   │   └── view_renderer.py # Dual view rendering
│   ├── detector/
│   │   └── yolo_detector.py # YOLOv8 wrapper
│   ├── tracker/
│   │   ├── core/           # ByteTrack core logic
│   │   │   ├── basetrack.py
│   │   │   ├── kalman_filter.py
│   │   │   └── matching.py
│   │   ├── byte_tracker.py  # ByteTrack implementation
│   │   └── bytetrack_wrapper.py # Wrapper interface
│   ├── trt_utils/          # TensorRT engine handling
│   ├── utils/              # Helpers (visualization, etc.)
│   ├── audio_director.py   # Audio processing
│   ├── config.py           # Configuration
│   └── aicamera_tracker.py # Main entry point
├── assets/                 # Demo videos
├── outputs/                # Saved videos go here
└── requirements.txt
```

### Acknowledgements

This project builds upon the [AI-Camera](https://github.com/abdur75648/AI-Camera) repository by Abdur Rahman (MIT License).

YOLOv8 + TensorRT integration was adapted and extended; ByteTrack and auto-framing modules were developed independently.
