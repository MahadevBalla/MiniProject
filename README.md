# Real-Time NObject Tracking: Detector Benchmarking with ByteTrack

## Project Summary

This project compares different object detectors for real-time video tracking using **ByteTrack**. It helps you test, analyze, and rank models for performance—including speed (FPS), latency, and stability—using the same videos and tracking pipeline. All results (stats, logs, plots, and outputs) are saved for later viewing or analysis.

### What It Does

- Runs tracking on videos using different detectors
- Logs per-frame statistics: detections, tracks, latency, FPS
- Exports both MOT-format tracking data and detailed logs for each run
- Makes automatic visual analysis plots and summary CSVs
- Ranks models by speed and efficiency

***

## Detector Architectures

| Detector         | Architecture                | Backend         |
| ---------------- | --------------------------- | --------------- |
| Haar Cascade     | Classical CV                | CPU             |
| MobileNet SSD v2 | Lightweight CNN             | CPU/CUDA        |
| SSD300 VGG16     | Larger CNN                  | CPU/CUDA        |
| YOLOv5 (n/s/m)   | CNN transformer hybrid      | PyTorch         |
| YOLOv8 (n/s/m)   | Next-gen CNN (Ultralytics)  | PyTorch/TRT     |
| YOLO TRT Engine  | Fast TensorRT variant       | TensorRT        |

*All models are run with ByteTrack for multi-object tracking.*

***

## Running a Benchmark

Try tracking on any video with any supported detector, for example:

```bash
python3 -m src.aicamera_tracker --input assets/test1.mp4 --detector yolov8 --show_display
```

Supported detectors:
- haar
- mobilenet
- ssd
- yolov5
- yolov8
- fasterrcnn
- unet

Other options (see script help for all):

- `--input` (video file path, or use webcam)
- `--detector` (choose from list above)
- `--device` (e.g. cuda:0 or cpu)
- `--show_display` (show live frames)
- `--output_dir` (where outputs are saved)
- More: See comments and help flags in `src/aicamera_tracker.py`.

***

## Outputs and Logging

After each run, you get:

- **Tracked Video**: Output with detections & tracks (`*_tracked_*.mp4`)
- **MOT Tracking File**: In standard MOT format (`*_mot.txt`)
- **Per-frame Log**: A JSON with all stats (see below) (`*_frame_log.json`)
- **Analysis Plots**: Automatic graphs for latency, FPS, and more (`outputs/analysis_plots/`)
- **Ranking CSV**: Detector performance ranking (`detector_performance_summary.csv`)

### Example Per-Frame Log (`*_frame_log.json`)

Each frame entry includes:

```json
{
  "frame_id": 5,
  "num_dets": 0,
  "num_tracks": 0,
  "det_time_ms": 19.72,
  "trk_time_ms": 0.19,
  "fps": 24.92
}
```

### Example MOT File (`*_mot.txt`)

Each row: frame, track_id, x, y, w, h, conf, ... (standard order):

```
131,2,280.0,115.6,53.2,53.2,1,-1,-1,-1
226,3,753.6,266.8,56.2,56.2,1,-1,-1,-1
```

***

## Directory Structure

```
MiniProject/
├── assets/                # Test videos (test1.mp4, test2.mp4, test3.mp4)
├── models/                # All detectors (XML, .pt, .onnx, .engine, etc.)
├── outputs/               # Results: plots, tracked videos, MOT files, logs
├── scripts/               # Model download & export helpers
├── src/                   # Source code: detector & tracker modules
├── detector_performance_summary.csv  # Auto rankings
├── model_comparison.ipynb            # Optional analysis notebook
└── requirements.txt
```

***

## Analysis and Metrics

Automatically generated:

- **Mean FPS** (average speed)
- **Detection latency**
- **Tracking latency**
- **Total frame latency**
- **Dropout rates**
- **Stability and variance**
- **Pareto efficiency** plots

Check `outputs/analysis_plots/` for all figures, and summary CSV for rankings.

***

## Technical Notes

- All detection results use **ByteTrack** for tracking (not DeepSORT as in the original repo).
- All outputs are standardized to make comparisons reproducible and fair.
- You can tune detection & tracking thresholds via command-line arguments.

***

## Acknowledgements

This project builds on the [AI-Camera repo by Abdur Rahman](https://github.com/abdur75648/AI-Camera), with substantial modifications to use ByteTrack for tracking and fully unified benchmarking.
