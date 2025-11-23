import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from . import config
from .tracker.bytetrack_wrapper import ByteTrackWrapper
from .utils import evaluation, io, visualize


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the tracker."""
    parser = argparse.ArgumentParser(
        description="AICamera: Real-time Object Detection & Tracking"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input video file. If None, tries to use webcam.",
    )
    parser.add_argument(
        "--webcam_id",
        type=int,
        default=0,
        help="Webcam ID to use if --input is not specified.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save the output video.",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,
        help="Name of the output video file. If None, generated from input name or timestamp.",
    )
    parser.add_argument(
        "--show_display",
        action="store_true",
        help="Show the processed video frames in a window.",
    )
    parser.add_argument(
        "--no_save", action="store_true", help="Do not save the output video."
    )
    parser.add_argument(
        "--yolo_engine",
        type=str,
        default=str(config.YOLO_ENGINE_PATH),
        help="Path to the YOLO TensorRT engine file.",
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=config.YOLO_CONF_THRESHOLD,
        help="Confidence threshold for YOLO detections.",
    )
    parser.add_argument(
        "--track_thresh",
        type=float,
        default=config.BYTETRACK_TRACK_THRESH,
        help="Tracking confidence threshold for ByteTrack.",
    )
    parser.add_argument(
        "--track_buffer",
        type=int,
        default=config.BYTETRACK_TRACK_BUFFER,
        help="Number of frames to keep lost tracks in ByteTrack.",
    )
    parser.add_argument(
        "--match_thresh",
        type=float,
        default=config.BYTETRACK_MATCH_THRESH,
        help="Matching threshold for ByteTrack.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for inference (e.g., 'cuda:0', 'cpu'). TensorRT typically requires CUDA.",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="yolo",
        choices=[
            "yolo",
            "fasterrcnn",
            "unet",
            "ssd",
            "mobilenet",
            "haar",
            "yolov5",
            "yolov8",
        ],
        help="Detector to use",
    )

    args = parser.parse_args()
    if args.device == "cpu" and Path(args.yolo_engine).exists():
        print(
            "Warning: TensorRT engine is specified, but device is 'cpu'. TRTEngine will not run on CPU."
        )
    return args


def main():
    """Main function to run the object detection and tracking pipeline."""
    args = parse_arguments()

    # --- Setup Device ---
    if args.device.lower() == "cpu":
        device = torch.device("cpu")
        print(
            "Running on CPU. Note: TensorRT specific features will not be used effectively."
        )
    elif torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        print(
            f"Warning: CUDA device '{args.device}' not available. Falling back to CPU."
        )
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print("Initializing detector...")
    try:
        if args.detector == "yolo":
            from .detector.yolo_detector import YOLODetector

            detector = YOLODetector(
                engine_path=args.yolo_engine,
                conf_threshold=args.conf_thresh,
                device=device,
            )
        elif args.detector == "fasterrcnn":
            from .detector.fasterrcnn_detector import FasterRCNNDetector

            detector = FasterRCNNDetector(
                conf_threshold=args.conf_thresh, device=device
            )
        elif args.detector == "ssd":
            from .detector.ssd_detector import SSDDetector

            detector = SSDDetector(conf_threshold=args.conf_thresh, device=device)
        elif args.detector == "haar":
            from .detector.haar_detector import HaarCascadeDetector

            detector = HaarCascadeDetector()
        elif args.detector == "mobilenet":
            from .detector.mobilenet_ssd_detector import MobileNetSSDDetector

            detector = MobileNetSSDDetector(
                prototxt_path=str(config.MOBILENET_SSD_PROTOTXT),
                weights_path=str(config.MOBILENET_SSD_WEIGHTS),
                conf_threshold=args.conf_thresh,
                use_cuda=(device.type == "cuda"),
            )
        elif args.detector == "unet":
            from .detector.unet_detector import UNetSegDetector

            detector = UNetSegDetector(conf_threshold=args.conf_thresh, device=device)
        elif args.detector == "yolov5":
            from .detector.yolov5_detector import YOLOv5Detector

            detector = YOLOv5Detector(
                variant="s", conf_threshold=args.conf_thresh, device=device
            )
        elif args.detector == "yolov8":
            from .detector.yolov8_detector import YOLOv8Detector

            detector = YOLOv8Detector(
                variant="n", conf_threshold=args.conf_thresh, device=device
            )
        else:
            raise ValueError(f"Unknown detector: {args.detector}")
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return

    # --- Setup Video Input ---
    if args.input:
        if not Path(args.input).exists():
            print(f"Error: Input video file not found: {args.input}")
            return
        video_source_name = Path(args.input).stem
        cap = cv2.VideoCapture(args.input)
        source_type = "video"
    else:
        print(
            f"No input video specified, attempting to use webcam ID: {args.webcam_id}"
        )
        cap = cv2.VideoCapture(args.webcam_id)
        video_source_name = f"webcam_{args.webcam_id}"
        source_type = "webcam"

    if not cap.isOpened():
        print(f"Error: Could not open video source ({video_source_name}).")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps == 0:  # Webcam might return 0
        source_fps = config.DEFAULT_OUTPUT_FPS
    print(
        f"Opened {source_type}: {video_source_name} ({frame_width}x{frame_height} @ {source_fps:.2f} FPS)"
    )

    # --- Initialize Tracker ---
    print("Initializing ByteTrack Tracker...")
    try:
        bytetrack_tracker = ByteTrackWrapper(
            track_thresh=args.track_thresh,
            track_buffer=args.track_buffer,
            match_thresh=args.match_thresh,
            frame_rate=source_fps,
        )
        print("ByteTrack initialized with:")
        print(f"  Track threshold: {args.track_thresh}")
        print(f"  Track buffer: {args.track_buffer}")
        print(f"  Match threshold: {args.match_thresh}")
        print(f"  Frame rate: {source_fps:.2f}")
    except Exception as e:
        print(f"Error initializing ByteTrack Tracker: {e}")
        return

    # --- Setup Video Output (if saving) ---
    video_writer = None
    if not args.no_save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.output_filename:
            output_video_name = args.output_filename
            if not output_video_name.lower().endswith((".mp4", ".avi")):
                output_video_name += ".mp4"  # Default to mp4
        else:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            base_name = f"{video_source_name}_{args.detector}_tracked_{timestamp}"
            output_video_name = base_name + ".mp4"

        output_video_path = output_dir / output_video_name

        # Use MP4V for .mp4, or XVID for .avi for broader compatibility
        fourcc = (
            cv2.VideoWriter_fourcc(*"mp4v")
            if output_video_name.lower().endswith(".mp4")
            else cv2.VideoWriter_fourcc(*"XVID")
        )
        video_writer = cv2.VideoWriter(
            str(output_video_path), fourcc, source_fps, (frame_width, frame_height)
        )
        if video_writer.isOpened():
            print(f"Output video will be saved to: {output_video_path}")
        else:
            print(
                f"Error: Could not open video writer for {output_video_path}. Video will not be saved."
            )
            video_writer = None  # Ensure it's None if opening failed

    # --- Main Processing Loop ---
    frame_idx = 0
    total_time_spent = 0
    display_fps = 0.0
    # --- Evaluation & Logging structures ---
    results_dict = {}  # For MOT-format results
    det_times = []  # Per-frame detector latency
    trk_times = []  # Per-frame tracker latency
    frame_log = []  # Debug JSON log

    try:
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                print("End of video stream or error reading frame.")
                break

            start_time_frame = time.time()
            det_start = time.time()
            # 1. Detection
            try:
                # detector.detect returns: bboxes_xyxy, scores, class_ids, filtered_indices
                det_bboxes, det_scores, det_class_ids, _ = detector.detect(frame_bgr)
                det_bboxes = det_bboxes if det_bboxes is not None else np.zeros((0, 4))
                det_scores = det_scores if det_scores is not None else np.zeros((0,))
                det_class_ids = (
                    det_class_ids if det_class_ids is not None else np.zeros((0,))
                )
                # det_bboxes = det_bboxes.astype(np.float32)
                # det_scores = det_scores.astype(np.float32)
                # det_class_ids = det_class_ids.astype(np.int64)
                det_bboxes = np.ascontiguousarray(det_bboxes, dtype=np.float32)
                det_scores = np.ascontiguousarray(det_scores, dtype=np.float32)
                det_class_ids = np.ascontiguousarray(det_class_ids, dtype=np.int64)

            except Exception as e:
                print(f"Error during detection on frame {frame_idx}: {e}")
                # Optionally, skip tracking for this frame or break
                if args.show_display:
                    cv2.imshow("AICamera Tracking", frame_bgr)  # Show original frame
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            det_time = time.time() - det_start
            det_times.append(det_time)

            trk_start = time.time()
            # 2. Tracking
            try:
                # bytetrack_tracker.update expects: yolo_bboxes_xyxy, yolo_confidences, yolo_class_ids, original_frame_bgr
                # It returns: List of (x1, y1, x2, y2, track_id, class_name, track_confidence)
                tracked_objects = bytetrack_tracker.update(
                    det_bboxes,
                    det_scores,
                    det_class_ids,
                    frame_bgr.copy(),  # Pass a copy if frame_bgr is modified by vis
                )
            except Exception as e:
                print(f"Error during tracking on frame {frame_idx}: {e}")
                tracked_objects = []  # Continue with no tracks for this frame
            trk_time = time.time() - trk_start
            trk_times.append(trk_time)

            end_time_frame = time.time()
            frame_processing_time = end_time_frame - start_time_frame
            total_time_spent += frame_processing_time
            if (
                frame_idx > 0 and total_time_spent > 0
            ):  # Avoid division by zero, smooth FPS
                display_fps = (frame_idx + 1) / total_time_spent
            elif frame_processing_time > 0:
                display_fps = 1.0 / frame_processing_time

            # 3. Visualization
            vis_frame = frame_bgr.copy()  # Draw on a copy

            # Convert tracked_objects to tlwhs/ids/scores format
            tlwhs, obj_ids, scores, frame_results = [], [], [], []
            for x1, y1, x2, y2, tid, cls_name, score in tracked_objects:
                tlwhs.append([x1, y1, x2 - x1, y2 - y1])
                obj_ids.append(tid)
                scores.append(score)
                w = x2 - x1
                h = y2 - y1
                frame_results.append(((x1, y1, w, h), tid))

            # Save MOT-format results for this frame
            results_dict[frame_idx + 1] = frame_results

            # Draw the boxes and IDs
            vis_frame = visualize.plot_tracking(vis_frame, tlwhs, obj_ids, scores)

            # Draw FPS and other info
            info_lines = [
                f"AICamera: {args.detector} + ByteTrack",
                f"Input: {video_source_name}",
                f"FPS: {display_fps:.2f}",
            ]
            vis_frame = visualize.draw_info_panel(vis_frame, info_lines)

            # 4. Display and Save
            if args.show_display:
                cv2.imshow("AICamera Tracking", vis_frame)
                key = cv2.waitKey(int(source_fps)) & 0xFF
                if key == ord("q"):
                    print("Exiting...")
                    break

            if video_writer and video_writer.isOpened():
                video_writer.write(vis_frame)

            frame_log.append(
                {
                    "frame_id": frame_idx,
                    "num_dets": len(det_bboxes),
                    "num_tracks": len(tracked_objects),
                    "det_time_ms": det_time * 1000,
                    "trk_time_ms": trk_time * 1000,
                    "fps": display_fps,
                }
            )

            frame_idx += 1
            if frame_idx % 100 == 0:  # Print progress every 100 frames
                print(f"Processed {frame_idx} frames. Current FPS: {display_fps:.2f}")

    except KeyboardInterrupt:
        print("Processing interrupted by user.")
    finally:
        # --- Cleanup ---
        if cap:
            cap.release()
            print("Video source released.")
        if video_writer and video_writer.isOpened():
            video_writer.release()
            print("Output video writer released.")
        if args.show_display:
            cv2.destroyAllWindows()
            print("Display windows closed.")

        # --- Save MOT results ---
        mot_txt_path = output_dir / f"{base_name}_mot.txt"
        io.write_results(str(mot_txt_path), results_dict, data_type="mot")
        print(f"Saved MOT results to {mot_txt_path}")

        # --- Save per-frame JSON for analysis ---
        frame_log_path = output_dir / f"{base_name}_frame_log.json"
        with open(frame_log_path, "w") as f:
            json.dump(frame_log, f, indent=2)
        print(f"Saved frame log to {frame_log_path}")

        # --- Print timing stats ---
        print(f"Avg detection time (ms): {1000 * np.mean(det_times):.2f}")
        print(f"Avg tracking time  (ms): {1000 * np.mean(trk_times):.2f}")

        # --- MOT evaluation (optional if dataset is available) ---
        try:
            DATA_ROOT = config.MOT_DATA_ROOT  # you must define this in config
            SEQ_NAME = config.MOT_SEQ_NAME  # e.g., "MOT17-04"
            evaluator = evaluation.Evaluator(DATA_ROOT, SEQ_NAME, "mot")
            acc = evaluator.eval_file(str(mot_txt_path))
            summary = evaluation.Evaluator.get_summary([acc], ["YOLOv8_ByteTrack"])
            print(summary)
            evaluation.Evaluator.save_summary(
                summary, str(output_dir / "eval_summary.xlsx")
            )
            print("Saved MOT evaluation to eval_summary.xlsx")
        except Exception as e:
            print(f"Skipping MOT evaluation: {e}")

        avg_fps = (frame_idx / total_time_spent) if total_time_spent > 0 else 0
        print("\n--- Processing Summary ---")
        print(f"Total frames processed: {frame_idx}")
        print(f"Total time: {total_time_spent:.2f} seconds")
        print(f"Average FPS: {avg_fps:.2f}")
        print("AICamera finished.")


if __name__ == "__main__":
    # This structure ensures that if this script is run directly,
    # the main() function is called.
    # For `python -m src.aicamera_tracker`, Python handles the module execution.
    main()
