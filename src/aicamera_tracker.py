import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from . import config
from .detector.yolo_detector import YOLODetector
from .tracker.bytetrack_wrapper import ByteTrackWrapper
from .utils import visualize

# Import autoframing components (only if enabled)
if config.AUTOFRAMING_ENABLED:
    from .autoframing import AutoFramer, ViewRenderer

# Import audio director (only if enabled)
if config.AUDIO_ENABLED and config.AUTOFRAMING_ENABLED:
    from .audio_director import AudioDirector


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the tracker."""
    parser = argparse.ArgumentParser(
        description="AICamera: Real-time Object Detection & Tracking with Autoframing"
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
        "--no_autoframing",
        action="store_true",
        help="Disable autoframing feature (use original single view).",
    )
    parser.add_argument(
        "--no_audio",
        action="store_true",
        help="Disable audio processing for speaker detection.",
    )

    args = parser.parse_args()
    if args.device == "cpu" and Path(args.yolo_engine).exists():
        print(
            "Warning: TensorRT engine is specified, but device is 'cpu'. TRTEngine will not run on CPU."
        )
    return args


def main():
    """Main function to run the object detection and tracking pipeline with autoframing."""
    args = parse_arguments()

    # Override config with command-line args
    autoframing_enabled = config.AUTOFRAMING_ENABLED and not args.no_autoframing
    audio_enabled = config.AUDIO_ENABLED and not args.no_audio and autoframing_enabled

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

    # --- Initialize Detector ---
    print("Initializing YOLOv8 Detector...")
    try:
        yolo_detector = YOLODetector(
            engine_path=args.yolo_engine,
            conf_threshold=args.conf_thresh,
            device=device,
        )
    except Exception as e:
        print(f"Error initializing YOLO Detector: {e}")
        print("Please ensure YOLO engine path is correct and TensorRT is set up.")
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
    if source_fps == 0:
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
        print(f"ByteTrack initialized")
    except Exception as e:
        print(f"Error initializing ByteTrack Tracker: {e}")
        return

    # --- Initialize Autoframing Components ---
    auto_framer = None
    view_renderer = None
    audio_director = None

    if autoframing_enabled:
        print("Initializing Autoframing Components...")

        try:
            # Initialize AutoFramer
            auto_framer = AutoFramer(
                frame_width=frame_width,
                frame_height=frame_height,
                smoothing_alpha_speech=config.AUTOFRAMING_SMOOTHING_SPEECH,
                smoothing_alpha_normal=config.AUTOFRAMING_SMOOTHING_NORMAL,
                speech_timeout=config.AUTOFRAMING_SPEECH_TIMEOUT,
            )

            # Initialize ViewRenderer
            view_renderer = ViewRenderer(
                frame_width=frame_width,
                frame_height=frame_height,
                layout=config.AUTOFRAMING_LAYOUT,
                zoom_padding=config.AUTOFRAMING_ZOOM_PADDING,
            )

            print("Autoframing components initialized")

        except Exception as e:
            print(f"Error initializing autoframing components: {e}")
            print("Continuing without autoframing...")
            autoframing_enabled = False

    # --- Initialize Audio Director ---
    if audio_enabled:
        print("Initializing Audio Director...")
        try:
            # Setup callbacks
            audio_callbacks = {
                "on_speech_start": lambda: (
                    auto_framer.on_speech_start() if auto_framer else None
                ),
                "on_speech_end": lambda: (
                    auto_framer.on_speech_end() if auto_framer else None
                ),
            }

            audio_director = AudioDirector(
                samplerate=config.AUDIO_SAMPLERATE,
                channels=config.AUDIO_CHANNELS,
                device=config.AUDIO_DEVICE,
                enable_doa=config.AUDIO_ENABLE_DOA,
                callbacks=audio_callbacks,
            )

            # Start audio processing
            audio_director.start()
            print("Audio Director initialized and started")

        except Exception as e:
            print(f"Error initializing Audio Director: {e}")
            print("Continuing without audio processing...")
            audio_enabled = False
            audio_director = None

    # --- Setup Video Output ---
    video_writer = None
    if not args.no_save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.output_filename:
            output_video_name = args.output_filename
            if not output_video_name.lower().endswith((".mp4", ".avi")):
                output_video_name += ".mp4"
        else:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            mode_suffix = "_autoframe" if autoframing_enabled else "_tracked"
            output_video_name = f"{video_source_name}{mode_suffix}_{timestamp}.mp4"

        output_video_path = output_dir / output_video_name

        # Calculate output dimensions
        if autoframing_enabled and view_renderer:
            output_width = view_renderer.output_width
            output_height = view_renderer.output_height
        else:
            output_width = frame_width
            output_height = frame_height

        fourcc = (
            cv2.VideoWriter_fourcc(*"mp4v")
            if output_video_name.lower().endswith(".mp4")
            else cv2.VideoWriter_fourcc(*"XVID")
        )
        video_writer = cv2.VideoWriter(
            str(output_video_path), fourcc, source_fps, (output_width, output_height)
        )
        if video_writer.isOpened():
            print(f"Output video will be saved to: {output_video_path}")
        else:
            print(
                f"Error: Could not open video writer for {output_video_path}. Video will not be saved."
            )
            video_writer = None

    # --- Main Processing Loop ---
    frame_idx = 0
    total_time_spent = 0
    display_fps = 0.0

    try:
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                print("End of video stream or error reading frame.")
                break

            start_time_frame = time.time()

            # 1. Detection
            try:
                det_bboxes, det_scores, det_class_ids, _ = yolo_detector.detect(
                    frame_bgr
                )
                det_bboxes = det_bboxes if det_bboxes is not None else np.zeros((0, 4))
                det_scores = det_scores if det_scores is not None else np.zeros((0,))
                det_class_ids = (
                    det_class_ids if det_class_ids is not None else np.zeros((0,))
                )
                det_bboxes = np.ascontiguousarray(det_bboxes, dtype=np.float32)
                det_scores = np.ascontiguousarray(det_scores, dtype=np.float32)
                det_class_ids = np.ascontiguousarray(det_class_ids, dtype=np.int64)

            except Exception as e:
                print(f"Error during detection on frame {frame_idx}: {e}")
                if args.show_display:
                    cv2.imshow("AICamera Tracking", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # 2. Tracking
            try:
                tracked_objects = bytetrack_tracker.update(
                    det_bboxes,
                    det_scores,
                    det_class_ids,
                    frame_bgr.copy(),
                )
            except Exception as e:
                print(f"Error during tracking on frame {frame_idx}: {e}")
                tracked_objects = []

            # 3. Autoframing Update (if enabled)
            active_speaker_info = None
            if autoframing_enabled and auto_framer:
                try:
                    # Get speech state from audio director or auto_framer
                    speech_active = auto_framer.speech_active if auto_framer else False

                    # Update active speaker tracking
                    active_speaker_info = auto_framer.update(
                        tracked_objects, speech_active=speech_active
                    )
                except Exception as e:
                    print(f"Error in autoframing update: {e}")

            end_time_frame = time.time()
            frame_processing_time = end_time_frame - start_time_frame
            total_time_spent += frame_processing_time
            if frame_idx > 0 and total_time_spent > 0:
                display_fps = (frame_idx + 1) / total_time_spent
            elif frame_processing_time > 0:
                display_fps = 1.0 / frame_processing_time

            # 4. Visualization
            vis_frame = frame_bgr.copy()

            # Convert tracked_objects to tlwhs/ids/scores for visualization
            tlwhs, obj_ids, scores = [], [], []
            for x1, y1, x2, y2, tid, cls_name, score in tracked_objects:
                tlwhs.append([x1, y1, x2 - x1, y2 - y1])
                obj_ids.append(tid)
                scores.append(score)

            # Draw tracking results
            vis_frame = visualize.plot_tracking(vis_frame, tlwhs, obj_ids, scores)

            # Draw info panel
            mode_text = (
                "Autoframing + Audio"
                if (autoframing_enabled and audio_enabled)
                else "Autoframing" if autoframing_enabled else "Tracking Only"
            )
            info_lines = [
                f"AICamera: YOLOv8 + ByteTrack [{mode_text}]",
                f"Input: {video_source_name}",
                f"FPS: {display_fps:.2f}",
            ]

            # Add audio status if enabled
            if audio_enabled and auto_framer:
                speech_status = "SPEAKING" if auto_framer.speech_active else "Silent"
                info_lines.append(f"Audio: {speech_status}")

            vis_frame = visualize.draw_info_panel(vis_frame, info_lines)

            # 5. Render dual view (if autoframing enabled)
            if autoframing_enabled and view_renderer and active_speaker_info:
                track_id, speaker_box = active_speaker_info
                speech_active = auto_framer.speech_active if auto_framer else False

                final_frame = view_renderer.render_dual_view(
                    vis_frame,
                    active_speaker_box=speaker_box,
                    track_id=track_id,
                    speech_active=speech_active,
                )
            elif autoframing_enabled and view_renderer:
                # No active speaker, but still render dual view with placeholder
                final_frame = view_renderer.render_dual_view(vis_frame)
            else:
                # Single view mode
                final_frame = vis_frame

            # 6. Display and Save
            if args.show_display:
                window_name = (
                    "AICamera - Autoframing"
                    if autoframing_enabled
                    else "AICamera Tracking"
                )
                cv2.imshow(window_name, final_frame)
                key = cv2.waitKey(int(1000 / source_fps)) & 0xFF
                if key == ord("q"):
                    print("Exiting...")
                    break

            if video_writer and video_writer.isOpened():
                video_writer.write(final_frame)

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx} frames. Current FPS: {display_fps:.2f}")

    except KeyboardInterrupt:
        print("Processing interrupted by user.")
    finally:
        # --- Cleanup ---
        if audio_director:
            audio_director.stop()
            print("Audio Director stopped.")

        if cap:
            cap.release()
            print("Video source released.")

        if video_writer and video_writer.isOpened():
            video_writer.release()
            print("Output video writer released.")

        if args.show_display:
            cv2.destroyAllWindows()
            print("Display windows closed.")

        avg_fps = (frame_idx / total_time_spent) if total_time_spent > 0 else 0
        print("\n--- Processing Summary ---")
        print(f"Total frames processed: {frame_idx}")
        print(f"Total time: {total_time_spent:.2f} seconds")
        print(f"Average FPS: {avg_fps:.2f}")
        print("AICamera finished.")


if __name__ == "__main__":
    main()
