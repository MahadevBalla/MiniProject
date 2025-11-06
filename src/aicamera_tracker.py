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


def parse_camera_indices(arg_webcam):
    """
    Accept either an int-like value or a comma-separated list of ints.
    Example: "0,2" -> [0,2]
    """
    if isinstance(arg_webcam, int):
        return [arg_webcam]
    s = str(arg_webcam).strip()
    if not s:
        return [0]
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip().isdigit()]
    if s.isdigit():
        return [int(s)]
    # fallback try casting
    try:
        return [int(s)]
    except Exception:
        return [0]


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the tracker."""
    parser = argparse.ArgumentParser(
        description="AICamera: Real-time Object Detection & Tracking with Autoframing"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input video file. If None, tries to use webcam(s).",
    )
    parser.add_argument(
        "--webcam_id",
        type=str,
        default="0",
        help="Webcam ID(s) to use if --input is not specified.",
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
    caps = []
    cam_indices = []
    source_type = "video"
    video_source_name = None

    if args.input:
        if not Path(args.input).exists():
            print(f"Error: Input video file not found: {args.input}")
            return
        video_source_name = Path(args.input).stem
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print(f"Error: Could not open input video {args.input}")
            return
        caps = [cap]
        cam_indices = [-1]  # -1 indicates file input
        source_type = "video"
    else:
        # Multi-webcam mode: parse webcam ids
        cam_indices = parse_camera_indices(args.webcam_id)
        caps = []
        for ci in cam_indices:
            cap = cv2.VideoCapture(ci)
            if cap.isOpened():
                caps.append(cap)
            else:
                print(f"Warning: Could not open webcam with ID {ci}. Skipping.")
        video_source_name = "cams_" + "_".join(map(str, cam_indices))
        source_type = "webcam_multi"

    # Validate at least one opened cap
    opened_caps = [c for c in caps if c is not None and c.isOpened()]
    if len(opened_caps) == 0:
        print(f"Error: No video sources could be opened ({video_source_name}).")
        return

    # adopt only opened caps and their indices
    caps_and_indices = []
    for idx, cap in zip(cam_indices, caps):
        if cap is None:
            continue
        if not cap.isOpened():
            continue
        caps_and_indices.append((idx, cap))
    if len(caps_and_indices) == 0:
        print("Error: no cameras available after initialization.")
        return

    # Use first opened cap to infer frame size & fps
    first_idx, first_cap = caps_and_indices[0]
    frame_width = (
        int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or config.YOLO_INPUT_SHAPE[1]
    )
    frame_height = (
        int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or config.YOLO_INPUT_SHAPE[0]
    )
    source_fps = first_cap.get(cv2.CAP_PROP_FPS) or config.DEFAULT_OUTPUT_FPS
    print(
        f"Opened {source_type}: {video_source_name} "
        f"({len(caps_and_indices)} source(s); first: id={first_idx}) "
        f"({frame_width}x{frame_height} @ {source_fps:.2f} FPS)"
    )

    # --- Initialize ByteTrack per camera (keeps track ids local to each cam) ---
    print("Initializing ByteTrack Tracker(s)...")
    bytetrack_trackers = {}
    for cam_id, _ in caps_and_indices:
        try:
            # instantiate one tracker per camera (frame_rate same for all)
            tracker = ByteTrackWrapper(
                track_thresh=args.track_thresh,
                track_buffer=args.track_buffer,
                match_thresh=args.match_thresh,
                frame_rate=source_fps,
            )
            bytetrack_trackers[cam_id] = tracker
            print(f"ByteTrack initialized for cam {cam_id}")
        except Exception as e:
            print(f"Error initializing ByteTrack for cam {cam_id}: {e}")
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
        # for saving we will use stitched top-row (frame_width * num_cams, frame_height)
        num_cams = len(caps_and_indices)
        output_width = frame_width * num_cams
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
    total_time_spent = 0.0
    display_fps = 0.0

    try:
        while True:
            start_time_frame = time.time()

            # Read frames from all opened caps
            frames_by_cam = []  # list of tuples (cam_id, frame_bgr)
            for cam_id, cap in caps_and_indices:
                ret, frame_bgr = cap.read()
                if not ret:
                    # if a camera file ended or camera failed, create a black frame to keep layout stable
                    frame_bgr = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                frames_by_cam.append((cam_id, frame_bgr))

            # Aggregate lists (two formats)
            tracked_objects_with_cam = []  # (x1,y1,x2,y2,tid,cls_name,score,cam_id)
            tracked_objects_simple = (
                []
            )  # (x1,y1,x2,y2,tid,cls_name,score) -- backward compatible

            # For visualization per-camera
            vis_frames = {}

            # 1. Per-camera detection + tracking
            for cam_id, frame_bgr in frames_by_cam:
                # run detection
                try:
                    det_bboxes, det_scores, det_class_ids, _ = yolo_detector.detect(
                        frame_bgr
                    )
                    det_bboxes = (
                        det_bboxes if det_bboxes is not None else np.zeros((0, 4))
                    )
                    det_scores = (
                        det_scores if det_scores is not None else np.zeros((0,))
                    )
                    det_class_ids = (
                        det_class_ids if det_class_ids is not None else np.zeros((0,))
                    )
                    det_bboxes = np.ascontiguousarray(det_bboxes, dtype=np.float32)
                    det_scores = np.ascontiguousarray(det_scores, dtype=np.float32)
                    det_class_ids = np.ascontiguousarray(det_class_ids, dtype=np.int64)
                except Exception as e:
                    print(
                        f"[cam {cam_id}] Error during detection frame {frame_idx}: {e}"
                    )
                    det_bboxes = np.zeros((0, 4), dtype=np.float32)
                    det_scores = np.zeros((0,), dtype=np.float32)
                    det_class_ids = np.zeros((0,), dtype=np.int64)

                # choose tracker for this cam (cam_id could be -1 for file input)
                tracker = bytetrack_trackers.get(cam_id)
                if tracker is None:
                    # fallback to the first tracker in dict
                    tracker = list(bytetrack_trackers.values())[0]

                # tracking
                try:
                    tracked = tracker.update(
                        det_bboxes, det_scores, det_class_ids, frame_bgr.copy()
                    )
                except Exception as e:
                    print(
                        f"[cam {cam_id}] Error during tracking frame {frame_idx}: {e}"
                    )
                    tracked = []

                # tracked items expected: list of tuples (x1,y1,x2,y2,tid,cls_name,score)
                # attach cam_id for fusion but keep simple list for compatibility
                for t in tracked:
                    if len(t) >= 7:
                        x1, y1, x2, y2, tid, cls_name, score = t[:7]
                    else:
                        # fallback protect
                        try:
                            x1, y1, x2, y2 = t[0:4]
                            tid = int(t[4]) if len(t) > 4 else -1
                            cls_name = str(t[5]) if len(t) > 5 else "obj"
                            score = float(t[6]) if len(t) > 6 else 0.0
                        except Exception:
                            x1, y1, x2, y2, tid, cls_name, score = (
                                0,
                                0,
                                0,
                                0,
                                -1,
                                "obj",
                                0.0,
                            )

                    tracked_objects_simple.append(
                        (x1, y1, x2, y2, tid, cls_name, score)
                    )
                    tracked_objects_with_cam.append(
                        (x1, y1, x2, y2, tid, cls_name, score, cam_id)
                    )

                # Prepare local visualization (draw tracks)
                vis_local = frame_bgr.copy()
                tlwhs, obj_ids, scores = [], [], []
                for x1, y1, x2, y2, tid, cls_name, score in tracked:
                    tlwhs.append([x1, y1, x2 - x1, y2 - y1])
                    obj_ids.append(tid)
                    scores.append(score)
                vis_local = visualize.plot_tracking(vis_local, tlwhs, obj_ids, scores)
                # draw camera id label top-left
                cv2.putText(
                    vis_local,
                    f"cam:{cam_id}",
                    (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                vis_frames[cam_id] = vis_local

            # 2. Autoframing / Active speaker update (use camera-aware tracked list)
            active_speaker_info = None
            if autoframing_enabled and auto_framer:
                try:
                    # use speech state if available
                    speech_active = (
                        auto_framer.speech_active
                        if hasattr(auto_framer, "speech_active")
                        else False
                    )
                    # NOTE: pass tracked_objects_with_cam (camera-aware) to AutoFramer
                    active_speaker_info = auto_framer.update(
                        tracked_objects_with_cam, speech_active=speech_active
                    )
                except Exception as e:
                    print(f"Error in autoframing update: {e}")
                    active_speaker_info = None

            # 3. Compose stitched preview (horizontal)
            # ensure ordering of cams is consistent with caps_and_indices
            stitched_cols = []
            for cam_id, _ in caps_and_indices:
                if cam_id in vis_frames:
                    stitched_cols.append(vis_frames[cam_id])
                else:
                    # black placeholder
                    black = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                    cv2.putText(
                        black,
                        f"cam:{cam_id} (no frame)",
                        (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    stitched_cols.append(black)
            try:
                stitched_preview = np.hstack(stitched_cols)
            except Exception:
                # if stacking fails, just show first frame
                stitched_preview = frames_by_cam[0][1].copy()

            # 4. Draw global info panel on the stitched preview
            mode_text = (
                "Autoframing + Audio"
                if (autoframing_enabled and audio_enabled)
                else "Autoframing" if autoframing_enabled else "Tracking Only"
            )
            info_lines = [
                f"AICamera: YOLOv8 + ByteTrack [{mode_text}]",
                f"Input: {video_source_name}",
            ]

            # compute FPS
            end_time_frame = time.time()
            frame_processing_time = max(1e-6, end_time_frame - start_time_frame)
            total_time_spent += frame_processing_time
            if frame_idx > 0 and total_time_spent > 0:
                display_fps = (frame_idx + 1) / total_time_spent
            elif frame_processing_time > 0:
                display_fps = 1.0 / frame_processing_time
            info_lines.append(f"FPS: {display_fps:.2f}")

            # Add audio status if enabled
            if audio_enabled and auto_framer:
                speech_status = (
                    "SPEAKING"
                    if getattr(auto_framer, "speech_active", False)
                    else "Silent"
                )
                info_lines.append(f"Audio: {speech_status}")

            stitched_preview = visualize.draw_info_panel(stitched_preview, info_lines)

            # 5. If autoframing + view_renderer and we have an active speaker, delegate rendering to view_renderer.
            if autoframing_enabled and view_renderer:
                try:
                    # build per-camera frames dict (cam_id -> frame)
                    per_camera_frames = {
                        cam_id: vis_frames.get(
                            cam_id,
                            np.zeros((frame_height, frame_width, 3), dtype=np.uint8),
                        )
                        for cam_id, _ in caps_and_indices
                    }

                    # view_renderer.render_dual_view returns stitched top-row (unchanged sizes)
                    final_frame = view_renderer.render_dual_view(
                        per_camera_frames=per_camera_frames,
                        active_speaker_info=active_speaker_info,
                        speech_active=(
                            auto_framer.speech_active if auto_framer else False
                        ),
                    )
                except Exception as e:
                    # fallback to stitched preview
                    print(f"Error in view rendering: {e}")
                    final_frame = stitched_preview
            else:
                final_frame = stitched_preview

            # 6. Display and Save
            if args.show_display:
                window_name = (
                    "AICamera - Autoframing"
                    if autoframing_enabled
                    else "AICamera Tracking"
                )
                try:
                    cv2.imshow(window_name, final_frame)
                except Exception:
                    pass
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Exiting...")
                    break

            if video_writer and video_writer.isOpened():
                # video_writer expects same dims as output; final_frame should match
                try:
                    video_writer.write(final_frame)
                except Exception:
                    # try a safe resize (last resort)
                    try:
                        resized_out = cv2.resize(
                            final_frame, (output_width, output_height)
                        )
                        video_writer.write(resized_out)
                    except Exception:
                        pass

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx} frames. Current FPS: {display_fps:.2f}")

    except KeyboardInterrupt:
        print("Processing interrupted by user.")
    finally:
        # --- Cleanup ---
        if audio_director:
            try:
                audio_director.stop()
                print("Audio Director stopped.")
            except Exception:
                pass

        # release all captures
        for cam_id, cap in caps_and_indices:
            try:
                if cap:
                    cap.release()
                    print(f"Video source {cam_id} released.")
            except Exception:
                pass

        # destroy ActiveSpeaker window if it exists
        try:
            cv2.destroyWindow("ActiveSpeaker")
        except Exception:
            pass

        if video_writer and video_writer.isOpened():
            try:
                video_writer.release()
                print("Output video writer released.")
            except Exception:
                pass

        if args.show_display:
            try:
                cv2.destroyAllWindows()
                print("Display windows closed.")
            except Exception:
                pass

        avg_fps = (frame_idx / total_time_spent) if total_time_spent > 0 else 0
        print("\n--- Processing Summary ---")
        print(f"Total frames processed: {frame_idx}")
        print(f"Total time: {total_time_spent:.2f} seconds")
        print(f"Average FPS: {avg_fps:.2f}")
        print("AICamera finished.")


if __name__ == "__main__":
    main()
