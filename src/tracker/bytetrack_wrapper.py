import numpy as np

from .. import config
from .byte_tracker import BYTETracker


class ByteTrackArgs:
    """Simple args container for ByteTracker initialization"""

    def __init__(self):
        self.track_thresh = 0.5
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.mot20 = False


class ByteTrackWrapper:
    """
    Wrapper for ByteTracker with standard tracking interface
    """

    def __init__(
        self, track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=30
    ):
        # Create args object for BYTETracker
        args = ByteTrackArgs()
        args.track_thresh = track_thresh
        args.track_buffer = track_buffer
        args.match_thresh = match_thresh
        args.mot20 = False

        # Initialize BYTETracker
        self.tracker = BYTETracker(args, frame_rate=frame_rate)
        self.frame_rate = frame_rate

    def update(self, bboxes_xyxy, scores, class_ids, frame_bgr):
        """
        Update tracker with new detections.

        Args:
            bboxes_xyxy: np.ndarray, shape (N, 4), format [x1, y1, x2, y2]
            scores: np.ndarray, shape (N,)
            class_ids: np.ndarray, shape (N,)
            frame_bgr: np.ndarray, original frame (H, W, 3) - not used by ByteTrack

        Returns:
            List of tracked objects: [(x1, y1, x2, y2, track_id, class_name, score), ...]
        """
        # Ensure all inputs are float32
        bboxes_xyxy = np.asarray(bboxes_xyxy, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        class_ids = np.asarray(class_ids, dtype=np.float32)

        # --- FILTER TO ONLY PERSON DETECTIONS ---
        if len(class_ids) > 0:
            mask = np.array(
                [
                    config.CLASSES[int(cid)] in config.CLASSES_TO_TRACK
                    for cid in class_ids
                ]
            )
            bboxes_xyxy = bboxes_xyxy[mask]
            scores = scores[mask]
            class_ids = class_ids[mask]

        # Prepare input for ByteTracker
        # ByteTracker expects: np.ndarray with shape (N, 6)
        # Format: [x1, y1, x2, y2, score, class_id]

        if len(bboxes_xyxy) == 0:
            # No detections, still update tracker to age out old tracks
            output_results = np.empty((0, 6), dtype=np.float32)
        else:
            # Combine bboxes, scores, and class_ids
            output_results = np.concatenate(
                [
                    bboxes_xyxy,
                    scores.reshape(-1, 1),
                    class_ids.reshape(-1, 1).astype(np.float32),
                ],
                axis=1,
            ).astype(np.float32)

        # Get frame dimensions from input frame
        img_h, img_w = frame_bgr.shape[:2]
        img_info = (img_h, img_w)
        img_size = (img_h, img_w)

        # Update tracker
        online_targets = self.tracker.update(output_results, img_info, img_size)

        #  Format ByteTrack output
        tracked_objects = []
        for track in online_targets:
            if not track.is_activated:
                continue

            # Get bounding box in xyxy format
            tlbr = track.tlbr  # [x1, y1, x2, y2]
            track_id = track.track_id
            score = track.score

            # Get class name from class_id stored in track
            if track.class_id is not None and 0 <= track.class_id < len(config.CLASSES):
                class_name = config.CLASSES[int(track.class_id)]
            else:
                class_name = "unknown"

            tracked_objects.append(
                (
                    float(tlbr[0]),  # x1
                    float(tlbr[1]),  # y1
                    float(tlbr[2]),  # x2
                    float(tlbr[3]),  # y2
                    int(track_id),
                    class_name,
                    float(score),
                )
            )

        return tracked_objects
