from typing import Optional, Tuple, Dict
import cv2
import numpy as np


class ViewRenderer:
    """
    Displays all normal camera feeds together (side-by-side),
    and shows the active speaker view in a separate OpenCV window.

    Behavior:
      - render_dual_view(...) returns the stitched top-row frame (unchanged resolution per camera).
      - It ALSO updates an OpenCV window named "ActiveSpeaker" with a cropped view (no resizing of top row).
      - The active view is a crop around the active speaker box (with optional small padding).
    """

    def __init__(self, frame_width: int, frame_height: int, zoom_padding: float = 0.3):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.zoom_padding = zoom_padding
        # we don't compute or enforce any global output_width/output_height here
        print("ViewRenderer initialized: separate active speaker window")

    def render_dual_view(
        self,
        per_camera_frames: Dict[int, np.ndarray],
        active_speaker_info: Optional[Tuple[int, np.ndarray, int]] = None,
        speech_active: bool = False,
    ) -> np.ndarray:
        """
        Build stitched top-row (side-by-side) and update separate ActiveSpeaker window.

        Returns:
            top_row (np.ndarray) -- stitched horizontal preview (unchanged per-camera sizes)
        """
        # Build stitched top row from cameras in sorted order
        if not per_camera_frames:
            top_row = self._create_placeholder_view()
        else:
            frames = []
            for cam_id in sorted(per_camera_frames.keys()):
                f = per_camera_frames[cam_id]
                # if frame missing/empty, give a same-size black placeholder
                if f is None or f.size == 0:
                    f = np.zeros(
                        (self.frame_height, self.frame_width, 3), dtype=np.uint8
                    )
                # Do NOT resize frames here; assume they're the expected per-camera size.
                # Draw a small camera label for clarity (in-place)
                try:
                    cv2.putText(
                        f,
                        f"Cam {cam_id}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                except Exception:
                    pass
                frames.append(f)

            # try:
            #     top_row = np.hstack(frames)
            # except Exception:
            #     # fallback to first camera if stacking fails
            #     top_row = frames[0]
            min_h = min(f.shape[0] for f in frames)
            frames = [
                cv2.resize(f, (int(f.shape[1] * min_h / f.shape[0]), min_h))
                for f in frames
            ]
            top_row = np.hstack(frames)

        # Prepare and show active speaker crop in a separate window
        active_view = self._create_placeholder_view()
        if active_speaker_info is not None:
            try:
                # expected (track_id, box, cam_id)
                _, box, cam_id = active_speaker_info
                cam_id = int(cam_id)
                cam_frame = per_camera_frames.get(cam_id)
                if cam_frame is not None and cam_frame.size != 0:
                    active_view = self._create_zoomed_view(
                        cam_frame, box, speech_active
                    )
            except Exception:
                # ignore and keep placeholder
                active_view = self._create_placeholder_view()

        # Update separate window. If the user closed the window manually, imshow will recreate it.
        try:
            cv2.imshow("ActiveSpeaker", active_view)
        except Exception:
            # ignore display errors
            pass

        return top_row

    def _create_zoomed_view(
        self, frame: np.ndarray, box: np.ndarray, speech_active: bool
    ) -> np.ndarray:
        """
        Crop the box area from `frame` with padding, do not scale the stitched (top row).
        Returns the cropped image (natural crop size).
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box)

        # Guard coordinates
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return self._create_placeholder_view()

        box_w = x2 - x1
        box_h = y2 - y1
        pad_w = int(box_w * self.zoom_padding)
        pad_h = int(box_h * self.zoom_padding)

        x1p = max(0, x1 - pad_w)
        y1p = max(0, y1 - pad_h)
        x2p = min(w, x2 + pad_w)
        y2p = min(h, y2 + pad_h)

        crop = frame[y1p:y2p, x1p:x2p].copy()
        if crop.size == 0:
            return self._create_placeholder_view()

        # Draw border and label on the crop (no resizing of the crop for active window)
        color = (0, 255, 0) if speech_active else (0, 255, 255)
        try:
            cv2.rectangle(
                crop, (2, 2), (crop.shape[1] - 3, crop.shape[0] - 3), color, 3
            )
            label = "Active Speaker" + (" [SPEAKING]" if speech_active else "")
            ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            cv2.rectangle(crop, (8, 8), (8 + ts[0] + 8, 8 + ts[1] + 8), (0, 0, 0), -1)
            cv2.putText(
                crop,
                label,
                (12, 12 + ts[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        except Exception:
            pass

        return crop

    def _create_placeholder_view(self) -> np.ndarray:
        img = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)
        text = "No Active Speaker"
        try:
            ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            cv2.putText(
                img,
                text,
                ((self.frame_width - ts[0]) // 2, self.frame_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (128, 128, 128),
                2,
                cv2.LINE_AA,
            )
        except Exception:
            pass
        return img
