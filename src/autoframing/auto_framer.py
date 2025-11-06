# File: MiniProject/src/autoframing/auto_framer.py
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .smoothing import adaptive_smooth_transition


class AutoFramer:
    """
    Manages active speaker detection by combining audio cues with visual tracking.
    Now camera-aware: expects tracked objects of the form:
      (x1, y1, x2, y2, track_id, class_name, score, cam_id)
    Returns active speaker info as:
      (track_id, smoothed_box, cam_id)
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        smoothing_alpha_speech: float = 0.5,
        smoothing_alpha_normal: float = 0.2,
        speech_timeout: float = 2.0,
    ):
        """
        Initialize AutoFramer.

        Args:
            frame_width: Width of video frame
            frame_height: Height of video frame
            smoothing_alpha_speech: Smoothing factor during speech (faster response)
            smoothing_alpha_normal: Smoothing factor during silence (slower)
            speech_timeout: Seconds after speech ends to keep tracking
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.smoothing_alpha_speech = smoothing_alpha_speech
        self.smoothing_alpha_normal = smoothing_alpha_normal
        self.speech_timeout = speech_timeout

        # Tracking state
        self.active_speaker_id: Optional[int] = None
        self.active_speaker_box: Optional[np.ndarray] = None  # Smoothed box
        self.last_speaker_box: Optional[np.ndarray] = None

        # Audio state
        self.speech_active: bool = False
        self.last_speech_time: float = 0.0

        # Track history for speaker identification
        # keys are (cam_id, track_id) to avoid id collisions across cams
        self.track_speech_scores: Dict[Tuple[int, int], float] = (
            {}
        )  # (cam_id,track_id) -> score

        print(f"AutoFramer initialized: {frame_width}x{frame_height}")

    def on_speech_start(self):
        """Callback when speech starts."""
        self.speech_active = True
        self.last_speech_time = time.time()

    def on_speech_end(self):
        """Callback when speech ends."""
        self.speech_active = False

    def update(
        self, tracked_objects: List[Tuple], speech_active: bool = None
    ) -> Optional[Tuple[int, np.ndarray, int]]:
        """
        Update active speaker tracking based on tracked objects and audio state.

        Args:
            tracked_objects: List of (x1, y1, x2, y2, track_id, class_name, score, cam_id)
            speech_active: Override speech state (if provided)

        Returns:
            Tuple of (active_track_id, smoothed_box, cam_id) or None if no active speaker
        """
        if speech_active is not None:
            if speech_active and not self.speech_active:
                self.on_speech_start()
            elif not speech_active and self.speech_active:
                self.on_speech_end()

        # Check if we should still be tracking (within timeout)
        time_since_speech = time.time() - self.last_speech_time
        tracking_active = self.speech_active or (
            time_since_speech < self.speech_timeout
        )

        if not tracking_active:
            # Reset tracking if timeout exceeded
            self.active_speaker_id = None
            self.active_speaker_box = None
            return None

        # Filter for person class only and ensure we have cam_id
        persons = []
        for obj in tracked_objects:
            # support both 7-field (old) and 8-field (new) formats
            if len(obj) == 7:
                x1, y1, x2, y2, track_id, class_name, score = obj
                cam_id = 0  # unknown cam; fallback to 0
            else:
                x1, y1, x2, y2, track_id, class_name, score, cam_id = obj
            if class_name == "person":
                persons.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                        int(track_id),
                        class_name,
                        float(score),
                        int(cam_id),
                    )
                )

        if not persons:
            # No persons detected, maintain last box if within timeout
            if self.active_speaker_box is not None and tracking_active:
                # return last known active speaker if still in timeout
                return (
                    self.active_speaker_id,
                    self.active_speaker_box,
                    getattr(self, "active_cam_id", 0),
                )
            return None

        # Build set of current (cam_id, track_id)
        current_keys = set()
        for x1, y1, x2, y2, track_id, class_name, score, cam_id in persons:
            current_keys.add((cam_id, track_id))
            if (cam_id, track_id) not in self.track_speech_scores:
                self.track_speech_scores[(cam_id, track_id)] = 0.0

        # Step 1: Decay ALL scores; remove disappeared tracks
        for key in list(self.track_speech_scores.keys()):
            if key in current_keys:
                self.track_speech_scores[key] *= 0.85  # Exponential decay
            else:
                # remove tracks that disappeared
                del self.track_speech_scores[key]

        # Step 2: Boost likely current speaker when speech_active
        if self.speech_active:
            likely_key = None

            # speech just started?
            speech_just_started = (time.time() - self.last_speech_time) < 0.3

            if (
                self.active_speaker_id is not None
                and hasattr(self, "active_cam_id")
                and (self.active_cam_id, self.active_speaker_id) in current_keys
                and not speech_just_started
            ):
                # continue the same speaker if still visible and not a brand-new speech
                likely_key = (self.active_cam_id, self.active_speaker_id)
            else:
                # choose largest person across all cameras
                likely_key = self._select_largest_person(persons)

            if likely_key is not None:
                if likely_key not in self.track_speech_scores:
                    self.track_speech_scores[likely_key] = 0.0
                # strong boost to likely speaker
                self.track_speech_scores[likely_key] += 3.0

        # SPEAKER SELECTION
        if self.speech_active:
            # pick highest-scored key
            if self.track_speech_scores:
                active_key = max(self.track_speech_scores.items(), key=lambda x: x[1])[
                    0
                ]
            else:
                active_key = self._select_largest_person(persons)
        else:
            # during timeout: prefer previous speaker if still visible
            if (
                self.active_speaker_id is not None
                and hasattr(self, "active_cam_id")
                and (self.active_cam_id, self.active_speaker_id) in current_keys
            ):
                active_key = (self.active_cam_id, self.active_speaker_id)
            else:
                if self.track_speech_scores:
                    active_key = max(
                        self.track_speech_scores.items(), key=lambda x: x[1]
                    )[0]
                else:
                    active_key = self._select_largest_person(persons)

        # active_key may be tuple (cam_id, track_id) or None
        if active_key is None:
            return None

        # unpack
        if isinstance(active_key, tuple) and len(active_key) == 2:
            active_cam_id, active_id = active_key
        else:
            # if returned as single int from legacy helper
            active_id = active_key
            active_cam_id = getattr(self, "active_cam_id", 0)

        # get bounding box for the chosen (cam_id,track_id)
        active_box = None
        for x1, y1, x2, y2, track_id, class_name, score, cam_id in persons:
            if cam_id == active_cam_id and track_id == active_id:
                active_box = np.array([x1, y1, x2, y2], dtype=np.float32)
                break

        if active_box is None:
            # if exact pair not found, try matching by track_id ignoring cam
            for x1, y1, x2, y2, track_id, class_name, score, cam_id in persons:
                if track_id == active_id:
                    active_box = np.array([x1, y1, x2, y2], dtype=np.float32)
                    active_cam_id = cam_id
                    break

        if active_box is None:
            return None

        # smoothing (boxes are in camera-local coordinates)
        # If previously active_speaker_box belonged to another camera, reset smoothing to avoid jumps.
        if getattr(self, "active_cam_id", None) != active_cam_id:
            # reset smoothing state for cross-camera jumps
            self.active_speaker_box = None

        self.active_speaker_box = adaptive_smooth_transition(
            self.active_speaker_box,
            active_box,
            speech_active=self.speech_active,
            alpha_speech=self.smoothing_alpha_speech,
            alpha_normal=self.smoothing_alpha_normal,
        )

        self.active_speaker_id = int(active_id)
        self.active_cam_id = int(active_cam_id)
        self.last_speaker_box = self.active_speaker_box.copy()

        # return (track_id, box, cam_id)
        return (self.active_speaker_id, self.active_speaker_box, self.active_cam_id)

    def _select_largest_person(self, persons: List[Tuple]) -> Optional[Tuple[int, int]]:
        """
        Select the person with the largest bounding box area across all cameras.

        Args:
            persons: List of (x1,y1,x2,y2,track_id,class_name,score,cam_id)

        Returns:
            (cam_id, track_id) of the largest person or None
        """
        if not persons:
            return None

        largest_area = 0
        largest_pair = None

        for x1, y1, x2, y2, track_id, class_name, score, cam_id in persons:
            area = max(0.0, (x2 - x1) * (y2 - y1))
            if area > largest_area:
                largest_area = area
                largest_pair = (cam_id, int(track_id))

        return largest_pair

    def get_active_speaker_info(self) -> Optional[Tuple[int, np.ndarray, int]]:
        """
        Get current active speaker information.

        Returns:
            Tuple of (speaker_id, bounding_box, cam_id) or None
        """
        if (
            hasattr(self, "active_speaker_id")
            and self.active_speaker_id is not None
            and self.active_speaker_box is not None
            and hasattr(self, "active_cam_id")
        ):
            return (self.active_speaker_id, self.active_speaker_box, self.active_cam_id)
        return None

    def reset(self):
        """Reset tracking state."""
        self.active_speaker_id = None
        self.active_speaker_box = None
        self.last_speaker_box = None
        self.speech_active = False
        self.track_speech_scores.clear()
        if hasattr(self, "active_cam_id"):
            del self.active_cam_id
