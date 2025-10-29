import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .smoothing import adaptive_smooth_transition


class AutoFramer:
    """
    Manages active speaker detection by combining audio cues with visual tracking.
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
        self.track_speech_scores: Dict[int, float] = {}  # track_id -> speech score

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
    ) -> Optional[Tuple[int, np.ndarray]]:
        """
        Update active speaker tracking based on tracked objects and audio state.

        Args:
            tracked_objects: List of (x1, y1, x2, y2, track_id, class_name, score)
            speech_active: Override speech state (if provided)

        Returns:
            Tuple of (active_speaker_id, smoothed_box) or None if no active speaker
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

        # Filter for person class only
        persons = [obj for obj in tracked_objects if obj[5] == "person"]

        if not persons:
            # No persons detected, maintain last box if within timeout
            if self.active_speaker_box is not None and tracking_active:
                return (self.active_speaker_id, self.active_speaker_box)
            return None

        # Get current track IDs
        current_track_ids = set()
        for x1, y1, x2, y2, track_id, class_name, score in persons:
            current_track_ids.add(track_id)

            if track_id not in self.track_speech_scores:
                self.track_speech_scores[track_id] = 0.0

        # Step 1: Decay ALL scores (recency bias)
        for tid in list(self.track_speech_scores.keys()):
            if tid in current_track_ids:
                self.track_speech_scores[tid] *= 0.85  # Exponential decay
            else:
                # Remove tracks that disappeared
                del self.track_speech_scores[tid]

        # Step 2: Boost ONLY the likely current speaker
        if self.speech_active:
            likely_speaker_id = None

            # Check if current speaker is still visible and speech just started
            speech_just_started = (time.time() - self.last_speech_time) < 0.3

            if (
                self.active_speaker_id is not None
                and self.active_speaker_id in current_track_ids
                and not speech_just_started
            ):
                # Continue tracking current speaker if recently active
                likely_speaker_id = self.active_speaker_id
            else:
                # New speech event - select largest person as likely speaker
                likely_speaker_id = self._select_largest_person(persons)

            # Boost only the likely speaker significantly
            if likely_speaker_id is not None:
                if likely_speaker_id not in self.track_speech_scores:
                    self.track_speech_scores[likely_speaker_id] = 0.0
                self.track_speech_scores[likely_speaker_id] += 3.0  # Strong boost

        # SPEAKER SELECTION
        if self.speech_active:
            # During active speech: select person with highest score
            if self.track_speech_scores:
                active_id = max(self.track_speech_scores.items(), key=lambda x: x[1])[0]
            else:
                # Fallback to largest person
                active_id = self._select_largest_person(persons)
        else:
            # During timeout period: maintain current speaker if still visible
            if (
                self.active_speaker_id is not None
                and self.active_speaker_id in current_track_ids
            ):
                active_id = self.active_speaker_id
            else:
                # Current speaker disappeared, select one with highest speech score
                if self.track_speech_scores:
                    active_id = max(
                        self.track_speech_scores.items(), key=lambda x: x[1]
                    )[0]
                else:
                    active_id = self._select_largest_person(persons)

        print(f"[DEBUG]: Scores: {self.track_speech_scores}")
        print(f"[DEBUG]: Selected active_id: {active_id}")

        # Get bounding box for active speaker
        active_box = None
        for x1, y1, x2, y2, track_id, class_name, score in persons:
            if track_id == active_id:
                active_box = np.array([x1, y1, x2, y2], dtype=np.float32)
                break

        if active_box is None:
            return None

        # Apply smoothing
        self.active_speaker_box = adaptive_smooth_transition(
            self.active_speaker_box,
            active_box,
            speech_active=self.speech_active,
            alpha_speech=self.smoothing_alpha_speech,
            alpha_normal=self.smoothing_alpha_normal,
        )

        self.active_speaker_id = active_id
        self.last_speaker_box = self.active_speaker_box.copy()

        return (self.active_speaker_id, self.active_speaker_box)

    def _select_largest_person(self, persons: List[Tuple]) -> int:
        """
        Select the person with the largest bounding box area.

        Args:
            persons: List of person detections

        Returns:
            Track ID of largest person
        """
        if not persons:
            return None

        largest_area = 0
        largest_id = None

        for x1, y1, x2, y2, track_id, class_name, score in persons:
            area = (x2 - x1) * (y2 - y1)
            if area > largest_area:
                largest_area = area
                largest_id = track_id

        return largest_id

    def get_active_speaker_info(self) -> Optional[Tuple[int, np.ndarray]]:
        """
        Get current active speaker information.

        Returns:
            Tuple of (speaker_id, bounding_box) or None
        """
        if self.active_speaker_id is not None and self.active_speaker_box is not None:
            return (self.active_speaker_id, self.active_speaker_box)
        return None

    def reset(self):
        """Reset tracking state."""
        self.active_speaker_id = None
        self.active_speaker_box = None
        self.last_speaker_box = None
        self.speech_active = False
        self.track_speech_scores.clear()
