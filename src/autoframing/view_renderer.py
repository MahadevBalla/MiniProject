from typing import Optional, Tuple

import cv2
import numpy as np


class ViewRenderer:
    """
    Renders dual-view layout with normal view and active speaker zoomed view.
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        layout: str = "horizontal",  # 'horizontal' or 'vertical'
        zoom_padding: float = 0.3,
    ):
        """
        Initialize ViewRenderer.

        Args:
            frame_width: Width of input frame
            frame_height: Height of input frame
            layout: Layout mode ('horizontal' for side-by-side, 'vertical' for stacked)
            zoom_padding: Padding around speaker box (0.3 = 30% padding)
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.layout = layout
        self.zoom_padding = zoom_padding

        # Calculate output dimensions
        if layout == "horizontal":
            self.output_width = frame_width * 2
            self.output_height = frame_height
        else:  # vertical
            self.output_width = frame_width
            self.output_height = frame_height * 2

        print(
            f"ViewRenderer initialized: {layout} layout, "
            f"output: {self.output_width}x{self.output_height}"
        )

    def render_dual_view(
        self,
        frame: np.ndarray,
        active_speaker_box: Optional[np.ndarray] = None,
        track_id: Optional[int] = None,
        speech_active: bool = False,
    ) -> np.ndarray:
        """
        Render dual view with normal frame and active speaker zoom.

        Args:
            frame: Original frame (BGR)
            active_speaker_box: Bounding box [x1, y1, x2, y2] or None
            track_id: Active speaker track ID
            speech_active: Whether speech is currently active

        Returns:
            Combined frame with dual view
        """
        # Create output canvas
        output = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)

        # Normal view (left or top)
        normal_view = frame.copy()

        # Draw active speaker indicator on normal view
        if active_speaker_box is not None:
            x1, y1, x2, y2 = map(int, active_speaker_box)
            # Draw box with special color for active speaker
            color = (
                (0, 255, 0) if speech_active else (0, 255, 255)
            )  # Green if speaking, yellow otherwise
            cv2.rectangle(normal_view, (x1, y1), (x2, y2), color, 3)

            # Draw label
            label = f"Active Speaker ID: {track_id}"
            if speech_active:
                label += " [SPEAKING]"

            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            label_y = max(y1 - 10, label_size[1] + 10)
            cv2.rectangle(
                normal_view,
                (x1, label_y - label_size[1] - 10),
                (x1 + label_size[0] + 10, label_y + 5),
                color,
                -1,
            )
            cv2.putText(
                normal_view,
                label,
                (x1 + 5, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )

        # Add "Normal View" label
        cv2.putText(
            normal_view,
            "Normal View",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        # Active speaker view (right or bottom)
        if active_speaker_box is not None:
            speaker_view = self._create_zoomed_view(
                frame, active_speaker_box, speech_active
            )
        else:
            # No active speaker, show placeholder
            speaker_view = self._create_placeholder_view(frame.shape[:2])

        # Combine views based on layout
        if self.layout == "horizontal":
            output[:, : self.frame_width] = normal_view
            output[:, self.frame_width :] = speaker_view
        else:  # vertical
            output[: self.frame_height, :] = normal_view
            output[self.frame_height :, :] = speaker_view

        return output

    def _create_zoomed_view(
        self, frame: np.ndarray, box: np.ndarray, speech_active: bool
    ) -> np.ndarray:
        """
        Create zoomed view of active speaker with padding.

        Args:
            frame: Original frame
            box: Bounding box [x1, y1, x2, y2]
            speech_active: Whether speech is active

        Returns:
            Zoomed and resized view
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box)

        # Calculate box dimensions
        box_w = x2 - x1
        box_h = y2 - y1

        # Add padding
        pad_w = int(box_w * self.zoom_padding)
        pad_h = int(box_h * self.zoom_padding)

        # Expand box with padding
        x1_padded = max(0, x1 - pad_w)
        y1_padded = max(0, y1 - pad_h)
        x2_padded = min(w, x2 + pad_w)
        y2_padded = min(h, y2 + pad_h)

        # Crop zoomed region
        zoomed = frame[y1_padded:y2_padded, x1_padded:x2_padded].copy()

        # Resize to match normal view size
        zoomed_resized = cv2.resize(
            zoomed,
            (self.frame_width, self.frame_height),
            interpolation=cv2.INTER_LINEAR,
        )

        # Add border to indicate active view
        border_color = (0, 255, 0) if speech_active else (0, 255, 255)
        border_thickness = 8
        cv2.rectangle(
            zoomed_resized,
            (border_thickness, border_thickness),
            (self.frame_width - border_thickness, self.frame_height - border_thickness),
            border_color,
            border_thickness,
        )

        # Add label
        label = "Active Speaker View"
        if speech_active:
            label += " [SPEAKING]"

        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        cv2.rectangle(
            zoomed_resized,
            (5, 5),
            (label_size[0] + 15, label_size[1] + 15),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            zoomed_resized,
            label,
            (10, 10 + label_size[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        return zoomed_resized

    def _create_placeholder_view(self, frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create placeholder view when no active speaker.

        Args:
            frame_shape: (height, width) of original frame

        Returns:
            Placeholder view
        """
        placeholder = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        placeholder[:] = (40, 40, 40)  # Dark gray

        # Add text
        text1 = "No Active Speaker"
        text2 = "Waiting for speech..."

        text_size1 = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        text_size2 = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]

        text_x1 = (self.frame_width - text_size1[0]) // 2
        text_y1 = (self.frame_height - text_size1[1]) // 2

        text_x2 = (self.frame_width - text_size2[0]) // 2
        text_y2 = text_y1 + 50

        cv2.putText(
            placeholder,
            text1,
            (text_x1, text_y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (128, 128, 128),
            2,
        )
        cv2.putText(
            placeholder,
            text2,
            (text_x2, text_y2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (100, 100, 100),
            2,
        )

        return placeholder

    def render_single_view_with_speaker(
        self,
        frame: np.ndarray,
        active_speaker_box: Optional[np.ndarray] = None,
        track_id: Optional[int] = None,
        speech_active: bool = False,
    ) -> np.ndarray:
        """
        Render single view with active speaker overlay (for compatibility).

        Args:
            frame: Original frame
            active_speaker_box: Active speaker bounding box
            track_id: Track ID
            speech_active: Speech activity flag

        Returns:
            Frame with active speaker overlay
        """
        output = frame.copy()

        if active_speaker_box is not None:
            x1, y1, x2, y2 = map(int, active_speaker_box)
            color = (0, 255, 0) if speech_active else (0, 255, 255)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)

            label = f"Active Speaker ID: {track_id}"
            if speech_active:
                label += " [SPEAKING]"

            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            label_y = max(y1 - 10, label_size[1] + 10)
            cv2.rectangle(
                output,
                (x1, label_y - label_size[1] - 10),
                (x1 + label_size[0] + 10, label_y + 5),
                color,
                -1,
            )
            cv2.putText(
                output,
                label,
                (x1 + 5, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )

        return output
