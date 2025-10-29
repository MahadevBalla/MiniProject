import numpy as np


def smooth_transition(prev_box, new_box, alpha=0.2):
    """
    Apply exponential smoothing to bounding box coordinates.

    Args:
        prev_box: Previous bounding box [x1, y1, x2, y2] or None
        new_box: New bounding box [x1, y1, x2, y2]
        alpha: Smoothing factor (0 = use prev_box, 1 = use new_box)

    Returns:
        Smoothed bounding box [x1, y1, x2, y2]
    """
    if prev_box is None:
        return np.array(new_box, dtype=np.float32)

    prev_box = np.array(prev_box, dtype=np.float32)
    new_box = np.array(new_box, dtype=np.float32)

    return alpha * new_box + (1 - alpha) * prev_box


def adaptive_smooth_transition(
    prev_box, new_box, speech_active=False, alpha_speech=0.5, alpha_normal=0.2
):
    """
    Apply adaptive smoothing based on speech activity.

    Args:
        prev_box: Previous bounding box or None
        new_box: New bounding box
        speech_active: Whether speech is currently detected
        alpha_speech: Smoothing factor during speech (faster response)
        alpha_normal: Smoothing factor during silence (slower response)

    Returns:
        Smoothed bounding box
    """
    alpha = alpha_speech if speech_active else alpha_normal
    return smooth_transition(prev_box, new_box, alpha)
