import numpy as np

def smooth_transition(prev_box, new_box, alpha=0.2):
    if prev_box is None:
        return new_box
    else:
        return alpha * new_box + (1 - alpha) * prev_box