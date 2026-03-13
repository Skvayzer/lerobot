"""Utilities for visual grounding in CraftNet.

Renders bounding boxes on camera frames to create grounded reference frames
for System 1 object targeting, following the Point-VLA approach.
"""

import numpy as np


def render_grounded_frame(
    frame: np.ndarray,
    bbox: list[float] | tuple[float, ...] | np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 3,
) -> np.ndarray:
    """Draw a bounding box on a camera frame to create a grounded reference.

    Args:
        frame: HWC uint8 numpy array (the camera image).
        bbox: [x1, y1, x2, y2] in normalized 0-1 coordinates.
        color: RGB color tuple for the rectangle.
        thickness: Line thickness in pixels.

    Returns:
        Copy of frame with bounding box drawn on it.
    """
    h, w = frame.shape[:2]
    x1 = int(bbox[0] * w)
    y1 = int(bbox[1] * h)
    x2 = int(bbox[2] * w)
    y2 = int(bbox[3] * h)

    # Clamp to image bounds
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)

    out = frame.copy()
    # Draw rectangle using numpy (no cv2 dependency)
    out[y1 : y1 + thickness, x1:x2] = color
    out[y2 - thickness : y2, x1:x2] = color
    out[y1:y2, x1 : x1 + thickness] = color
    out[y1:y2, x2 - thickness : x2] = color
    return out
