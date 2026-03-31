"""
YOLOE open-vocabulary visual verification utility.

Uses YOLOE-11L-Seg to check whether a queried object is visible in a camera frame.
Returns (found: bool, annotated_frame: np.ndarray | None).
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional

_model = None
_WEIGHTS_CANDIDATES = [
    Path("/workspace/yoloe-11l-seg.pt"),
    Path(__file__).resolve().parents[4] / "yoloe-11l-seg.pt",  # repo root
]


def _load_model():
    global _model
    if _model is not None:
        return _model
    from ultralytics import YOLO
    for p in _WEIGHTS_CANDIDATES:
        if p.exists():
            _model = YOLO(str(p))
            return _model
    raise FileNotFoundError(f"YOLOE weights not found in {_WEIGHTS_CANDIDATES}")


def yoloe_verify(
    frame_rgb: np.ndarray,
    target_name: str,
    conf_thresh: float = 0.25,
) -> Tuple[bool, Optional[np.ndarray]]:
    """Run YOLOE on a single RGB frame and check for target_name.

    Args:
        frame_rgb: (H, W, 3) uint8 RGB image.
        target_name: open-vocabulary text query (e.g. "laptop").
        conf_thresh: minimum confidence to consider a detection valid.

    Returns:
        (found, annotated_frame) where annotated_frame is the frame with
        bounding boxes drawn, or None if detection failed.
    """
    model = _load_model()
    model.set_classes([target_name])
    results = model.predict(frame_rgb, conf=conf_thresh, verbose=False)

    if not results or len(results[0].boxes) == 0:
        return False, frame_rgb.copy()

    ann = results[0].plot()  # BGR annotated image
    # Convert back to RGB for consistency
    ann_rgb = ann[:, :, ::-1].copy()
    return True, ann_rgb
