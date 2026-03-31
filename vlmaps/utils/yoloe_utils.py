"""
YOLOE open-vocabulary visual verification utility.

Runs YOLOE in a subprocess to avoid CUDA/segfault conflicts with habitat-sim.
Communicates via temporary files.
"""

import subprocess
import sys
import tempfile
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

import cv2

_WEIGHTS_CANDIDATES = [
    Path("/workspace/yoloe-11l-seg.pt"),
    Path(__file__).resolve().parents[4] / "yoloe-11l-seg.pt",  # repo root
]

_WORKER_SCRIPT = Path(__file__).resolve().parent / "_yoloe_worker.py"


def _find_weights() -> str:
    for p in _WEIGHTS_CANDIDATES:
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"YOLOE weights not found in {_WEIGHTS_CANDIDATES}")


def yoloe_verify(
    frame_rgb: np.ndarray,
    target_name: str,
    conf_thresh: float = 0.25,
) -> Tuple[bool, Optional[np.ndarray]]:
    """Run YOLOE on a single RGB frame via subprocess.

    Args:
        frame_rgb: (H, W, 3) uint8 RGB image.
        target_name: open-vocabulary text query (e.g. "laptop").
        conf_thresh: minimum confidence to consider a detection valid.

    Returns:
        (found, annotated_frame) where annotated_frame is the RGB frame
        with bounding boxes drawn, or the original frame if nothing found.
    """
    weights = _find_weights()

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = Path(tmpdir) / "frame.png"
        out_path = Path(tmpdir) / "result.png"
        flag_path = Path(tmpdir) / "found.txt"

        # Save input frame (BGR for cv2)
        cv2.imwrite(str(in_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        result = subprocess.run(
            [
                sys.executable, str(_WORKER_SCRIPT),
                "--weights", weights,
                "--input", str(in_path),
                "--output", str(out_path),
                "--flag", str(flag_path),
                "--target", target_name,
                "--conf", str(conf_thresh),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            print(f"  YOLOE subprocess error: {result.stderr[:300]}")
            return False, frame_rgb.copy()

        found = flag_path.exists()
        if out_path.exists():
            ann_bgr = cv2.imread(str(out_path))
            ann_rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
            return found, ann_rgb

        return found, frame_rgb.copy()
