"""
YOLOE open-vocabulary visual verification utility.

Two modes:
  - YoloeSession  : persistent subprocess (model loaded once).  Preferred.
    Use .check(frame) for blocking inference or the async API for real-time scans.
  - yoloe_verify  : legacy one-shot subprocess (kept for compatibility).
    Slow (~30 s first call) because it reloads the model every time.
"""

import queue
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

_WEIGHTS_CANDIDATES = [
    Path("/workspace/yoloe-11l-seg.pt"),
    Path(__file__).resolve().parents[4] / "yoloe-11l-seg.pt",  # repo root
]

_WORKER_SCRIPT = Path(__file__).resolve().parent / "_yoloe_worker.py"
_PERSISTENT_WORKER_SCRIPT = Path(__file__).resolve().parent / "_yoloe_persistent_worker.py"


def runtime_conf_thresh(default: float = 0.30) -> float:
    raw = os.environ.get("VLMAPS_YOLOE_CONF_THRESH")
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _find_weights() -> str:
    for p in _WEIGHTS_CANDIDATES:
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"YOLOE weights not found in {_WEIGHTS_CANDIDATES}")


# ── Persistent session ────────────────────────────────────────────────────────

class YoloeSession:
    """Persistent YOLOE subprocess — loads model once, inference on demand.

    Usage (blocking):
        session = YoloeSession(weights, target)
        session.start()          # blocks until model is loaded (~10 s first time)
        found, ann = session.check(frame_rgb)
        session.stop()

    Usage (async, for real-time scans):
        session.start()
        session.start_bg_thread()
        # inside rotation loop:
        session.push_frame(frame_rgb)   # non-blocking submit
        found, ann = session.poll_result()  # non-blocking read
        session.stop_bg_thread()
        session.stop()
    """

    def __init__(self, weights: str, target: str, conf_thresh: float = 0.25):
        self.weights = weights
        self.target = target
        self.conf_thresh = conf_thresh
        self._proc: Optional[subprocess.Popen] = None
        self._tmpdir = tempfile.TemporaryDirectory()
        self._in_path = Path(self._tmpdir.name) / "frame.png"
        self._out_path = Path(self._tmpdir.name) / "result.png"
        self._io_lock = threading.Lock()
        self._ready = False
        # async state
        self._frame_q: queue.Queue = queue.Queue(maxsize=2)
        self._result_q: queue.Queue = queue.Queue(maxsize=2)
        self._bg_running = False
        self._bg_thread: Optional[threading.Thread] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Launch the persistent worker and wait until the model is loaded."""
        self._proc = subprocess.Popen(
            [
                sys.executable, str(_PERSISTENT_WORKER_SCRIPT),
                "--weights", self.weights,
                "--target", self.target,
                "--conf", str(self.conf_thresh),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )
        # Read until "READY" or EOF — diagnostic lines go to stderr, not stdout
        self._ready = False
        while True:
            line = self._proc.stdout.readline()
            if not line:
                # EOF — worker crashed; drain stderr for diagnosis
                try:
                    err = self._proc.stderr.read(4000)
                    if err:
                        print(f"  [YOLOE] Worker stderr:\n{err.strip()}", flush=True)
                except Exception:
                    pass
                break
            if line.strip() == "READY":
                self._ready = True
                break
            # Any other stdout line from the worker — print it for visibility
            print(f"  [YOLOE] worker: {line.rstrip()}", flush=True)
        return self._ready

    def stop(self):
        """Shutdown the persistent worker cleanly."""
        self.stop_bg_thread()
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.write("QUIT\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
        try:
            self._tmpdir.cleanup()
        except Exception:
            pass

    # ── Blocking inference ────────────────────────────────────────────────────

    def check(
        self, frame_rgb: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray], Optional[Tuple[float, float]]]:
        """Run YOLOE synchronously.

        Returns:
            found:       True if the target object was detected.
            ann_rgb:     Annotated frame (RGB) with bounding boxes drawn.
            bbox_center: (cx, cy) pixel coordinates of the best detection,
                         or None if not found.

        After the first call (model already loaded), typically <500 ms.
        """
        if not self._ready or self._proc is None:
            return False, frame_rgb.copy(), None

        with self._io_lock:
            cv2.imwrite(
                str(self._in_path),
                cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
            )
            self._proc.stdin.write(f"{self._in_path}|{self._out_path}\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline().strip()

        # Parse response: "0" or "1|cx|cy"
        parts = line.split("|")
        found = (parts[0] == "1")
        bbox_center: Optional[Tuple[float, float]] = None
        if found and len(parts) == 3:
            try:
                bbox_center = (float(parts[1]), float(parts[2]))
            except ValueError:
                pass

        ann_rgb = frame_rgb.copy()
        if self._out_path.exists():
            ann_bgr = cv2.imread(str(self._out_path))
            if ann_bgr is not None:
                ann_rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)

        return found, ann_rgb, bbox_center

    # ── Async (background thread) API ─────────────────────────────────────────

    def start_bg_thread(self):
        """Start a background thread that feeds frames to the subprocess."""
        if self._bg_running:
            return
        self._bg_running = True
        # Clear stale items
        while not self._frame_q.empty():
            try:
                self._frame_q.get_nowait()
            except queue.Empty:
                break
        while not self._result_q.empty():
            try:
                self._result_q.get_nowait()
            except queue.Empty:
                break
        self._bg_thread = threading.Thread(target=self._bg_loop, daemon=True)
        self._bg_thread.start()

    def stop_bg_thread(self):
        """Stop the background thread (waits for current inference to finish)."""
        if not self._bg_running:
            return
        self._bg_running = False
        if self._bg_thread:
            self._bg_thread.join(timeout=3)
            self._bg_thread = None

    def push_frame(self, frame_rgb: np.ndarray):
        """Submit a frame for async detection. Drops the oldest frame if full."""
        frame = frame_rgb.copy()
        # Replace stale frame rather than blocking
        try:
            self._frame_q.put_nowait(frame)
        except queue.Full:
            try:
                self._frame_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_q.put_nowait(frame)
            except queue.Full:
                pass

    def poll_result(
        self,
    ) -> Tuple[Optional[bool], Optional[np.ndarray], Optional[Tuple[float, float]]]:
        """Return the latest detection result without blocking.

        Returns (None, None, None) if no result is available yet.
        """
        try:
            return self._result_q.get_nowait()
        except queue.Empty:
            return None, None, None

    def _bg_loop(self):
        while self._bg_running:
            try:
                frame = self._frame_q.get(timeout=0.1)
            except queue.Empty:
                continue
            found, ann, bbox_center = self.check(frame)
            # Drop stale result and put the new one
            try:
                self._result_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._result_q.put_nowait((found, ann, bbox_center))
            except queue.Full:
                pass


# ── Module-level session cache ────────────────────────────────────────────────

_session: Optional[YoloeSession] = None


def get_session(target: str, conf_thresh: float = 0.25) -> Optional[YoloeSession]:
    """Return (and lazily create) a persistent YOLOE session for *target*.

    If the target changes, the old session is stopped and a new one started.
    The first call blocks while the model loads (~10 s); subsequent calls
    return immediately.
    """
    global _session

    if _session is not None and _session.target == target:
        return _session

    # Stop previous session if target changed
    if _session is not None:
        _session.stop()
        _session = None

    try:
        weights = _find_weights()
    except FileNotFoundError as e:
        print(f"  [YOLOE] weights not found: {e}")
        return None

    print(f"  [YOLOE] Starting persistent session for '{target}' (loading model…)")
    sess = YoloeSession(weights, target, conf_thresh)
    ok = sess.start()
    if not ok:
        print("  [YOLOE] Worker failed to start.")
        return None

    print("  [YOLOE] Session ready.")
    _session = sess
    return _session


def shutdown_session():
    """Stop the global session (call on program exit)."""
    global _session
    if _session is not None:
        _session.stop()
        _session = None


# ── Legacy one-shot helper (kept for compatibility) ───────────────────────────

def yoloe_verify(
    frame_rgb: np.ndarray,
    target_name: str,
    conf_thresh: float = 0.25,
) -> Tuple[bool, Optional[np.ndarray]]:
    """One-shot YOLOE via a fresh subprocess.

    Slow on first call (model reload ~30 s).  Prefer get_session() instead.
    """
    weights = _find_weights()

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = Path(tmpdir) / "frame.png"
        out_path = Path(tmpdir) / "result.png"
        flag_path = Path(tmpdir) / "found.txt"

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
            timeout=60,
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
