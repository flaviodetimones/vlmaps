"""
Persistent YOLOE worker — loads the model ONCE, then processes frames on demand.

Protocol (stdin -> stdout, line-buffered):
  startup  : prints "READY\\n" when model is loaded.
  request  : reads "<input_path>|<output_path>\\n" from stdin.
  response : prints "1|cx|cy\\n" (found, bbox center px) or "0\\n" (not found).
  shutdown : reads "QUIT\\n" -> exits cleanly.
"""

import argparse
import shutil
import sys
from pathlib import Path


def _ensure_mobileclip() -> None:
    """Ensure mobileclip_blt.ts is in the ultralytics assets dir.

    Searches candidate repo locations for a known-good copy and installs it.
    If no source is found, removes any corrupt cached file (< 100 MB) so that
    ultralytics can re-download a fresh copy on next model load.
    """
    # _yoloe_persistent_worker.py lives at:
    #   <submodule_root>/vlmaps/utils/_yoloe_persistent_worker.py
    # parents[2] = <submodule_root>  (e.g. /workspace/third_party/vlmaps)
    # parents[4] = workspace root    (e.g. /workspace)
    this = Path(__file__).resolve()
    _MOBILECLIP_CANDIDATES = [
        this.parents[2] / "mobileclip_blt.ts",  # submodule root
        this.parents[4] / "mobileclip_blt.ts",  # /workspace (outer repo root)
        Path("/workspace/mobileclip_blt.ts"),    # explicit absolute fallback
    ]

    src = next((p for p in _MOBILECLIP_CANDIDATES if p.exists()), None)

    try:
        import ultralytics.utils as _uu
        settings = _uu.SETTINGS

        # ultralytics stores assets in 'weights_dir' (v8.x) or 'assets_dir'
        assets_dir: Path | None = None
        for key in ("weights_dir", "assets_dir"):
            raw = settings.get(key, "")
            if raw:
                candidate = Path(str(raw)).expanduser()
                if str(candidate) not in ("", "."):
                    assets_dir = candidate
                    break

        if assets_dir is None:
            # Fallback: use the ultralytics package directory
            assets_dir = Path(_uu.__file__).parent / "assets"

        assets_dir.mkdir(parents=True, exist_ok=True)
        dst = assets_dir / "mobileclip_blt.ts"

        if src is not None:
            # Replace if missing or size differs (corrupt/partial download)
            if not dst.exists() or abs(dst.stat().st_size - src.stat().st_size) > 1024:
                print(
                    f"[worker] Copying mobileclip_blt.ts ({src.stat().st_size // 1_000_000} MB) "
                    f"→ {dst}",
                    flush=True,
                )
                shutil.copy2(src, dst)
            else:
                print(f"[worker] mobileclip_blt.ts already present at {dst}", flush=True)
        else:
            # No source available — remove corrupt cached file (< 100 MB) so
            # ultralytics will re-download a fresh copy
            _MIN_MOBILECLIP_BYTES = 100 * 1024 * 1024  # 100 MB
            if dst.exists() and dst.stat().st_size < _MIN_MOBILECLIP_BYTES:
                print(
                    f"[worker] Removing corrupt mobileclip_blt.ts at {dst} "
                    f"({dst.stat().st_size // 1024} KB < 100 MB) to force re-download",
                    flush=True,
                )
                dst.unlink()
            elif not dst.exists():
                print("[worker] mobileclip_blt.ts not found locally — ultralytics will download it", flush=True)
            else:
                print(f"[worker] mobileclip_blt.ts cached at {dst} ({dst.stat().st_size // 1_000_000} MB)", flush=True)

    except Exception as exc:
        sys.stderr.write(f"[worker] mobileclip pre-copy warning: {exc}\n")


import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    # Ensure the mobileclip text-encoder is available before loading the model
    _ensure_mobileclip()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.set_classes([args.target])

    sys.stdout.write("READY\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "QUIT":
            break

        try:
            input_path, output_path = line.split("|", 1)
        except ValueError:
            sys.stdout.write("0\n")
            sys.stdout.flush()
            continue

        img = cv2.imread(input_path)
        if img is None:
            sys.stdout.write("0\n")
            sys.stdout.flush()
            continue

        results = model.predict(img, conf=args.conf, verbose=False)
        found = bool(results and len(results[0].boxes) > 0)
        ann = results[0].plot() if results else img
        cv2.imwrite(output_path, ann)

        if found:
            # Return bbox center of the highest-confidence detection
            boxes = results[0].boxes
            best = int(boxes.conf.argmax())
            x1, y1, x2, y2 = boxes.xyxy[best].tolist()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            sys.stdout.write(f"1|{cx:.2f}|{cy:.2f}\n")
        else:
            sys.stdout.write("0\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
