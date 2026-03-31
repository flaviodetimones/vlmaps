"""
Subprocess worker for YOLOE inference.
Called by yoloe_utils.yoloe_verify() to isolate CUDA context.
"""

import argparse
import sys
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--flag", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.set_classes([args.target])

    img = cv2.imread(args.input)
    if img is None:
        sys.exit(1)

    results = model.predict(img, conf=args.conf, verbose=False)

    found = results and len(results[0].boxes) > 0

    # Save annotated image
    ann = results[0].plot() if results else img
    cv2.imwrite(args.output, ann)

    # Write flag file only if found
    if found:
        Path(args.flag).write_text("1")


if __name__ == "__main__":
    main()
