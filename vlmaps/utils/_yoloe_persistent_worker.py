"""
Persistent YOLOE worker — loads the model ONCE, then processes frames on demand.

Protocol (stdin -> stdout, line-buffered):
  startup  : prints "READY\\n" when model is loaded.
  request  : reads "<input_path>|<output_path>\\n" from stdin.
  response : prints "1|cx|cy\\n" (found, bbox center px) or "0\\n" (not found).
  shutdown : reads "QUIT\\n" -> exits cleanly.
"""

import argparse
import sys

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

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
