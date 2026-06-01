"""Draw the projected gate (center + bbox) onto captured frames for visual QA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="e.g. dataset/smoke")
    ap.add_argument("--out", default=None, help="output dir (default <run_dir>/overlay)")
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    run = Path(args.run_dir)
    out = Path(args.out) if args.out else run / "overlay"
    out.mkdir(parents=True, exist_ok=True)

    labels = [json.loads(l) for l in (run / "labels.jsonl").read_text().splitlines()]
    drawn = 0
    for lab in labels:
        if drawn >= args.limit:
            break
        img_path = run / "frames" / f"{lab['frame']:06d}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        if lab["center_px"]:
            u, v = int(lab["center_px"][0]), int(lab["center_px"][1])
            cv2.circle(img, (u, v), 6, (0, 0, 255), 2)
        if lab["bbox_px"]:
            u0, v0, u1, v1 = (int(x) for x in lab["bbox_px"])
            cv2.rectangle(img, (u0, v0), (u1, v1), (0, 255, 0), 2)
        cv2.putText(img, f"r={lab['range_m']:.1f}m in={lab['in_frame']}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imwrite(str(out / f"{lab['frame']:06d}.jpg"), img)
        drawn += 1
    print(f"wrote {drawn} overlays to {out}")


if __name__ == "__main__":
    main()
