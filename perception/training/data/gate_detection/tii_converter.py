"""TII drone racing dataset converter.

Converts raw TII gate-detection data to the shared on-disk format
defined in :mod:`perception.training.data.gate_detection.format`.

TII dataset layout::

    autonomous/
      flight-{name}/
        camera_flight-{name}/   ← JPEG images
          000001.jpg
          ...
        labels_flight-{name}/   ← YOLO-style .txt labels (one line per gate)
          000001.txt
          ...

Each label line has 17 tokens:
  class cx cy w h  tlx tly tlv  trx try trv  brx bry brv  blx bly blv

All coordinate values are normalised to [0, 1].  Visibility flags:
  0 = outside image, 2 = inside image.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from perception.training.data.gate_detection.format import (
    generate_corner_heatmaps,
    render_gate_mask,
    save_sample,
)

log = logging.getLogger(__name__)

# Physical gate dimensions used by TII (metres).
_TII_GATE_DIMS_M = [1.52, 1.52]


# ------------------------------------------------------------------ #
# Label parsing
# ------------------------------------------------------------------ #


def parse_tii_label(line: str, img_width: int, img_height: int) -> list[dict]:
    """Parse a single TII label line into gate dicts.

    A line encodes exactly one gate.  The gate is *included* only when
    all four corners have visibility == 2 (inside image).

    Args:
        line: Space-separated label string (17 tokens).
        img_width: Image width in pixels (for denormalisation).
        img_height: Image height in pixels (for denormalisation).

    Returns:
        A list containing zero or one gate dict::

            {"gate_id": int, "corners": [[x, y] x 4], "confidence": 1.0}
    """
    tokens = line.strip().split()
    if len(tokens) != 17:
        log.warning("Skipping malformed label line (%d tokens): %s", len(tokens), line.strip())
        return []

    class_id = int(tokens[0])
    # tokens[1:5] are bbox (cx, cy, w, h) — not needed for corner coords.

    # Parse the four corners (TL, TR, BR, BL) starting at token index 5.
    corners: list[list[float]] = []
    for i in range(4):
        base = 5 + i * 3
        nx = float(tokens[base])
        ny = float(tokens[base + 1])
        vis = int(float(tokens[base + 2]))

        if vis != 2:
            # At least one corner is not visible — drop the whole gate.
            return []

        # Denormalise to pixel space.
        px = nx * img_width
        py = ny * img_height
        corners.append([px, py])

    return [
        {
            "gate_id": class_id,
            "corners": corners,
            "confidence": 1.0,
        }
    ]


# ------------------------------------------------------------------ #
# Flight converter
# ------------------------------------------------------------------ #


def convert_tii_flight(
    flight_dir: str | Path,
    output_dir: str | Path,
    max_samples: int | None = None,
) -> int:
    """Convert a single TII flight directory to the shared format.

    Args:
        flight_dir: Path to a flight directory, e.g.
            ``autonomous/flight-{name}``.  Must contain
            ``camera_flight-{name}/`` and ``labels_flight-{name}/``.
        output_dir: Where to write converted samples.
        max_samples: Stop after writing this many samples (``None`` = no limit).

    Returns:
        Number of samples written.
    """
    flight_dir = Path(flight_dir)
    output_dir = Path(output_dir)

    # Discover the image and label sub-directories.
    img_dirs = sorted(flight_dir.glob("camera_flight-*"))
    lbl_dirs = sorted(flight_dir.glob("labels_flight-*"))

    if not img_dirs:
        log.warning("No camera_flight-* directory found in %s", flight_dir)
        return 0
    if not lbl_dirs:
        log.warning("No labels_flight-* directory found in %s", flight_dir)
        return 0

    img_dir = img_dirs[0]
    lbl_dir = lbl_dirs[0]

    # Collect image paths sorted by name.
    img_paths = sorted(img_dir.glob("*.jpg"))
    if not img_paths:
        log.warning("No JPEG images found in %s", img_dir)
        return 0

    written = 0

    for img_path in img_paths:
        if max_samples is not None and written >= max_samples:
            break

        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"

        if not lbl_path.exists():
            log.debug("No label for %s — skipping.", stem)
            continue

        # Read and parse all label lines.
        img = cv2.imread(str(img_path))
        if img is None:
            log.warning("Failed to read image %s — skipping.", img_path)
            continue

        h, w = img.shape[:2]
        label_text = lbl_path.read_text()
        all_gates: list[dict] = []
        for line in label_text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            all_gates.extend(parse_tii_label(line, w, h))

        if not all_gates:
            log.debug("No visible gates for %s — skipping.", stem)
            continue

        # Generate derived data.
        gate_mask = render_gate_mask(all_gates, h, w)
        corner_heatmaps = generate_corner_heatmaps(all_gates, h, w)
        obstacle_mask = np.zeros((h, w), dtype=np.uint8)

        metadata = {
            "source": "tii",
            "flight": flight_dir.name,
            "frame": stem,
            "gate_dims_m": _TII_GATE_DIMS_M,
        }

        sample_dir = output_dir / f"{flight_dir.name}_{stem}"
        save_sample(
            sample_dir,
            rgb=img,
            gate_mask=gate_mask,
            obstacle_mask=obstacle_mask,
            corner_coords=all_gates,
            corner_heatmaps=corner_heatmaps,
            metadata=metadata,
        )

        written += 1
        log.debug("Wrote sample %d: %s", written, sample_dir.name)

    log.info("Converted %d samples from %s", written, flight_dir.name)
    return written


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #


def main() -> None:
    """CLI entrypoint for TII dataset conversion."""
    parser = argparse.ArgumentParser(
        description="Convert TII drone racing data to gate-detection training format."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to raw TII data root (contains autonomous/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write converted samples.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=30,
        help="Max samples to convert (default: 30).",
    )
    parser.add_argument(
        "--flights",
        nargs="*",
        default=None,
        help="Optional list of flight names to process (e.g. flight-1 flight-2).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    autonomous_dir = args.data_dir / "autonomous"
    if not autonomous_dir.is_dir():
        log.error("autonomous/ directory not found in %s", args.data_dir)
        raise SystemExit(1)

    # Discover or filter flights.
    if args.flights:
        flight_dirs = [autonomous_dir / f for f in args.flights]
        flight_dirs = [d for d in flight_dirs if d.is_dir()]
    else:
        flight_dirs = sorted(
            d for d in autonomous_dir.iterdir() if d.is_dir()
        )

    if not flight_dirs:
        log.error("No flight directories found.")
        raise SystemExit(1)

    log.info("Processing %d flight(s): %s", len(flight_dirs), [d.name for d in flight_dirs])

    total = 0
    remaining = args.n_samples

    for flight_dir in flight_dirs:
        if remaining is not None and remaining <= 0:
            break

        count = convert_tii_flight(
            flight_dir,
            args.output_dir,
            max_samples=remaining,
        )
        total += count
        if remaining is not None:
            remaining -= count

    log.info("Total: %d samples written to %s", total, args.output_dir)


if __name__ == "__main__":
    main()
