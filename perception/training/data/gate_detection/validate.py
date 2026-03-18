"""Visual validation tool for gate detection training data.

Loads the manifest, prints dataset statistics, renders annotated sample
images, and cross-checks that corner coordinates land on gate mask pixels.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from perception.training.data.gate_detection.format import CORNER_NAMES, load_sample

log = logging.getLogger(__name__)

# Colors for corner types (BGR for OpenCV): TL=blue, TR=green, BR=red, BL=yellow
CORNER_COLORS_BGR = {
    "TL": (255, 0, 0),
    "TR": (0, 255, 0),
    "BR": (0, 0, 255),
    "BL": (0, 255, 255),
}


def overlay_annotations(
    rgb: np.ndarray,
    gate_mask: np.ndarray,
    corner_coords: list[dict],
) -> np.ndarray:
    """Draw gate mask overlay and corner markers on an RGB image.

    Args:
        rgb: (H, W, 3) uint8 BGR image (will not be mutated).
        gate_mask: (H, W) uint8 instance mask (0=bg, 1+=gate IDs).
        corner_coords: List of gate corner annotation dicts.

    Returns:
        (H, W, 3) uint8 annotated BGR image.
    """
    vis = rgb.copy()

    # Green overlay for gate mask regions
    gate_overlay = np.zeros_like(vis)
    gate_overlay[:, :, 1] = 255  # green channel
    mask_bool = gate_mask > 0
    alpha = 0.35
    vis[mask_bool] = cv2.addWeighted(
        vis[mask_bool], 1.0 - alpha, gate_overlay[mask_bool], alpha, 0
    )

    # Draw corners and gate ID labels
    for gate in corner_coords:
        gate_id = gate["gate_id"]
        corners = gate["corners"]  # [[x, y], ...] TL, TR, BR, BL

        for idx, (cx, cy) in enumerate(corners):
            name = CORNER_NAMES[idx]
            color = CORNER_COLORS_BGR[name]
            center = (int(round(cx)), int(round(cy)))
            cv2.circle(vis, center, 5, color, -1)
            cv2.circle(vis, center, 5, (0, 0, 0), 1)  # black outline

        # Label at the centroid of the four corners
        centroid_x = int(round(np.mean([c[0] for c in corners])))
        centroid_y = int(round(np.mean([c[1] for c in corners])))
        label = f"G{gate_id}"
        cv2.putText(
            vis,
            label,
            (centroid_x + 6, centroid_y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            vis,
            label,
            (centroid_x + 6, centroid_y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

    return vis


def _check_corners_on_mask(
    gate_mask: np.ndarray,
    corner_coords: list[dict],
    tolerance_px: int = 3,
) -> int:
    """Check that every corner coordinate lands within tolerance of a gate mask pixel.

    Returns the number of corners that fail the check.
    """
    h, w = gate_mask.shape[:2]
    fail_count = 0

    for gate in corner_coords:
        for cx, cy in gate["corners"]:
            ix = int(round(cx))
            iy = int(round(cy))

            # Build a search window clamped to image bounds
            y_lo = max(0, iy - tolerance_px)
            y_hi = min(h, iy + tolerance_px + 1)
            x_lo = max(0, ix - tolerance_px)
            x_hi = min(w, ix + tolerance_px + 1)

            patch = gate_mask[y_lo:y_hi, x_lo:x_hi]
            if patch.size == 0 or not np.any(patch > 0):
                fail_count += 1

    return fail_count


def validate_dataset(
    data_dir: str | Path,
    n_vis: int = 10,
    output_dir: str | Path | None = None,
) -> None:
    """Run visual validation on a gate detection dataset.

    1. Loads ``manifest.parquet`` and prints dataset statistics.
    2. Renders annotated sample images for visual inspection.
    3. Cross-checks that corners land on gate mask pixels.

    Args:
        data_dir: Root data directory containing source subdirs and
            ``manifest.parquet``.
        n_vis: Number of random samples to visualize per source.
        output_dir: Directory to write annotated images. Defaults to
            ``data_dir / "validation"``.
    """
    data_dir = Path(data_dir)
    manifest_path = data_dir / "manifest.parquet"

    if output_dir is None:
        output_dir = data_dir / "validation"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Read manifest and print stats ─────────────────────────────
    table = pq.read_table(str(manifest_path))
    sample_ids = table.column("sample_id").to_pylist()
    sources = table.column("source").to_pylist()
    n_gates_col = table.column("n_gates").to_pylist()

    total = len(sample_ids)
    source_counts = Counter(sources)
    gate_dist = Counter(n_gates_col)

    print(f"\n{'='*60}")
    print(f"  Dataset Validation: {data_dir}")
    print(f"{'='*60}")
    print(f"  Total samples: {total}")
    print()
    print("  Per-source counts:")
    for src, cnt in sorted(source_counts.items()):
        print(f"    {src:20s} {cnt:6d}")
    print()
    print("  Gate count distribution:")
    for n_g in sorted(gate_dist.keys()):
        print(f"    {n_g} gate(s): {gate_dist[n_g]:6d}")
    print(f"{'='*60}\n")

    # Group samples by source
    source_samples: dict[str, list[str]] = {}
    for sid, src in zip(sample_ids, sources):
        source_samples.setdefault(src, []).append(sid)

    # ── 2. Visual samples ────────────────────────────────────────────
    rng = np.random.default_rng(seed=42)
    total_fail = 0
    total_checked = 0

    for src in sorted(source_samples.keys()):
        sids = source_samples[src]
        k = min(n_vis, len(sids))
        chosen = rng.choice(len(sids), size=k, replace=False)
        chosen_ids = [sids[i] for i in chosen]

        print(f"  [{src}] Visualizing {k} samples ...")

        for sid in chosen_ids:
            sample_dir = data_dir / sid
            try:
                sample = load_sample(sample_dir)
            except Exception as exc:
                log.warning("Failed to load %s: %s", sid, exc)
                continue

            rgb = sample["rgb"]
            gate_mask = sample["gate_mask"]
            corner_coords = sample["corner_coords"]

            # ── 3. Cross-check corners vs mask ───────────────────────
            fails = _check_corners_on_mask(gate_mask, corner_coords)
            n_corners = sum(len(g["corners"]) for g in corner_coords)
            total_fail += fails
            total_checked += n_corners

            if fails > 0:
                log.warning(
                    "%s: %d / %d corners NOT on gate mask (3px tolerance)",
                    sid,
                    fails,
                    n_corners,
                )

            # Render and save
            vis = overlay_annotations(rgb, gate_mask, corner_coords)
            # Sanitize sample_id for filename (replace / with _)
            safe_name = sid.replace("/", "_")
            out_path = output_dir / f"{safe_name}.png"
            cv2.imwrite(str(out_path), vis)
            log.info("Saved %s", out_path)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n  Cross-check summary:")
    print(f"    Corners checked : {total_checked}")
    print(f"    Corners failing : {total_fail}")
    if total_fail > 0:
        print(f"    WARNING: {total_fail} corner(s) do not land within 3px of a gate mask pixel.")
    else:
        print(f"    All corners pass (within 3px of gate mask).")
    print(f"\n  Annotated images saved to: {output_dir}\n")


def main() -> None:
    """CLI entry point for dataset validation."""
    parser = argparse.ArgumentParser(
        description="Validate gate detection training dataset with visual QA."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root data directory containing source subdirs and manifest.parquet.",
    )
    parser.add_argument(
        "--n-vis",
        type=int,
        default=10,
        help="Number of random samples to visualize per source (default: 10).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for annotated images (default: DATA_DIR/validation).",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    validate_dataset(
        data_dir=args.data_dir,
        n_vis=args.n_vis,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
