#!/usr/bin/env python3
"""Post-run quality checks comparing TUM estimate to EuRoC ground truth.

Usage: python3 validate_trajectory.py <tum_estimate.txt> <euroc_gt.csv>
"""

import argparse
import math
import sys


def load_tum(path):
    """Load TUM trajectory file. Returns list of (timestamp, tx, ty, tz, qx, qy, qz, qw)."""
    poses = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            poses.append([float(c) for c in cols[:8]])
    return poses


def load_euroc_gt(path):
    """Load EuRoC ground truth CSV. Returns list of timestamps in seconds."""
    timestamps = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split(",")
            ts_ns = float(cols[0])
            timestamps.append(ts_ns / 1e9)
    return timestamps


def main():
    parser = argparse.ArgumentParser(
        description="Validate TUM trajectory against EuRoC ground truth."
    )
    parser.add_argument("tum_estimate", help="TUM format estimate trajectory file")
    parser.add_argument("euroc_gt_csv", help="EuRoC ground truth CSV file")
    args = parser.parse_args()

    results = []

    # Check 1: File exists and non-empty
    try:
        poses = load_tum(args.tum_estimate)
        passed = len(poses) > 0
    except Exception as e:
        poses = []
        passed = False
    results.append(("File exists and non-empty", passed, f"{len(poses)} poses loaded"))

    if not poses:
        # Cannot run further checks
        results.append(("No NaN", False, "no data"))
        results.append(("Monotonic timestamps", False, "no data"))
        results.append(("Sufficient poses (>3000)", False, "no data"))
        results.append(("Duration coverage (>=90%)", False, "no data"))
        results.append(("Position sanity (<100m)", False, "no data"))
        print_summary(results)
        sys.exit(1)

    # Check 2: No NaN
    nan_found = False
    for pose in poses:
        for v in pose:
            if math.isnan(v):
                nan_found = True
                break
        if nan_found:
            break
    results.append(("No NaN", not nan_found, "NaN detected" if nan_found else "clean"))

    # Check 3: Monotonic timestamps
    monotonic = True
    for i in range(1, len(poses)):
        if poses[i][0] <= poses[i - 1][0]:
            monotonic = False
            break
    results.append(("Monotonic timestamps", monotonic,
                     "OK" if monotonic else f"violation at index {i}"))

    # Check 4: Sufficient poses (>3000)
    count = len(poses)
    results.append(("Sufficient poses (>3000)", count > 3000, f"{count} poses"))

    # Check 5: Duration coverage (>=90% of ground truth)
    gt_timestamps = load_euroc_gt(args.euroc_gt_csv)
    est_duration = poses[-1][0] - poses[0][0]
    if len(gt_timestamps) >= 2:
        gt_duration = gt_timestamps[-1] - gt_timestamps[0]
        if gt_duration > 0:
            coverage = est_duration / gt_duration
        else:
            coverage = 0.0
    else:
        gt_duration = 0.0
        coverage = 0.0
    results.append(("Duration coverage (>=90%)", coverage >= 0.9,
                     f"{coverage * 100:.1f}% (est {est_duration:.1f}s / gt {gt_duration:.1f}s)"))

    # Check 6: Position sanity (<100m)
    max_pos = 0.0
    for pose in poses:
        for v in pose[1:4]:  # tx, ty, tz
            if abs(v) > max_pos:
                max_pos = abs(v)
    results.append(("Position sanity (<100m)", max_pos < 100.0,
                     f"max |pos| = {max_pos:.3f} m"))

    print_summary(results)
    all_pass = all(r[1] for r in results)
    sys.exit(0 if all_pass else 1)


def print_summary(results):
    print("\n=== Trajectory Validation ===")
    for name, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {detail}")
    all_pass = all(r[1] for r in results)
    print(f"\nResult: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")


if __name__ == "__main__":
    main()
