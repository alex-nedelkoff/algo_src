#!/usr/bin/env python3
"""Compute VIO KPIs using the evo Python API.

Usage:
    python3 evaluate_kpi.py \
        --estimate <tum_estimate> \
        --groundtruth <euroc_gt_csv> \
        --estimate-raw <raw_openvins_output> \
        --output-dir <output_directory>
"""

import argparse
import json
import os
import sys

import copy

import numpy as np
from evo.core import metrics, sync
from evo.core.geometry import umeyama_alignment
from evo.core.metrics import PoseRelation, Unit
from evo.tools import file_interface


def align_trajectory(traj_est, traj_gt):
    """Align estimate to ground truth via Umeyama (rotation + translation + scale)."""
    r, t, s = umeyama_alignment(
        traj_est.positions_xyz.T, traj_gt.positions_xyz.T, with_scale=True
    )
    aligned = copy.deepcopy(traj_est)
    aligned.scale(s)
    aligned.transform(np.vstack([
        np.hstack([r, t.reshape(3, 1)]),
        [0, 0, 0, 1],
    ]))
    return aligned


def compute_ate_translation(traj_est, traj_gt):
    """ATE translation RMSE after SE(3) Umeyama alignment with scale correction."""
    traj_est_aligned = align_trajectory(traj_est, traj_gt)
    data = (traj_est_aligned, traj_gt)
    ape_metric = metrics.APE(PoseRelation.translation_part)
    ape_metric.process_data(data)
    return ape_metric.get_statistic(metrics.StatisticsType.rmse), traj_est_aligned


def compute_ate_rotation(traj_est_aligned, traj_gt):
    """ATE rotation RMSE in degrees on the already-aligned trajectory."""
    data = (traj_est_aligned, traj_gt)
    ape_metric = metrics.APE(PoseRelation.rotation_angle_deg)
    ape_metric.process_data(data)
    return ape_metric.get_statistic(metrics.StatisticsType.rmse)


def compute_rpe_translation(traj_est_aligned, traj_gt):
    """RPE translation RMSE at ~1s delta (20 frames at 20Hz)."""
    data = (traj_est_aligned, traj_gt)
    rpe_metric = metrics.RPE(
        PoseRelation.translation_part, delta=20, delta_unit=Unit.frames
    )
    rpe_metric.process_data(data)
    return rpe_metric.get_statistic(metrics.StatisticsType.rmse)


def compute_completeness(num_associated, num_gt_total):
    """Completeness as fraction of GT poses that have an associated estimate."""
    return num_associated / num_gt_total


def compute_latency(raw_estimate_path):
    """Parse timestamps from raw OpenVINS output and compute inter-frame latency."""
    timestamps = []
    with open(raw_estimate_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            timestamps.append(float(cols[0]))

    if len(timestamps) < 2:
        return 0.0, 0.0

    deltas_ms = np.diff(timestamps) * 1000.0
    return float(np.median(deltas_ms)), float(np.percentile(deltas_ms, 95))


def write_aligned_tum(traj_aligned, output_path):
    """Write aligned trajectory in TUM format."""
    file_interface.write_tum_trajectory_file(output_path, traj_aligned)


def main():
    parser = argparse.ArgumentParser(
        description="Compute VIO KPIs using the evo Python API."
    )
    parser.add_argument("--estimate", required=True, help="TUM format estimate trajectory")
    parser.add_argument("--groundtruth", required=True, help="EuRoC ground truth CSV")
    parser.add_argument("--estimate-raw", required=True, help="Raw OpenVINS total state output")
    parser.add_argument("--output-dir", required=True, help="Directory for output files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load trajectories
    traj_est = file_interface.read_tum_trajectory_file(args.estimate)
    traj_gt = file_interface.read_euroc_csv_trajectory(args.groundtruth)
    traj_gt_full_len = len(traj_gt.timestamps)

    # Time-associate
    traj_gt, traj_est = sync.associate_trajectories(traj_gt, traj_est)

    # ATE translation (also produces aligned trajectory)
    ate_trans, traj_est_aligned = compute_ate_translation(traj_est, traj_gt)

    # ATE rotation
    ate_rot = compute_ate_rotation(traj_est_aligned, traj_gt)

    # RPE @ 1s
    rpe_trans = compute_rpe_translation(traj_est_aligned, traj_gt)

    # Completeness
    completeness = compute_completeness(len(traj_est.timestamps), traj_gt_full_len)

    # Latency
    latency_median, latency_p95 = compute_latency(args.estimate_raw)

    # Build report
    report = {
        "ate_translation_rmse": round(ate_trans, 6),
        "ate_rotation_rmse_deg": round(ate_rot, 4),
        "rpe_translation_rmse": round(rpe_trans, 6),
        "completeness_pct": round(completeness * 100.0, 2),
        "latency_median_ms": round(latency_median, 3),
        "latency_p95_ms": round(latency_p95, 3),
    }

    # Write outputs
    report_path = os.path.join(args.output_dir, "kpi_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    aligned_path = os.path.join(args.output_dir, "aligned_trajectory.tum")
    write_aligned_tum(traj_est_aligned, aligned_path)

    # Stdout summary
    print("\n=== VIO KPI Report ===")
    print(f"  ATE translation RMSE:  {report['ate_translation_rmse']:.6f} m")
    print(f"  ATE rotation RMSE:     {report['ate_rotation_rmse_deg']:.4f} deg")
    print(f"  RPE translation RMSE:  {report['rpe_translation_rmse']:.6f} m")
    print(f"  Completeness:          {report['completeness_pct']:.2f} %")
    print(f"  Latency median:        {report['latency_median_ms']:.3f} ms")
    print(f"  Latency p95:           {report['latency_p95_ms']:.3f} ms")
    print(f"\nOutputs written to {args.output_dir}/")
    print(f"  - kpi_report.json")
    print(f"  - aligned_trajectory.tum")


if __name__ == "__main__":
    main()
