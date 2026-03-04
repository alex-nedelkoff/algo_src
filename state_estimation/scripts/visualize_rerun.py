#!/usr/bin/env python3
"""Rerun.io visualization for VIO trajectory evaluation.

Usage:
    python3 visualize_rerun.py \
        --estimate /results/aligned_trajectory.tum \
        --groundtruth /data/euroc/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv \
        [--output /results/vio_visualization.rrd] \
        [--serve]
"""

import argparse
import csv
import os
import sys

import numpy as np
import rerun as rr
from evo.core import sync
from evo.tools import file_interface
from PIL import Image


def load_trajectories(estimate_path, groundtruth_path):
    """Load and time-associate estimate and ground truth trajectories."""
    traj_est = file_interface.read_tum_trajectory_file(estimate_path)
    traj_gt = file_interface.read_euroc_csv_trajectory(groundtruth_path)
    traj_gt, traj_est = sync.associate_trajectories(traj_gt, traj_est)
    return traj_est, traj_gt


def load_cam_timestamps(cam_dir):
    """Build sorted array of (timestamp_sec, image_path) from EuRoC cam data.csv."""
    csv_path = os.path.join(cam_dir, "data.csv")
    img_dir = os.path.join(cam_dir, "data")
    entries = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0].startswith("#"):
                continue
            ts_ns = int(row[0])
            filename = row[1].strip()
            entries.append((ts_ns / 1e9, os.path.join(img_dir, filename)))
    entries.sort()
    return np.array([e[0] for e in entries]), [e[1] for e in entries]


def find_nearest_image(cam_timestamps, cam_paths, query_t, max_dt=0.05):
    """Find the nearest camera image to query_t. Returns path or None."""
    idx = np.searchsorted(cam_timestamps, query_t)
    best_idx = None
    best_dt = max_dt
    for candidate in [idx - 1, idx]:
        if 0 <= candidate < len(cam_timestamps):
            dt = abs(cam_timestamps[candidate] - query_t)
            if dt < best_dt:
                best_dt = dt
                best_idx = candidate
    return cam_paths[best_idx] if best_idx is not None else None


def log_trajectories(traj_est, traj_gt, cam_dir=None):
    """Log full 3D trajectory strips and per-frame temporal data."""
    est_xyz = traj_est.positions_xyz
    gt_xyz = traj_gt.positions_xyz
    timestamps = traj_est.timestamps

    cam_timestamps, cam_paths = None, None
    if cam_dir:
        cam_timestamps, cam_paths = load_cam_timestamps(cam_dir)
        print(f"Loaded {len(cam_timestamps)} camera frames from {cam_dir}")

    # Full trajectory line strips (static, logged once)
    rr.log(
        "/world/ground_truth",
        rr.LineStrips3D([gt_xyz], colors=[[0, 200, 0]]),
        static=True,
    )
    rr.log(
        "/world/estimate",
        rr.LineStrips3D([est_xyz], colors=[[220, 40, 40]]),
        static=True,
    )

    # Per-frame temporal logging
    t0 = timestamps[0]
    image_cache = {}
    for i in range(len(timestamps)):
        t = timestamps[i]
        rr.set_time("sim_time", duration=t - t0)

        gt_pos = gt_xyz[i]
        est_pos = est_xyz[i]

        rr.log("/world/gt_pose", rr.Points3D([gt_pos], colors=[[0, 200, 0]], radii=[0.04]))
        rr.log("/world/est_pose", rr.Points3D([est_pos], colors=[[220, 40, 40]], radii=[0.04]))

        # Error line connecting GT and estimate at this frame
        rr.log(
            "/world/error_line",
            rr.LineStrips3D([[gt_pos, est_pos]], colors=[[255, 255, 0]]),
        )

        # Translation error scalar
        error = float(np.linalg.norm(est_pos - gt_pos))
        rr.log("/metrics/ate_error_m", rr.Scalars(error))

        # Camera image synced to this pose timestamp
        if cam_timestamps is not None:
            img_path = find_nearest_image(cam_timestamps, cam_paths, t)
            if img_path:
                # Cache JPEG-encoded bytes to reduce gRPC server memory usage
                if img_path not in image_cache:
                    import io
                    buf = io.BytesIO()
                    Image.open(img_path).save(buf, format="JPEG", quality=80)
                    image_cache[img_path] = buf.getvalue()
                rr.log("/camera/image", rr.EncodedImage(contents=image_cache[img_path], media_type="image/jpeg"))


def main():
    parser = argparse.ArgumentParser(
        description="Rerun.io visualization for VIO trajectory evaluation."
    )
    parser.add_argument("--estimate", required=True, help="Aligned TUM estimate trajectory")
    parser.add_argument("--groundtruth", required=True, help="EuRoC ground truth CSV")
    parser.add_argument("--cam0", default=None, help="EuRoC cam0 directory (contains data.csv and data/)")
    parser.add_argument("--output", default=None, help="Path to save .rrd file")
    parser.add_argument("--serve", action="store_true", help="Launch rr.serve() for browser viewing")
    args = parser.parse_args()

    # Default to saving .rrd if neither flag given
    if not args.serve and args.output is None:
        args.output = "/results/vio_visualization.rrd"

    # Initialize Rerun
    rr.init("vio_trajectory", spawn=False)

    if args.output:
        rr.save(args.output)
        print(f"Will save to {args.output}")

    if args.serve:
        server_uri = rr.serve_grpc(server_memory_limit="4GiB")
        rr.serve_web_viewer(open_browser=False)
        print("Rerun gRPC server + web viewer started")

    # Load and process trajectories
    traj_est, traj_gt = load_trajectories(args.estimate, args.groundtruth)
    print(f"Loaded {len(traj_est.timestamps)} synced poses")

    # Log all data
    log_trajectories(traj_est, traj_gt, cam_dir=args.cam0)
    print("Visualization complete")

    if args.output:
        print(f"Saved: {args.output}")

    if args.serve:
        from urllib.parse import quote
        # Replace container-internal address with localhost for browser access
        browser_uri = server_uri.replace("0.0.0.0", "localhost").replace("127.0.0.1", "localhost")
        viewer_url = f"http://localhost:9090/?url={quote(browser_uri)}"
        print(f"Open in browser: {viewer_url}")
        print("Press Ctrl+C to stop")
        try:
            import signal
            signal.pause()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
