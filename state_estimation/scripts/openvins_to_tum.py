#!/usr/bin/env python3
"""Convert OpenVINS save_total_state output to TUM trajectory format.

OpenVINS total state format (space-separated):
  timestamp qx qy qz qw px py pz vx vy vz bg0 bg1 bg2 ba0 ba1 ba2 [calibration...]

TUM trajectory format:
  timestamp tx ty tz qx qy qz qw
"""

import argparse
import math
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Convert OpenVINS total state output to TUM trajectory format."
    )
    parser.add_argument("input_file", help="OpenVINS save_total_state output file")
    parser.add_argument("output_file", help="Output TUM trajectory file")
    args = parser.parse_args()

    poses = []
    nan_count = 0
    line_num = 0

    with open(args.input_file, "r") as f:
        for raw_line in f:
            line_num += 1
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            cols = line.split()
            if len(cols) < 8:
                print(f"WARNING: line {line_num}: expected >=8 columns, got {len(cols)}, skipping")
                continue

            values = []
            has_nan = False
            for c in cols[:8]:
                v = float(c)
                if math.isnan(v):
                    has_nan = True
                values.append(v)

            if has_nan:
                nan_count += 1
                print(f"WARNING: line {line_num}: NaN detected, skipping")
                continue

            timestamp = values[0]
            qx, qy, qz, qw = values[1], values[2], values[3], values[4]
            px, py, pz = values[5], values[6], values[7]

            # TUM format: timestamp tx ty tz qx qy qz qw
            poses.append((timestamp, px, py, pz, qx, qy, qz, qw))

    with open(args.output_file, "w") as f:
        for pose in poses:
            ts = pose[0]
            f.write(f"{ts:.9f} {pose[1]:.6f} {pose[2]:.6f} {pose[3]:.6f} "
                    f"{pose[4]:.6f} {pose[5]:.6f} {pose[6]:.6f} {pose[7]:.6f}\n")

    # Summary
    n = len(poses)
    if n > 0:
        first_ts = poses[0][0]
        last_ts = poses[-1][0]
        duration = last_ts - first_ts
        print(f"Converted {n} poses")
        print(f"  First timestamp: {first_ts:.9f}")
        print(f"  Last timestamp:  {last_ts:.9f}")
        print(f"  Duration:        {duration:.3f} s")
    else:
        print("WARNING: No poses converted")

    if nan_count > 0:
        print(f"  NaN lines skipped: {nan_count}")


if __name__ == "__main__":
    main()
