#!/usr/bin/env python3
"""Convert EuRoC MAV ASL format dataset to a ROS2 bag file.

Usage:
    python3 euroc_to_rosbag2.py <input_mav0_dir> <output_bag_dir>

Example:
    python3 euroc_to_rosbag2.py /data/euroc/MH_01_easy/mav0 /data/euroc/MH_01_easy_bag

Requires: rosbags, PIL/Pillow, numpy
"""

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore


def read_cam0_entries(mav0_dir: Path) -> list[tuple[int, Path]]:
    """Read cam0/data.csv and return (timestamp_ns, image_path) pairs."""
    csv_path = mav0_dir / "cam0" / "data.csv"
    entries = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            ts_ns = int(row[0])
            filename = row[1].strip()
            img_path = mav0_dir / "cam0" / "data" / filename
            entries.append((ts_ns, img_path))
    return entries


def read_imu0_entries(mav0_dir: Path) -> list[tuple[int, float, float, float, float, float, float]]:
    """Read imu0/data.csv and return (timestamp_ns, wx, wy, wz, ax, ay, az) tuples."""
    csv_path = mav0_dir / "imu0" / "data.csv"
    entries = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            ts_ns = int(row[0])
            wx, wy, wz = float(row[1]), float(row[2]), float(row[3])
            ax, ay, az = float(row[4]), float(row[5]), float(row[6])
            entries.append((ts_ns, wx, wy, wz, ax, ay, az))
    return entries


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_mav0_dir> <output_bag_dir>")
        sys.exit(1)

    mav0_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not mav0_dir.is_dir():
        print(f"Error: input directory does not exist: {mav0_dir}")
        sys.exit(1)

    # Set up ROS2 Humble typestore
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    # Get message types
    Header = typestore.types["std_msgs/msg/Header"]
    ImageMsg = typestore.types["sensor_msgs/msg/Image"]
    ImuMsg = typestore.types["sensor_msgs/msg/Imu"]
    Vector3 = typestore.types["geometry_msgs/msg/Vector3"]
    Quaternion = typestore.types["geometry_msgs/msg/Quaternion"]
    Time = typestore.types["builtin_interfaces/msg/Time"]

    # Read input data
    print(f"Reading cam0 entries from {mav0_dir / 'cam0' / 'data.csv'} ...")
    cam_entries = read_cam0_entries(mav0_dir)
    print(f"  Found {len(cam_entries)} camera frames")

    print(f"Reading imu0 entries from {mav0_dir / 'imu0' / 'data.csv'} ...")
    imu_entries = read_imu0_entries(mav0_dir)
    print(f"  Found {len(imu_entries)} IMU samples")

    # Build merged timeline: (timestamp_ns, 'cam', index) or (timestamp_ns, 'imu', index)
    timeline = []
    for i, (ts_ns, _) in enumerate(cam_entries):
        timeline.append((ts_ns, "cam", i))
    for i, (ts_ns, *_) in enumerate(imu_entries):
        timeline.append((ts_ns, "imu", i))
    timeline.sort(key=lambda x: x[0])

    print(f"Total messages to write: {len(timeline)}")

    # Write ROS2 bag
    with Writer(output_dir, version=8) as writer:
        cam_conn = writer.add_connection(
            "/cam0/image_raw",
            ImageMsg.__msgtype__,
            typestore=typestore,
            serialization_format="cdr",
        )
        imu_conn = writer.add_connection(
            "/imu0",
            ImuMsg.__msgtype__,
            typestore=typestore,
            serialization_format="cdr",
        )

        msg_count = 0
        cam_count = 0
        imu_count = 0

        for ts_ns, sensor, idx in timeline:
            stamp = Time(sec=ts_ns // 10**9, nanosec=ts_ns % 10**9)

            if sensor == "cam":
                _, img_path = cam_entries[idx]
                img = Image.open(img_path).convert("L")
                raw_bytes = img.tobytes()
                height, width = img.size[1], img.size[0]

                msg = ImageMsg(
                    header=Header(stamp=stamp, frame_id="cam0"),
                    height=height,
                    width=width,
                    encoding="mono8",
                    is_bigendian=0,
                    step=width,
                    data=np.frombuffer(raw_bytes, dtype=np.uint8),
                )
                serialized = typestore.serialize_cdr(msg, ImageMsg.__msgtype__)
                writer.write(cam_conn, ts_ns, serialized)
                cam_count += 1

            elif sensor == "imu":
                _, wx, wy, wz, ax, ay, az = imu_entries[idx]

                msg = ImuMsg(
                    header=Header(stamp=stamp, frame_id="imu0"),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=0.0),
                    orientation_covariance=np.array([-1.0] + [0.0] * 8),
                    angular_velocity=Vector3(x=wx, y=wy, z=wz),
                    angular_velocity_covariance=np.zeros(9),
                    linear_acceleration=Vector3(x=ax, y=ay, z=az),
                    linear_acceleration_covariance=np.zeros(9),
                )
                serialized = typestore.serialize_cdr(msg, ImuMsg.__msgtype__)
                writer.write(imu_conn, ts_ns, serialized)
                imu_count += 1

            msg_count += 1
            if msg_count % 500 == 0:
                print(f"  Written {msg_count}/{len(timeline)} messages ...")

    # Summary
    all_ts = [t[0] for t in timeline]
    duration_s = (max(all_ts) - min(all_ts)) / 1e9
    print(f"\nDone! Wrote ROS2 bag to {output_dir}")
    print(f"  Images: {cam_count}")
    print(f"  IMU:    {imu_count}")
    print(f"  Total:  {msg_count}")
    print(f"  Duration: {duration_s:.2f} s")


if __name__ == "__main__":
    main()
