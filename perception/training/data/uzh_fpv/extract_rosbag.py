"""Extract stereo images + ground truth from UZH-FPV rosbags.

Uses the pure-Python `rosbags` library (no ROS dependency).

Usage:
  python -m perception.training.data.uzh_fpv.extract_rosbag \
      --bag ~/corvidx/data/uzh-fpv/raw/indoor_forward_3_snapdragon_with_gt.bag \
      --output-dir ~/corvidx/data/uzh-fpv/extracted/indoor_forward_3_snapdragon/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

from .constants import LEFT_IMAGE_TOPIC, RIGHT_IMAGE_TOPIC

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def extract_images(
    bag_path: Path,
    output_dir: Path,
    left_topic: str = LEFT_IMAGE_TOPIC,
    right_topic: str = RIGHT_IMAGE_TOPIC,
) -> dict[str, int]:
    """Extract stereo images and ground truth from a rosbag.

    Returns:
        Dictionary with counts: {"left": N, "right": M, "gt_poses": P}
    """
    typestore = get_typestore(Stores.ROS1_NOETIC)

    left_dir = output_dir / "left"
    right_dir = output_dir / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    left_timestamps = []
    right_timestamps = []
    left_count = 0
    right_count = 0

    log.info("Opening rosbag: %s", bag_path)

    with Reader(bag_path) as reader:
        # Log available topics
        topics = {c.topic: c.msgtype for c in reader.connections}
        log.info("Available topics: %s", topics)

        # Find ground truth topic — prefer PoseStamped over Odometry
        gt_topic = None
        for topic in topics:
            if "groundtruth" in topic.lower() or "ground_truth" in topic.lower():
                if gt_topic is None or "pose" in topic.lower():
                    gt_topic = topic

        for connection, timestamp, rawdata in reader.messages():
            topic = connection.topic

            if topic in (left_topic, right_topic):
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                h, w = msg.height, msg.width
                encoding = msg.encoding

                # Convert raw data to numpy array
                if encoding in ("mono8", "8UC1"):
                    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w)
                elif encoding in ("bgr8", "rgb8"):
                    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
                    if encoding == "rgb8":
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    log.warning("Unknown encoding %s, treating as mono8", encoding)
                    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w)

                # Timestamp in nanoseconds
                stamp_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec

                if topic == left_topic:
                    out_path = left_dir / f"{left_count:06d}.png"
                    left_timestamps.append(stamp_ns)
                    left_count += 1
                else:
                    out_path = right_dir / f"{right_count:06d}.png"
                    right_timestamps.append(stamp_ns)
                    right_count += 1

                cv2.imwrite(str(out_path), img)

                total = left_count + right_count
                if total % 500 == 0:
                    log.info("  Extracted %d images (%d L, %d R) ...",
                             total, left_count, right_count)

            elif gt_topic and topic == gt_topic:
                # Ground truth will be extracted separately below
                pass

        # Extract ground truth poses
        gt_poses = []
        if gt_topic:
            log.info("Extracting ground truth from topic: %s", gt_topic)
            # Re-read for GT (rosbags Reader supports re-iteration)

        # Re-open to extract GT
        with Reader(bag_path) as reader2:
            gt_msgtype = topics.get(gt_topic, "") if gt_topic else ""
            for connection, timestamp, rawdata in reader2.messages():
                if gt_topic and connection.topic == gt_topic:
                    msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                    stamp_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec

                    # Handle both PoseStamped and Odometry message types
                    if "Odometry" in gt_msgtype:
                        p = msg.pose.pose.position
                        q = msg.pose.pose.orientation
                    else:
                        p = msg.pose.position
                        q = msg.pose.orientation

                    gt_poses.append([
                        stamp_ns,
                        p.x, p.y, p.z,
                        q.x, q.y, q.z, q.w,
                    ])

    # Save timestamps
    np.savetxt(
        str(output_dir / "left_timestamps.txt"),
        np.array(left_timestamps, dtype=np.int64),
        fmt="%d",
    )
    np.savetxt(
        str(output_dir / "right_timestamps.txt"),
        np.array(right_timestamps, dtype=np.int64),
        fmt="%d",
    )

    # Save ground truth
    if gt_poses:
        gt_array = np.array(gt_poses)
        header = "timestamp_ns tx ty tz qx qy qz qw"
        np.savetxt(
            str(output_dir / "groundtruth.txt"),
            gt_array,
            fmt="%.6f",
            header=header,
        )
        log.info("Saved %d ground truth poses", len(gt_poses))
    else:
        log.warning("No ground truth topic found in bag")

    log.info(
        "Extraction complete: %d left, %d right images",
        left_count, right_count,
    )
    return {"left": left_count, "right": right_count, "gt_poses": len(gt_poses)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract stereo images + GT from UZH-FPV rosbag"
    )
    parser.add_argument("--bag", required=True, type=Path, help="Path to .bag file")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--left-topic", default=LEFT_IMAGE_TOPIC, help="Left camera topic")
    parser.add_argument("--right-topic", default=RIGHT_IMAGE_TOPIC, help="Right camera topic")

    args = parser.parse_args()
    extract_images(args.bag, args.output_dir, args.left_topic, args.right_topic)


if __name__ == "__main__":
    main()
