"""Build a compact DPVO scale calibration from a fresh two-tick corpus."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import struct

from .live.dpvo_route import (
    FULL_INTRINSICS,
    DpvoSessionConfig,
    calibration_identity,
    fit_tick_scale,
    scale_intrinsics,
)


RACE_STATUS = struct.Struct("<BQqqIq")
# Full-resolution camera model; intrinsics scale with the requested input size
# exactly as dpvo_bridge_replay does, so the calibration identity matches the
# poses produced at that resolution.
FULL_SIZE = (640, 360)


def decode_race_rows(mavlink_rows) -> list[dict]:
    decoded = []
    for message in mavlink_rows:
        if message.get("mavpackettype") != "ENCAPSULATED_DATA":
            continue
        try:
            data = bytes.fromhex(message["data"])
            if len(data) < RACE_STATUS.size or data[0] != 1:
                continue
            fields = RACE_STATUS.unpack_from(data)
            wall = float(message["_rx_wall"])
        except (KeyError, TypeError, ValueError, struct.error):
            continue
        decoded.append(
            {
                "clock_ms": int(fields[1]),
                "gate_idx": int(fields[4]),
                "tickstamp": int(fields[5]),
                "rx_wall_ns": int(round(wall * 1_000_000_000)),
            }
        )
    return decoded


def find_fresh_tick_pair(race_rows: list[dict]) -> tuple[dict, dict]:
    if not race_rows:
        raise ValueError("no RACE_STATUS rows")
    resets = [
        index
        for index in range(1, len(race_rows))
        if race_rows[index]["clock_ms"]
        < race_rows[index - 1]["clock_ms"] - 5000
    ]
    bounds = [0, *resets, len(race_rows)]
    for segment_index in range(len(bounds) - 1):
        segment = race_rows[bounds[segment_index] : bounds[segment_index + 1]]
        is_fresh = segment_index > 0 or segment[0]["clock_ms"] < 5000
        if not is_fresh or segment[0]["gate_idx"] != 0:
            continue
        seen = segment[0]["gate_idx"]
        gate1 = None
        for row in segment[1:]:
            gate_idx = row["gate_idx"]
            if gate_idx == seen:
                continue
            if seen == 0 and gate_idx == 1:
                gate1 = row
            elif seen == 1 and gate_idx == 2 and gate1 is not None:
                return gate1, row
            seen = gate_idx
    raise ValueError("no fresh post-reset two-tick segment")


def _nearest_pose(poses: list[dict], target_ns: int, tolerance_ns: int) -> dict:
    pose = min(poses, key=lambda row: abs(row["frame_ns"] - target_ns))
    if abs(pose["frame_ns"] - target_ns) > tolerance_ns:
        raise ValueError("no DPVO pose within tick alignment tolerance")
    return pose


def build_calibration(
    mavlink_rows,
    pose_rows,
    config: DpvoSessionConfig,
    model_sha256: str,
    gate1_world=(6.3, 0.0, -1.35),
    gate2_world=(11.7, 5.2, -1.35),
    tolerance_ms: float = 150.0,
) -> dict:
    identity = calibration_identity(config, model_sha256)
    poses = []
    for row in pose_rows:
        if row.get("identity") != identity:
            raise ValueError("pose identity does not match calibration session")
        try:
            frame_ns = int(row["frame_ns"])
            p = tuple(float(value) for value in row["p"])
        except (KeyError, TypeError, ValueError):
            continue
        if len(p) == 3 and all(math.isfinite(value) for value in p):
            poses.append({"frame_ns": frame_ns, "p": p})
    if len(poses) < 2:
        raise ValueError("insufficient DPVO poses")
    poses.sort(key=lambda row: row["frame_ns"])

    gate1_tick, gate2_tick = find_fresh_tick_pair(decode_race_rows(mavlink_rows))
    tolerance_ns = int(float(tolerance_ms) * 1_000_000)
    gate1_pose = _nearest_pose(poses, gate1_tick["rx_wall_ns"], tolerance_ns)
    gate2_pose = _nearest_pose(poses, gate2_tick["rx_wall_ns"], tolerance_ns)
    scale = fit_tick_scale(
        gate1_pose["p"], gate2_pose["p"], gate1_world, gate2_world
    )

    tick_records = []
    for tick, pose in ((gate1_tick, gate1_pose), (gate2_tick, gate2_pose)):
        tick_records.append(
            {
                "gate_idx": tick["gate_idx"],
                "race_ms": tick["clock_ms"],
                "tickstamp": tick["tickstamp"],
                "rx_wall_ns": tick["rx_wall_ns"],
                "pose_frame_ns": pose["frame_ns"],
                "sample_offset_ms": round(
                    (pose["frame_ns"] - tick["rx_wall_ns"]) / 1_000_000.0, 3
                ),
                "raw_p": list(pose["p"]),
            }
        )
    return {
        "version": 1,
        "control_approved": False,
        "validation": {
            "status": "observe_only",
            "reason": "temporal_stability_not_approved",
        },
        "identity": identity,
        "model_sha256": str(model_sha256),
        "config": asdict(config),
        "scale": scale,
        "gate1_world": [float(value) for value in gate1_world],
        "gate2_world": [float(value) for value in gate2_world],
        "ticks": tick_records,
        "sample_separation_ms": round(
            (gate2_pose["frame_ns"] - gate1_pose["frame_ns"]) / 1_000_000.0,
            3,
        ),
    }


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mavlink_jsonl", type=Path)
    parser.add_argument("poses_jsonl", type=Path)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--patches", type=int, default=32)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--removal-window", type=int, default=22)
    parser.add_argument("--optimization-window", type=int, default=10)
    arguments = parser.parse_args(argv)
    config = DpvoSessionConfig(
        patches=arguments.patches,
        width=arguments.width,
        height=arguments.height,
        intrinsics=scale_intrinsics(
            FULL_INTRINSICS, FULL_SIZE, (arguments.width, arguments.height)
        ),
        removal_window=arguments.removal_window,
        optimization_window=arguments.optimization_window,
    )
    calibration = build_calibration(
        _read_jsonl(arguments.mavlink_jsonl),
        _read_jsonl(arguments.poses_jsonl),
        config,
        arguments.model_sha256,
    )
    rendered = json.dumps(calibration, sort_keys=True, separators=(",", ":"))
    if arguments.output:
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
