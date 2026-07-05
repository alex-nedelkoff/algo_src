"""Deterministic offline replay of a recorded VQ2 corpus through an estimator.

Usage:
    python -m vq2.replay <corpus_dir> [<corpus_dir> ...]

Validation without ground truth (no pose oracle exists in any VQ2 mode):
  * rest windows are self-labeling (gyro ~ 0, |f| ~ g): there the
    accel-implied attitude IS truth, and true velocity is zero;
  * so end-to-end attitude drift = estimator vs accel-implied attitude at
    the FINAL rest, and velocity drift = estimator velocity at final rest;
  * camera-vs-IMU clock offset is estimated from rx_wall as a by-product.

Everything here is deterministic: same corpus in, same numbers out.
"""
from __future__ import annotations

import math
import statistics
import sys

from . import corpus as corpus_mod
from .estimators import BaselineImuEstimator, accel_implied_attitude, is_at_rest

DEG = 180.0 / math.pi


def find_rest_windows(imu: list, min_len: int = 30) -> list:
    """Contiguous runs of at-rest samples, as (start_idx, end_idx) inclusive."""
    windows = []
    start = None
    for i, s in enumerate(imu):
        if is_at_rest(s):
            if start is None:
                start = i
        else:
            if start is not None and i - start >= min_len:
                windows.append((start, i - 1))
            start = None
    if start is not None and len(imu) - start >= min_len:
        windows.append((start, len(imu) - 1))
    return windows


def replay_corpus(root: str) -> dict:
    c = corpus_mod.load(root)
    seg = c.flight_segment
    report = {"root": root, "segments": len(c.segments), "frames": len(c.frames)}
    if seg is None or not seg.imu:
        report["error"] = "no IMU data"
        return report

    report["seg_rows"] = len(seg.imu)
    report["seg_duration_s"] = round(seg.duration_s, 1)
    report["imu_gaps"] = seg.imu_gaps()

    rests = find_rest_windows(seg.imu)
    report["rest_windows"] = [
        {
            "t0_s": round((seg.imu[a].t_us - seg.imu[0].t_us) / 1e6, 1),
            "t1_s": round((seg.imu[b].t_us - seg.imu[0].t_us) / 1e6, 1),
            "n": b - a + 1,
        }
        for a, b in rests
    ]
    if not rests:
        report["error"] = "no rest window to seed from"
        return report

    est = BaselineImuEstimator()
    seed_idx = rests[0][1]  # end of the first rest window
    est.seed_from_rest(seg.imu[seed_idx])
    seed_att = (est.state.roll, est.state.pitch)
    report["seed"] = {
        "roll_deg": round(seed_att[0] * DEG, 2),
        "pitch_deg": round(seed_att[1] * DEG, 2),
        "at_s": round((seg.imu[seed_idx].t_us - seg.imu[0].t_us) / 1e6, 1),
    }

    # indices where a rest window midpoint falls -> snapshot the state there
    snap_at = {}
    for a, b in rests:
        mid = (a + b) // 2
        if mid > seed_idx:
            snap_at[mid] = (a, b)

    max_tilt = 0.0
    max_v = 0.0
    max_gyro = 0.0
    snapshots = []
    for i, s in enumerate(seg.imu[seed_idx + 1 :], start=seed_idx + 1):
        est.predict(s)
        max_tilt = max(max_tilt, est.state.tilt)
        max_v = max(max_v, abs(est.state.vx_b), abs(est.state.vy_b), abs(est.state.vz_up))
        max_gyro = max(max_gyro, max(abs(g) for g in s.gyr))
        if i in snap_at:
            truth_r, truth_p = accel_implied_attitude(s)
            snapshots.append(
                {
                    "t_s": round((s.t_us - seg.imu[0].t_us) / 1e6, 1),
                    "est_v": [round(est.state.vx_b, 2), round(est.state.vy_b, 2), round(est.state.vz_up, 2)],
                    "att_err_deg_if_grounded": [
                        round((est.state.roll - truth_r) * DEG, 2),
                        round((est.state.pitch - truth_p) * DEG, 2),
                    ],
                }
            )
    report["max_tilt_deg"] = round(max_tilt * DEG, 1)
    report["max_speed_est"] = round(max_v, 2)
    report["max_gyro_rad_s"] = round(max_gyro, 2)
    report["rest_snapshots"] = snapshots

    # end-to-end checks at the final rest window (if distinct from the seed one)
    if len(rests) > 1 or rests[-1][1] != seed_idx:
        a, b = rests[-1]
        mid = seg.imu[(a + b) // 2]
        truth_r, truth_p = accel_implied_attitude(mid)
        report["final_rest"] = {
            "att_err_deg": {
                "roll": round((est.state.roll - truth_r) * DEG, 3),
                "pitch": round((est.state.pitch - truth_p) * DEG, 3),
            },
            "vel_drift": {
                "vx_b": round(est.state.vx_b, 3),
                "vy_b": round(est.state.vy_b, 3),
                "vz_up": round(est.state.vz_up, 3),
            },
            "yaw_open_loop_deg": round(est.state.yaw * DEG, 2),
        }

    # GateNet detection stats (when detections.jsonl is present)
    if c.detections:
        n = len(c.detections)
        with_det = [d for d in c.detections if d.insts]
        solved = [i for d in with_det for i in d.insts if i.get("solved")]
        good = [i for i in solved if not i.get("low_confidence")]
        ranges = sorted(
            math.sqrt(sum(x * x for x in i["t_cam"])) for i in solved if i.get("t_cam")
        )
        report["gatenet"] = {
            "frames_scored": n,
            "det_rate": round(len(with_det) / n, 3),
            "insts_per_det_frame": round(
                sum(len(d.insts) for d in with_det) / max(1, len(with_det)), 2
            ),
            "pnp_solved": len(solved),
            "pnp_confident": len(good),
            "range_m_p10_p50_p90": [
                round(ranges[int(q * (len(ranges) - 1))], 1) for q in (0.1, 0.5, 0.9)
            ]
            if ranges
            else None,
        }

    # camera<->IMU clock bridge estimate from rx_wall (coarse: receive jitter)
    if c.frames and seg.imu:
        imu_offsets = [s.rx_wall - s.t_us / 1e6 for s in seg.imu[:: max(1, len(seg.imu) // 500)]]
        stamped = [fr for fr in c.frames if fr.rx_wall > 0]
        cam_offsets = [fr.rx_wall - fr.sim_ns / 1e9 for fr in stamped[:: max(1, len(stamped) // 500)]]
        report["clock_bridge"] = {
            "imu_boot_to_wall_s": round(statistics.median(imu_offsets), 4),
            "cam_epoch_to_wall_s": round(statistics.median(cam_offsets), 4),
            "cam_offset_spread_ms": round(
                1000 * (max(cam_offsets) - min(cam_offsets)), 1
            ),
        }
    return report


def main(argv: list) -> int:
    if not argv:
        print(__doc__)
        return 2
    for root in argv:
        report = replay_corpus(root)
        print(f"\n=== {root} ===")
        for k, v in report.items():
            if k == "root":
                continue
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
