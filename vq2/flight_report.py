"""One-command flight forensics over a recorded corpus.

    python3 -m vq2.flight_report <corpus_dir> [--policy radius|huber]

Outputs into <corpus_dir>:
    report.json   -- scorecard (all metrics, machine-readable, trended)
    report.md     -- human summary
    report.npz    -- timeline arrays for the rrd emitter (vq2/rrd_emit.py,
                     run under the rerun-0.33 venv python)

Every metric here was paid for at least once -- see the experiment log
(VQ2-STAB-01 and neighbours) for the incidents that motivated each.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

from . import corpus as corpus_mod
from .estimators import accel_implied_attitude, is_at_rest
from .fusion import GATES, FusionConfig, run_fusion
from .replay import find_rest_windows

NOMINAL_HZ = 144.0
RATE_CLEAN = 1.0          # rad/s: attitude chain proven clean below this
GAP_FACTOR = 1.5          # gap > 1.5 nominal periods counts as a drop


def _ingest_metrics(seg, frames, bridge):
    period_us = 1e6 / NOMINAL_HZ
    gaps, gapxrate = [], []
    for a, b in zip(seg.imu, seg.imu[1:]):
        g = b.t_us - a.t_us
        if g > GAP_FACTOR * period_us:
            rate = max(abs(x) for x in b.gyr)
            gaps.append(g / 1e3)
            gapxrate.append(((g - period_us) / 1e6) * rate)
    dur = seg.duration_s or 1.0
    return {
        "imu_rows": len(seg.imu),
        "imu_rate_hz": round(len(seg.imu) / dur, 1),
        "gap_count": len(gaps),
        "gap_ms_p90": round(float(np.percentile(gaps, 90)), 1) if gaps else 0.0,
        "gap_ms_max": round(max(gaps), 1) if gaps else 0.0,
        # attitude-error budget: sum of (dropped time x rate)/2, degrees
        "gap_attitude_budget_deg": round(
            math.degrees(sum(gapxrate) / 2), 2) if gapxrate else 0.0,
        "frames": len(frames),
        "clock_bridge_spread_ms": round(bridge[1], 1) if bridge else None,
    }


def _attitude_metrics(seg):
    rests = find_rest_windows(seg.imu)
    rates = np.array([[abs(g) for g in s.gyr] for s in seg.imu])
    out = {
        "rate_max_deg_s": round(float(np.degrees(rates.max())), 1),
        "rate_p99_deg_s": round(float(np.degrees(np.percentile(rates.max(axis=1), 99))), 1),
        "time_above_clean_s": round(float(
            np.sum(rates.max(axis=1) > RATE_CLEAN) / NOMINAL_HZ), 2),
        "rest_windows": len(rests),
    }
    # attitude closure: gyro-integrated (wfix) start-rest -> end-rest vs
    # accel-implied at both ends (only meaningful with two rest windows)
    if len(rests) >= 2:
        a0 = seg.imu[(rests[0][0] + rests[0][1]) // 2]
        a1 = seg.imu[(rests[-1][0] + rests[-1][1]) // 2]
        r0, p0 = accel_implied_attitude(a0)
        r1, p1 = accel_implied_attitude(a1)
        roll, pitch = r0, p0
        last = a0.t_us
        for s in seg.imu:
            if s.t_us <= last or s.t_us > a1.t_us:
                continue
            dt = (s.t_us - last) / 1e6
            last = s.t_us
            roll += s.gyr[0] * dt
            pitch += -s.gyr[1] * dt
        out["closure_err_deg"] = {
            "roll": round(math.degrees(roll - r1), 2),
            "pitch": round(math.degrees(pitch - p1), 2),
        }
    return out


def _waterfall(res):
    stages = {}
    for e in res.obs_events:
        stages[e["stage"]] = stages.get(e["stage"], 0) + 1
    acc = [e for e in res.obs_events if e["stage"] == "accepted"]
    nis = [e["nis"] for e in acc if "nis" in e and np.isfinite(e["nis"])]
    miss = [e["miss"] for e in res.obs_events]
    t_acc = sorted(e["t_boot_s"] for e in acc)
    starve = max(np.diff(t_acc)) if len(t_acc) > 1 else None
    return {
        "stages": stages,
        "produced": len(res.obs_events),
        "accept_rate": round(len(acc) / len(res.obs_events), 3)
        if res.obs_events else None,
        "miss_p50": round(float(np.percentile(miss, 50)), 2) if miss else None,
        "miss_p90": round(float(np.percentile(miss, 90)), 2) if miss else None,
        "nis_p50": round(float(np.percentile(nis, 50)), 2) if nis else None,
        "nis_frac_gt_gate": round(float(np.mean([n > 11.34 for n in nis])), 3)
        if nis else None,
        "max_fix_starvation_s": round(float(starve), 1) if starve else None,
    }


def _control_metrics(root, seg):
    """Only for corpora that logged commands (cmds.jsonl: rr/pr/yr/thr)."""
    path = os.path.join(root, "cmds.jsonl")
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path)]
    if not rows or "thr" not in rows[0]:
        return None
    thr = np.array([r["thr"] for r in rows])
    rr = np.array([r.get("rr", 0.0) for r in rows])
    pr = np.array([r.get("pr", 0.0) for r in rows])
    t = np.array([r["t"] for r in rows])
    dt = np.diff(t) * 1e3
    # quasi-hover collective: thr while commanded rates ~ 0 (rotor-fitness /
    # collective-deficit diagnostic, SysID-notebook method)
    quiet = (np.abs(rr) < 0.05) & (np.abs(pr) < 0.05)
    return {
        "cmd_rows": len(rows),
        "cmd_dt_ms_p99": round(float(np.percentile(dt, 99)), 1) if len(dt) else None,
        "cmd_dt_ms_max": round(float(dt.max()), 1) if len(dt) else None,
        "thr_hover_mean": round(float(thr[quiet].mean()), 4) if quiet.any() else None,
        "rate_cmd_sat_frac": round(float(np.mean(
            (np.abs(rr) > 0.59) | (np.abs(pr) > 0.59))), 3),
        "thr_clip_frac": round(float(np.mean((thr <= 0.051) | (thr >= 0.599))), 3),
    }


def _crossing(res, gate_xy=(11.0, 0.0)):
    p = np.asarray(res.p)
    if len(p) < 2:
        return None
    out = []
    for i in range(1, len(p)):
        if p[i - 1][0] < gate_xy[0] <= p[i][0]:
            out.append({"t_boot_s": round(float(res.t_s[i]), 2),
                        "y": round(float(p[i][1] - gate_xy[1]), 2),
                        "z": round(float(p[i][2]), 2)})
    return out or None


def build_report(root: str, policy: str = "radius") -> dict:
    c = corpus_mod.load(root)
    seg = c.flight_segment
    if seg is None or not seg.imu:
        return {"error": "no IMU data"}
    bridge = corpus_mod.clock_bridge(seg, c.frames)
    res = run_fusion(root, FusionConfig(use_flow=False, use_vision_pos=True,
                                        accept_policy=policy))
    p = np.asarray(res.p)
    report = {
        "root": root,
        "policy": policy,
        "ingest": _ingest_metrics(seg, c.frames, bridge),
        "attitude": _attitude_metrics(seg),
        "waterfall": _waterfall(res),
        "control": _control_metrics(root, seg),
        "estimator": {
            "dist_traveled_m": round(float(np.sum(np.linalg.norm(
                np.diff(p[:, :2], axis=0), axis=1))), 1) if len(p) > 1 else 0.0,
            "final_speed": round(float(np.linalg.norm(res.v[-1])), 2)
            if res.v else None,
        },
        "crossings": _crossing(res),
    }
    # timeline arrays for the rrd emitter
    np.savez_compressed(
        os.path.join(root, "report.npz"),
        t_s=np.asarray(res.t_s), p=np.asarray(res.p), v=np.asarray(res.v),
        a_w_xy=np.asarray(res.a_w_xy),
        obs_t=np.asarray([e["t_boot_s"] for e in res.obs_events]),
        obs_miss=np.asarray([e["miss"] for e in res.obs_events]),
        obs_stage=np.asarray([e["stage"] for e in res.obs_events]),
        obs_nis=np.asarray([e.get("nis", float("nan"))
                            for e in res.obs_events]),
        gates=np.asarray(GATES),
    )
    with open(os.path.join(root, "report.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    with open(os.path.join(root, "report.md"), "w") as f:
        f.write(_to_md(report))
    return report


def _to_md(r: dict) -> str:
    lines = [f"# Flight report — {os.path.basename(r['root'])} "
             f"(policy: {r['policy']})", ""]
    for section in ("ingest", "attitude", "waterfall", "control",
                    "estimator", "crossings"):
        lines.append(f"## {section}")
        v = r.get(section)
        if v is None:
            lines.append("_n/a (no data recorded for this section)_")
        elif isinstance(v, dict):
            for k, val in v.items():
                lines.append(f"- **{k}**: {val}")
        else:
            lines.append(f"- {v}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list) -> int:
    if not argv:
        print(__doc__)
        return 2
    policy = "radius"
    if "--policy" in argv:
        i = argv.index("--policy")
        policy = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    r = build_report(argv[0], policy)
    print(json.dumps(r, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
