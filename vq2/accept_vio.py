"""VIO acceptance harness (the GO/NO-GO gate from HANDOFF_vq2_racing.md):
IMU + floor-flow trajectory (NO vision position fixes) must stay within
0.5 m XY of every vision anchor inside the first 12 m traveled.

Usage:
    python3 -m vq2.accept_vio <corpus_dir>            # verdict at Z_FLOOR
    python3 -m vq2.accept_vio <corpus_dir> --sweep    # calibrate z_floor
"""
from __future__ import annotations

import sys

import numpy as np

from .fusion import FusionConfig, run_fusion

PASS_BAR_M = 0.5
PASS_DIST_M = 12.0


def evaluate(root: str, z_floor: float) -> dict:
    res = run_fusion(root, FusionConfig(use_flow=True, use_vision_pos=False,
                                        z_floor=z_floor))
    out = {"z_floor": z_floor, "flow_updates": res.flow_updates,
           "flow_rejects": res.flow_rejects, "n_anchors": len(res.anchors)}
    if not res.p or not res.anchors:
        out.update(max_err_xy_12m=float("inf"), p90_err_xy=float("inf"),
                   dist_traveled_m=0.0, passed=False)
        return out
    p = np.asarray(res.p)
    seg_d = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
    dist_at = np.concatenate([[0.0], np.cumsum(seg_d)])
    t = np.asarray(res.t_s)
    errs_12 = []
    errs_all = []
    for a in res.anchors:
        i = int(np.searchsorted(t, a["t_boot_s"]))
        i = min(i, len(dist_at) - 1)
        errs_all.append(a["err_xy"])
        if dist_at[i] <= PASS_DIST_M:
            errs_12.append(a["err_xy"])
    max12 = max(errs_12) if errs_12 else float("inf")
    out.update(
        max_err_xy_12m=round(float(max12), 3),
        p90_err_xy=round(float(np.percentile(errs_all, 90)), 3),
        dist_traveled_m=round(float(dist_at[-1]), 2),
        passed=bool(max12 <= PASS_BAR_M),
    )
    return out


def sweep_z_floor(root: str, lo: float = 0.0, hi: float = 0.45,
                  step: float = 0.05) -> list:
    rows = []
    z = lo
    while z <= hi + 1e-9:
        res = run_fusion(root, FusionConfig(use_flow=True, z_floor=z))
        errs = [a["err_xy"] for a in res.anchors]
        rows.append((round(z, 3),
                     round(float(np.mean(errs)) if errs else float("inf"), 3)))
        z += step
    return rows


def main(argv: list) -> int:
    if not argv:
        print(__doc__)
        return 2
    root = argv[0]
    if "--sweep" in argv:
        rows = sweep_z_floor(root)
        print("z_floor  mean_err_xy")
        for z, e in rows:
            print(f"  {z:5.2f}   {e}")
        best = min(rows, key=lambda r: r[1])
        print(f"best: z_floor={best[0]} (mean_err={best[1]})")
        print("-> update Z_FLOOR in vq2/flow_vel.py to this value")
        return 0
    from .flow_vel import Z_FLOOR
    r = evaluate(root, Z_FLOOR)
    for k, v in r.items():
        print(f"  {k}: {v}")
    # baseline comparison: how bad is IMU-only? (expect the ~0.6x compression)
    base = run_fusion(root, FusionConfig(use_flow=False))
    if base.anchors:
        errs = [a["err_xy"] for a in base.anchors]
        print(f"  imu_only_p90_err_xy: {round(float(np.percentile(errs, 90)), 3)}")
    print("VERDICT:", "PASS — wire into vq2wp.py (Task 7)" if r["passed"]
          else "FAIL — see fallbacks in the plan header; do NOT go live")
    return 0 if r["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
