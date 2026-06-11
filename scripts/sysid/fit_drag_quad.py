"""Fit quadratic body drag (+ z-axis thrust/drag) on recorded VQ runs (COR-127 REPLAY-01).

Model (body frame, live io chart; drag is odd in v so the chart's y-flip is absorbed):
  acc_x = -(Dx + qx*|v|) * vbx
  acc_y = -(Dy + qy*|v|) * vby
  acc_z = -(f0 + df*thr) - (Dz + qz*|v|) * vbz        (FRD: thrust specific force = -c on z)

acc = recorded IMU specific force (the holdout lesson: fit/validate against IMU, never
differentiated odometry). Fit on BLOCK MEANS (policy recordings carry +-100 rad/s^2
alternating 1-step dither; 0.25 s means kill it). lstsq per axis; x/y fit Dx,qx jointly
(the linear coeff re-fits since the old Dx absorbed quadratic effects over its 0-8 m/s
fit range); z re-fits the thrust map (f0, df) jointly with (Dz, qz) and prints the low-thr
residual so the thrust-floor hypothesis can be judged from the same data.

Usage: python scripts/sysid/fit_drag_quad.py /tmp/vq_replay/*.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BLOCK_S = 0.25


def blocks(t, arr, block_s=BLOCK_S):
    """Block means on the wall-clock grid."""
    out = []
    t0 = t[0]
    while t0 < t[-1]:
        m = (t >= t0) & (t < t0 + block_s)
        if m.sum() >= 3:
            out.append(arr[m].mean(axis=0))
        t0 += block_s
    return np.array(out)


def main():
    paths = [p for p in sys.argv[1:] if p.endswith(".npz")]
    V, A, TH = [], [], []
    for p in paths:
        d = np.load(p)
        t = d["t_wall"] - d["t_wall"][0]
        thr = d["cmd"][:, 3:4]
        V.append(blocks(t, d["vel"]))
        A.append(blocks(t, d["acc"]))
        TH.append(blocks(t, thr))
        print(f"  {Path(p).stem}: {len(V[-1])} blocks, vmax {np.linalg.norm(d['vel'], axis=1).max():.1f}")
    vb = np.concatenate(V)
    acc = np.concatenate(A)
    thr = np.concatenate(TH)[:, 0]
    vmag = np.linalg.norm(vb, axis=1)
    print(f"total {len(vb)} blocks, |v| range {vmag.min():.1f}-{vmag.max():.1f}")

    # x / y axes: acc_ax = -D*vb - q*|v|*vb
    for name, i in (("x", 0), ("y", 1)):
        X = np.stack([-vb[:, i], -vmag * vb[:, i]], axis=1)
        coef, res, *_ = np.linalg.lstsq(X, acc[:, i], rcond=None)
        pred = X @ coef
        rms = float(np.sqrt(np.mean((acc[:, i] - pred) ** 2)))
        rms0 = float(np.sqrt(np.mean(acc[:, i] ** 2)))
        print(f"{name}: D={coef[0]:+.4f}  q={coef[1]:+.5f}  resid RMS {rms:.2f} (raw {rms0:.2f}) m/s^2")

    # z axis: acc_z = -(f0 + df*thr) - (Dz + qz*|v|)*vbz
    Xz = np.stack([-np.ones_like(thr), -thr, -vb[:, 2], -vmag * vb[:, 2]], axis=1)
    coef, *_ = np.linalg.lstsq(Xz, acc[:, 2], rcond=None)
    pred = Xz @ coef
    rms = float(np.sqrt(np.mean((acc[:, 2] - pred) ** 2)))
    print(f"z: f0={coef[0]:+.3f}  df={coef[1]:+.3f}  Dz={coef[2]:+.4f}  qz={coef[3]:+.5f}  resid RMS {rms:.2f} m/s^2")

    # thrust-floor check: low-thr blocks' z residual (positive resid = more lift than the linear map)
    for lo, hi in ((0.0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 0.6)):
        m = (thr >= lo) & (thr < hi)
        if m.sum() >= 5:
            r = acc[m, 2] - pred[m]
            print(f"  thr [{lo:.2f},{hi:.2f}): n={m.sum():4d}  z-resid mean {r.mean():+.2f}  rms {np.sqrt((r**2).mean()):.2f}")


if __name__ == "__main__":
    main()
