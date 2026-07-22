"""Offline VO replay harness (COR-147, Phase 2 seed).

corpus frames -> MonoVO -> per-keyframe VoSteps -> integrated UNIT-scale
trajectory. Reports VO health and, when a p_est reference is present in
livelog.jsonl, the direction agreement between the VO path and the estimator
path (scale-free, since monocular VO is unit-scale until calibrated).

Usage:
  python -m vq2.tools.vo_replay <corpus_dir> [--max N] [--kf-flow PX]

This is a diagnostic, not flight code. Scoring is shape/direction agreement,
NOT ticks — that comes later, in the sim. Read the line-transit methodology
warning before trusting any proxy metric here.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import cv2

from vq2.live.vo_cv import MonoVO, VoStep


def load_frame_order(corpus: str):
    """(sim_ns, path) in capture order, from frames_dedup.jsonl if present."""
    fd = os.path.join(corpus, "frames_dedup.jsonl")
    fdir = os.path.join(corpus, "frames")
    out = []
    if os.path.exists(fd):
        for line in open(fd):
            try:
                ns = json.loads(line)["sim_ns"]
            except (json.JSONDecodeError, KeyError):
                continue
            p = os.path.join(fdir, f"{ns}.jpg")
            if os.path.exists(p):
                out.append((ns, p))
    else:
        import glob
        for p in sorted(glob.glob(os.path.join(fdir, "*.jpg"))):
            out.append((int(os.path.splitext(os.path.basename(p))[0]), p))
    return out


def load_p_est(corpus: str):
    """(sim_ns, xyz) estimator reference from livelog.jsonl, if available."""
    lp = os.path.join(corpus, "livelog.jsonl")
    if not os.path.exists(lp):
        return None
    ts, ps = [], []
    for line in open(lp):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        p = d.get("p_est") or d.get("p")
        ns = d.get("ns")
        if p is not None and ns is not None and len(p) >= 3:
            ts.append(ns)
            ps.append(p[:3])
    if len(ts) < 2:
        return None
    return np.array(ts), np.array(ps, dtype=float)


def integrate(steps):
    """Chain unit-scale VoSteps into a body-frame trajectory (arbitrary scale).
    Each step contributes t_body_unit rotated by the accumulated heading."""
    C = np.eye(3)          # accumulated body-frame rotation
    pos = np.zeros(3)
    traj = [(None, pos.copy())]
    for s in steps:
        pos = pos + C @ s.t_body_unit
        C = C @ s.R_body
        traj.append((s.t1, pos.copy()))
    return traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--max", type=int, default=100000)
    ap.add_argument("--kf-flow", type=float, default=9.0)
    args = ap.parse_args()

    frames = load_frame_order(args.corpus)[: args.max]
    print(f"{os.path.basename(args.corpus)}: {len(frames)} frames")

    vo = MonoVO(kf_flow_px=args.kf_flow)
    steps: list[VoStep] = []
    for ns, path in frames:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        s = vo.step(ns * 1e-9, img)   # sim_ns -> seconds
        if s is not None:
            steps.append(s)

    if not steps:
        print("NO KEYFRAMES CLOSED — check motion window / kf-flow")
        return

    inl = np.array([s.n_inliers for s in steps])
    flow = np.array([s.median_flow_px for s in steps])
    traj = np.array([p for _, p in integrate(steps)])
    print(f"keyframes closed     : {len(steps)}")
    print(f"inliers  p10/50/90   : {np.percentile(inl,10):.0f} / "
          f"{np.percentile(inl,50):.0f} / {np.percentile(inl,90):.0f}")
    print(f"kf median flow px    : {np.median(flow):.1f}")
    print(f"traj extent (unit-sc): dx={np.ptp(traj[:,0]):.1f} "
          f"dy={np.ptp(traj[:,1]):.1f} dz={np.ptp(traj[:,2]):.1f}")

    ref = load_p_est(args.corpus)
    if ref is not None and len(steps) >= 3:
        ref_ts, ref_ps = ref
        # match each VO keyframe time to nearest p_est sample; compare the
        # per-step displacement DIRECTIONS (scale-free) in the body xy plane
        agree = []
        for s in steps:
            t_ns = s.t1 * 1e9
            j = int(np.argmin(np.abs(ref_ts - t_ns)))
            j0 = int(np.argmin(np.abs(ref_ts - s.t0 * 1e9)))
            if j == j0:
                continue
            d_ref = ref_ps[j] - ref_ps[j0]
            if np.linalg.norm(d_ref[:2]) < 1e-3:
                continue
            d_vo = s.t_body_unit
            cos = np.dot(d_ref[:2], d_vo[:2]) / (
                np.linalg.norm(d_ref[:2]) * np.linalg.norm(d_vo[:2]) + 1e-9)
            agree.append(cos)
        if agree:
            agree = np.array(agree)
            print(f"VO-vs-p_est xy dir   : mean cos={agree.mean():.2f} "
                  f"(n={len(agree)}, frac>0.5={np.mean(agree>0.5):.2f})")
    else:
        print("no p_est reference (livelog absent/empty) — health-only report")


if __name__ == "__main__":
    main()
