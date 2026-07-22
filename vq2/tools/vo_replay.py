"""Offline VO replay harness (COR-147, Phase 2 seed).

corpus frames -> MonoVO -> per-keyframe VoSteps -> integrated UNIT-scale
trajectory. Reports VO health, a truth-by-construction forward-motion check,
and (when livelog.jsonl yields an estimator reference via carry-forward ns)
the globally-aligned direction agreement between VO and the estimator path.

CAVEAT: the estimator path is NOT ground truth — it is the DR estimate this
project exists to correct, and standalone monocular VO integration drifts
(no loop closure). So the estimator-agreement number is a coarse sanity
signal, not an acceptance test. Trust the per-step forward-motion check and
recoverPose health; the real acceptance test is VO factors IN the smoother
vs a truth trajectory (controlled sim flight).

Usage:
  python -m vq2.tools.vo_replay <corpus_dir> [--max N] [--kf-flow PX]

This is a diagnostic, not flight code. Scoring is shape/direction agreement,
NOT ticks — that comes later, in the sim. Read the line-transit methodology
warning before trusting any proxy metric here.
"""
from __future__ import annotations

import argparse
import math
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


# livelog record kinds whose `p` field is an estimator position (no `ns`)
_POS_KINDS = ("kf_upd", "mapfollow", "mfdr_seed", "tick_fix")


def load_reference(corpus: str):
    """Estimator reference trajectory (sim_ns, xyz) from livelog.jsonl.

    livelog is a heterogeneous, time-ordered stream: position records
    (`att.p_est`, `kf_upd.p`, ...) carry NO timestamp, while `obs*` records
    carry `ns`. So we CARRY FORWARD the last-seen `ns` onto each position
    record (the flight loop emits obs then the estimator update, so the
    last-seen ns bounds the position time from below; sparsity of gate obs
    makes this loose by up to ~0.5s — fine for coarse direction agreement).

    Returns (ns[M], xyz[M,3]) with strictly increasing unique ns, or None.
    """
    lp = os.path.join(corpus, "livelog.jsonl")
    if not os.path.exists(lp):
        return None
    last_ns = None
    ts, ps = [], []
    for line in open(lp):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        ns = d.get("ns")
        if isinstance(ns, (int, float)):
            last_ns = int(ns)
        p = d.get("p_est")
        if p is None and (d.get("kind") in _POS_KINDS):
            p = d.get("p")
        if isinstance(p, list) and len(p) >= 3 and last_ns is not None:
            ts.append(last_ns)
            ps.append([float(x) for x in p[:3]])
    if len(ts) < 3:
        return None
    ts = np.array(ts)
    ps = np.array(ps, dtype=float)
    # collapse carry-forward duplicates: keep the LAST position per unique ns
    uniq, idx = np.unique(ts, return_index=False), {}
    for i, t in enumerate(ts):
        idx[t] = i
    keep = np.array([idx[t] for t in uniq])
    return uniq, ps[keep]


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

    # per-step physics check (truth-by-construction, no estimator needed):
    # a gate transit is forward-dominant, so body-frame +x should dominate.
    tb = np.array([s.t_body_unit for s in steps])
    fwd_dom = float(np.mean(np.argmax(np.abs(tb), axis=1) == 0))
    fwd_pos = float(np.mean(tb[:, 0] > 0))
    print(f"fwd-motion check     : +x dominant {fwd_dom:.0%}, +x>0 {fwd_pos:.0%} "
          f"(|t| axis mean {np.abs(tb).mean(axis=0).round(2)})")

    ref = load_reference(args.corpus)
    if ref is not None:
        _compare_to_reference(steps, traj, ref)
    else:
        print("no estimator reference (livelog absent/sparse) — health-only")


def _compare_to_reference(steps, traj, ref):
    """Direction agreement between the VO path and the (sparse) estimator
    path, evaluated over the reference's OWN native intervals. VO is unit-
    scale, so we compare DIRECTIONS (cos angle) in the xy plane — z is the
    weakest VO axis and carries init outliers in the reference. VO position
    is linearly interpolated to each reference timestamp; displacement is
    then differenced over consecutive reference samples and weighted by the
    reference step length (long, confident moves dominate)."""
    ref_ts, ref_ps = ref
    kf_ns = np.array([s.t1 * 1e9 for s in steps])          # keyframe close times
    kf_pos = traj[1:]                                       # VO pos at each close
    if len(kf_ns) < 2:
        print("too few keyframes for reference comparison")
        return
    lo, hi = kf_ns[0], kf_ns[-1]
    D_ref, D_vo, wts = [], [], []
    for a in range(len(ref_ts) - 1):
        na, nb = ref_ts[a], ref_ts[a + 1]
        if nb <= na or na < lo or nb > hi:               # only where VO covers
            continue
        d_ref = (ref_ps[a + 1] - ref_ps[a])[:2]
        if np.linalg.norm(d_ref) < 0.05:                  # ignore ~stationary ref
            continue
        va = np.array([np.interp(na, kf_ns, kf_pos[:, i]) for i in range(2)])
        vb = np.array([np.interp(nb, kf_ns, kf_pos[:, i]) for i in range(2)])
        d_vo = vb - va
        if np.linalg.norm(d_vo) < 1e-6:
            continue
        D_ref.append(d_ref)
        D_vo.append(d_vo)
        wts.append(float(np.linalg.norm(d_ref)))
    if not D_ref:
        print("reference overlaps no VO-covered interval")
        return
    D_ref, D_vo, wts = np.array(D_ref), np.array(D_vo), np.array(wts)

    # The VO path is in its own start frame; the estimator path is reset-NED.
    # Solve the single 2D rotation theta that best aligns VO->ref (weighted
    # Procrustes on the displacement set): theta = atan2(sum w (vo x ref),
    # sum w (vo . ref)). Report agreement AFTER alignment (the meaningful
    # "does VO track the estimator's shape" metric) + the recovered yaw.
    s = np.sum(wts * (D_vo[:, 0] * D_ref[:, 1] - D_vo[:, 1] * D_ref[:, 0]))
    c = np.sum(wts * (D_vo[:, 0] * D_ref[:, 0] + D_vo[:, 1] * D_ref[:, 1]))
    theta = math.atan2(s, c)
    ct, stt = math.cos(theta), math.sin(theta)
    R = np.array([[ct, -stt], [stt, ct]])
    D_vo_a = D_vo @ R.T
    cos_a = np.sum(D_vo_a * D_ref, axis=1) / (
        np.linalg.norm(D_vo_a, axis=1) * np.linalg.norm(D_ref, axis=1))
    cos_raw = np.sum(D_vo * D_ref, axis=1) / (
        np.linalg.norm(D_vo, axis=1) * np.linalg.norm(D_ref, axis=1))
    wmean_raw = float(np.sum(cos_raw * wts) / np.sum(wts))
    wmean_a = float(np.sum(cos_a * wts) / np.sum(wts))
    print(f"VO-vs-est xy dir     : raw wmean cos={wmean_raw:+.2f} -> "
          f"aligned={wmean_a:+.2f} (yaw offset {math.degrees(theta):+.0f} deg)")
    print(f"  aligned agreement  : n={len(cos_a)}, frac>0.7={np.mean(cos_a > 0.7):.2f}, "
          f"frac>0.5={np.mean(cos_a > 0.5):.2f}")


if __name__ == "__main__":
    main()
