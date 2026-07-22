"""Offline VO replay harness (COR-147, Phase 2 seed).

corpus frames -> MonoVO -> per-keyframe VoSteps -> integrated UNIT-scale
trajectory. Reports VO health, a truth-by-construction forward-motion check,
and (when livelog.jsonl yields an estimator reference via the capture-time
join in vq2.tools.livelog_join) the globally-aligned direction agreement and
relative drift between VO and the estimator path.

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
from vq2.tools.livelog_join import load_estimator_reference


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
    ap.add_argument("--full-flight", action="store_true",
                    help="score the whole flight; default scores the pre-tick "
                         "straight approach leg (single-rotation fit is only "
                         "valid before the gate-1 turn).")
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

    ref = load_estimator_reference(args.corpus, pre_tick_only=not args.full_flight)
    if ref is not None:
        scope = "whole flight" if args.full_flight else "pre-tick approach leg"
        print(f"estimator ref scope  : {scope} (n={len(ref[0])} p_est samples)")
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

    # DRIFT: VO is unit-scale, so recover the single best-fit metric scale on the
    # SAME (aligned) displacement set (weighted least squares), then measure how
    # far the scaled VO track drifts from the estimator track. NOTE both tracks
    # are dead-reckoning-ish (estimator = DR, VO = un-loop-closed integration),
    # so this is RELATIVE divergence between two imperfect tracks, NOT absolute
    # VO error against truth. Reported per the join-aligned motion window only.
    denom = np.sum(wts * np.sum(D_vo_a ** 2, axis=1))
    if denom <= 0:
        return
    scale = float(np.sum(wts * np.sum(D_vo_a * D_ref, axis=1)) / denom)
    D_vo_s = scale * D_vo_a
    leg_len = np.linalg.norm(D_ref, axis=1)
    leg_res = np.linalg.norm(D_vo_s - D_ref, axis=1)
    leg_drift = leg_res / leg_len                      # per-leg fractional drift
    wmed_drift = float(np.percentile(leg_drift, 50))
    # cumulative endpoint drift over the covered window / estimator path length
    ref_path = float(np.sum(leg_len))
    endpoint_res = float(np.linalg.norm(np.sum(D_vo_s - D_ref, axis=0)))
    cum_drift = endpoint_res / ref_path if ref_path > 0 else float("nan")
    print(f"VO-vs-est drift      : fit scale={scale:.3f} (unit->m), "
          f"per-leg |resid|/|leg| med={wmed_drift:.0%}, "
          f"endpoint {endpoint_res:.2f}m / {ref_path:.1f}m path = {cum_drift:.0%}")


if __name__ == "__main__":
    main()
