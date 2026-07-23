"""Metric-scale A/B for the unit-scale MonoVO stream (COR-147 Phase 2).

The MonoVO OdomDelta stream is monocular = direction only (scale_locked=False).
This module puts METRIC scale on it two independent ways and lets the harness
compare them offline:

  A  climb-calibration (legacy, vq2/live/dpvo_odom.py): scale = ||Δp_KF|| /
     ||Δp_VO|| over the early climb window, where p_KF is the noiseless-IMU DR
     estimate (independent of VO — VO never feeds back offline). Available
     BEFORE gate 1. This is what truth_crossings assumes (CLIMB_TRUE anchor).

  B  gate-scale / GNSCALE (legacy vq2/live/dpvo_gate_scale.py): scale from the
     gate-PnP metric RANGE closing as the drone approaches a gate, vs the VO
     unit displacement over the same capture-time interval. Range ||t_cam|| is a
     scalar (rotation-frame-free), so the turn does not bias it.

MEASURED DATA REALITY (test95-99/46/140, see COR-147 notes): gate 1 is flown
BLIND (NOFIX=1) — pre-tick obs sit at a CONSTANT ~6.2 m gate-1 range (no closing
to scale against). The only range-CLOSING PnP is the gate-2 approach, which is
POST the gate-1 tick. So B is structurally unavailable until after gate 1, while
A is available during the climb. That asymmetry is a first-class result, not a
tuning detail.

Estimators are pure functions of (VO capture-ns, VO unit trajectory) + the
per-corpus livelog series, so they unit-test on synthetic arrays. `scale_report`
runs MonoVO and prints the per-corpus A/B/p_est table.
"""
from __future__ import annotations

import os

import numpy as np

from vq2.tools.livelog_join import (
    read_rows, build_capture_clock, map_t_to_capture_s, first_tick_capture_ns,
)


def _interp_xyz(kf_ns, traj, ns):
    return np.array([np.interp(ns, kf_ns, traj[:, i]) for i in range(3)])


def climb_cal_scale(kf_ns, traj, ref_ns, ref_xyz, min_disp_m=0.8):
    """Strategy A. `ref_*` = the climb-cluster KF positions (metric, DR) on the
    capture-ns axis. Walk from the first ref sample inside VO coverage until the
    KF has moved >= min_disp_m; scale = that KF displacement / the VO unit
    displacement over the same ns interval.

    Returns dict(scale, d_kf, d_vo, n, ns0, ns1) or dict(scale=None, reason=...).
    """
    if len(kf_ns) < 2 or len(ref_ns) < 2:
        return {"scale": None, "reason": "too few samples"}
    lo, hi = kf_ns[0], kf_ns[-1]
    idx = [i for i in range(len(ref_ns)) if lo <= ref_ns[i] <= hi]
    if len(idx) < 2:
        return {"scale": None, "reason": "climb window outside VO coverage"}
    i0 = idx[0]
    p0 = ref_xyz[i0]
    for k in idx[1:]:
        d_kf = float(np.linalg.norm(ref_xyz[k] - p0))
        if d_kf >= min_disp_m:
            v0 = _interp_xyz(kf_ns, traj, ref_ns[i0])
            v1 = _interp_xyz(kf_ns, traj, ref_ns[k])
            d_vo = float(np.linalg.norm(v1 - v0))
            if d_vo < 1e-6:
                return {"scale": None, "reason": "degenerate VO displacement"}
            return {"scale": d_kf / d_vo, "d_kf": d_kf, "d_vo": d_vo,
                    "n": k - i0 + 1, "ns0": float(ref_ns[i0]),
                    "ns1": float(ref_ns[k])}
    return {"scale": None, "reason": f"climb disp < {min_disp_m} m"}


def gate_scale(kf_ns, traj, obs_ns, obs_range, min_baseline_m=2.0, min_pts=3):
    """Strategy B. `obs_*` = gate-PnP samples (capture ns, metric range ||t_cam||).
    Extract the longest monotone-decreasing (approaching) run inside VO coverage,
    then take the median of pairwise (Δrange / VO-chord) ratios over that run.
    Δrange is a scalar (rotation-free); VO-chord is the unit displacement norm.

    Returns dict(scale, baseline_m, d_vo, n, ns0, ns1) or dict(scale=None,...).
    """
    if len(kf_ns) < 2 or len(obs_ns) < 2:
        return {"scale": None, "reason": "too few samples"}
    lo, hi = kf_ns[0], kf_ns[-1]
    seq = [(obs_ns[i], obs_range[i]) for i in range(len(obs_ns))
           if lo <= obs_ns[i] <= hi]
    seq.sort()
    if len(seq) < min_pts:
        return {"scale": None, "reason": "no gate obs in VO coverage"}
    # longest monotone-decreasing (range closing) run
    best_a = best_b = 0
    a = 0
    for b in range(1, len(seq)):
        if seq[b][1] < seq[b - 1][1]:
            if b - a > best_b - best_a:
                best_a, best_b = a, b
        else:
            a = b
    run = seq[best_a:best_b + 1]
    if len(run) < min_pts:
        return {"scale": None, "reason": "closing run too short"}
    baseline = run[0][1] - run[-1][1]
    if baseline < min_baseline_m:
        return {"scale": None, "reason": f"range baseline < {min_baseline_m} m"}
    vo = {ns: _interp_xyz(kf_ns, traj, ns) for ns, _ in run}
    ratios = []
    for i in range(len(run)):
        for j in range(i + 1, len(run)):
            dr = run[i][1] - run[j][1]
            chord = float(np.linalg.norm(vo[run[j][0]] - vo[run[i][0]]))
            if dr > 1e-6 and chord > 1e-6:
                ratios.append(dr / chord)
    if not ratios:
        return {"scale": None, "reason": "no usable pairs"}
    d_vo = float(np.linalg.norm(vo[run[-1][0]] - vo[run[0][0]]))
    return {"scale": float(np.median(ratios)), "baseline_m": float(baseline),
            "d_vo": d_vo, "n": len(run), "ns0": float(run[0][0]),
            "ns1": float(run[-1][0])}


def load_climb_ref(corpus):
    """(ns[K], xyz[K,3]) from PRE-TICK kf_upd rows = the climb cluster (metric DR)."""
    rows = read_rows(os.path.join(corpus, "livelog.jsonl"))
    clock = build_capture_clock(rows)
    if clock is None:
        return None
    hi = first_tick_capture_ns(rows, clock)
    ns, xyz = [], []
    for d in rows:
        if d.get("kind") == "kf_upd" and isinstance(d.get("p"), list) and len(d["p"]) >= 3:
            cs = map_t_to_capture_s(d["t"], clock)
            if cs is None:
                continue
            n = int(round(cs * 1e9))
            if hi is not None and n > hi:
                continue
            ns.append(n)
            xyz.append([float(x) for x in d["p"][:3]])
    if len(ns) < 2:
        return None
    return np.asarray(ns, dtype=np.int64), np.asarray(xyz, dtype=float)


def load_gate_obs(corpus):
    """(ns[M], range[M]) from POST-TICK obs rows carrying t_cam (gate-2 approach).
    range = ||t_cam|| (metric PnP range). Pre-tick gate-1 is blind (constant
    range) so it is excluded via the first-tick boundary."""
    rows = read_rows(os.path.join(corpus, "livelog.jsonl"))
    clock = build_capture_clock(rows)
    if clock is None:
        return None
    tk = first_tick_capture_ns(rows, clock)
    ns, rng = [], []
    for d in rows:
        if d.get("kind") == "obs" and isinstance(d.get("t_cam"), list) and len(d["t_cam"]) >= 3:
            cs = map_t_to_capture_s(d["t"], clock)
            if cs is None:
                continue
            n = int(round(cs * 1e9))
            if tk is not None and n < tk:
                continue
            ns.append(n)
            rng.append(float(np.linalg.norm(d["t_cam"])))
    if len(ns) < 2:
        return None
    order = np.argsort(ns)
    return np.asarray(ns, dtype=np.int64)[order], np.asarray(rng, dtype=float)[order]


def pest_fit_scale(kf_ns, traj, ref_ns, ref_xyz):
    """SECONDARY reference scale: the metric scale that best matches the VO unit
    path to the p_est (DR) reference over the covered window, via a weighted 2D
    similarity fit on the xy displacement set (same rotation+scale Procrustes as
    vo_replay). p_est is DR not truth, so this corroborates, never decides.

    Returns dict(scale, n) or dict(scale=None, reason=...).
    """
    if len(kf_ns) < 2 or len(ref_ns) < 2:
        return {"scale": None, "reason": "too few samples"}
    lo, hi = kf_ns[0], kf_ns[-1]
    kf_pos = traj
    D_ref, D_vo, wts = [], [], []
    for a in range(len(ref_ns) - 1):
        na, nb = ref_ns[a], ref_ns[a + 1]
        if nb <= na or na < lo or nb > hi:
            continue
        d_ref = (ref_xyz[a + 1] - ref_xyz[a])[:2]
        if np.linalg.norm(d_ref) < 0.05:
            continue
        va = np.array([np.interp(na, kf_ns, kf_pos[:, i]) for i in range(2)])
        vb = np.array([np.interp(nb, kf_ns, kf_pos[:, i]) for i in range(2)])
        d_vo = vb - va
        if np.linalg.norm(d_vo) < 1e-6:
            continue
        D_ref.append(d_ref); D_vo.append(d_vo); wts.append(float(np.linalg.norm(d_ref)))
    if len(D_ref) < 1:
        return {"scale": None, "reason": "no overlapping interval"}
    D_ref, D_vo, wts = np.array(D_ref), np.array(D_vo), np.array(wts)
    s = np.sum(wts * (D_vo[:, 0] * D_ref[:, 1] - D_vo[:, 1] * D_ref[:, 0]))
    c = np.sum(wts * (D_vo[:, 0] * D_ref[:, 0] + D_vo[:, 1] * D_ref[:, 1]))
    theta = np.arctan2(s, c)
    ct, st = np.cos(theta), np.sin(theta)
    R = np.array([[ct, -st], [st, ct]])
    D_vo_a = D_vo @ R.T
    denom = np.sum(wts * np.sum(D_vo_a ** 2, axis=1))
    if denom <= 0:
        return {"scale": None, "reason": "degenerate"}
    scale = float(np.sum(wts * np.sum(D_vo_a * D_ref, axis=1)) / denom)
    return {"scale": scale, "n": len(D_ref)}


def _integrate_vo(corpus, kf_flow=9.0, max_frames=100000):
    """Run MonoVO over corpus frames -> (kf_ns[N] float, traj[N,3] unit)."""
    import cv2
    from vq2.live.vo_cv import MonoVO
    from vq2.tools.vo_replay import load_frame_order, integrate
    frames = load_frame_order(corpus)[:max_frames]
    vo = MonoVO(kf_flow_px=kf_flow)
    steps = []
    for ns, path in frames:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        s = vo.step(ns * 1e-9, img)
        if s is not None:
            steps.append(s)
    if len(steps) < 2:
        return None
    traj = np.array([p for _, p in integrate(steps)])
    kf_ns = np.array([s.t1 * 1e9 for s in steps])   # keyframe close capture ns
    return kf_ns, traj[1:]                            # VO pos at each keyframe close


def _fmt(v):
    return "  n/a " if v is None else f"{v:6.3f}"


def main():
    import argparse
    from vq2.tools.livelog_join import load_estimator_reference
    ap = argparse.ArgumentParser(description="COR-147 Phase 2 scale A/B")
    ap.add_argument("corpora", nargs="+")
    ap.add_argument("--kf-flow", type=float, default=9.0)
    args = ap.parse_args()
    hdr = (f"{'corpus':<14}{'A climb':>9}{'B gate':>9}{'s_pest':>9}"
           f"{'A/pest':>9}{'B/pest':>9}{'A/B':>8}   notes")
    print(hdr); print("-" * len(hdr))
    rows_out = []
    for corpus in args.corpora:
        name = os.path.basename(os.path.normpath(corpus))
        vo = _integrate_vo(corpus, kf_flow=args.kf_flow)
        if vo is None:
            print(f"{name:<14} no VO"); continue
        kf_ns, traj = vo
        climb = load_climb_ref(corpus)
        gates = load_gate_obs(corpus)
        pref = load_estimator_reference(corpus, pre_tick_only=True)
        a = climb_cal_scale(kf_ns, traj, *climb) if climb else {"scale": None, "reason": "no climb ref"}
        b = gate_scale(kf_ns, traj, *gates) if gates else {"scale": None, "reason": "no gate obs"}
        p = pest_fit_scale(kf_ns, traj, *pref) if pref else {"scale": None, "reason": "no p_est ref"}
        sA, sB, sP = a.get("scale"), b.get("scale"), p.get("scale")
        aP = sA / sP if sA and sP else None
        bP = sB / sP if sB and sP else None
        aB = sA / sB if sA and sB else None
        note = a.get("reason", "") or b.get("reason", "")
        print(f"{name:<14}{_fmt(sA)}{_fmt(sB)}{_fmt(sP)}{_fmt(aP)}{_fmt(bP)}"
              f"{_fmt(aB):>8}   {note}")
        rows_out.append((name, sA, sB, sP))
    # summary (n with both A and B available; median ratios)
    both = [(a, b, p) for _, a, b, p in rows_out if a and b]
    print("-" * len(hdr))
    print(f"corpora: {len(rows_out)}  |  A available: {sum(1 for _,a,_,_ in rows_out if a)}"
          f"  B available: {sum(1 for _,_,b,_ in rows_out if b)}  |  both: {len(both)}")
    if both:
        aB = np.array([a / b for a, b, _ in both])
        print(f"A/B ratio over corpora with both: median {np.median(aB):.3f} "
              f"[{aB.min():.3f}..{aB.max():.3f}] (n={len(both)})")


if __name__ == "__main__":
    main()

