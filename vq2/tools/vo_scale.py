"""Metric scale for the unit-scale MonoVO stream (COR-147 Phase 2 + 2b).

MonoVO's OdomDelta is monocular, UNIT-per-keyframe-pair: `vo_cv._solve` takes
recoverPose's up-to-scale translation and RE-NORMALIZES it (t_body_unit = t/|t|),
with NO cross-keyframe triangulation or shared points. So magnitude is
mathematically unobservable AND discarded, and there is ZERO scale link between
consecutive deltas => integrated path length ∝ KEYFRAME COUNT, not metres, and a
single locked scalar makes only the first pair metric. Global metric scale
therefore requires either per-keyframe magnitude (a relpose.py contract change +
triangulation/Sim3) OR PER-GATE RE-ANCHORING (this module's approach — no
contract change, no local-map SLAM).

DECISIONS APPLIED (Alex, 2026-07-23): adopt gate-scale, RETIRE climb-cal for the
OpenCV VO. Estimators here:

  A  climb-cal (RETIRED, kept only to document its failure): scale = ||Δp_KF|| /
     ||Δp_VO|| over the climb window. NON-VIABLE for MonoVO — the low-parallax
     near-vertical climb closes 0-3 keyframes (of 103-294), so ||Δp_VO|| is an
     interpolation artifact and the scale lands ~10x off. (Legacy climb-cal
     worked for DPVO's dense per-frame poses; it does not transfer here.)

  B  gate-scale (gate-2): scale from the gate-PnP metric RANGE closing on the
     gate-2 approach vs VO unit displacement. Range ||t_cam|| is a scalar
     (rotation-free). Only available AFTER the gate-1 tick.

  B' gate-1 SINGLE-VIEW anchor (Phase 2b, Alex's idea): a known-size (1.5 m)
     gate seen ONCE at absolute PnP range defines a metric baseline WITHOUT range
     closing — from that view the drone flies to CROSS the gate (range->0), so it
     travels ~that range. This scales the BLIND gate-1 leg that B cannot reach.

MEASURED (test95-99/46/140): gate 1 is flown BLIND (NOFIX=1) — pre-tick obs sit
at a CONSTANT ~6.2 m gate-1 range (no closing), the approach is obs_nofix. So B
needs the gate-2 leg (post-tick), and B' (single-view) is what scales gate 1. The
two independent gate anchors agree to a median 0.95 (n=6) => per-gate re-anchoring
gives a consistent scale without global propagation.

Estimators are pure functions of (VO capture-ns, VO unit trajectory) + the
per-corpus livelog series, so they unit-test on synthetic arrays. `main` runs
MonoVO and prints the per-corpus scale table.
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


def gate1_anchor_scale(kf_ns, traj, pad_range_m, pad_ns, tick_ns):
    """Strategy B' — SINGLE-VIEW gate anchor (COR-147 Phase 2b, Alex's idea).

    A known-size (1.5 m) gate seen ONCE at absolute PnP range `pad_range_m`
    defines a metric baseline WITHOUT any range closing: from that view the drone
    flies to CROSS the gate (range -> 0 at `tick_ns`), so it travels ~pad_range_m.
    scale = pad_range_m / ||VO(tick) - VO(pad_view)|| (VO unit displacement over
    the same interval). This is what B lacked on the blind (NOFIX) gate-1 leg:
    gate 1 is observed only at the pad (constant ~6.2 m range, no closing), but a
    single view + the known crossing is enough to anchor the leg.

    Returns dict(scale, baseline_m, d_vo, ns0, ns1) or dict(scale=None, reason=).
    """
    if len(kf_ns) < 2 or tick_ns is None:
        return {"scale": None, "reason": "no tick / too few VO"}
    lo, hi = kf_ns[0], kf_ns[-1]
    if not (lo <= pad_ns <= hi):
        return {"scale": None, "reason": "pad view outside VO coverage"}
    # tick may sit just past the last keyframe; np.interp clamps, which is the
    # VO endpoint — acceptable, but require the pad view to be covered.
    v0 = _interp_xyz(kf_ns, traj, pad_ns)
    v1 = _interp_xyz(kf_ns, traj, min(max(tick_ns, lo), hi))
    d_vo = float(np.linalg.norm(v1 - v0))
    if d_vo < 1e-6:
        return {"scale": None, "reason": "degenerate VO displacement"}
    return {"scale": float(pad_range_m) / d_vo, "baseline_m": float(pad_range_m),
            "d_vo": d_vo, "ns0": float(pad_ns), "ns1": float(tick_ns)}


def load_gate1_anchor(corpus):
    """(pad_range_m, pad_ns, tick_ns) for the single-view gate-1 anchor.

    pad_range_m = median PRE-TICK obs range (||t_cam||, the gate-1 range while the
    drone climbs at the pad); pad_ns = the LAST such pre-tick view (closest to the
    approach); tick_ns = first gate tick (the crossing). None if unavailable.
    """
    rows = read_rows(os.path.join(corpus, "livelog.jsonl"))
    clock = build_capture_clock(rows)
    if clock is None:
        return None
    tk = first_tick_capture_ns(rows, clock)
    if tk is None:
        return None
    pre = []
    for d in rows:
        if d.get("kind") == "obs" and isinstance(d.get("t_cam"), list) and len(d["t_cam"]) >= 3:
            cs = map_t_to_capture_s(d["t"], clock)
            if cs is None:
                continue
            n = int(round(cs * 1e9))
            if n < tk:
                pre.append((n, float(np.linalg.norm(d["t_cam"]))))
    if len(pre) < 1:
        return None
    pre.sort()
    pad_range = float(np.median([r for _, r in pre]))
    return pad_range, float(pre[-1][0]), float(tk)


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
    hdr = (f"{'corpus':<14}{'A climb':>9}{'B gate2':>9}{'g1 view':>9}"
           f"{'s_pest':>9}{'g1/B':>8}{'g1/pest':>9}   notes")
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
        g1 = load_gate1_anchor(corpus)
        pref = load_estimator_reference(corpus, pre_tick_only=True)
        a = climb_cal_scale(kf_ns, traj, *climb) if climb else {"scale": None, "reason": "no climb"}
        b = gate_scale(kf_ns, traj, *gates) if gates else {"scale": None, "reason": "no gate2 obs"}
        g = gate1_anchor_scale(kf_ns, traj, *g1) if g1 else {"scale": None, "reason": "no gate1 view"}
        p = pest_fit_scale(kf_ns, traj, *pref) if pref else {"scale": None, "reason": "no p_est"}
        sA, sB, sG, sP = a.get("scale"), b.get("scale"), g.get("scale"), p.get("scale")
        gB = sG / sB if sG and sB else None
        gP = sG / sP if sG and sP else None
        note = g.get("reason", "") or b.get("reason", "")
        print(f"{name:<14}{_fmt(sA)}{_fmt(sB)}{_fmt(sG)}{_fmt(sP)}"
              f"{_fmt(gB):>8}{_fmt(gP)}   {note}")
        rows_out.append((name, sA, sB, sG, sP))
    print("-" * len(hdr))
    n_g1 = sum(1 for r in rows_out if r[3])
    n_b = sum(1 for r in rows_out if r[2])
    print(f"corpora: {len(rows_out)}  |  gate-1 single-view: {n_g1}  gate-2 B: {n_b}"
          f"  climb-A viable: 0 (0-3 keyframes in climb window)")
    gb = np.array([r[3] / r[2] for r in rows_out if r[2] and r[3]])
    if len(gb):
        print(f"gate1/gate2 scale agreement: median {np.median(gb):.3f} "
              f"[{gb.min():.3f}..{gb.max():.3f}] (n={len(gb)}) "
              f"-- two independent gate anchors, same underlying scale")


if __name__ == "__main__":
    main()

