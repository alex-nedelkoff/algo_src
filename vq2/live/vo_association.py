"""Associate VO measurements into the SlidingSmoother (COR-147, Phase 3).

The smoother's push_odom already lowers a yaw-relative delta to an OdomFactor,
and VO's t_body_unit is inherently yaw-relative (body-x = forward regardless
of world heading). So association is thin — its real jobs are:

  1. SCALE: monocular VO is unit-scale. Recover the metric scale from a known
     leg (gate crossings: surveyed pad->G1 or G1->G2 distance / VO leg length),
     or plug in a climb-calibrated scale. Applied before every push.
  2. INFORMATION-WEIGHTED SIGMA: the smoother design wants flow-inlier-scaled
     odometry sigma (a keyframe with 300 inliers is tighter than one with 40).

Frame note: VO t_body_unit is in the body frame at t0 (x-fwd, y-right, z-down);
push_odom applies the state's yaw to rotate it into the world frame. dyaw is
the VO relative-yaw increment. No extra rotation needed.
"""
from __future__ import annotations

import math

import numpy as np

from .vo_cv import VoStep

# reference inlier count at which sigma == base (healthy keyframe)
_REF_INLIERS = 120.0


def integrate_leg(steps: list[VoStep], t0_ns: float, t1_ns: float) -> np.ndarray:
    """Unit-scale VO displacement (3,) between two capture times, integrating
    keyframe deltas in the accumulated body frame (same chaining as the
    replay harness)."""
    C = np.eye(3)
    pos = np.zeros(3)
    at0 = at1 = None
    t0 = None
    for s in steps:
        if t0 is None:
            t0 = s.t0
        # position at the START of this step's interval is `pos`
        if at0 is None and s.t1 * 1e9 >= t0_ns:
            at0 = pos.copy()
        pos = pos + C @ s.t_body_unit
        C = C @ s.R_body
        if at1 is None and s.t1 * 1e9 >= t1_ns:
            at1 = pos.copy()
    if at0 is None or at1 is None:
        raise ValueError("leg endpoints not covered by VO keyframes")
    return at1 - at0


def calibrate_scale(steps: list[VoStep], t0_ns: float, t1_ns: float,
                    known_dist_m: float) -> float:
    """Metric scale = known leg distance / VO unit leg length. Use a gate->gate
    leg (surveyed) or pad->gate. Raises if the VO leg is degenerate."""
    leg = integrate_leg(steps, t0_ns, t1_ns)
    ulen = float(np.linalg.norm(leg))
    if ulen < 1e-6:
        raise ValueError("degenerate VO leg (unit length ~0)")
    return known_dist_m / ulen


def sigma_for(step: VoStep, base_sigma_p: float) -> float:
    """Information-weighted position sigma: tighter for high-inlier keyframes,
    looser for sparse ones. sqrt scaling on the inlier ratio, clamped."""
    r = _REF_INLIERS / max(step.n_inliers, 1)
    return float(base_sigma_p * min(max(math.sqrt(r), 0.5), 3.0))


def feed_smoother(smoother, steps: list[VoStep], scale: float,
                  base_sigma_p: float = 0.15, sigma_yaw: float = 0.05) -> int:
    """Push every VoStep into the smoother as a scaled, info-weighted odom
    factor. Returns the number pushed. Deltas are metric (scale applied)."""
    n = 0
    for s in steps:
        dp = scale * s.t_body_unit
        smoother.push_odom(s.t1, dp, s.dyaw,
                           sigma_p=sigma_for(s, base_sigma_p),
                           sigma_yaw=sigma_yaw)
        n += 1
    return n
