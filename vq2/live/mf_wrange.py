"""Width-as-range correction for the staged-leg dead reckoning (COR-147).

WHY (measured, 2026-07-25). The gate-2 staging leg is flown on MF_DR:
position from COMMANDED speed x gyro yaw, seeded at the gate-1 tick
(vq2wp.py, `_vn = 1.2 if pitching else 0.35`). Nothing measures the actual
displacement, so the believed range to gate-2 drifts from the truth. Scale-free
audit of the banked corpora (fractional closure, vision vs MF_DR, over the leg):

    vq2_mig2   vision closed  9%   MF_DR believed 39%   over-credit 4.3x
    vq2_mig3   vision closed 53%   MF_DR believed 76%   over-credit 1.4x
    vq2_mig4   vision closed 52%   MF_DR believed 53%   consistent
    vq2_mig8   vision closed -14%  MF_DR believed 68%   (range GREW)

MF_DR converges on its staging waypoint -- believing it has arrived -- while
the drone is still far short in the world. It then stops driving, the gate
never grows past the vision-release width, and the leg dies in the staged
go-around.

WHAT. The detector already publishes the gate's apparent width `fg_w` at
~30 Hz, and range is proportional to 1/width. That is a direct measurement of
the exact quantity MF_DR gets wrong. This module turns it into a bounded
correction along the gate bearing.

SCALE-FREE BY CONSTRUCTION. The true width of whatever the detector locks
onto is NOT the 1.5 m aperture -- at the gate-1 tick the geometry gives 6.68 m
to gate-2 at fg_w=41, implying ~0.86 m. So no constant is assumed: C = r*w is
calibrated once per leg, from the DR's own believed range at the instant just
after the seed, when the DR is at its most trustworthy. Everything after that
is a RELATIVE range correction, immune to what the detector is actually
measuring -- as long as it keeps measuring the same thing.

RANGE ONLY. Width measures range, not bearing. The correction moves the DR
along the gate bearing and never laterally.

FAILS CLOSED. Stale detection, small width, uncalibrated, degenerate range,
or an implausible residual -> zero correction and a reason string.
"""
from __future__ import annotations

import math

# Defaults are deliberately timid: this authority competes with a guidance
# loop, so a bad detection must never be able to teleport the DR.
W_MIN = 30.0        # px: below this the width estimate is noise
MAX_AGE = 0.4       # s: same freshness the vision-release gate uses
# The caller runs at CMD_HZ = 50, and a detection persists for several updates,
# so the per-update gain sets a time constant of 1/(50*GAIN). The drift being
# corrected accumulates over ~5 s, so the correction must be SLOWER than the
# detection cadence, not faster -- at 0.15 it would be 0.13 s and would slave
# the DR to every individual width reading, throwing away the DR's smoothing
# and letting one bad detection dominate. 0.02 gives ~1 s.
GAIN = 0.02         # fraction of the residual applied per update
MAX_STEP = 0.05     # m per update: 2.5 m/s slew cap at 50 Hz
MAX_ERR = 15.0      # m: larger residual = wrong object, not drift
MIN_RANGE = 0.5     # m: below this the bearing unit vector is unstable


def calibrate(mf_p, gate_w, fg_w, fg_age, *, w_min=W_MIN, max_age=MAX_AGE):
    """C = r_believed * w, anchored just after the MF_DR seed.

    Returns (C, reason). C is None when the sample is not usable.
    """
    if fg_age is None or fg_age > max_age:
        return None, 'stale'
    if not fg_w or fg_w < w_min:
        return None, 'narrow'
    r = math.hypot(float(gate_w[0]) - float(mf_p[0]),
                   float(gate_w[1]) - float(mf_p[1]))
    if r < MIN_RANGE:
        return None, 'degenerate'
    return r * float(fg_w), 'ok'


def step(mf_p, gate_w, fg_w, fg_age, cal_c, *, gain=GAIN, max_step=MAX_STEP,
         w_min=W_MIN, max_age=MAX_AGE, max_err=MAX_ERR):
    """One correction step. Returns (dx, dy, diag).

    diag carries the reason and, when it fired, the measured residual so the
    correction is auditable offline from the livelog alone.

    Sign: residual = r_vision - r_dr. Positive means the DR believes it is
    CLOSER than the imagery says, so the DR is pushed back AWAY from the gate
    (the over-credit case above).
    """
    if cal_c is None:
        return 0.0, 0.0, {'why': 'uncalibrated'}
    if fg_age is None or fg_age > max_age:
        return 0.0, 0.0, {'why': 'stale'}
    if not fg_w or fg_w < w_min:
        return 0.0, 0.0, {'why': 'narrow'}
    dx = float(mf_p[0]) - float(gate_w[0])
    dy = float(mf_p[1]) - float(gate_w[1])
    r_dr = math.hypot(dx, dy)
    if r_dr < MIN_RANGE:
        return 0.0, 0.0, {'why': 'degenerate'}
    r_vis = cal_c / float(fg_w)
    resid = r_vis - r_dr
    if abs(resid) > max_err:
        return 0.0, 0.0, {'why': 'implausible', 'resid': round(resid, 2)}
    move = max(-max_step, min(max_step, resid * gain))
    ux, uy = dx / r_dr, dy / r_dr        # unit vector gate -> DR position
    return (ux * move, uy * move,
            {'why': 'ok', 'r_vis': round(r_vis, 2), 'r_dr': round(r_dr, 2),
             'resid': round(resid, 2), 'move': round(move, 3)})
