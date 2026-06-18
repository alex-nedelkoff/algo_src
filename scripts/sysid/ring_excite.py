"""Open-loop body-rate excitation for high-tilt rate-loop ID: ramped chirp per tilt
setpoint. Pure numpy; the live --ringid deploy mode consumes this schedule."""
from __future__ import annotations
import numpy as np


def excite_schedule(axis, dt, tilt_setpoints, chirp_f0, chirp_f1, seg_s, amp) -> np.ndarray:
    seg_n = int(seg_s / dt)
    t = np.arange(seg_n) * dt
    # linear chirp phase
    phase = 2 * np.pi * (chirp_f0 * t + 0.5 * (chirp_f1 - chirp_f0) / seg_s * t ** 2)
    base = np.sin(phase)
    ramp = np.clip(t / (0.2 * seg_s), 0.0, 1.0)             # ramp-in over first 20%
    seg_cmd = amp * ramp * base
    rows = []
    for tilt in tilt_setpoints:
        block = np.zeros((seg_n, 4))
        block[:, axis] = seg_cmd
        block[:, 3] = float(tilt)
        rows.append(block)
    return np.vstack(rows)
