"""Waypoint navigation API — importable, testable wrapper around goto.py's proven control.

Pure control law (cruise_accel / settle_accel / attitude_command) is separated from the
flight loop so it can be unit-tested with synthetic states. WaypointNavigator owns the
blocking loop + IO; the caller owns lifecycle (sim reset + arm) and hands in a live Store
and Commander.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

from aigp.control_math import (accel_to_thrust_norm, attitude_error_quat,
                               collective_accel, desired_attitude, mat_to_quat)
from aigp.geometry import quat_to_R

G = 9.81


@dataclass
class NavGains:
    """All goto.py tuning constants; defaults are goto.py's proven live values."""
    KP_ATT: np.ndarray = field(default_factory=lambda: np.array([0.5, 1.6, 1.0]))
    KP_YAW: float = 3.0
    KD_YAW: float = 0.3
    KP_CT: float = 0.5
    KD_CT: float = 1.2
    AL_MAX: float = 0.5
    KD_AL: float = 1.2
    KP_Z: float = 1.8
    KD_Z: float = 3.0
    WMAX: float = 4.0
    TILT_MAX_DEG: float = 15.0
    ABORT_TILT_DEG: float = 80.0
    WP_TIMEOUT: float = 40.0
    MAX_SPEED: float = 1.2
    ARRIVE: float = 1.5
    DECEL_MAX: float = 2.0
    SETTLE_T: float = 2.5
    SETTLE_V: float = 0.4
    LOOP_DT: float = 0.004


def load_plant(path: str = "sysid/sim_response.json"):
    """Probed plant params from sysid: (hover_thrust, k_a, rate_gain_axes[roll,pitch,yaw])."""
    r = json.load(open(path))
    hover = float(r["hover_thrust"])
    k_a = float(r["k_a"])
    rg = np.array([r["rate_gain_axes"]["roll"],
                   r["rate_gain_axes"]["pitch"],
                   r["rate_gain_axes"]["yaw"]], float)
    return hover, k_a, rg


def cruise_accel(state, leg_start, target, gains):
    """Horizontal accel for a leg: along-track ramp-to-stop + cross-track line-follow.
    Returns (a2, tangent, speed_cmd). Mirrors goto.py:fly_leg's inner law."""
    pos = state.pos_ned
    vel = state.vel_ned
    leg_start = np.asarray(leg_start, float)
    target = np.asarray(target, float)
    seg = (target - leg_start)[:2]
    seglen = float(np.linalg.norm(seg))
    rel = target - pos
    if seglen > 1e-3:
        tv = seg / seglen
    else:
        tv = rel[:2] / (np.linalg.norm(rel[:2]) + 1e-9)
    lat = np.array([-tv[1], tv[0]])
    spd = float(np.clip(0.6 * float(rel[:2] @ tv), 0.0, gains.MAX_SPEED))
    a_al = float(np.clip(gains.KD_AL * (spd - float(vel[:2] @ tv)),
                         -gains.DECEL_MAX, gains.AL_MAX))
    a_ct = -gains.KP_CT * float((pos - leg_start)[:2] @ lat) - gains.KD_CT * float(vel[:2] @ lat)
    a2 = a_al * tv + a_ct * lat
    return a2, tv, spd


def settle_accel(state, target, gains):
    """Horizontal accel to kill momentum at target: pull-to-point + velocity damp.
    Mirrors goto.py:settle."""
    target = np.asarray(target, float)
    err = (target - state.pos_ned)[:2]
    return np.clip(gains.KP_CT * err, -1.0, 1.0) - gains.KD_CT * state.vel_ned[:2]
