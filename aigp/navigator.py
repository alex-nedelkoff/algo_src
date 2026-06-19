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
    RAMP_K: float = 0.6   # along-track distance -> commanded speed gain
    # Per-axis sign on the body-rate command (roll, pitch, yaw). The VQ sim's rate-loop
    # roll axis is inverted vs our FRD convention (sysID: roll sign = -1), so a positive
    # roll setpoint drives the drone the wrong way -> lateral runaway. -1 on roll fixes it.
    RATE_SIGN: np.ndarray = field(default_factory=lambda: np.array([-1.0, 1.0, 1.0]))
    ARRIVE: float = 1.5
    DECEL_MAX: float = 2.0
    SETTLE_T: float = 2.5
    SETTLE_V: float = 0.4
    LOOP_DT: float = 0.004


def load_plant(path: str = "sysid/sim_response.json"):
    """Probed plant params from sysid: (hover_thrust, k_a, rate_gain_axes[roll,pitch,yaw])."""
    with open(path) as f:
        r = json.load(f)
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
    spd = float(np.clip(gains.RAMP_K * float(rel[:2] @ tv), 0.0, gains.MAX_SPEED))
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


def attitude_command(state, a2, z_sp, yaw_ref, plant, gains):
    """Inner law (pure goto.py:cmd): horizontal accel -> tilt-clamp -> desired attitude ->
    normalized body-rate command (clipped) + thrust. Yaw rate steers toward yaw_ref.
    Returns (rate_cmd_norm, thrust, tilt_deg, dbg)."""
    hover, k_a, rg = plant
    a = np.zeros(3)
    a[:2] = np.asarray(a2, float)
    tilt_max_acc = np.tan(np.radians(gains.TILT_MAX_DEG)) * G
    n = float(np.linalg.norm(a[:2]))
    if n > tilt_max_acc:
        a[:2] = a[:2] / n * tilt_max_acc
    a[2] = gains.KP_Z * (z_sp - state.pos_ned[2]) + gains.KD_Z * (0.0 - state.vel_ned[2])

    R = quat_to_R(state.quat_wxyz)
    yaw_cur = float(np.arctan2(R[1, 0], R[0, 0]))
    q_des = mat_to_quat(desired_attitude(a, yaw_cur))
    w_des = gains.KP_ATT * attitude_error_quat(state.quat_wxyz, q_des)
    w_des[2] = (gains.KP_YAW * ((yaw_ref - yaw_cur + np.pi) % (2 * np.pi) - np.pi)
                - gains.KD_YAW * float(state.omega[2]))
    thr = accel_to_thrust_norm(collective_accel(a, state.quat_wxyz), hover, k_a)
    rate_cmd_norm = np.clip(w_des * gains.RATE_SIGN / rg, -gains.WMAX, gains.WMAX)
    tilt = float(np.degrees(np.arccos(max(-1.0, min(1.0, R[2, 2])))))
    return rate_cmd_norm, thr, tilt, {"a": a, "w_des": w_des, "q_des": q_des, "thr": thr}


class WaypointNavigator:
    """Blocking waypoint navigation over the proven attitude-rate control. Caller owns lifecycle
    (sim reset + arm) and passes a live store + commander. See module docstring."""

    def __init__(self, store, commander, plant, *, gains=None, flog=None):
        self.store = store
        self.commander = commander
        self.plant = plant
        self.gains = gains if gains is not None else NavGains()
        self.flog = flog
        self._origin_pos = None
        self._origin_yaw = None
        self._look_point = None   # world-NED point the camera tracks in yaw='lookat'
        self._t0 = None

    # --- pure helpers (IO-free) ---
    def set_origin(self, pos_ned=None, yaw=None):
        """Capture the body-frame reference pose. Missing fields are read from current drone state."""
        if pos_ned is None or yaw is None:
            ds = self.store.get_drone()
            if ds is None:
                raise RuntimeError("no drone state available to capture origin")
            if pos_ned is None:
                pos_ned = ds.pos_ned.copy()
            if yaw is None:
                R = quat_to_R(ds.quat_wxyz)
                yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        self._origin_pos = np.asarray(pos_ned, float)
        self._origin_yaw = float(yaw)

    def _resolve(self, wp, frame):
        wp = np.asarray(wp, float)
        if frame == "world":
            return wp
        if frame == "body":
            if self._origin_pos is None:
                self.set_origin()
            cy, sy = np.cos(self._origin_yaw), np.sin(self._origin_yaw)
            fwd = np.array([-cy, -sy])
            right = np.array([-sy, cy])
            off = np.array([wp[0] * fwd[0] + wp[1] * right[0],
                            wp[0] * fwd[1] + wp[1] * right[1],
                            wp[2]])
            return self._origin_pos + off
        raise ValueError(f"frame must be 'world' or 'body', got {frame!r}")

    def _yaw_ref(self, yaw_mode, target):
        """Desired NOSE (body-x) heading. Camera is -body_x, so camera-pointing modes add pi.
        Modes: 'hold' (fixed origin heading, strafe), 'face' (nose at target),
        'lookat' (camera at self._look_point, else target), 'course' (camera along travel)."""
        if yaw_mode == "hold":
            return self._origin_yaw if self._origin_yaw is not None else 0.0
        ds = self.store.get_drone()
        pos = ds.pos_ned if ds is not None else (self._origin_pos
                                                 if self._origin_pos is not None else np.zeros(3))
        if yaw_mode == "face":                       # point the NOSE at the target
            rel = np.asarray(target, float) - pos
            return float(np.arctan2(rel[1], rel[0]))
        if yaw_mode == "lookat":                     # point the CAMERA (-body_x) at a fixed point
            pt = self._look_point if self._look_point is not None else np.asarray(target, float)
            rel = np.asarray(pt, float) - pos
            return float(np.arctan2(rel[1], rel[0]) + np.pi)
        if yaw_mode == "course":                     # point the CAMERA along horizontal travel
            v = ds.vel_ned[:2] if ds is not None else np.zeros(2)
            if float(np.linalg.norm(v)) > 0.3:
                return float(np.arctan2(v[1], v[0]) + np.pi)
            rel = np.asarray(target, float) - pos    # too slow to have a heading -> aim at target
            return float(np.arctan2(rel[1], rel[0]) + np.pi)
        raise ValueError(f"yaw must be 'hold'|'face'|'lookat'|'course', got {yaw_mode!r}")

    def _current_pos(self):
        ds = self.store.get_drone()
        if ds is not None:
            return ds.pos_ned.copy()
        return self._origin_pos.copy() if self._origin_pos is not None else np.zeros(3)

    # --- blocking flight loop (live sim; not unit-tested) ---
    def goto(self, wp, *, frame="world", yaw="hold", look_point=None):
        if self._origin_pos is None:
            self.set_origin()
        if self._t0 is None:
            self._t0 = time.time()
        if look_point is not None:
            self._look_point = self._resolve(look_point, frame)
        target = self._resolve(wp, frame)
        return self._fly_leg(target, self._origin_pos, yaw)

    def follow(self, wps, *, frame="world", yaw="hold", settle=True, look_point=None):
        if self._origin_pos is None:
            self.set_origin()
        if self._t0 is None:
            self._t0 = time.time()
        if look_point is not None:
            self._look_point = self._resolve(look_point, frame)
        targets = [self._resolve(w, frame) for w in wps]
        if self.flog is not None:
            self.flog.set_path(np.vstack([self._current_pos()] + targets))
        leg_start = self._origin_pos.copy()
        for i, target in enumerate(targets):
            res = self._fly_leg(target, leg_start, yaw)
            print(f"   {res.upper()} wp{i}", flush=True)
            if res != "reached":
                return res
            if settle:
                self.settle(target, yaw)
            leg_start = self._current_pos()
        return "reached"

    def settle(self, target=None, yaw="hold"):
        if target is None:
            target = self._current_pos()
        target = np.asarray(target, float)
        if self._t0 is None:
            self._t0 = time.time()
        t0 = time.time()
        while time.time() - t0 < self.gains.SETTLE_T:
            ds = self.store.get_drone()
            if ds is not None:
                if float(np.linalg.norm(ds.vel_ned[:2])) < self.gains.SETTLE_V:
                    return
                a2 = settle_accel(ds, target, self.gains)
                rate, thr, tilt, dbg = attitude_command(ds, a2, float(target[2]),
                                                         self._yaw_ref(yaw, target), self.plant, self.gains)
                self.commander.send_attitude_target(rate, thr)
                if self.flog is not None:
                    self.flog.push(time.time() - self._t0, ds, dbg, nearest=target, cruise=0.0,
                                   running=self.store.get_race_live(), armed=True)
                if tilt > self.gains.ABORT_TILT_DEG:
                    print(f"  ABORT settle tilt={tilt:.0f}", flush=True)
                    return
            time.sleep(self.gains.LOOP_DT)

    def _fly_leg(self, target, leg_start, yaw_mode):
        g = self.gains
        t_leg = time.time()
        last = -1
        while time.time() - t_leg < g.WP_TIMEOUT:
            ds = self.store.get_drone()
            if ds is not None:
                rel = target - ds.pos_ned
                if float(np.linalg.norm(rel)) < g.ARRIVE:
                    return "reached"
                a2, tv, spd = cruise_accel(ds, leg_start, target, g)
                rate, thr, tilt, dbg = attitude_command(ds, a2, float(target[2]),
                                                         self._yaw_ref(yaw_mode, target),
                                                         self.plant, g)
                self.commander.send_attitude_target(rate, thr)
                if self.flog is not None:
                    self.flog.push(time.time() - self._t0, ds, dbg, nearest=target, tangent=tv,
                                   cruise=spd, running=self.store.get_race_live(), armed=True)
                if tilt > g.ABORT_TILT_DEG:
                    print(f"  ABORT tilt={tilt:.0f}", flush=True)
                    return "abort"
                k = int((time.time() - t_leg) / 2.0)
                if k != last:
                    last = k
                    print(f"  dist={float(np.linalg.norm(rel)):5.1f} "
                          f"spd={float(np.linalg.norm(ds.vel_ned[:2])):4.1f} tilt={tilt:3.0f}",
                          flush=True)
            time.sleep(g.LOOP_DT)
        return "timeout"
