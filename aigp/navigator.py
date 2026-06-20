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
from gate_traj import GateTrajectory

G = 9.81

WFIX = np.array([1.0, -1.0, 1.0])   # sim rate-frame mirror (pitch axis), proven (fly_gate3/vq_waypoint2)


class ZVDShaper:
    """Zero-Vibration-Derivative input shaper on the body-rate command — cancels the underdamped
    rate-loop ring (EXP-20a/b weights for zeta~0.14). Per-axis 3-impulse:
    out[ax] = A0*r[t] + A1*r[t-d] + A2*r[t-2d], d = ZVD_DELAY[ax] in loop frames.
    Amplitudes are the proven deploy values (closed_loop_policy.py); the delay is loop-rate
    dependent (deploy used (7,7,4) at 72 Hz -> ~(24,24,14) at the navigator's ~250 Hz; tune live)."""
    AMP = np.array([0.371, 0.476, 0.153])

    def __init__(self, delay=(14, 14, 14)):
        self.delay = tuple(int(x) for x in delay)
        self.buf = np.zeros((2 * max(self.delay) + 1, 3))

    def shape(self, rates):
        self.buf = np.roll(self.buf, 1, axis=0)
        self.buf[0] = np.asarray(rates, float)
        a, d = self.AMP, self.delay
        return np.array([a[0] * self.buf[0, ax] + a[1] * self.buf[d[ax], ax]
                         + a[2] * self.buf[2 * d[ax], ax] for ax in range(3)])


def _qfix(q):
    """Live sim-wire quat (wxyz) -> TRUE attitude quat (wxyz): q_true = q_wire[[1,2,3,0]]."""
    q = np.asarray(q, float)
    return q[[1, 2, 3, 0]]


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
    YAW_WMAX: float = 1.0   # gentle yaw-rate cap: fast yaw steps excite the rate loop -> tilt spike
    TILT_MAX_DEG: float = 15.0
    ABORT_TILT_DEG: float = 80.0
    WP_TIMEOUT: float = 40.0
    MAX_SPEED: float = 1.2
    RAMP_K: float = 0.6   # along-track distance -> commanded speed gain
    # Per-axis sign on the body-rate command (roll, pitch, yaw). Default [1,1,1]: live tests
    # showed [-1,..] DESTABILISES the forward leg, so the sim's roll axis matches our FRD
    # convention. (The lateral-leg runaway is the weathervane on SIDEWAYS flight, not a roll
    # sign — fly nose-first via yaw='lookat'/'course' to avoid it.) Kept as a knob for probing.
    RATE_SIGN: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))
    ARRIVE: float = 1.5
    DECEL_MAX: float = 2.0
    SETTLE_T: float = 2.5
    SETTLE_V: float = 0.4
    LOOP_DT: float = 0.004
    KD_ATT: float = 0.3
    SPLINE_AL_MAX: float = 4.0
    MARGIN: float = 0.6
    C_DRAG: float = 0.057
    V_RAMP_RATE: float = 1.2   # startup speed ramp (m/s^2): avoids a violent max-tilt launch
    ZVD: bool = False
    ZVD_DELAY: tuple = (24, 24, 14)


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
    rate_cmd_norm[2] = np.clip(rate_cmd_norm[2], -gains.YAW_WMAX, gains.YAW_WMAX)   # gentle yaw
    tilt = float(np.degrees(np.arccos(max(-1.0, min(1.0, R[2, 2])))))
    return rate_cmd_norm, thr, tilt, {"a": a, "w_des": w_des, "q_des": q_des, "thr": thr}


def attitude_command_tf(state, a2, z_sp, yaw_ref, plant, gains, s_cam, ymirror=False):
    """TRUE-frame attitude for STRAFING (camera not along travel). Port of vq_waypoint2's
    true-frame block: qfix attitude, s_cam yaw basis, WFIX rate mirror, tilt-conditional
    collective cap. a2 = desired horizontal accel in live-world (pos_ned); yaw_ref = desired
    NOSE heading in the true-cam frame. Returns (rate_cmd_norm, thrust, tilt_deg, dbg)."""
    hover, k_a, rg = plant
    a = np.zeros(3)
    a[:2] = np.asarray(a2, float)
    a[2] = gains.KP_Z * (z_sp - state.pos_ned[2]) + gains.KD_Z * (0.0 - state.vel_ned[2])
    tilt_max_acc = np.tan(np.radians(gains.TILT_MAX_DEG)) * G
    n = float(np.linalg.norm(a[:2]))
    if n > tilt_max_acc:
        a[:2] = a[:2] / n * tilt_max_acc
    q_t = _qfix(state.quat_wxyz)
    R_t = quat_to_R(q_t)
    tilt = float(np.degrees(np.arccos(max(-1.0, min(1.0, R_t[2, 2])))))
    yaw_cur = float(np.arctan2(s_cam * R_t[1, 0], s_cam * R_t[0, 0]))
    yaw_body = yaw_ref if s_cam > 0 else yaw_ref + np.pi
    if ymirror:
        a[1] = -a[1]
    q_des = mat_to_quat(desired_attitude(a, yaw_body))
    om_t = np.asarray(state.omega, float) * WFIX
    w = gains.KP_ATT * attitude_error_quat(q_t, q_des)
    w[0] = float(np.clip(w[0] - gains.KD_ATT * om_t[0], -2.0, 2.0))
    w[1] = float(np.clip(w[1] - gains.KD_ATT * om_t[1], -2.0, 2.0))
    w[2] = float(np.clip(gains.KP_YAW * ((yaw_body - yaw_cur + np.pi) % (2 * np.pi) - np.pi)
                         - gains.KD_YAW * om_t[2], -1.5, 1.5))
    w = w * WFIX
    c_max = 10.0 if tilt > 40.0 else 18.0
    thr = accel_to_thrust_norm(min(collective_accel(a, q_t), c_max), hover, k_a)
    rate = np.clip(w / rg, -gains.WMAX, gains.WMAX)
    return rate, thr, tilt, {"a": a, "w_des": w, "q_des": q_des, "thr": thr}


def _strafe_recompose(a2, cam_live, lat_course, fwd, s_lat):
    """Bridge a live-world horizontal accel (pos_ned) into the TRUE-frame heading chart that
    desired_attitude expects, for camera-decoupled STRAFE flight. Verbatim vq_waypoint2 L231-233:
    decompose a2 onto the FIXED spawn course axes (cam_live forward, lat_course perp), then re-emit
    those scalar intents along the drone's CURRENT true-frame heading basis (fwd, lat=[-fwd_y,fwd_x]),
    with the calibrated lateral sign s_lat. For nose-aligned fixed-heading flight a_h == a2."""
    a_al = float(np.asarray(a2, float) @ np.asarray(cam_live, float))
    a_lat = float(np.asarray(a2, float) @ np.asarray(lat_course, float)) * (s_lat if s_lat else 1.0)
    fwd = np.asarray(fwd, float)
    lat = np.array([-fwd[1], fwd[0]])
    return a_al * fwd + a_lat * lat


def _spline_accel(state, ref, gains):
    """traj_track's cross/along law to the spline reference. Returns (a2 horiz, travel unit)."""
    tang = np.asarray(ref["tang"], float)[:2]
    travel = tang / max(float(np.linalg.norm(tang)), 1e-9)
    lat_hat = np.array([-travel[1], travel[0]])
    d = (state.pos_ned - np.asarray(ref["pos"], float))[:2]
    v_al = float(state.vel_ned[:2] @ travel)
    v_ct = float(state.vel_ned[:2] @ lat_hat)
    p_ct = float(d @ lat_hat)
    tilt_max_acc = np.tan(np.radians(gains.TILT_MAX_DEG)) * G
    a_al = float(np.clip(gains.KD_AL * (ref["v"] - v_al), -4.0, gains.SPLINE_AL_MAX))
    a_ct_max = max(0.0, tilt_max_acc * gains.MARGIN - gains.C_DRAG * v_al * v_al)
    a_ct = float(np.clip(-gains.KP_CT * p_ct - gains.KD_CT * v_ct, -a_ct_max, a_ct_max))
    return a_al * travel + a_ct * lat_hat, travel


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
        self._s_cam = 1.0
        self._yaw0_t = 0.0
        self._cam_live = np.array([1.0, 0.0])    # FIXED spawn course-forward (world XY), set in set_origin
        self._lat_course = np.array([0.0, 1.0])  # FIXED spawn course-perp (world XY)
        self._s_lat = 0.0                         # strafe lateral sign (0 = unprobed); locked by _probe_s_lat
        self._tf_ymirror = True                   # world-y mirror in the true frame (proven by vq_waypoint2 --square)
        self._zvd = None

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

        # true-frame calibration (verbatim from vq_waypoint2): camera-forward sign + true-cam spawn yaw
        ds = self.store.get_drone()
        if ds is not None:
            yaw0 = self._origin_yaw
            q_t0 = _qfix(ds.quat_wxyz)
            R_t0 = quat_to_R(q_t0)
            cam_live = -np.array([np.cos(yaw0), np.sin(yaw0)])
            self._s_cam = 1.0 if float(R_t0[:2, 0] @ cam_live) > 0 else -1.0
            self._yaw0_t = float(np.arctan2(self._s_cam * R_t0[1, 0], self._s_cam * R_t0[0, 0]))
        else:
            self._s_cam = 1.0
            self._yaw0_t = float(self._origin_yaw or 0.0)

        # FIXED spawn course frame for strafe (camera-decoupled) modes (verbatim vq_waypoint2 L103/L110):
        # cam_live = the camera's world-XY direction at spawn; lat_course = its perpendicular.
        self._cam_live = -np.array([np.cos(self._origin_yaw), np.sin(self._origin_yaw)])
        self._lat_course = np.array([-self._cam_live[1], self._cam_live[0]])

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

    @staticmethod
    def _tan(a, b):
        """Unit horizontal direction a->b, or None if degenerate."""
        d = (np.asarray(b, float) - np.asarray(a, float))[:2]
        n = float(np.linalg.norm(d))
        return d / n if n > 1e-6 else None

    def _yaw_ref(self, yaw_mode, target, tangent=None):
        """Desired NOSE (body-x) heading. Camera is -body_x, so camera-pointing modes add pi.
        Modes: 'hold' (fixed origin heading, strafe), 'face' (nose at target),
        'lookat' (camera at self._look_point if set, else camera along the leg), 'course'
        (camera along travel). Travel-facing modes follow the FIXED leg `tangent` when given
        (no parallax/velocity-noise thrash); they fall back to live bearing/velocity only when
        no tangent is available, holding the current heading when that vector is degenerate."""
        ds = self.store.get_drone()
        pos = ds.pos_ned if ds is not None else (self._origin_pos
                                                 if self._origin_pos is not None else np.zeros(3))
        cur_yaw = (float(np.arctan2(*(quat_to_R(ds.quat_wxyz)[[1, 0], 0])))
                   if ds is not None else (self._origin_yaw or 0.0))
        if yaw_mode == "hold":
            return self._origin_yaw if self._origin_yaw is not None else 0.0
        if yaw_mode == "face":                       # point the NOSE at the target
            rel = np.asarray(target, float) - pos
            return cur_yaw if np.linalg.norm(rel[:2]) < 1.0 else float(np.arctan2(rel[1], rel[0]))
        if yaw_mode == "lookat" and self._look_point is not None:   # CAMERA on a fixed point
            rel = np.asarray(self._look_point, float) - pos
            return cur_yaw if np.linalg.norm(rel[:2]) < 1.0 else float(np.arctan2(rel[1], rel[0]) + np.pi)
        if yaw_mode in ("course", "lookat"):         # CAMERA (-body_x) along the leg / travel
            if tangent is not None:                  # fixed leg direction = thrash-free
                return float(np.arctan2(tangent[1], tangent[0]) + np.pi)
            v = ds.vel_ned[:2] if ds is not None else np.zeros(2)
            if float(np.linalg.norm(v)) > 1.5:       # moving: along velocity
                return float(np.arctan2(v[1], v[0]) + np.pi)
            rel = np.asarray(target, float) - pos    # slow + no tangent: aim at target
            return cur_yaw if np.linalg.norm(rel[:2]) < 1.0 else float(np.arctan2(rel[1], rel[0]) + np.pi)
        raise ValueError(f"yaw must be 'hold'|'face'|'lookat'|'course', got {yaw_mode!r}")

    def _current_pos(self):
        ds = self.store.get_drone()
        if ds is not None:
            return ds.pos_ned.copy()
        return self._origin_pos.copy() if self._origin_pos is not None else np.zeros(3)

    # --- blocking flight loop (live sim; not unit-tested) ---
    def goto(self, wp, *, yaw="course", look_point=None, v_cruise=2.5,
             engine="spline", frame="body"):
        if self._origin_pos is None:
            self.set_origin()
        if self._t0 is None:
            self._t0 = time.time()
        if look_point is not None:
            self._look_point = self._resolve(look_point, frame)
        target = self._resolve(wp, frame)
        if self._is_strafe(yaw):       # camera-decoupled strafe must use the gentle legs engine
            engine = "legs"            # (the spline path rings the rate loop)
        if engine == "spline":
            return self._follow_spline([target], yaw=yaw, v_cruise=v_cruise)
        # legs engine
        self._zvd = ZVDShaper(self.gains.ZVD_DELAY) if self.gains.ZVD else None
        if self._is_strafe(yaw):
            self._probe_s_lat(float(np.asarray(target, float)[2]))
        self._align_yaw(target, yaw, self._origin_pos)   # turn-in-place (no-op for strafe)
        return self._fly_leg(target, self._origin_pos, yaw)

    def follow(self, wps, *, yaw="course", look_point=None, v_cruise=2.5,
               engine="spline", frame="body"):
        if self._origin_pos is None:
            self.set_origin()
        if self._t0 is None:
            self._t0 = time.time()
        if look_point is not None:
            self._look_point = self._resolve(look_point, frame)
        targets = [self._resolve(w, frame) for w in wps]
        if self._is_strafe(yaw):       # camera-decoupled strafe -> gentle legs engine (spline rings)
            engine = "legs"
        if engine == "legs":
            return self._follow_legs(targets, yaw=yaw, settle=True)
        return self._follow_spline(targets, yaw=yaw, v_cruise=v_cruise)

    def _probe_s_lat(self, z_sp):
        """Lock the strafe lateral sign before flying camera-decoupled legs: command a fixed lateral
        push (~1.5 s) along the FIXED course-perp and read the achieved world-lateral velocity sign.
        Port of vq_waypoint2's s_lat probe. No-op once locked (self._s_lat != 0)."""
        if self._s_lat != 0.0:
            return
        ds0 = self.store.get_drone()
        if ds0 is None:
            self._s_lat = 1.0
            return
        v0 = float(ds0.vel_ned[:2] @ self._lat_course)
        t0 = time.time()
        while time.time() - t0 < 1.5:
            ds = self.store.get_drone()
            if ds is not None:
                rate, thr, tilt, dbg = self._strafe_attitude(ds, 1.2 * self._lat_course, z_sp,
                                                             self._yaw0_t)
                if self._zvd is not None:
                    rate = self._zvd.shape(rate)
                self.commander.send_attitude_target(rate, thr)
                if self.flog is not None:
                    self.flog.push(time.time() - self._t0, ds, dbg, cruise=0.0,
                                   running=self.store.get_race_live(), armed=True)
                if tilt > self.gains.ABORT_TILT_DEG:
                    break
            time.sleep(self.gains.LOOP_DT)
        ds = self.store.get_drone()
        v1 = float(ds.vel_ned[:2] @ self._lat_course) if ds is not None else v0
        self._s_lat = 1.0 if (v1 - v0) > 0 else -1.0
        print(f"  s_lat locked: {self._s_lat:+.0f} (dv {v1 - v0:+.2f})", flush=True)

    def _follow_legs(self, targets, *, yaw, settle=True):
        """Original legs engine: aligned yaw + point-to-point fly + settle at each wp. For strafe
        (camera-decoupled) modes, legs run in the TRUE frame (no turn-in-place; s_lat probed once)."""
        if self.flog is not None:
            self.flog.set_path(np.vstack([self._current_pos()] + targets))
        self._zvd = ZVDShaper(self.gains.ZVD_DELAY) if self.gains.ZVD else None
        if self._is_strafe(yaw):
            self._probe_s_lat(float(np.asarray(targets[0], float)[2]))   # lock lateral sign first
        leg_start = self._origin_pos.copy()
        for i, target in enumerate(targets):
            leg_tan = self._tan(leg_start, target)    # fixed leg direction (camera holds it)
            self._align_yaw(target, yaw, leg_start)   # turn-in-place to face the leg (nose-first)
            res = self._fly_leg(target, leg_start, yaw)
            print(f"   {res.upper()} wp{i}", flush=True)
            if res != "reached":
                return res
            if settle:
                self.settle(target, yaw, tangent=leg_tan)
            leg_start = self._current_pos()
        return "reached"

    def _follow_spline(self, targets, *, yaw, v_cruise):
        g = self.gains
        self._zvd = ZVDShaper(g.ZVD_DELAY) if g.ZVD else None
        gates = np.vstack([self._origin_pos] + [np.asarray(t, float) for t in targets])
        traj = GateTrajectory(gates, v_cruise=v_cruise, tilt_budget_deg=25.0,
                              c_drag=g.C_DRAG, margin=g.MARGIN)
        if self.flog is not None:
            self.flog.set_path(traj._P)

        # pre-stabilize: damp the spawn tilt + warm the ZVD buffer before tracking, else the
        # launch step rings the underdamped rate loop (tilt 18 -> 40 -> abort).
        t_pre = time.time()
        while time.time() - t_pre < 3.0:
            ds = self.store.get_drone()
            if ds is not None:
                R = quat_to_R(ds.quat_wxyz)
                yaw_hold = float(np.arctan2(R[1, 0], R[0, 0]))
                a2 = -0.6 * ds.vel_ned[:2]                       # damp horizontal velocity, stay level
                rate, thr, tilt, dbg = attitude_command(ds, a2, float(ds.pos_ned[2]),
                                                         yaw_hold, self.plant, g)
                if self._zvd is not None:
                    rate = self._zvd.shape(rate)
                self.commander.send_attitude_target(rate, thr)
                if self.flog is not None:
                    self.flog.push(time.time() - self._t0, ds, dbg, cruise=0.0,
                                   running=self.store.get_race_live(), armed=True)
                if tilt > g.ABORT_TILT_DEG:
                    print(f"  ABORT prestab tilt={tilt:.0f}", flush=True)
                    return "abort"
                if tilt < 5.0 and float(np.linalg.norm(ds.vel_ned[:2])) < 0.5:
                    print(f"  prestab done tilt={tilt:.0f}", flush=True)
                    break
            time.sleep(g.LOOP_DT)

        t_run = time.time()
        last = -1
        s_ref = 0.0
        nloop = 0
        while time.time() - t_run < g.WP_TIMEOUT * max(2, len(targets)):
            ds = self.store.get_drone()
            if ds is not None:
                nloop += 1
                lo = int(np.searchsorted(traj._s, s_ref - 2.0))
                hi = int(np.searchsorted(traj._s, s_ref + 5.0))
                hi = max(hi, lo + 1)
                seg = traj._P[lo:hi] - ds.pos_ned
                s_ref = float(traj._s[lo + int(np.argmin(np.einsum("ij,ij->i", seg, seg)))])
                ref = traj.sample(s_ref)
                ref = dict(ref)
                ref["v"] = min(ref["v"], g.V_RAMP_RATE * (time.time() - t_run))
                a2, travel = _spline_accel(ds, ref, g)
                if yaw == "course":
                    rate, thr, tilt, dbg = attitude_command(ds, a2, float(ref["pos"][2]),
                                                            ref["yaw"], self.plant, g)
                else:
                    yref = self._yaw_ref_tf(yaw, ds)
                    rate, thr, tilt, dbg = attitude_command_tf(ds, a2, float(ref["pos"][2]),
                                                               yref, self.plant, g, self._s_cam,
                                                               ymirror=self._tf_ymirror)
                if self._zvd is not None:
                    rate = self._zvd.shape(rate)
                self.commander.send_attitude_target(rate, thr)
                if self.flog is not None:
                    self.flog.push(time.time() - self._t0, ds, dbg, nearest=ref["pos"],
                                   tangent=travel, cruise=ref["v"],
                                   running=self.store.get_race_live(), armed=True)
                if tilt > g.ABORT_TILT_DEG:
                    print(f"  ABORT tilt={tilt:.0f}", flush=True)
                    return "abort"
                if s_ref >= traj.s_max - g.ARRIVE:
                    print("  REACHED end", flush=True)
                    return "reached"
                k = int((time.time() - t_run) / 1.0)
                if k != last:
                    last = k
                    print(f"  s={s_ref:5.1f}/{traj.s_max:.0f} v={float(np.linalg.norm(ds.vel_ned[:2])):4.1f}"
                          f"/{ref['v']:.1f} tilt={tilt:3.0f}"
                          f" hz={nloop / max(time.time() - t_run, 1e-6):.0f}", flush=True)
            time.sleep(g.LOOP_DT)
        return "timeout"

    def _yaw_ref_tf(self, yaw, ds):
        """NOSE heading in the true-cam frame for strafe (camera-decoupled) modes. course/face are
        handled in the raw nose-first path. 'hold'/'fixed' hold the spawn true-cam heading."""
        if yaw in ("hold", "fixed"):
            return self._yaw0_t
        raise NotImplementedError(yaw)   # 'lookat' (camera about a point) is a follow-up

    @staticmethod
    def _is_strafe(yaw_mode):
        """Camera-DECOUPLED modes that fly sideways (heading independent of travel) -> true frame.
        'hold'/'fixed' = fixed heading. Nose-first modes (course/face) use the raw path."""
        return yaw_mode in ("hold", "fixed")

    def _strafe_attitude(self, ds, a2, z_sp, yaw_ref_tf):
        """True-frame attitude for a strafe step: bridge the world accel a2 into the true-heading
        chart (_strafe_recompose with the locked s_lat), then run the proven vq_waypoint2 attitude
        block (attitude_command_tf, ymirror). Returns (rate_cmd_norm, thrust, tilt_deg, dbg)."""
        R_t = quat_to_R(_qfix(ds.quat_wxyz))
        yaw_cur_t = float(np.arctan2(self._s_cam * R_t[1, 0], self._s_cam * R_t[0, 0]))
        fwd = np.array([np.cos(yaw_cur_t), np.sin(yaw_cur_t)])
        a_h = _strafe_recompose(a2, self._cam_live, self._lat_course, fwd, self._s_lat)
        return attitude_command_tf(ds, a_h, z_sp, yaw_ref_tf, self.plant, self.gains,
                                   self._s_cam, ymirror=self._tf_ymirror)

    def settle(self, target=None, yaw="hold", tangent=None):
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
                if self._is_strafe(yaw):
                    rate, thr, tilt, dbg = self._strafe_attitude(ds, a2, float(target[2]),
                                                                 self._yaw_ref_tf(yaw, ds))
                else:
                    rate, thr, tilt, dbg = attitude_command(ds, a2, float(target[2]),
                                                            self._yaw_ref(yaw, target, tangent),
                                                            self.plant, self.gains)
                if self._zvd is not None:
                    rate = self._zvd.shape(rate)
                self.commander.send_attitude_target(rate, thr)
                if self.flog is not None:
                    self.flog.push(time.time() - self._t0, ds, dbg, nearest=target, cruise=0.0,
                                   running=self.store.get_race_live(), armed=True)
                if tilt > self.gains.ABORT_TILT_DEG:
                    print(f"  ABORT settle tilt={tilt:.0f}", flush=True)
                    return
            time.sleep(self.gains.LOOP_DT)

    def _align_yaw(self, target, yaw_mode, pos_hold, tol_deg=15.0, t_max=4.0):
        """Travel-facing modes ('face'/'course'/'lookat'-at-target): rotate IN PLACE at pos_hold
        until the heading faces the leg direction, BEFORE translating — so the drone flies each
        leg nose-first (no strafe = no weathervane runaway). No-op for 'hold' and fixed-point
        'lookat' (those intentionally hold/strafe). Station-keeps at pos_hold while turning."""
        if self._is_strafe(yaw_mode):     # strafe modes hold heading -> never turn-in-place
            return
        if yaw_mode == "lookat" and self._look_point is not None:
            return
        pos_hold = np.asarray(pos_hold, float)
        tangent = self._tan(pos_hold, target)   # fixed leg direction to align to
        t0 = time.time()
        while time.time() - t0 < t_max:
            ds = self.store.get_drone()
            if ds is not None:
                yaw_ref = self._yaw_ref(yaw_mode, target, tangent)
                R = quat_to_R(ds.quat_wxyz)
                yaw_cur = float(np.arctan2(R[1, 0], R[0, 0]))
                err = (yaw_ref - yaw_cur + np.pi) % (2 * np.pi) - np.pi
                # turn LEVEL (no position-fighting tilt) so the rotation can't excite the
                # weathervane; only damp residual horizontal velocity. The leg's cross-track
                # then corrects any small drift once we translate.
                a2 = -0.6 * ds.vel_ned[:2]
                rate, thr, tilt, dbg = attitude_command(ds, a2, float(pos_hold[2]),
                                                         yaw_ref, self.plant, self.gains)
                self.commander.send_attitude_target(rate, thr)
                if self.flog is not None:
                    self.flog.push(time.time() - self._t0, ds, dbg, nearest=pos_hold, cruise=0.0,
                                   running=self.store.get_race_live(), armed=True)
                if tilt > self.gains.ABORT_TILT_DEG:
                    print(f"  ABORT align tilt={tilt:.0f}", flush=True)
                    return
                if abs(err) < np.radians(tol_deg) and float(np.linalg.norm(ds.vel_ned[:2])) < 0.6:
                    return
            time.sleep(self.gains.LOOP_DT)

    def _fly_leg(self, target, leg_start, yaw_mode):
        g = self.gains
        tangent = self._tan(leg_start, target)   # fixed leg direction for travel-facing yaw
        t_leg = time.time()
        last = -1
        while time.time() - t_leg < g.WP_TIMEOUT:
            ds = self.store.get_drone()
            if ds is not None:
                rel = target - ds.pos_ned
                if float(np.linalg.norm(rel)) < g.ARRIVE:
                    return "reached"
                a2, tv, spd = cruise_accel(ds, leg_start, target, g)
                if self._is_strafe(yaw_mode):     # camera-decoupled: strafe in the true frame
                    rate, thr, tilt, dbg = self._strafe_attitude(ds, a2, float(target[2]),
                                                                 self._yaw_ref_tf(yaw_mode, ds))
                else:
                    rate, thr, tilt, dbg = attitude_command(ds, a2, float(target[2]),
                                                            self._yaw_ref(yaw_mode, target, tangent),
                                                            self.plant, g)
                if self._zvd is not None:
                    rate = self._zvd.shape(rate)
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
