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
    # camera-decoupled strafe (fixed course-frame guidance; vq_waypoint2 mission-1 values)
    KV_AL: float = 0.7      # along-track position->speed gain
    KV_LAT: float = 0.9     # lateral position->speed gain
    KD_LAT: float = 1.4     # lateral speed->accel gain
    VLAT_MAX: float = 1.5   # lateral speed cap (m/s)
    FWD_AMAX: float = 0.6   # forward accel cap (m/s^2) -> terminal ~3 m/s vs drag
    REV_SP: float = 0.8     # max reverse along-track speed (m/s)
    BRAKE_MARGIN: float = 0.85   # stop-profile: brake as if slightly under DECEL (covers vel-est lag -> no overshoot)
    KP_Z: float = 1.8
    KD_Z: float = 3.0
    KI_Z: float = 0.8       # gated z-integral gain: cancels the ~1.2 m analytic-z sag for 3D waypoints
    Z_INT_GATE: float = 1.5  # only integrate within this altitude error (no climb-transient windup)
    Z_INT_CLIP: float = 3.0  # z-integral accel clamp (m/s^2)
    Z_FF: float = -2.0      # constant collective feed-forward (m/s^2, up) = the VQ steady deficit;
    #                         instant so the integral doesn't lag -> no transient high-tilt sag
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
    CAPTURE: float = 2.5    # strafe flow: advance to the next wp at this radius (no stop -> smooth corners)
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


def attitude_command_tf(state, a2, z_sp, yaw_ref, plant, gains, s_cam, ymirror=False, z_int=0.0,
                        z_ff=0.0):
    """TRUE-frame attitude for STRAFING (camera not along travel). Port of vq_waypoint2's
    true-frame block: qfix attitude, s_cam yaw basis, WFIX rate mirror, tilt-conditional
    collective cap. a2 = desired horizontal accel in live-world (pos_ned); yaw_ref = desired
    NOSE heading in the true-cam frame. z_int = gated z-integral accel; z_ff = constant collective
    feed-forward (the VQ steady deficit, negative = up) so the integral need not chase it. Returns
    (rate_cmd_norm, thrust, tilt_deg, dbg)."""
    hover, k_a, rg = plant
    a = np.zeros(3)
    a[:2] = np.asarray(a2, float)
    a[2] = (gains.KP_Z * (z_sp - state.pos_ned[2]) + gains.KD_Z * (0.0 - state.vel_ned[2])
            + z_int + z_ff)
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


def _course_guidance(pos, vel, target, cam_live, lat_course, gains):
    """Forward + lateral accel INTENTS in the FIXED spawn course frame (cam_live forward,
    lat_course perp), for camera-decoupled strafe legs. Port of vq_waypoint2 mission-1
    (L176-202/230): every leg decomposes against the SAME course frame, so a single s_lat is
    valid for all legs (a per-leg tangent decomposition flips the cross-track sign between legs
    -> the bug that made forward legs want +1 and strafe legs want -1). Returns (a_al, a_lat);
    a_lat is UNSIGNED (the calibrated s_lat is applied at recompose)."""
    err = (np.asarray(target, float) - np.asarray(pos, float))[:2]
    v = np.asarray(vel, float)[:2]
    e_al = float(err @ cam_live); e_lat = float(err @ lat_course)
    v_al = float(v @ cam_live); v_lat = float(v @ lat_course)
    tilt_max_acc = np.tan(np.radians(gains.TILT_MAX_DEG)) * G
    v_al_sp = float(np.clip(gains.KV_AL * e_al, -gains.REV_SP, gains.MAX_SPEED))
    a_al = float(np.clip(gains.KD_AL * (v_al_sp - v_al), -gains.DECEL_MAX, gains.FWD_AMAX))
    v_lat_sp = float(np.clip(gains.KV_LAT * e_lat, -gains.VLAT_MAX, gains.VLAT_MAX))
    a_lat = float(np.clip(gains.KD_LAT * (v_lat_sp - v_lat), -tilt_max_acc, tilt_max_acc))
    return a_al, a_lat


def _dr_vel(vw_prev, pos_xy, pos_prev_xy, dt, alpha=0.85):
    """EMA-filtered world velocity dead-reckoned from position deltas. vq_waypoint2 closes the
    strafe loop on THIS (not ds.vel_ned) because the live-world velocity's lateral sign convention
    fights the push chain; position-derived vel is in the SAME frame as the position error, so the
    lateral damping has the correct sign. Returns the filtered (vx, vy)."""
    if dt <= 1e-4:
        return np.asarray(vw_prev, float)
    vw_raw = (np.asarray(pos_xy, float) - np.asarray(pos_prev_xy, float)) / dt
    return alpha * np.asarray(vw_prev, float) + (1.0 - alpha) * vw_raw


def _z_int_step(z_int, ze, dt, ki, gate, clip):
    """One gated z-integrator step (clamped to +-clip). Cancels the VQ analytic z-loop's ~1.2 m sag
    (collective deficit, worse at higher tilt) so altitude-changing waypoints reach z. Anti-windup:
    charge (grow |z_int|) ONLY within |ze| < gate (no windup during the big climb/descent transient),
    but always allow UNWINDING (a step that shrinks |z_int|) even when far -- else a value charged
    on a climb can't bleed off on the next descent and fights it (drone stuck off-altitude)."""
    delta = ki * ze * dt
    unwinding = (z_int * delta < 0.0)   # delta opposes the current bias -> moving toward zero
    if abs(ze) < gate or unwinding:
        z_int = float(np.clip(z_int + delta, -clip, clip))
    return z_int


def _line_guidance(pos, vw, leg_start, target, cam_live, lat_course, gains, brake_to_stop=False):
    """Straight-LINE segment following (not point-seeking) for crisp legs. Tracks the segment
    leg_start->target: along-track speed control + a STIFF cross-track term that holds the drone ON
    the line (the property point-seeking lacks on diagonal legs -> bowed paths). Computes the world
    accel in the leg's along/cross basis, then projects onto the FIXED course axes (cam_live,
    lat_course) for the true-frame recompose (so s_lat stays consistent). `vw` = position-derived
    world velocity (frame-consistent with the position error). brake_to_stop=True uses a constant-
    deceleration STOP profile (v_sp = sqrt(2*DECEL_MAX*e_along)) -> carry full speed, then brake hard
    to a crisp halt at the target (stop-on-a-dime; no asymptotic creep). Returns (a_al, a_lat)."""
    pos = np.asarray(pos, float)[:2]; vw = np.asarray(vw, float)[:2]
    leg_start = np.asarray(leg_start, float)[:2]; target = np.asarray(target, float)[:2]
    seg = target - leg_start
    seglen = float(np.linalg.norm(seg))
    if seglen > 1e-6:
        t_hat = seg / seglen
    else:
        rel = target - pos
        t_hat = rel / (float(np.linalg.norm(rel)) + 1e-9)
    n_hat = np.array([-t_hat[1], t_hat[0]])
    e_along = float((target - pos) @ t_hat)
    e_cross = float((pos - leg_start) @ n_hat)
    v_along = float(vw @ t_hat); v_cross = float(vw @ n_hat)
    tilt_max_acc = np.tan(np.radians(gains.TILT_MAX_DEG)) * G
    if brake_to_stop:
        v_cap = float(np.sqrt(2.0 * gains.DECEL_MAX * gains.BRAKE_MARGIN * abs(e_along)))   # stoppable speed
        v_along_sp = float(np.clip(np.sign(e_along) * min(gains.MAX_SPEED, v_cap),
                                   -gains.REV_SP, gains.MAX_SPEED))
    else:
        v_along_sp = float(np.clip(gains.KV_AL * e_along, -gains.REV_SP, gains.MAX_SPEED))
    a_along = float(np.clip(gains.KD_AL * (v_along_sp - v_along), -gains.DECEL_MAX, gains.FWD_AMAX))
    a_cross = float(np.clip(-gains.KP_CT * e_cross - gains.KD_CT * v_cross, -tilt_max_acc, tilt_max_acc))
    a_world = a_along * t_hat + a_cross * n_hat
    return float(a_world @ np.asarray(cam_live, float)), float(a_world @ np.asarray(lat_course, float))


def _strafe_recompose(a_al, a_lat, fwd, s_lat):
    """Emit course-frame intents (a_al forward, a_lat course-perp) along the drone's CURRENT
    true-heading basis (fwd, lat=[-fwd_y,fwd_x]) with the calibrated lateral sign s_lat. Verbatim
    vq_waypoint2 L231-233: a[:2] = a_al*fwd + (s_lat*a_lat)*lat."""
    fwd = np.asarray(fwd, float)
    lat = np.array([-fwd[1], fwd[0]])
    return a_al * fwd + (s_lat if s_lat else 1.0) * a_lat * lat


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
        self._s_lat = 0.0                         # strafe lateral sign (0 = unprobed); locked by _probe_inflight
        self._tf_ymirror = True                   # world-y mirror in the true frame (proven by vq_waypoint2 --square)
        self._zvd = None
        # dead-reckoned world velocity for the strafe loop (frame-consistent with position error)
        self._vw = np.zeros(2)
        self._pos_prev_vw = None
        self._t_prev_vw = None
        self._z_int = 0.0          # gated z-integral accel (m/s^2) for 3D waypoints
        self._t_prev_zi = None
        self._stop_each = False    # True: stop-on-a-dime at EVERY waypoint (else only the final)

    def _world_vel(self, pos):
        """Stateful position-derived world velocity (see _dr_vel). Call once per loop step with the
        live pos_ned; returns the filtered (vx, vy). Only advances on a FRESH odometry sample (pos
        actually changed) -- the control loop runs ~250 Hz but odometry updates slower, so
        differentiating every loop injects a zero-velocity sawtooth (tilt jitter)."""
        now = time.time()
        pos = np.asarray(pos, float)
        if self._pos_prev_vw is None:
            self._pos_prev_vw = pos.copy(); self._t_prev_vw = now; self._vw = np.zeros(2)
            return self._vw
        if np.allclose(pos[:2], self._pos_prev_vw[:2]):    # no new odometry sample -> hold estimate
            return self._vw
        dt = min(now - self._t_prev_vw, 0.05); self._t_prev_vw = now
        self._vw = _dr_vel(self._vw, pos[:2], self._pos_prev_vw[:2], dt)
        self._pos_prev_vw = pos.copy()
        return self._vw

    @staticmethod
    def _advance_wp(rel_norm, is_last, gains):
        """Strafe flow: advance to the next waypoint at the loose CAPTURE radius (still moving ->
        rounded corners); the FINAL waypoint completes only at the tight ARRIVE radius."""
        return float(rel_norm) < (gains.ARRIVE if is_last else gains.CAPTURE)

    def _reset_world_vel(self):
        self._vw = np.zeros(2); self._pos_prev_vw = None; self._t_prev_vw = None

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
            return self._fly_strafe_course([target], yaw)   # line-following (probes s_lat in-flight)
        self._align_yaw(target, yaw, self._origin_pos)   # turn-in-place (nose-first)
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

    def _probe_inflight(self, target, leg_start, yaw):
        """Lock the strafe lateral sign WHILE flying the first leg (vq_waypoint2-style): drive forward
        toward the first waypoint via line-following + superimpose a brief FIXED lateral nudge, then
        read the achieved world-lateral velocity sign. No separate stationary sideways push -> no
        start hook. No-op once locked (self._s_lat != 0)."""
        if self._s_lat != 0.0:
            return
        ds0 = self.store.get_drone()
        if ds0 is None:
            self._s_lat = 1.0
            return
        self._reset_world_vel()
        yref = self._yaw_ref_tf(yaw, ds0)
        z_sp = float(np.asarray(target, float)[2])
        v0 = float(self._world_vel(ds0.pos_ned) @ self._lat_course)
        t0 = time.time(); vw = self._vw
        while time.time() - t0 < 1.5:
            ds = self.store.get_drone()
            if ds is not None:
                vw = self._world_vel(ds.pos_ned)
                a_al, _ = _line_guidance(ds.pos_ned, vw, leg_start, target,
                                         self._cam_live, self._lat_course, self.gains)
                # forward toward wp0 + a FIXED lateral nudge (s_lat still 0 -> recompose uses +1 = raw push)
                rate, thr, tilt, dbg = self._strafe_attitude(ds, a_al, 1.2, z_sp, yref)
                if self._zvd is not None:
                    rate = self._zvd.shape(rate)
                self.commander.send_attitude_target(rate, thr)
                if self.flog is not None:
                    self.flog.push(time.time() - self._t0, ds, dbg, nearest=target,
                                   tangent=self._cam_live, cruise=float(np.linalg.norm(vw)),
                                   running=self.store.get_race_live(), armed=True)
                if tilt > self.gains.ABORT_TILT_DEG:
                    break
            time.sleep(self.gains.LOOP_DT)
        v1 = float(vw @ self._lat_course)
        self._s_lat = 1.0 if (v1 - v0) > 0 else -1.0
        print(f"  s_lat locked: {self._s_lat:+.0f} (dv {v1 - v0:+.2f})", flush=True)

    def _follow_legs(self, targets, *, yaw, settle=True):
        """Original legs engine: aligned yaw + point-to-point fly + settle at each wp. For strafe
        (camera-decoupled) modes, legs run in the TRUE frame (no turn-in-place; s_lat probed once)."""
        if self.flog is not None:
            self.flog.set_path(np.vstack([self._current_pos()] + targets))
        self._zvd = ZVDShaper(self.gains.ZVD_DELAY) if self.gains.ZVD else None
        if self._is_strafe(yaw):
            return self._fly_strafe_course(targets, yaw)   # continuous flow (probes s_lat in-flight)
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

    def _fly_strafe_course(self, targets, yaw):
        """Continuous camera-decoupled flight through all waypoints: ONE loop, no per-corner stop.
        Heading is fixed (no turn-in-place), so the drone flows through corners -- advance to the
        next wp at the loose CAPTURE radius while still moving (rounded corners, steady speed = no
        stop-and-go lurch); only the final wp uses the tight ARRIVE + a settle. Guidance/attitude
        reuse the proven fixed-course-frame strafe (_course_guidance + _strafe_attitude)."""
        g = self.gains
        n = len(targets)
        i = 0
        leg_start = self._origin_pos.copy()   # straight-line segment is leg_start -> targets[i]
        self._z_int = 0.0; self._t_prev_zi = None   # fresh z-integral for this course
        self._probe_inflight(np.asarray(targets[0], float), leg_start, yaw)   # lock s_lat on the way to wp0
        t0 = time.time()
        last_log = -1
        while i < n and time.time() - t0 < g.WP_TIMEOUT * n:
            ds = self.store.get_drone()
            if ds is not None:
                target = np.asarray(targets[i], float)
                is_last = (i == n - 1)
                stop_here = is_last or self._stop_each   # crisp halt at the final wp (or every wp in --stop)
                rel = target - ds.pos_ned
                if self._advance_wp(float(np.linalg.norm(rel)), stop_here, g):
                    print(f"   REACHED wp{i}", flush=True)
                    if stop_here and not is_last:
                        self.settle(target, yaw)         # full halt before the next leg
                    leg_start = target.copy()   # next leg starts at the waypoint just reached
                    i += 1
                    continue
                vw = self._world_vel(ds.pos_ned)
                a_al, a_lat = _line_guidance(ds.pos_ned, vw, leg_start, target,
                                             self._cam_live, self._lat_course, g,
                                             brake_to_stop=stop_here)
                rate, thr, tilt, dbg = self._strafe_attitude(ds, a_al, a_lat, float(target[2]),
                                                             self._yaw_ref_tf(yaw, ds))
                if self._zvd is not None:
                    rate = self._zvd.shape(rate)
                self.commander.send_attitude_target(rate, thr)
                if self.flog is not None:
                    self.flog.push(time.time() - self._t0, ds, dbg, nearest=target,
                                   tangent=self._cam_live, cruise=float(np.linalg.norm(vw)),
                                   running=self.store.get_race_live(), armed=True)
                if tilt > g.ABORT_TILT_DEG:
                    print(f"  ABORT tilt={tilt:.0f}", flush=True)
                    return "abort"
                k = int((time.time() - t0) / 1.0)
                if k != last_log:
                    last_log = k
                    print(f"  wp{i} dist={float(np.linalg.norm(rel)):4.1f} "
                          f"dxy={float(np.linalg.norm(rel[:2])):4.1f} "
                          f"z={float(ds.pos_ned[2]):+4.1f}/{float(target[2]):+4.1f} "
                          f"zi={self._z_int:+4.1f} thr={dbg['thr']:.2f} "
                          f"v={float(np.linalg.norm(vw)):4.1f} tilt={tilt:3.0f}", flush=True)
            time.sleep(g.LOOP_DT)
        if i >= n:
            self.settle(np.asarray(targets[-1], float), yaw)   # park at the final wp
            return "reached"
        return "timeout"

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

    def _strafe_attitude(self, ds, a_al, a_lat, z_sp, yaw_ref_tf):
        """True-frame attitude for a strafe step: emit the course-frame intents (a_al, a_lat) along
        the drone's current true-heading basis (_strafe_recompose with the locked s_lat), then run
        the proven vq_waypoint2 attitude block (attitude_command_tf, ymirror). Also advances the
        gated z-integral (3D-waypoint altitude hold)."""
        g = self.gains
        now = time.time()
        ze = float(z_sp) - float(ds.pos_ned[2])
        if self._t_prev_zi is not None:
            dt = min(now - self._t_prev_zi, 0.05)
            self._z_int = _z_int_step(self._z_int, ze, dt, g.KI_Z, g.Z_INT_GATE, g.Z_INT_CLIP)
        self._t_prev_zi = now
        R_t = quat_to_R(_qfix(ds.quat_wxyz))
        yaw_cur_t = float(np.arctan2(self._s_cam * R_t[1, 0], self._s_cam * R_t[0, 0]))
        fwd = np.array([np.cos(yaw_cur_t), np.sin(yaw_cur_t)])
        a_h = _strafe_recompose(a_al, a_lat, fwd, self._s_lat)
        return attitude_command_tf(ds, a_h, z_sp, yaw_ref_tf, self.plant, g,
                                   self._s_cam, ymirror=self._tf_ymirror, z_int=self._z_int,
                                   z_ff=g.Z_FF)

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
                if self._is_strafe(yaw):
                    vw = self._world_vel(ds.pos_ned)
                    a_al, a_lat = _course_guidance(ds.pos_ned, np.array([vw[0], vw[1], 0.0]), target,
                                                   self._cam_live, self._lat_course, self.gains)
                    rate, thr, tilt, dbg = self._strafe_attitude(ds, a_al, a_lat, float(target[2]),
                                                                 self._yaw_ref_tf(yaw, ds))
                else:
                    a2 = settle_accel(ds, target, self.gains)
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
                if self._is_strafe(yaw_mode):     # camera-decoupled: fixed-course-frame strafe
                    vw = self._world_vel(ds.pos_ned)
                    a_al, a_lat = _course_guidance(ds.pos_ned, np.array([vw[0], vw[1], 0.0]), target,
                                                   self._cam_live, self._lat_course, g)
                    rate, thr, tilt, dbg = self._strafe_attitude(ds, a_al, a_lat, float(target[2]),
                                                                 self._yaw_ref_tf(yaw_mode, ds))
                    tv = self._cam_live; spd = float(np.linalg.norm(vw))
                else:
                    a2, tv, spd = cruise_accel(ds, leg_start, target, g)
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
