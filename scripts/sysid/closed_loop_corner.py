"""Closed-loop matched-sim sanity (COR-127, MATCHED-MIMO-FIX follow-up).

Runs a faithful port of the laptop `corner_speed.py` control law (EXP-28 3-term
coordinated turn + WV-DYNAMIC weathervane-FF + tilt cap + optional -KD*omega damping)
against `VQMatchedDynamics` (MIMO rate loop from sysid/vq_model.json) and checks the
sim reproduces the LIVE corner behavior:

  live kd=0.0 @ R10 v6 : entry speed-pump to ~26.6 m/s, cmd/actual tilt 12/29, tumble
  live kd=0.3 @ R10 v6 : pump halved ~12.8 m/s, tilt tracks (17/16), ~139 deg turned,
                         residual beta-leak loop still tumbles ~t22

Frame facts honored (canonical note / HANDOFF + data-verified 06-09 on corner runs):
  - the PLANT (vq_matched / fits) lives in the qfix/wfix frame, which IS the physical
    one: quat-derivative == wfix*omega (slopes +1.00/-1.00/+1.00) and world vel ==
    R(qfix)@vel (FRAME-FIX). The LIVE io_layer frame the controller consumes is the
    stored representation: q_live = q_true[[3,0,1,2]] (inverse qfix shuffle, NOT a
    rotation -- yaw-dependent), om_live = wfix*om_true = [w0,-w1,w2], vel = body frame.
  - live spawn true-attitude is yaw~180 deg (maps to live-frame ~identity), so the sim
    spawns at q_true=[0,0,0,1] to put the controller at its live operating point.
  - rate cmd path: controller sends wcmd = w_des / RG (sim_response gains) RAW; the
    plant applies the identified rate loop (MIMO A,B fit on raw wcmd) -- no transform.

Usage: python scripts/sysid/closed_loop_corner.py [R] [CRUISE] [--kd X] [--no-wvff] [--diag]
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sim.dynamics.vq_matched import VQMatchedDynamics, POS, VEL, QUAT, OMEGA  # noqa: E402

G = 9.81
DT = 1.0 / 72.0

# --- gains/constants verbatim from laptop corner_speed.py (EXP-28 / CORNER-01) ---
ROLL_WV0, ROLL_WV1, YAW_WV = -0.105, -0.019, -0.149
WVFF_CLIP = 1.0
KP_ATT = np.array([0.5, 1.6, 1.0]); KP_YAW = 3.0; KD_YAW = 0.3; KP_Z = 1.8; KD_Z = 3.0
KV = 1.5; ACCEL_MAX = 0.6; DECEL_MAX = 2.0
WMAX = 4.0; ABORT_TILT = 80.0; T_ENTRY = 5.0; MAX_T = 60.0
RAMP = 4.0
MAX_BANK_DEG = 70.0; TILT_MAX_ACC = np.tan(np.radians(MAX_BANK_DEG)) * G
TD2 = 0.055; _K = np.exp(-0.14 * np.pi / np.sqrt(1 - 0.14 ** 2)); _D = 1 + 2 * _K + _K * _K
ZVD_A = [1 / _D, 2 * _K / _D, _K * _K / _D]; ZVD_T = [0.0, TD2, 2 * TD2]
YR_CAP = 1.5


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_mul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def mat_to_quat(R):
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0); w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s; y = (R[0, 2] - R[2, 0]) * s; z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def attitude_error_quat(q_cur, q_des):
    q_err = quat_mul(quat_conj(np.asarray(q_cur, float)), np.asarray(q_des, float))
    if q_err[0] < 0:
        q_err = -q_err
    return 2.0 * q_err[1:4]


def desired_attitude(a_des, yaw_sp):
    t_vec = np.asarray(a_des, float) - np.array([0.0, 0.0, G])
    zb = -t_vec / np.linalg.norm(t_vec)
    x_c = np.array([np.cos(yaw_sp), np.sin(yaw_sp), 0.0])
    yb = np.cross(zb, x_c); yb = yb / np.linalg.norm(yb)
    xb = np.cross(yb, zb)
    return np.column_stack([xb, yb, zb])


def collective_accel(a_des, quat):
    cos_tilt = max(float(quat_to_R(quat)[2, 2]), 0.5)
    return float((G - np.asarray(a_des, float)[2]) / cos_tilt)


def accel_to_thrust_norm(c, hover, k_a):
    return float(np.clip(hover + (c - G) / k_a, 0.0, 1.0))


def tilt_deg(quat):
    return float(np.degrees(np.arccos(np.clip(quat_to_R(quat)[2, 2], -1.0, 1.0))))


def sideslip_deg(quat, vel_body):
    v = np.asarray(vel_body, float)[:2]
    if float(np.linalg.norm(v)) < 1e-3:
        return 0.0
    R = quat_to_R(quat)
    cam = -R[:2, 0]
    nc = np.linalg.norm(cam)
    if nc < 1e-9:
        return 0.0
    cam = cam / nc
    vd = v / np.linalg.norm(v)
    return float(np.degrees(np.arctan2(cam[0] * vd[1] - cam[1] * vd[0], float(cam @ vd))))


def weathervane_ff(vbx, vby, g_roll, g_yaw, gain=1.0, sign=1.0):
    roll_ff = -sign * gain * (ROLL_WV0 + ROLL_WV1 * vbx) * vby / g_roll
    yaw_ff = -sign * gain * YAW_WV * vby / g_yaw
    return float(roll_ff), float(yaw_ff)


class Controller:
    """Per-step corner_speed law; state mirrors the live script's loop variables."""

    def __init__(self, R_turn, cruise, kd, wvff, rg, hover, ka):
        self.Rt = R_turn; self.cruise = cruise; self.kd = kd; self.wvff = wvff
        self.RG = rg; self.hover = hover; self.ka = ka
        self.SDIR = -1.0
        self.PHI_SAFE = np.radians(40.0)
        self.A_CENT_MAX = G * np.tan(self.PHI_SAFE)
        C_DRAG = 0.057
        self.V_MAX = float(np.sqrt(G * np.tan(self.PHI_SAFE) / (C_DRAG + 1.0 / R_turn)))
        self.phase = "ENTRY"; self.t_eng = None; self.turned = 0.0; self.yaw_prev = None
        self.peak_beta = 0.0; self.peak_spd = 0.0
        self.peak_tilt = 0.0; self.peak_tilt_cmd = 0.0
        self._ybuf = collections.deque()
        # spawn-frame vectors (yaw0 = 0 at reset)
        self.yaw0 = 0.0
        fwd = -np.array([np.cos(self.yaw0), np.sin(self.yaw0)])
        self.push = -fwd; self.latp = np.array([-self.push[1], self.push[0]])
        self.spawn = np.zeros(3); self.z_sp = 0.0

    def shape_yaw(self, yr_raw, now):
        self._ybuf.append((now, yr_raw))
        while self._ybuf and now - self._ybuf[0][0] > 0.3:
            self._ybuf.popleft()
        out = 0.0
        for a, d in zip(ZVD_A, ZVD_T):
            tgt = now - d; val = yr_raw; best = 1e9
            for tb, vb in self._ybuf:
                e = abs(tb - tgt)
                if e < best:
                    best = e; val = vb
            out += a * val
        return out

    def step(self, t, pos, vel_body, quat, omega):
        """Returns (action [thr,wx,wy,wz], info). vel_body = R^T v_world (the live ds.vel_ned)."""
        vel = vel_body[:2]; v = float(np.linalg.norm(vel))
        beta = sideslip_deg(quat, vel_body)
        if self.phase == "ENTRY":
            v_al = float(vel @ self.push); v_ct = float(vel @ self.latp)
            p_ct = float((pos[:2] - self.spawn[:2]) @ self.latp)
            a2 = (float(np.clip(1.2 * (self.cruise - v_al), -2.0, 0.5)) * self.push
                  + (-0.6 * p_ct - 1.2 * v_ct) * self.latp)
            yaw_sp = self.yaw0; yaw_ff = 0.0; pitch_ff = 0.0
            if t > T_ENTRY and abs(beta) < 25 and v > 0.6 * self.cruise:
                self.phase = "TURN"; self.t_eng = t
        else:
            rr = min(1.0, (t - self.t_eng) / RAMP)
            vhat = vel / max(v, 1e-6)
            inside = self.SDIR * np.array([-vhat[1], vhat[0]])
            a_cent = rr * min(v * v / self.Rt, self.A_CENT_MAX)
            phi = np.arctan2(a_cent, G)
            psidot = a_cent / max(v, 0.5)
            a_along = float(np.clip(KV * (min(self.cruise, self.V_MAX) - v), -DECEL_MAX, ACCEL_MAX))
            turn_a2 = a_along * vhat + a_cent * inside
            entry_a2 = float(np.clip(1.2 * (self.cruise - float(vel @ self.push)), -2.0, 0.5)) * self.push
            a2 = (1.0 - rr) * entry_a2 + rr * turn_a2
            yaw_sp = float(np.arctan2(-vhat[1], -vhat[0]))
            yaw_ff = self.SDIR * psidot
            pitch_ff = self.SDIR * np.sin(phi) * psidot
            vh_hdg = float(np.arctan2(vhat[1], vhat[0]))
            if self.yaw_prev is not None:
                self.turned += ((vh_hdg - self.yaw_prev + np.pi) % (2 * np.pi)) - np.pi
            self.yaw_prev = vh_hdg
            self.peak_beta = max(self.peak_beta, abs(beta)); self.peak_spd = max(self.peak_spd, v)

        # --- cmd() verbatim ---
        a = np.zeros(3); a[:2] = np.asarray(a2, float)
        n = float(np.linalg.norm(a[:2]))
        if n > TILT_MAX_ACC:
            a[:2] = a[:2] / n * TILT_MAX_ACC
        a[2] = KP_Z * (self.z_sp - pos[2]) + KD_Z * (0.0 - vel_body[2])
        Rc = quat_to_R(quat); yaw_cur = float(np.arctan2(Rc[1, 0], Rc[0, 0]))
        q_des = mat_to_quat(desired_attitude(a, yaw_sp))
        w_des = KP_ATT * attitude_error_quat(quat, q_des)
        w_des[1] += pitch_ff if self.phase == "TURN" else 0.0
        if self.kd:
            w_des[0] -= self.kd * float(omega[0])
            w_des[1] -= self.kd * float(omega[1])
        yr_raw = (KP_YAW * ((yaw_sp - yaw_cur + np.pi) % (2 * np.pi) - np.pi)
                  - KD_YAW * float(omega[2]) + yaw_ff)
        w_des[2] = float(np.clip(self.shape_yaw(yr_raw, t), -YR_CAP, YR_CAP))
        wcmd = w_des / self.RG
        if self.wvff:
            roll_ff, yaw_ff_wv = weathervane_ff(float(vel_body[0]), float(vel_body[1]),
                                                self.RG[0], self.RG[2])
            wcmd[0] += float(np.clip(roll_ff, -WVFF_CLIP, WVFF_CLIP))
            wcmd[2] += float(np.clip(yaw_ff_wv, -WVFF_CLIP, WVFF_CLIP))
        thr = accel_to_thrust_norm(collective_accel(a, quat), self.hover, self.ka)
        tilt = tilt_deg(quat)
        self.peak_tilt = max(self.peak_tilt, tilt)
        self.peak_tilt_cmd = max(self.peak_tilt_cmd, tilt_deg(q_des))
        action = np.array([thr, *np.clip(wcmd, -WMAX, WMAX)])
        return action, {"tilt": tilt, "tilt_cmd": tilt_deg(q_des), "beta": beta, "v": v}


def run(R_turn, cruise, kd, wvff, use_mimo=True):
    model = json.load(open(ROOT / "sysid" / "vq_model.json"))
    if not use_mimo:
        model = {k: v for k, v in model.items() if k != "rate_loop_mimo"}
    r = json.load(open(ROOT / "sysid" / "sim_response.json"))
    rg = np.array([r["rate_gain_axes"]["roll"], r["rate_gain_axes"]["pitch"], r["rate_gain_axes"]["yaw"]])
    dyn = VQMatchedDynamics(model, dt=DT, frame="NED")
    ctl = Controller(R_turn, cruise, kd, wvff, rg, r["hover_thrust"], r["k_a"])
    s = dyn.reset(1)
    s[0, QUAT] = np.array([0.0, 0.0, 0.0, 1.0])  # true-frame yaw 180 = live-frame identity
    n_steps = int(MAX_T / DT)
    outcome = "MAX_T"
    last = -1
    peak_true_tilt = 0.0
    for k in range(n_steps):
        t = k * DT
        pos = s[0, POS]; vel_w = s[0, VEL]; q_true = s[0, QUAT]; om_true = s[0, OMEGA]
        Rb = quat_to_R(q_true)
        vel_body = Rb.T @ vel_w
        # plant (qfix/wfix true frame) -> live io_layer frame the controller was built on
        quat = q_true[[3, 0, 1, 2]]                       # stored = q_true[[3,0,1,2]]
        om = om_true * np.array([1.0, -1.0, 1.0])         # om_live = wfix * om_true
        true_tilt = tilt_deg(q_true)
        peak_true_tilt = max(peak_true_tilt, true_tilt)
        action, info = ctl.step(t, pos, vel_body, quat, om)
        kk = int(t / 0.5)
        if kk != last:
            last = kk
            print(f"t={t:4.1f} {ctl.phase[0]} v={info['v']:4.1f} tiltC/A={info['tilt_cmd']:3.0f}/{info['tilt']:3.0f} "
                  f"trueT={true_tilt:3.0f} beta={info['beta']:+4.0f} turned={np.degrees(ctl.turned):+5.0f}", flush=True)
        if info["tilt"] > ABORT_TILT or true_tilt > 110.0:
            outcome = f"TUMBLE t={t:.1f}"
            break
        if ctl.phase == "TURN" and abs(ctl.turned) >= 4 * np.pi:
            outcome = f"SUCCESS t={t:.1f}"
            break
        s = dyn.step(s, action[None, :])
    print(f"[{'MIMO' if use_mimo else 'DIAG'} kd={kd} wvff={wvff}] {outcome} "
          f"turned={np.degrees(ctl.turned):.0f}deg peak_spd={ctl.peak_spd:.1f} "
          f"peak|beta|={ctl.peak_beta:.0f} peak tiltC/A={ctl.peak_tilt_cmd:.0f}/{ctl.peak_tilt:.0f} "
          f"peak_trueT={peak_true_tilt:.0f}", flush=True)
    return outcome, ctl


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    R_turn = float(argv[0]) if len(argv) > 0 else 10.0
    cruise = float(argv[1]) if len(argv) > 1 else 6.0
    kd = float(sys.argv[sys.argv.index("--kd") + 1]) if "--kd" in sys.argv else 0.0
    wvff = "--no-wvff" not in sys.argv
    use_mimo = "--diag" not in sys.argv
    print(f"closed_loop_corner R={R_turn} CRUISE={cruise} kd={kd} wvff={wvff} mimo={use_mimo}")
    run(R_turn, cruise, kd, wvff, use_mimo)
