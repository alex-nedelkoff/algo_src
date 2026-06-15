"""Standalone spline FF anchor for the residual deploy mirror (COR-127 RESIDUAL-ASYM-01).

ff_batch + aim are copied VERBATIM from rl_finetune.py so the live deploy applies the
EXACT same analytic anchor the residual policy was trained on (train==deploy). Constants
are derived from argv + sysid/vq_model.json, so running the deploy with the SAME flags
(--maxw 6 --thrmax 0.6 --vdes 3 --delta_scale 0.15) reproduces training's anchor.

Self-contained except `gate_traj` (GateTrajectory, numpy+scipy) -- drop both files next to
the deploy script. No sb3 / env imports (laptop aigp env has numpy+scipy+torch only).

residual_action(delta, S, gates3, gi, trajs) -> clip(ff_anchor(S) + tanh(delta)*scale, -1, 1).
"""
import sys
import json

import numpy as np
try:
    from gate_traj import GateTrajectory      # laptop deploy: file dropped next to the script
except ImportError:
    from sim.gate_traj import GateTrajectory  # repo/pod layout


def _argf(f, d):
    return float(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else d


G = 9.81
POS = slice(0, 3); VEL = slice(3, 6); QUAT = slice(6, 10); OMEGA = slice(10, 13)
_M = json.load(open("sysid/vq_model.json"))
GAIN = np.array([_M["rate_loop"][n]["gain_G"] for n in ("roll", "pitch", "yaw")])
F0 = _M["thrust"]["f0"]; DF = _M["thrust"]["df_dthr"]
MAXW = _argf("--maxw", 17.45)
THRMAX = _argf("--thrmax", 1.0)
VDES = _argf("--vdes", 3.0)
DELTA_SCALE = _argf("--delta_scale", 0.15)
NG = 6
ROLL_WV0, ROLL_WV1, YAW_WV = -0.105, -0.019, -0.149
YR_CAP = 1.5
KP_POS = np.array([0.6, 0.6, 2.0]); KD_POS = np.array([1.2, 1.2, 3.0])
KP_ATT = np.array([6.0, 6.0, 4.0]); KD_ATT = np.array([1.2, 1.2, 0.0]); KP_YAW = 4.0; KD_YAW = 0.5
TILT_MAX = np.tan(np.radians(35)) * G


def quat_to_rotmat_batch(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    n = q.shape[0]; R = np.empty((n, 3, 3))
    R[:, 0, 0] = 1 - 2 * (yy + zz); R[:, 0, 1] = 2 * (xy - wz); R[:, 0, 2] = 2 * (xz + wy)
    R[:, 1, 0] = 2 * (xy + wz); R[:, 1, 1] = 1 - 2 * (xx + zz); R[:, 1, 2] = 2 * (yz - wx)
    R[:, 2, 0] = 2 * (xz - wy); R[:, 2, 1] = 2 * (yz + wx); R[:, 2, 2] = 1 - 2 * (xx + yy)
    return R


def mat_to_quat_batch(m):
    N = m.shape[0]; t = m[:, 0, 0]+m[:, 1, 1]+m[:, 2, 2]; q = np.zeros((N, 4))
    cA = t > 0; cB = (~cA) & (m[:, 0, 0] >= m[:, 1, 1]) & (m[:, 0, 0] >= m[:, 2, 2])
    cC = (~cA) & (~cB) & (m[:, 1, 1] >= m[:, 2, 2]); cD = (~cA) & (~cB) & (~cC)
    sA = np.sqrt(np.maximum(t+1, 1e-12))*2
    q[cA, 0] = .25*sA[cA]; q[cA, 1] = (m[cA, 2, 1]-m[cA, 1, 2])/sA[cA]; q[cA, 2] = (m[cA, 0, 2]-m[cA, 2, 0])/sA[cA]; q[cA, 3] = (m[cA, 1, 0]-m[cA, 0, 1])/sA[cA]
    sB = np.sqrt(np.maximum(1+m[:, 0, 0]-m[:, 1, 1]-m[:, 2, 2], 1e-12))*2
    q[cB, 0] = (m[cB, 2, 1]-m[cB, 1, 2])/sB[cB]; q[cB, 1] = .25*sB[cB]; q[cB, 2] = (m[cB, 0, 1]+m[cB, 1, 0])/sB[cB]; q[cB, 3] = (m[cB, 0, 2]+m[cB, 2, 0])/sB[cB]
    sC = np.sqrt(np.maximum(1+m[:, 1, 1]-m[:, 0, 0]-m[:, 2, 2], 1e-12))*2
    q[cC, 0] = (m[cC, 0, 2]-m[cC, 2, 0])/sC[cC]; q[cC, 1] = (m[cC, 0, 1]+m[cC, 1, 0])/sC[cC]; q[cC, 2] = .25*sC[cC]; q[cC, 3] = (m[cC, 1, 2]+m[cC, 2, 1])/sC[cC]
    sD = np.sqrt(np.maximum(1+m[:, 2, 2]-m[:, 0, 0]-m[:, 1, 1], 1e-12))*2
    q[cD, 0] = (m[cD, 1, 0]-m[cD, 0, 1])/sD[cD]; q[cD, 1] = (m[cD, 0, 2]+m[cD, 2, 0])/sD[cD]; q[cD, 2] = (m[cD, 1, 2]+m[cD, 2, 1])/sD[cD]; q[cD, 3] = .25*sD[cD]
    return q/np.linalg.norm(q, axis=1, keepdims=True)


def ff_batch(S, tgt_pos, tgt_vel, tgt_yaw):
    pos = S[:, POS]; vel = S[:, VEL]; q = S[:, QUAT]; om = S[:, OMEGA]
    a = KP_POS*(tgt_pos-pos) + KD_POS*(tgt_vel-vel)
    n = np.linalg.norm(a[:, :2], axis=1); sc = np.where(n > TILT_MAX, TILT_MAX/np.maximum(n, 1e-9), 1.0)
    a[:, 0] *= sc; a[:, 1] *= sc
    R = quat_to_rotmat_batch(q); yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    t = a + np.array([0, 0, G]); zb = t/np.linalg.norm(t, axis=1, keepdims=True)
    xc = np.stack([np.cos(tgt_yaw), np.sin(tgt_yaw), np.zeros_like(tgt_yaw)], axis=1)
    yb = np.cross(zb, xc); yb /= np.linalg.norm(yb, axis=1, keepdims=True); xb = np.cross(yb, zb)
    q_des = mat_to_quat_batch(np.stack([xb, yb, zb], axis=2))
    w0, x0, y0, z0 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]; w1, x1, y1, z1 = q_des[:, 0], q_des[:, 1], q_des[:, 2], q_des[:, 3]
    qe = np.stack([w0*w1+x0*x1+y0*y1+z0*z1, w0*x1-x0*w1-y0*z1+z0*y1, w0*y1+x0*z1-y0*w1-z0*x1, w0*z1-x0*y1+y0*x1-z0*w1], axis=1)
    qe[qe[:, 0] < 0] *= -1
    w = KP_ATT*(2*qe[:, 1:4]) - KD_ATT*om
    w[:, 2] = KP_YAW*(((tgt_yaw-yaw+np.pi) % (2*np.pi))-np.pi) - KD_YAW*om[:, 2]
    vb = np.einsum("nji,nj->ni", R, vel)
    w[:, 0] += (ROLL_WV0 + ROLL_WV1*vb[:, 0]) * vb[:, 1]
    w[:, 2] += -YAW_WV * vb[:, 1]
    cos_t = np.maximum(R[:, 2, 2], 0.5); c = (G + a[:, 2])/cos_t
    tilt_deg = np.degrees(np.arccos(np.clip(R[:, 2, 2], -1, 1)))
    c_max = np.where(tilt_deg > 40.0, 10.0, 18.0)
    c = np.minimum(c, c_max)
    thr = np.clip((F0+c)/(-DF), 0.0, 1.0)
    u = np.empty((S.shape[0], 4)); u[:, 0] = np.clip(2*thr/THRMAX-1, -1, 1); u[:, 1:4] = np.clip(w/GAIN/MAXW, -1, 1)
    cap = YR_CAP/abs(GAIN[2])/MAXW
    u[:, 3] = np.clip(u[:, 3], -cap, cap)
    return u.astype(np.float32)


def aim(S, gates, gp, trajs):
    n_envs = len(gp); LEAD = 2.5
    out_pos = np.zeros((n_envs, 3)); out_vel = np.zeros((n_envs, 3)); out_yaw = np.zeros(n_envs)
    for i in range(n_envs):
        traj = trajs[i]
        s_d = traj.nearest_s(S[i, POS])
        s_ref = float(min(s_d + LEAD, traj.s_max))
        ref = traj.sample(s_ref)
        out_pos[i] = ref["pos"]; out_vel[i] = float(ref["v"]) * ref["tang"]; out_yaw[i] = float(ref["yaw"])
    return out_pos, out_vel, out_yaw


def build_trajs(gates3, spawn3, vdes=VDES):
    """gates3 (NG,3) world ENU, spawn3 (3,). Matches rl_finetune.make_env exactly."""
    spawn3 = np.asarray(spawn3, float)
    return [GateTrajectory(np.vstack([spawn3[None, :], np.asarray(gates3, float)]),
                           v_cruise=vdes, tilt_budget_deg=35.0, c_drag=0.057, margin=0.6)]


def residual_action(delta, S, gates3, gi, trajs, delta_scale=DELTA_SCALE):
    anchor = ff_batch(S, *aim(S, gates3, np.array([gi]), trajs))
    return np.clip(anchor + np.tanh(np.asarray(delta)) * delta_scale, -1.0, 1.0)
