"""Closed-loop DEPLOY-CHAIN check: run vq_deploy2's exact obs/action code against the
matched plant (COR-127). The plant is NED/qfix-frame VQMatchedDynamics; states are converted
to the LIVE io_layer representation (closed_loop_corner adapter), then through vq_deploy2's
to_enu + gate-yaw obs + B action map. If the policy races here (it scores ~6/6 on this plant
in the training env), the deploy chain is faithful and any live failure is plant-side; if it
diverges like the live runs, the bug is in the chain and can be fixed offline.

Usage: python scripts/sysid/closed_loop_policy.py [--policy ft_sym_v6c_std_best.zip] [--turn 0.2]
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sim.dynamics.vq_matched import VQMatchedDynamics, POS, VEL, QUAT, OMEGA  # noqa: E402

G = 9.81
DT = 1.0 / 72.0
NG = 6
GATE_R = 1.5
MAX_T = 33.0


def argf(f, d):
    return float(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else d


MAXW = argf("--maxw", 17.45)    # must match the policy's training-time action cap
THRMAX = argf("--thrmax", 1.0)  # must match the policy's training-time thrust cap
ZVD = "--zvd" in sys.argv       # ZVD shaper on wire rate cmds; must match training (EXP-20b ring killer)
ZVD_AMP = np.array([0.371, 0.476, 0.153])
ZVD_DELAY = (7, 7, 4)           # frames per axis at 72 Hz (roll, pitch, yaw)
SLEW = argf("--slew", 0.0)      # rad/s^2 wire-rate slew cap (RING-06); must match training


def args_(f, d):
    return sys.argv[sys.argv.index(f) + 1] if f in sys.argv else d


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


# ---- vq_deploy2 code path, verbatim ----------------------------------------------------
def qfix(q):
    return np.array([q[1], q[2], q[3], q[0]])


B = np.array([1.0, -1.0, -1.0])
bq = np.array([0.0, 1.0, 0.0, 0.0])
bqc = np.array([0.0, -1.0, 0.0, 0.0])


def quat_to_euler(q):
    w, x, y, zz = q
    return (np.arctan2(2 * (w * x + y * zz), 1 - 2 * (x * x + y * y)),
            np.arcsin(np.clip(2 * (w * y - zz * x), -1, 1)),
            np.arctan2(2 * (w * zz + x * y), 1 - 2 * (y * y + zz * zz)))


def to_enu(quat_live, vel_live, pos_live, om_live):
    q_true = qfix(quat_live)
    Rt = quat_to_R(q_true)
    vel_w = Rt @ vel_live
    om_t = om_live * np.array([1.0, -1.0, 1.0])
    return pos_live * B, vel_w * B, quat_mul(quat_mul(bq, q_true), bqc), om_t * B


def rotxy(v, yaw):
    c, sn = np.cos(-yaw), np.sin(-yaw)
    return np.array([c * v[0] - sn * v[1], sn * v[0] + c * v[1]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    policy_zip = args_("--policy", str(ROOT / "ft_sym_v6c_std_best.zip"))
    turn = argf("--turn", 0.2)
    z = zipfile.ZipFile(policy_zip)
    sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu")
    Wt = {k: v.numpy() for k, v in sd.items()}

    def policy(obs):
        h = obs.astype(np.float64)
        for i in (0, 2, 4):
            h = np.tanh(Wt[f"mlp_extractor.policy_net.{i}.weight"] @ h + Wt[f"mlp_extractor.policy_net.{i}.bias"])
        return Wt["action_net.weight"] @ h + Wt["action_net.bias"]

    model = json.load(open(args_("--model", str(ROOT / "sysid" / "vq_model.json"))))
    lag = argf("--lag", 0.0); lat = argf("--lat", 0.0)
    dyn = VQMatchedDynamics(model, dt=DT, frame="NED", latency_s=lat, thrust_lag_s=lag)
    s = dyn.reset(1)
    if lag > 0:    # thrust-lag state rides the 17-state motor slot
        s = np.concatenate([s, np.zeros((1, 4))], axis=1)
        s[0, 13] = (-9.81 - model["thrust"]["f0"]) / model["thrust"]["df_dthr"]  # start at hover thr
    s[0, QUAT] = np.array([0.0, 0.0, 0.0, 1.0])  # live-frame identity spawn (closed_loop_corner)
    vback = argf("--vback", 0.0)   # spawn drifting backward (away from gate 0) at this speed -- the live LVL handoff state
    if vback > 0:
        s[0, VEL] = quat_to_R(s[0, QUAT])[:, 0] * vback   # body +x = away from camera-forward course

    # course along the NOSE in ENU, exactly as vq_deploy2 builds it
    q_true0 = s[0, QUAT]
    quat_live0 = q_true0[[3, 0, 1, 2]]
    pe0, _, qe0, _ = to_enu(quat_live0, np.zeros(3), s[0, POS].copy(), np.zeros(3))
    camfwd = -(quat_to_R(qfix(quat_live0))[:, 0]) * B
    camfwd[2] = 0
    camfwd /= np.linalg.norm(camfwd) + 1e-9
    hd0 = float(np.arctan2(-camfwd[1], -camfwd[0]))
    if "--nosebranch" not in sys.argv:
        # LIVE branch by default (HYPO-01): vq_deploy4's course puts the nose ANTI-course
        # (obs[8] ~ pi, the trained camera-forward attitude). The old chain default flew the
        # mirrored nose-toward-gates branch (obs[8] ~ 0) and masked the live entry behavior.
        hd0 += np.pi
    gates = []
    gyaws = []
    hd = hd0
    p = pe0.copy()
    space = argf("--space", 10.0)   # training first leg is 8+sp ~ 17-21 m; 10 is OOD-short (ENV-RT-01)
    for _ in range(NG):
        hd += turn
        p = p + space * np.array([np.cos(hd), np.sin(hd), 0.0])
        gates.append(p.copy())
        gyaws.append(hd)
    gates = np.array(gates)
    gyaws = np.array(gyaws)
    print(f"spawn_enu={pe0.round(1)} camfwd={camfwd.round(2)} hd0={hd0:.2f} gates0={gates[0].round(1)}")

    prev_act = np.zeros(4)
    gi = 0
    betas = []
    last = -1
    n_steps = int(MAX_T / DT)
    # obs staleness: build the policy's obs from the state OBSDELAY frames ago (live ODOMETRY is
    # ~22 ms stale + loop jitter; training obs are same-tick truth). --obsjitter draws the delay
    # uniformly in [0, OBSDELAY] each step instead of holding it fixed.
    obsdelay = int(argf("--obsdelay", 0))
    obsjitter = "--obsjitter" in sys.argv
    actevery = int(argf("--actevery", 1))   # policy acts every N frames, cmd held (live loop-rate sag: dt p99 ~3 frames)
    rng = np.random.default_rng(3)
    hist: list[NDArray] = []
    held_action = None
    zvd_buf = np.zeros((15, 3))
    slew_prev = None
    for k in range(n_steps):
        t = k * DT
        hist.append(s.copy())
        if len(hist) > 8:
            hist.pop(0)
        d = int(rng.integers(0, obsdelay + 1)) if (obsjitter and obsdelay > 0) else obsdelay
        so = hist[max(0, len(hist) - 1 - d)]
        # plant truth -> live io_layer representation (data-verified adapter)
        q_true = so[0, QUAT]
        quat_live = q_true[[3, 0, 1, 2]]
        om_live = so[0, OMEGA] * np.array([1.0, -1.0, 1.0])
        Rb = quat_to_R(q_true)
        vel_live = Rb.T @ so[0, VEL]         # ds.vel_ned is BODY frame
        pos_live = so[0, POS]
        # ---- vq_deploy2 obs build ----
        pe, ve, qe, ome = to_enu(quat_live, vel_live, pos_live, om_live)
        gate = gates[gi]
        gi1 = (gi + 1) % NG   # training env wraps lookahead; min() gave a zero next-gate vector at the last gate
        gyaw = gyaws[gi]
        roll, pitch, dyaw = quat_to_euler(qe)
        obs = np.zeros(27, dtype=np.float32)
        obs[0:2] = rotxy((pe - gate)[:2], gyaw)
        obs[2] = pe[2] - gate[2]
        obs[3:5] = rotxy(ve[:2], gyaw)
        obs[5] = ve[2]
        obs[6] = roll
        obs[7] = pitch
        obs[8] = wrap(dyaw - gyaw)
        obs[9:12] = ome
        obs[12:16] = 0.5114015
        obs[16:20] = prev_act
        dg = gates[gi1] - gate
        obs[20:22] = rotxy(dg[:2], gyaw)
        obs[22] = dg[2]
        obs[23] = wrap(gyaws[gi1] - gyaw)
        obs[24] = 2.0
        obs[25] = 2.0
        obs[26] = 12.0
        if "--noobsclip" not in sys.argv:
            # OOD guard (run-7 flee): clip gate-rel position to the training-covered range
            obs[0:2] = np.clip(obs[0:2], -44.0, 44.0)
            obs[2] = np.clip(obs[2], -8.0, 8.0)
        u = policy(obs)
        uc = np.clip(u, -1, 1)
        prev_act = uc
        thr = float((uc[0] + 1) / 2 * THRMAX)
        rates = np.array([uc[1], -uc[2], -uc[3]]) * MAXW   # ENU -> VQ rate cmd (B)
        if ZVD:
            zvd_buf = np.roll(zvd_buf, 1, axis=0); zvd_buf[0] = rates
            rates = np.array([ZVD_AMP[0] * zvd_buf[0, ax] + ZVD_AMP[1] * zvd_buf[ZVD_DELAY[ax], ax]
                              + ZVD_AMP[2] * zvd_buf[2 * ZVD_DELAY[ax], ax] for ax in range(3)])
        if SLEW > 0:
            if slew_prev is None:
                slew_prev = rates.copy()
            mx = SLEW * DT
            rates = slew_prev + np.clip(rates - slew_prev, -mx, mx); slew_prev = rates.copy()
        action = np.array([[thr, *rates]])
        if actevery > 1:                     # hold the previous cmd between policy ticks
            if held_action is None or k % actevery == 0:
                held_action = action
            action = held_action
        # diagnostics
        tilt_true = float(np.degrees(np.arccos(np.clip(Rb[2, 2], -1, 1))))
        vmag = float(np.linalg.norm(ve))
        if np.linalg.norm(ve[:2]) > 0.5:
            nose = quat_to_R(qe)[:2, 0]
            vv = ve[:2]
            beta = abs(np.degrees(np.arctan2(nose[0] * vv[1] - nose[1] * vv[0], nose @ vv)))
            betas.append(beta)
        else:
            beta = 0.0
        dist = float(np.linalg.norm((gate - pe)[:2]))
        if dist < GATE_R:
            gi += 1
            print(f"  GATE {gi}/{NG} t={t:.1f} v={vmag:.1f} tilt={tilt_true:.0f} |beta|={beta:.0f}")
            if gi >= NG:
                break
        if tilt_true > 110.0:
            print(f"  ABORT tilt={tilt_true:.0f} t={t:.1f} gate {gi}/{NG}")
            break
        if int(t * 2) != last:
            last = int(t * 2)
            print(f"  t={t:4.1f} gate{gi} dist={dist:4.1f} v={vmag:4.1f} tilt={tilt_true:3.0f} |beta|={beta:3.0f} "
                  f"u=[{u[0]:+.2f},{u[1]:+.2f},{u[2]:+.2f},{u[3]:+.2f}]")
        s = dyn.step(s, action)
    print(f"OFFLINE DEPLOY-CHAIN: reached {gi}/{NG} gates | mean|beta|={np.mean(betas) if betas else 0:.0f}")


if __name__ == "__main__":
    main()
