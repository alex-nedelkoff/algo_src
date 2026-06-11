"""Diverge-hunt (ENV-RT-01): same rollout, obs computed by BOTH pipelines each step.

Replicates the closed_loop_policy.py chain rollout exactly, and at every step also
force-sets a 1-env GateRaceEnv (whose track = the chain's course) to the equivalent
ENU state and reads back env._compute_obs_batched(). Prints the max-|diff| obs element
per step (motor dims [12:16] excluded — the chain hardcodes 0.5114). The first
persistently diverging element is the right-turn defect's address.

Usage: PYTHONPATH=. python scripts/sysid/diverge_probe.py --policy X.zip --turn 0.2 [--steps 200]
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
from sim.envs.gate_race_env import GateRaceEnv  # noqa: E402
from sim.tracks import Track, GateState  # noqa: E402
from scripts.sysid.closed_loop_policy import (  # noqa: E402
    quat_to_R, quat_mul, qfix, B, bq, bqc, quat_to_euler, to_enu, rotxy, wrap,
    DT, MAXW, NG, argf, args_,
)

GATE_R = 1.5


def _yaw_quat(th):
    return np.array([np.cos(th / 2), 0.0, 0.0, np.sin(th / 2)])


def main():
    policy_zip = args_("--policy", str(ROOT / "ft_env_v6c_best.zip"))
    turn = argf("--turn", 0.2)
    n_steps = int(argf("--steps", 200))
    z = zipfile.ZipFile(policy_zip)
    sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu")
    Wt = {k: v.numpy() for k, v in sd.items()}

    def policy(obs):
        h = obs.astype(np.float64)
        for i in (0, 2, 4):
            h = np.tanh(Wt[f"mlp_extractor.policy_net.{i}.weight"] @ h + Wt[f"mlp_extractor.policy_net.{i}.bias"])
        return Wt["action_net.weight"] @ h + Wt["action_net.bias"]

    model = json.load(open(ROOT / "sysid" / "vq_model.json"))
    dyn = VQMatchedDynamics(model, dt=DT, frame="NED")
    s = dyn.reset(1)
    s[0, QUAT] = np.array([0.0, 0.0, 0.0, 1.0])

    # ---- chain course (verbatim from closed_loop_policy) ----
    q_true0 = s[0, QUAT]
    quat_live0 = q_true0[[3, 0, 1, 2]]
    pe0, _, qe0, _ = to_enu(quat_live0, np.zeros(3), s[0, POS].copy(), np.zeros(3))
    camfwd = -(quat_to_R(qfix(quat_live0))[:, 0]) * B
    camfwd[2] = 0
    camfwd /= np.linalg.norm(camfwd) + 1e-9
    hd0 = float(np.arctan2(-camfwd[1], -camfwd[0]))
    gates, gyaws = [], []
    hd = hd0
    p = pe0.copy()
    for _ in range(NG):
        hd += turn
        p = p + 10.0 * np.array([np.cos(hd), np.sin(hd), 0.0])
        gates.append(p.copy())
        gyaws.append(hd)
    gates = np.array(gates)
    gyaws = np.array(gyaws)

    # ---- mirror env with the SAME course ----
    gs = [GateState(position=gates[k].copy(), orientation=_yaw_quat(gyaws[k])) for k in range(NG)]
    track = Track(gates=gs, name="probe", start_position=pe0.copy())
    env = GateRaceEnv(n_envs=1, dt=DT, max_steps=10000, action_mode="vq_rate",
                      vq_model_path=str(ROOT / "sysid" / "vq_model.json"), tracks=[track],
                      random_gate_start=False, gate_collision=False, gate_passage_radius=1.0,
                      arena_bounds=120.0)
    env.reset()

    cmp_idx = [i for i in range(27) if i not in (12, 13, 14, 15)]
    prev_act = np.zeros(4)
    gi = 0
    print(f"turn={turn:+.2f} hd0={hd0:.2f} gate0={gates[0].round(1)} gyaw0={gyaws[0]:+.2f}")
    for k in range(n_steps):
        t = k * DT
        q_true = s[0, QUAT]
        quat_live = q_true[[3, 0, 1, 2]]
        om_live = s[0, OMEGA] * np.array([1.0, -1.0, 1.0])
        Rb = quat_to_R(q_true)
        vel_live = Rb.T @ s[0, VEL]
        pos_live = s[0, POS]
        pe, ve, qe, ome = to_enu(quat_live, vel_live, pos_live, om_live)

        # ---- chain obs (verbatim) ----
        gate = gates[gi]
        gi1 = min(gi + 1, NG - 1)
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

        # ---- env obs on the SAME physical state ----
        env._states[0, POS] = pe
        env._states[0, VEL] = ve
        env._states[0, QUAT] = qe / np.linalg.norm(qe)
        env._states[0, OMEGA] = ome
        env._gate_indices[0] = gi
        env._prev_actions[0] = prev_act
        obs_env = env._compute_obs_batched()[0]

        d = np.abs(obs[cmp_idx] - obs_env[cmp_idx])
        j = int(np.argmax(d))
        if k % 9 == 0 or d[j] > 0.05:
            jj = cmp_idx[j]
            print(f"k={k:3d} t={t:4.2f} gi={gi} maxdiff obs[{jj}] chain={obs[jj]:+.3f} env={obs_env[jj]:+.3f} "
                  f"|d|={d[j]:.4f}  beta_v=({ve[0]:+.1f},{ve[1]:+.1f})")

        u = policy(obs)
        uc = np.clip(u, -1, 1)
        prev_act = uc
        thr = float((uc[0] + 1) / 2)
        rates = np.array([uc[1], -uc[2], -uc[3]]) * MAXW
        s = dyn.step(s, np.array([[thr, *rates]]))

        dist = float(np.linalg.norm((gate - pe)[:2]))
        if dist < GATE_R:
            gi += 1
            print(f"  GATE {gi}/{NG} t={t:.2f}")
            if gi >= NG:
                break
        tilt_true = float(np.degrees(np.arccos(np.clip(Rb[2, 2], -1, 1))))
        if tilt_true > 110.0:
            print(f"  ABORT tilt t={t:.2f} gi={gi}")
            break


if __name__ == "__main__":
    main()
