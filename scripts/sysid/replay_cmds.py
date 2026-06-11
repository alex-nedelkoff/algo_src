"""Open-loop replay of a recorded deploy run's commands through VQMatchedDynamics.

DEPLOY-03 discriminator: run 20260610T230939 hit |v|=35.6 m/s live under the policy's
commands. If the matched plant does NOT reach that speed under the SAME commands, the
thrust/drag model is wrong in the high-|u|/high-v regime (a simpler story than any
weathervane refinement). |v| is frame-invariant, so the comparison dodges the live
io-frame adapter entirely; only the initial state needs converting (live shuffled quat
q_true = q_live[[1,2,3,0]], om_true = om_live*[1,-1,1], vel recorded in BODY frame).

Usage: python scripts/sysid/replay_cmds.py /tmp/vq_replay/20260610T230939.npz [--lag 0.085] [--lat 0.019]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sim.dynamics.vq_matched import VQMatchedDynamics, POS, VEL, QUAT, OMEGA  # noqa: E402
from scripts.sysid.closed_loop_policy import quat_to_R  # noqa: E402

DT = 1.0 / 72.0


def argf(f, d):
    return float(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else d


def main():
    npz = sys.argv[1]
    d = np.load(npz)
    lag = argf("--lag", 0.0)
    lat = argf("--lat", 0.0)
    t = d["t_wall"] - d["t_wall"][0]
    cmd = d["cmd"]            # [wx, wy, wz, thrust] as sent
    vel_b = d["vel"]          # live io: BODY frame
    quat_live = d["quat"]     # wxyz, live shuffled chart
    om_live = d["omega"]

    model = json.load(open(ROOT / "sysid" / "vq_model.json"))
    dyn = VQMatchedDynamics(model, dt=DT, frame="NED", latency_s=lat, thrust_lag_s=lag)
    s = dyn.reset(1)
    if lag > 0:
        s = np.concatenate([s, np.zeros((1, 4))], axis=1)
        s[0, 13] = (-9.81 - model["thrust"]["f0"]) / model["thrust"]["df_dthr"]

    # initial state from sample 0, converted live io -> true frame
    q0 = quat_live[0][[1, 2, 3, 0]]
    q0 = q0 / np.linalg.norm(q0)
    R0 = quat_to_R(q0)
    s[0, QUAT] = q0
    s[0, VEL] = R0 @ vel_b[0]
    s[0, OMEGA] = om_live[0] * np.array([1.0, -1.0, 1.0])
    s[0, POS] = 0.0

    v_live = np.linalg.norm(vel_b, axis=1)
    n_steps = int(t[-1] / DT)
    v_sim = np.zeros(n_steps)
    om_sim = np.zeros((n_steps, 3))
    j = 0
    for k in range(n_steps):
        tk = k * DT
        while j + 1 < len(t) and t[j + 1] <= tk:   # zero-order hold on the sent cmd
            j += 1
        action = np.array([[cmd[j, 3], cmd[j, 0], cmd[j, 1], cmd[j, 2]]])
        s = dyn.step(s, action)
        v_sim[k] = np.linalg.norm(s[0, VEL])
        om_sim[k] = s[0, OMEGA]

    print(f"replay {Path(npz).stem}  T={t[-1]:.1f}s  lag={lag} lat={lat}")
    print(f"  |v| max:   live {v_live.max():5.1f}   sim {v_sim.max():5.1f}")
    for tc in np.arange(1.0, t[-1], 1.0):
        kl = np.searchsorted(t, tc)
        ks = min(int(tc / DT), n_steps - 1)
        print(f"  t={tc:4.1f}  |v| live {v_live[kl]:5.1f}  sim {v_sim[ks]:5.1f}"
              f"   thr_cmd {cmd[min(kl, len(cmd)-1), 3]:+.2f}")


if __name__ == "__main__":
    main()
