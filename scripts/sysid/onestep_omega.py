"""1-step omega prediction test on a deploy recording (DEPLOY-04 closed-loop gap).

For every recorded transition (state_k, cmd_k) -> omega_{k+1}: set the matched plant to the
recorded state, step once with the recorded command, compare predicted vs recorded omega.
This is the rate-loop's predictive power ON THE POLICY'S OWN AGGRESSION DISTRIBUTION —
open-loop velocity replay is now honest (ENVELOPE-01), so what's left of the live gap
must show up here. Reports per-axis RMSE by |omega| band, plus the |W| reference
(RMSE ~= |W| means zero predictive power, the DEPLOY-01 finding).

Usage: python scripts/sysid/onestep_omega.py /tmp/vq_replay/<run>.npz
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


def main():
    d = np.load(sys.argv[1])
    t = d["t_us"] / 1e6
    _, uidx = np.unique(d["t_us"], return_index=True)   # dedup stale recorder rows
    uidx = np.sort(uidx)
    t = t[uidx]
    cmd = d["cmd"][uidx]
    vel_b = d["vel"][uidx]
    quat_live = d["quat"][uidx]
    om_live = d["omega"][uidx]

    model = json.load(open(ROOT / "sysid" / "vq_model.json"))
    dyn = VQMatchedDynamics(model, dt=DT, frame="NED")

    SGN = np.array([1.0, -1.0, 1.0])
    n = len(t) - 1
    pred = np.full((n, 3), np.nan)
    meas = np.full((n, 3), np.nan)
    wmag = np.full(n, np.nan)
    for k in range(n):
        dtk = t[k + 1] - t[k]
        if not (0.5 * DT < dtk < 2.5 * DT):
            continue
        q0 = quat_live[k][[1, 2, 3, 0]]
        nq = np.linalg.norm(q0)
        if nq < 1e-6 or np.isnan(nq):
            continue
        s = dyn.reset(1)
        s[0, QUAT] = q0 / nq
        s[0, VEL] = quat_to_R(s[0, QUAT]) @ vel_b[k]
        s[0, OMEGA] = om_live[k] * SGN
        s[0, POS] = 0.0
        a = np.array([[cmd[k, 3], cmd[k, 0], cmd[k, 1], cmd[k, 2]]])
        s2 = dyn.step(s, a, dt=dtk)
        pred[k] = s2[0, OMEGA]
        meas[k] = om_live[k + 1] * SGN
        wmag[k] = np.linalg.norm(om_live[k] * SGN)

    ok = ~np.isnan(wmag)
    print(f"{Path(sys.argv[1]).stem}: {ok.sum()}/{n} valid transitions")
    for lo, hi in ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 50.0)):
        m = ok & (wmag >= lo) & (wmag < hi)
        if m.sum() < 10:
            continue
        e = pred[m] - meas[m]
        rmse = np.sqrt((e ** 2).mean(axis=0))
        wref = np.sqrt((meas[m] ** 2).mean(axis=0))
        print(f"  |w| [{lo:3.1f},{hi:4.1f}): n={m.sum():4d}  RMSE r/p/y "
              f"{rmse[0]:.3f}/{rmse[1]:.3f}/{rmse[2]:.3f}   |W|ref {wref[0]:.2f}/{wref[1]:.2f}/{wref[2]:.2f}")


if __name__ == "__main__":
    main()
