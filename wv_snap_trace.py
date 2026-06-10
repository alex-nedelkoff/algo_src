"""wv_snap_trace.py -- phase 1b: time-series view of the DEPLOY-02 braking snap. For each deploy
recording, find the snap (beta crossing ~90 deg at speed) and dump the surrounding window at full
rate: t, vb, |v|, beta, TRUE tilt (qfix), omega, wcmd, MIMO residual/dt. Goal: see the unmodeled
moment's structure vs (vbx, vby, beta) WHILE the vehicle is still controlled (tilt < ~45), separating
the snap path-in from tumble thrash. Frames: qfix/WFIX per fit_model (SOLVED)."""
import glob
import json

import numpy as np

from aigp.geometry import quat_to_R
from aigp.recorder import load_run, data_root
from fit_model import WFIX, qfix

DT = 1.0 / 72.0
model = json.load(open("sysid/vq_model.json"))
A = np.array(model["rate_loop_mimo"]["A"], float)
B = np.array(model["rate_loop_mimo"]["B"], float)

root = str(data_root())
RUNS = sorted(glob.glob(root + "/*_vq_deploy4*")) + sorted(glob.glob(root + "/*_vq_deploy5"))

for run in RUNS:
    d = load_run(run)
    tu = d["t_us"]
    _, u = np.unique(tu, return_index=True); u = np.sort(u)
    t = tu[u] / 1e6
    W = d["omega"][u] * WFIX; C = d["cmd"][u][:, :3]; vb = d["vel"][u]; Q = d["quat"][u]
    ok = ~np.isnan(vb).any(1) & ~np.isnan(W).any(1) & ~np.isnan(C).any(1) & ~np.isnan(Q).any(1)
    t, W, C, vb, Q = t[ok], W[ok], C[ok], vb[ok], Q[ok]
    tilt = np.degrees(np.arccos(np.clip([quat_to_R(qfix(q))[2, 2] for q in Q], -1, 1)))
    spd = np.linalg.norm(vb[:, :2], axis=1)
    beta = np.degrees(np.arctan2(vb[:, 1], vb[:, 0]))          # 0 = velocity along +body_x
    resid = np.full_like(W, np.nan)
    good = np.diff(t) < 1.4 * DT
    resid[1:][good] = (W[1:] - (W[:-1] @ A.T + C[:-1] @ B.T))[good]

    # snap = |beta| crossing 90 deg while at speed (>2.5) and still controlled (tilt<60)
    ab = np.abs(beta)
    cross = np.flatnonzero((ab[:-1] > 90) & (ab[1:] <= 90) & (spd[1:] > 2.5) & (tilt[1:] < 60))
    rid = run.replace("\\", "/").split("/")[-1]
    if len(cross) == 0:
        print(f"\n#### {rid}: no snap crossing found (spd>2.5, tilt<60)", flush=True)
        continue
    k = int(cross[0])
    lo = np.searchsorted(t, t[k] - 1.2); hi = np.searchsorted(t, t[k] + 0.6)
    print(f"\n#### {rid}: snap beta-crossing at t={t[k]:.2f} (spd {spd[k]:.1f}); window rows:", flush=True)
    print("      t    vbx    vby   |v|  beta tilt |  om: r     p     y   | wcmd: r     p     y   | resid/dt: r      p      y", flush=True)
    for i in range(lo, min(hi, len(t)), 2):
        r = resid[i] / DT
        print(f"  {t[i]:6.2f} {vb[i,0]:+6.2f} {vb[i,1]:+6.2f} {spd[i]:5.2f} {beta[i]:+5.0f} {tilt[i]:4.0f} |"
              f" {W[i,0]:+5.2f} {W[i,1]:+5.2f} {W[i,2]:+5.2f} |"
              f" {C[i,0]:+5.2f} {C[i,1]:+5.2f} {C[i,2]:+5.2f} |"
              f" {r[0]:+7.2f} {r[1]:+7.2f} {r[2]:+7.2f}", flush=True)
