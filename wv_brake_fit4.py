"""wv_brake_fit4.py -- phase 4: block-mean residual regression. Finding so far: with recorded cmds
the old model reproduces most of the snap's yaw FLIP (it is commanded/velocity-swing); the gap is the
PRE-SNAP sustained bias (~1 s, smooth cmds, tilt climbing 44->56 deg, residual 2-4 rad/s^2 all axes)
that over-tilts the live plant. 1-step fits drown in cmd-timing noise (alternating +/-100); 0.5 s
BLOCK MEANS cancel it (zero-mean dither) and keep the bias.

Method: split every run into contiguous 0.5 s blocks (|W|<6, pre-collision, t>3.5); per block store
mean residual (rad/s^2, MIMO-pred, no wv subtraction) + mean state features. Regress block-mean
moments on weathervane bases; deploy pre-snap blocks tagged for LORO + regime breakdown.
Frames per fit_model (SOLVED). Usage: python wv_brake_fit4.py
"""
import glob
import json

import numpy as np

from aigp.geometry import quat_to_R
from aigp.recorder import load_run, data_root
from fit_model import WFIX, qfix

DT = 1.0 / 72.0
BLK = 36                                                  # 0.5 s
model = json.load(open("sysid/vq_model.json"))
A = np.array(model["rate_loop_mimo"]["A"], float)
B = np.array(model["rate_loop_mimo"]["B"], float)


def blocks(run_dir, tag):
    d = load_run(run_dir)
    tu = d["t_us"]; t = d["t_wall"]; cs = d["coll_seq"]
    _, u = np.unique(tu, return_index=True); u = np.sort(u)
    t, cs = t[u], cs[u]
    W = d["omega"][u] * WFIX; C4 = d["cmd"][u]; C = C4[:, :3]; thr = C4[:, 3]
    vb = d["vel"][u]; Q = d["quat"][u]
    tt = tu[u] / 1e6
    ok = (t > 3.5) & ~np.isnan(vb).any(1) & ~np.isnan(W).any(1) & ~np.isnan(C).any(1) & ~np.isnan(Q).any(1)
    idx = np.flatnonzero(ok)
    if len(idx) < BLK + 1:
        return []
    cs0 = cs[idx[0]]
    bad = np.flatnonzero(cs[idx] != cs0)
    if len(bad):
        idx = idx[: bad[0]]
    if len(idx) < BLK + 1:
        return []
    W, C, thr, vb, Q, tt = W[idx], C[idx], thr[idx], vb[idx], Q[idx], tt[idx]
    resid = (W[1:] - (W[:-1] @ A.T + C[:-1] @ B.T)) / DT
    dt_ok = np.diff(tt) < 1.4 * DT
    calm = (np.abs(W[:-1]) < 6).all(1)
    out = []
    for s0 in range(0, len(resid) - BLK, BLK):
        sl = slice(s0, s0 + BLK)
        if not (dt_ok[sl].all() and calm[sl].all()):
            continue
        tilt = np.degrees(np.arccos(np.clip(quat_to_R(qfix(Q[s0 + BLK // 2]))[2, 2], -1, 1)))
        out.append(dict(r=resid[sl].mean(0), vb=vb[sl].mean(0), thr=float(thr[sl].mean()),
                        spd=float(np.linalg.norm(vb[sl], axis=1).mean()), tilt=float(tilt),
                        t=float(tt[s0]), run=run_dir.replace("\\", "/").split("/")[-1], tag=tag))
    return out


root = str(data_root())
BL = []
for pat, tag in (("/*_vq_deploy4*", "dep"), ("/*_vq_deploy5", "dep"),
                 ("/*_collect_vq_brake", "brake"),
                 ("/*_collect_vq_crab", "crab"), ("/*_corner_speed", "corner"),
                 ("/*_collect_vq_hr", "hr"), ("/*_collect_vq", "cvq"),
                 ("/*_collect_vq_ramp", "ramp"), ("/*_collect_vq_lateral", "lat")):
    for r in sorted(glob.glob(root + pat)):
        BL += blocks(r, tag)
print(f"blocks: {len(BL)} total; by tag: "
      + " ".join(f"{tg}={sum(1 for b in BL if b['tag'] == tg)}" for tg in ("dep", "brake", "crab", "corner", "hr", "cvq", "ramp", "lat")),
      flush=True)

R = np.array([b["r"] for b in BL]); VB = np.array([b["vb"] for b in BL])
TH = np.array([b["thr"] for b in BL]); SP = np.array([b["spd"] for b in BL])
TL = np.array([b["tilt"] for b in BL]); TAG = np.array([b["tag"] for b in BL])
vbx, vby, vbz = VB[:, 0], VB[:, 1], VB[:, 2]

print("\n== deploy + brake blocks (the regime to fix) ==", flush=True)
for b in BL:
    if b["tag"] in ("dep", "brake") and (abs(b["vb"][1]) > 0.8 or b["tag"] == "dep"):
        print(f"  {b['run']:<30} t={b['t']:5.2f} vb=[{b['vb'][0]:+5.2f} {b['vb'][1]:+5.2f} {b['vb'][2]:+5.2f}] "
              f"spd={b['spd']:4.1f} tilt={b['tilt']:3.0f} thr={b['thr']:.2f} resid=[{b['r'][0]:+6.2f} {b['r'][1]:+6.2f} {b['r'][2]:+6.2f}]",
              flush=True)

# odd-in-vby features (roll/yaw); even features (pitch)
FEAT_RY = {"vby": vby, "vby*vbx": vby * vbx, "vby*|vbx|": vby * np.abs(vbx),
           "vby*|v|": vby * SP, "vby*thr": vby * TH, "vby*vbz": vby * vbz}
FEAT_P = {"vbx": vbx, "vbx*|v|": vbx * SP, "vbx*thr": vbx * TH, "vbz": vbz,
          "vbz*|v|": vbz * SP, "vby^2": vby ** 2}


def fit_report(axn, ax, FEATS, bases):
    print(f"\n================ {axn} (block means) ================", flush=True)
    y = R[:, ax]
    dep = (TAG == "dep") | (TAG == "brake")
    print(f"{'basis':<34} {'R2all':>6} {'R2dep':>6} {'dep-RMSE':>8} {'cru-RMSE':>8}  coeffs", flush=True)
    base_rmse_dep = float(np.sqrt(np.mean(y[dep] ** 2)))
    base_rmse_cru = float(np.sqrt(np.mean(y[~dep] ** 2)))
    print(f"{'(zero model)':<34} {'-':>6} {'-':>6} {base_rmse_dep:>8.3f} {base_rmse_cru:>8.3f}", flush=True)
    for bn, keys in bases.items():
        X = np.column_stack([FEATS[k] for k in keys])
        sol, *_ = np.linalg.lstsq(X, y, rcond=None)
        p = X @ sol
        r2a = 1 - np.var(y - p) / np.var(y)
        r2d = 1 - np.mean((y - p)[dep] ** 2) / max(np.var(y[dep]), 1e-12)
        rd = float(np.sqrt(np.mean((y - p)[dep] ** 2))); rc = float(np.sqrt(np.mean((y - p)[~dep] ** 2)))
        cs = " ".join(f"{k}={v:+.3f}" for k, v in zip(keys, sol))
        print(f"{bn:<34} {r2a:>6.3f} {r2d:>6.3f} {rd:>8.3f} {rc:>8.3f}  {cs}", flush=True)


fit_report("ROLL", 0, FEAT_RY,
           {"[vby]": ["vby"], "[vby,vby*vbx]": ["vby", "vby*vbx"],
            "[vby,vby*vbx,vby*|vbx|]": ["vby", "vby*vbx", "vby*|vbx|"],
            "[vby,vby*thr]": ["vby", "vby*thr"],
            "[all6]": list(FEAT_RY)})
fit_report("YAW", 2, FEAT_RY,
           {"[vby]": ["vby"], "[vby,vby*vbx]": ["vby", "vby*vbx"],
            "[vby,vby*vbx,vby*|vbx|]": ["vby", "vby*vbx", "vby*|vbx|"],
            "[vby,vby*thr]": ["vby", "vby*thr"],
            "[all6]": list(FEAT_RY)})
fit_report("PITCH", 1, FEAT_P,
           {"[vbx]": ["vbx"], "[vbx,vbz]": ["vbx", "vbz"],
            "[vbx,vbz,vbx*|v|]": ["vbx", "vbz", "vbx*|v|"],
            "[all6]": list(FEAT_P)})

# tilt as the hidden variable? correlation of |resid| with tilt per axis
print("\ncorr(|block resid|, tilt): "
      + " ".join(f"{axn}={np.corrcoef(np.abs(R[:, ax]), TL)[0, 1]:+.2f}"
                 for axn, ax in (("roll", 0), ("pitch", 1), ("yaw", 2))), flush=True)
