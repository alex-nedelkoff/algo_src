"""diag_coupling.py -- DIAGNOSE what roll/yaw coupling vq_matched's per-axis first-order rate loop
misses in the turn. Pool corner runs; residual r = W[k+1] - (a*W[k] + (1-a)*G*C[k]) (the part the
first-order model can't explain); regress r_roll & r_yaw (turn region, tilt>15) against candidate
features -> which form explains it (linear cross-rate? cross-cmd? gyroscopic w x w?). Frame per
wv_validate (WFIX omega, odom-vel=body). Usage: python diag_coupling.py <run_dir> [<run_dir> ...]"""
import sys, json, glob
import numpy as np
from aigp.recorder import load_run
from fit_model import qfix, WFIX
from aigp.geometry import quat_to_R

model = json.load(open("sysid/vq_model.json")); rl = model["rate_loop"]
G = np.array([rl[k]["gain_G"] for k in ("roll", "pitch", "yaw")])
TAU = np.array([rl[k]["tau_ms"] for k in ("roll", "pitch", "yaw")]) / 1e3
DT = 1.0 / 72.0; a = np.exp(-DT / TAU)

runs = sys.argv[1:]
Wl, Cl, Vl, Tl = [], [], [], []
for run in runs:
    d = load_run(run)
    _, u = np.unique(d["t_us"], return_index=True); u = np.sort(u)
    W = d["omega"][u] * WFIX; C = d["cmd"][u]; vb = d["vel"][u]; Q = d["quat"][u]
    tilt = np.degrees(np.arccos(np.clip([quat_to_R(q)[2, 2] for q in Q], -1, 1)))
    Wl.append(W); Cl.append(C); Vl.append(vb); Tl.append(tilt)

# build per-transition arrays (k -> k+1), tag turn region by tilt[k]
Wk = np.vstack([W[:-1] for W in Wl]); Wk1 = np.vstack([W[1:] for W in Wl])
Ck = np.vstack([C[:-1] for C in Cl]); tk = np.concatenate([t[:-1] for t in Tl])
resid = Wk1 - (a * Wk + (1 - a) * G * Ck[:, :3])          # what first-order misses, per axis (WFIX frame)

m = tk > 15                                                # turn/over-tilt region
wr, wp, wy = Wk[m, 0], Wk[m, 1], Wk[m, 2]
cr, cp, cy = Ck[m, 0], Ck[m, 1], Ck[m, 2]
feats = {
    "w_roll": wr, "w_pitch": wp, "w_yaw": wy,
    "c_roll": cr, "c_pitch": cp, "c_yaw": cy,
    "wp*wy": wp * wy, "wr*wy": wr * wy, "wr*wp": wr * wp,
}


def fit(y, names):
    X = np.column_stack([feats[n] for n in names] + [np.ones_like(y)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta; ss = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - pred) ** 2) / ss if ss > 0 else 0.0
    return r2, beta


print(f"pooled {len(runs)} runs, turn-region rows (tilt>15) = {int(m.sum())}", flush=True)
print(f"resid std [roll,pitch,yaw] = {np.round(resid[m].std(0),3)}  (this is what we must explain)", flush=True)
for axis, y, lab in [(0, resid[m, 0], "ROLL resid"), (2, resid[m, 2], "YAW resid")]:
    print(f"\n== {lab} (std {y.std():.3f}) ==", flush=True)
    for names in [["w_roll", "w_pitch", "w_yaw"], ["c_roll", "c_pitch", "c_yaw"],
                  ["wp*wy", "wr*wy", "wr*wp"],
                  ["w_roll", "w_pitch", "w_yaw", "c_roll", "c_pitch", "c_yaw"],
                  ["w_roll", "w_pitch", "w_yaw", "c_roll", "c_pitch", "c_yaw", "wp*wy", "wr*wy", "wr*wp"]]:
        r2, beta = fit(y, names)
        print(f"  R2={r2:5.2f}  feats={names}", flush=True)
        if r2 > 0.3:
            print(f"          coef={dict(zip(names + ['const'], np.round(beta, 4)))}", flush=True)
