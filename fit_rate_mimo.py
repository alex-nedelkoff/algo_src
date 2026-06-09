"""fit_rate_mimo.py -- refit the VQ rate loop as a MIMO linear map (3x3 A,B) so it captures the
roll<->yaw cross-coupling the diagonal first-order loop misses in a coordinated turn (diag_coupling:
linear rate+cmd -> R2 0.94-0.99; gyroscopic w x w negligible). Model: om[k+1] = A@om[k] + B@wcmd[k]
+ weathervane(v_body_y). Fit on the weathervane-SUBTRACTED target so the existing wv term stays.
Frame per wv_validate (WFIX omega, raw cmd, odom-vel=body). Reports banded R2 + leave-one-run-out
test R2, and writes 'rate_loop_mimo' {A,B} into sysid/vq_model.json. Usage: python fit_rate_mimo.py <run_dir>..."""
import sys, json
import numpy as np
from aigp.recorder import load_run
from fit_model import qfix, WFIX
from aigp.geometry import quat_to_R

model = json.load(open("sysid/vq_model.json")); rl = model["rate_loop"]
G = np.array([rl[k]["gain_G"] for k in ("roll", "pitch", "yaw")])
TAU = np.array([rl[k]["tau_ms"] for k in ("roll", "pitch", "yaw")]) / 1e3
DT = 1.0 / 72.0
rw0, rw1, yw = -0.105, -0.019, model["weathervane"]["wv_coeff"]


def transitions(run):
    d = load_run(run)
    _, u = np.unique(d["t_us"], return_index=True); u = np.sort(u)
    W = d["omega"][u] * WFIX; C = d["cmd"][u][:, :3]; vb = d["vel"][u]; Q = d["quat"][u]
    tilt = np.degrees(np.arccos(np.clip([quat_to_R(q)[2, 2] for q in Q], -1, 1)))
    wv = np.zeros_like(W[:-1])
    wv[:, 0] = (rw0 + rw1 * vb[:-1, 0]) * vb[:-1, 1] * DT
    wv[:, 2] = yw * vb[:-1, 1] * DT
    return W[:-1], C[:-1], W[1:] - wv, tilt[:-1]      # Xw, Xc, target(=W[k+1]-wv), tilt[k]


def fit(Xw, Xc):                                       # target = A@om + B@wcmd  -> stack [om,wcmd]
    X = np.hstack([Xw, Xc])
    return X                                            # caller solves per-axis


a_diag = np.exp(-DT / TAU)
PRIOR = np.vstack([np.diag(a_diag), np.diag((1 - a_diag) * G)])   # (6,3) diagonal first-order prior


def solve(X, Y, ridge=0.0):
    if ridge > 0:                                       # ridge toward the diagonal first-order prior (conditioning + stability)
        return np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ Y + ridge * PRIOR)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)        # (6,3): rows 0:3=A^T, 3:6=B^T
    return beta


def r2(X, Y, beta, m=None):
    if m is not None:
        X, Y = X[m], Y[m]
    if len(Y) < 5:
        return np.full(3, np.nan)
    pred = X @ beta; ss = np.sum((Y - Y.mean(0)) ** 2, axis=0)
    return 1 - np.sum((Y - pred) ** 2, axis=0) / np.where(ss > 0, ss, 1)


import glob as _glob
from aigp.recorder import data_root
RID = float(sys.argv[sys.argv.index("--ridge") + 1]) if "--ridge" in sys.argv else 0.0
args = [a for a in sys.argv[1:] if not a.startswith("--") and a not in (str(RID),)]
if args and args[0] == "AUTO":
    root = str(data_root())
    runs = sorted(_glob.glob(root + "/*_collect_vq")) + sorted(_glob.glob(root + "/*_corner_speed"))
else:
    runs = args
TR = [transitions(r) for r in runs]
Xw = np.vstack([t[0] for t in TR]); Xc = np.vstack([t[1] for t in TR])
Y = np.vstack([t[2] for t in TR]); tilt = np.concatenate([t[3] for t in TR])
X = np.hstack([Xw, Xc])
beta = solve(X, Y, RID)
A = beta[:3].T; B = beta[3:].T                          # (3,3) each: om[k+1]=A@om+B@wcmd (+wv)
eig = np.abs(np.linalg.eigvals(A))
print(f"ridge={RID}  |eig(A)| = {np.round(np.sort(eig)[::-1],4)}  (max {eig.max():.4f}; <1 = open-loop stable)", flush=True)
print(f"diag(B) MIMO={np.round(np.diag(B),3)}  vs prior (1-a)G={np.round((1-a_diag)*G,3)}  (should be same ballpark/sign)", flush=True)

print(f"pooled {len(runs)} runs, {len(Y)} transitions", flush=True)
print(f"MIMO train R2  all={np.round(r2(X,Y,beta),3)}  tilt<15={np.round(r2(X,Y,beta,tilt<15),3)}  tilt>15={np.round(r2(X,Y,beta,tilt>15),3)}", flush=True)
# baseline diagonal for comparison
a = np.exp(-DT / TAU)
base = a * Xw + (1 - a) * G * Xc
bss = lambda m: 1 - np.sum((Y[m]-base[m])**2,0)/np.maximum(np.sum((Y[m]-Y[m].mean(0))**2,0),1e-9)
print(f"diag  train R2  all={np.round(bss(np.ones(len(Y),bool)),3)}  tilt<15={np.round(bss(tilt<15),3)}  tilt>15={np.round(bss(tilt>15),3)}", flush=True)
# leave-one-run-out test R2 (honesty: no overfit)
if len(runs) > 1:
    tk2 = []
    for i in range(len(runs)):
        tr = [TR[j] for j in range(len(runs)) if j != i]
        Xtr = np.hstack([np.vstack([t[0] for t in tr]), np.vstack([t[1] for t in tr])])
        Ytr = np.vstack([t[2] for t in tr]); bi = solve(Xtr, Ytr, RID)
        Xte = np.hstack([TR[i][0], TR[i][1]]); Yte = TR[i][2]; tlt = TR[i][3]
        m = tlt > 15
        if m.sum() > 5:
            tk2.append(r2(Xte, Yte, bi, m))
    tk2 = np.array(tk2)
    print(f"LORO test R2 (tilt>15, {len(tk2)} runs w/ turn): min={np.round(tk2.min(0),2)} median={np.round(np.median(tk2,0),2)}", flush=True)
print("\nA=\n", np.round(A, 4), "\nB=\n", np.round(B, 4), flush=True)

model["rate_loop_mimo"] = {"A": A.tolist(), "B": B.tolist(),
                           "note": "om[k+1]=A@om+B@wcmd+weathervane; MIMO refit incl corner runs (roll<->yaw coupling); WFIX omega, raw cmd frame",
                           "n_transitions": int(len(Y)), "source_runs": [r.split(chr(92))[-1] for r in runs]}
json.dump(model, open("sysid/vq_model.json", "w"), indent=2)
print("\nwrote rate_loop_mimo -> sysid/vq_model.json", flush=True)
