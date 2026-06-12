"""MIMO 2nd-order rate loop fit: om[k+1] = A1@om[k] + A2@om[k-1] + B0@w[k] + B1@w[k-1],
mirror-symmetrized (M=diag(-1,1,-1)), pooled deploy+tumble recordings. Validation:
open-loop omega rollout on run5's ring segment vs live."""
import json, sys, glob
import numpy as np

SGN = np.array([1.0, -1.0, 1.0])
DT = 1/72
runs = sorted(glob.glob("/tmp/vq_replay/2026061*_vq_deploy4/data.npz")) + \
       sorted(glob.glob("/tmp/vq_replay/*_tumble.npz")) + \
       ["/tmp/vq_replay/20260611T000346.npz", "/tmp/vq_replay/20260611T000430.npz"]
X, Y = [], []
for p in runs:
    d = np.load(p)
    t = d["t_us"]/1e6
    _, u = np.unique(d["t_us"], return_index=True); u = np.sort(u)
    om = d["omega"][u] * SGN
    wc = d["cmd"][u][:, 0:3]
    tt = t[u]
    for k in range(1, len(om)-1):
        if 0.5*DT < tt[k+1]-tt[k] < 2.0*DT and 0.5*DT < tt[k]-tt[k-1] < 2.0*DT:
            X.append(np.concatenate([om[k], om[k-1], wc[k], wc[k-1]])); Y.append(om[k+1])
X = np.array(X); Y = np.array(Y)
print(f"transitions: {len(X)} from {len(runs)} runs")
K, *_ = np.linalg.lstsq(X, Y, rcond=None)        # (12,3) -> Y = X @ K
K = K.T                                           # rows = output axis
A1, A2, B0, B1 = K[:, 0:3], K[:, 3:6], K[:, 6:9], K[:, 9:12]
M = np.diag([-1.0, 1.0, -1.0])
A1 = (A1 + M@A1@M)/2; A2 = (A2 + M@A2@M)/2; B0 = (B0 + M@B0@M)/2; B1 = (B1 + M@B1@M)/2
pred = X[:, 0:3]@A1.T + X[:, 3:6]@A2.T + X[:, 6:9]@B0.T + X[:, 9:12]@B1.T
rmse = np.sqrt(np.mean((Y-pred)**2, axis=0))
print(f"sym 2nd-order MIMO 1-step RMSE r/p/y: {rmse.round(3)}")
for nm, mat in (("A1", A1), ("A2", A2), ("B0", B0), ("B1", B1)):
    print(nm, np.array2string(mat, precision=3, suppress_small=True))
# companion-form poles
C = np.zeros((6, 6)); C[0:3, 0:3] = A1; C[0:3, 3:6] = A2; C[3:6, 0:3] = np.eye(3)
ev = np.linalg.eigvals(C)
print("poles |.|:", np.abs(ev).round(3), " freq Hz:", (np.abs(np.angle(ev))/(2*np.pi*DT)).round(1))
json.dump({"A1": A1.tolist(), "A2": A2.tolist(), "B0": B0.tolist(), "B1": B1.tolist(),
           "note": "RING-01 06-12: MIMO 2nd-order rate loop (underdamped ~7 Hz pole pair); "
                   "fit on deploy+tumble pooled, mirror-symmetrized; dt=1/72"},
          open("/tmp/rate_loop_2nd.json", "w"))

# validation rollout: run5 first 1.5 s, omega only (attitude effects ignored -- short horizon)
d = np.load("/tmp/vq_replay/20260612T114220_vq_deploy4/data.npz")
t = d["t_us"]/1e6; _, u = np.unique(d["t_us"], return_index=True); u = np.sort(u)
om = d["omega"][u]*SGN; wc = d["cmd"][u][:, 0:3]
o_prev, o_cur = om[0], om[1]
print("\n k  live_omy   2nd_omy   1st_omy")
mdl = json.load(open("sysid/vq_model.json"))
Am = np.array(mdl["rate_loop_mimo"]["A"]); Bm = np.array(mdl["rate_loop_mimo"]["B"])
o1 = om[1].copy()
for k in range(1, 30):
    o_new = A1@o_cur + A2@o_prev + B0@wc[k] + B1@wc[k-1]
    o1 = Am@o1 + Bm@wc[k]
    o_prev, o_cur = o_cur, o_new
    if k % 2 == 0:
        print(f"{k:3d} {om[k+1][1]:+8.2f} {o_new[1]:+8.2f} {o1[1]:+8.2f}")
