"""fit_wv_v2.py -- joint refit: MIMO rate loop (A,B) + speed-dependent weathervane v2.

Findings behind this (wv_brake_fit1-5, 06-10): the braking-snap regime shows NO vbx~0 weathervane
blowup (dedicated collect_vq_brake data, both branches); instead the wv coefficient is a function of
TOTAL airspeed, decaying/sign-flipping with |v| -- which unifies the old directional roll fit
(decay in disguise), WV-DYNAMIC's high-v flip, and the const yaw -0.149 (mid-curve value).

Model v2 (mirror-symmetric: coeff even in v, *vby odd):
    wv_ax = (a_ax + b_ax * min(|v|, VCLIP)) * v_body_y       ax in {roll, yaw}
Joint fit: ONE lstsq per axis with columns [om(3), wcmd(3), vby*dt, vby*min(|v|,VCLIP)*dt] -- the
iterate-and-subtract scheme DIVERGES (A,B reabsorb the wv each round via vby<->omega collinearity).
Pitch is fit without wv columns (mirror symmetry: pitch is even in vby). A,B symmetrized after.
A,B set = fit_rate_mimo AUTO (collect_vq + corner + deploy + hr) + the new brake runs.

Validation gates printed: turn-band 1-step RMSE on corner runs (must stay ~<=[0.06,0.04,0.06]),
residual g(|v|) -> ~0 after fit, brake-window anchored rollout RMSE old vs new.
Writes: vq_model.json 'rate_loop_mimo' (updated) + 'weathervane_v2' + 'dr_rate_disturbance'
(tilt-binned residual RMS = measured DR spec). Old 'weathervane' key kept for back-compat.
Usage: python fit_wv_v2.py [--ridge 1.0]
"""
import glob
import json
import sys

import numpy as np

from aigp.geometry import quat_to_R
from aigp.recorder import load_run, data_root
from fit_model import WFIX, qfix

DT = 1.0 / 72.0
BLK = 36
VCLIP = 8.0
RID = float(sys.argv[sys.argv.index("--ridge") + 1]) if "--ridge" in sys.argv else 1.0

model = json.load(open("sysid/vq_model.json"))
rl = model["rate_loop"]
G = np.array([rl[k]["gain_G"] for k in ("roll", "pitch", "yaw")])
TAU = np.array([rl[k]["tau_ms"] for k in ("roll", "pitch", "yaw")]) / 1e3
a_diag = np.exp(-DT / TAU)
PRIOR = np.vstack([np.diag(a_diag), np.diag((1 - a_diag) * G)])


def load_clean(run_dir):
    d = load_run(run_dir)
    tu = d["t_us"]; t = d["t_wall"]; cs = d["coll_seq"]
    _, u = np.unique(tu, return_index=True); u = np.sort(u)
    t, cs = t[u], cs[u]
    W = d["omega"][u] * WFIX; C = d["cmd"][u][:, :3]; vb = d["vel"][u]; Q = d["quat"][u]
    tt = tu[u] / 1e6
    ok = (t > 3.5) & ~np.isnan(vb).any(1) & ~np.isnan(W).any(1) & ~np.isnan(C).any(1) & ~np.isnan(Q).any(1)
    idx = np.flatnonzero(ok)
    if len(idx) < BLK + 1:
        return None
    cs0 = cs[idx[0]]
    bad = np.flatnonzero(cs[idx] != cs0)
    if len(bad):
        idx = idx[: bad[0]]
    if len(idx) < BLK + 1:
        return None
    return dict(W=W[idx], C=C[idx], vb=vb[idx], tt=tt[idx], Q=Q[idx],
                run=run_dir.replace("\\", "/").split("/")[-1])


def wv_term(vb, p):
    """(n,3) weathervane omega increment per step (rad/s): coeff(|v|)*vby*DT on roll & yaw."""
    spd = np.minimum(np.linalg.norm(vb, axis=1), VCLIP)
    out = np.zeros((len(vb), 3))
    out[:, 0] = (p["roll"]["a"] + p["roll"]["b"] * spd) * vb[:, 1] * DT
    out[:, 2] = (p["yaw"]["a"] + p["yaw"]["b"] * spd) * vb[:, 1] * DT
    return out


def fit_joint(runs_dat):
    """One lstsq: Y = [om,cmd,vby*dt,vby*spdc*dt] @ theta. Ridge only on the A,B block (toward the
    diagonal first-order PRIOR); wv columns unpenalized (well-identified at n~70k)."""
    Xs, Ys = [], []
    for d in runs_dat:
        W, C, vb, tt = d["W"], d["C"], d["vb"], d["tt"]
        m = np.diff(tt) < 1.4 * DT
        spdc = np.minimum(np.linalg.norm(vb[:-1], axis=1), VCLIP)
        f = np.column_stack([vb[:-1, 1] * DT, vb[:-1, 1] * spdc * DT])
        Xs.append(np.hstack([W[:-1][m], C[:-1][m], f[m]]))
        Ys.append(W[1:][m])
    X = np.vstack(Xs); Y = np.vstack(Ys)
    reg = np.diag([RID] * 6 + [1e-9] * 2)
    P8 = np.vstack([PRIOR, np.zeros((2, 3))])                # ridge target: diagonal prior, wv free
    theta = np.linalg.solve(X.T @ X + reg, X.T @ Y + RID * P8)
    A = theta[:3].T; B = theta[3:6].T
    M = np.diag([-1.0, 1.0, -1.0])
    A = (A + M @ A @ M) / 2; B = (B + M @ B @ M) / 2
    p = {"roll": {"a": float(theta[6, 0]), "b": float(theta[7, 0])},
         "yaw": {"a": float(theta[6, 2]), "b": float(theta[7, 2])}}
    # pitch wv columns: mirror-violating for pitch -> discarded (theta[6:,1] reported only)
    return A, B, p, len(Y), (float(theta[6, 1]), float(theta[7, 1]))


def block_g(runs_dat, A, B):
    """Block-mean residual -> g(|v|) points (per axis), weighted; then weighted linear fit a+b*|v|."""
    pts = {0: [], 2: []}
    sig_rows = []
    for d in runs_dat:
        W, C, vb, tt, Q = d["W"], d["C"], d["vb"], d["tt"], d["Q"]
        resid = (W[1:] - (W[:-1] @ A.T + C[:-1] @ B.T)) / DT
        dt_ok = np.diff(tt) < 1.4 * DT
        calm = (np.abs(W[:-1]) < 6).all(1)
        for s0 in range(0, len(resid) - BLK, BLK):
            sl = slice(s0, s0 + BLK)
            if not (dt_ok[sl].all() and calm[sl].all()):
                continue
            r = resid[sl].mean(0); v = vb[sl].mean(0)
            spd = float(np.linalg.norm(vb[sl], axis=1).mean())
            tilt = float(np.degrees(np.arccos(np.clip(quat_to_R(qfix(Q[s0 + BLK // 2]))[2, 2], -1, 1))))
            sig_rows.append((r, v, spd, tilt))
            if abs(v[1]) > 0.5:
                for ax in (0, 2):
                    pts[ax].append((spd, r[ax], v[1]))
    fit = {}
    for ax, nm in ((0, "roll"), (2, "yaw")):
        P = np.array(pts[ax])                       # spd, resid, vby
        g_obs = P[:, 1] / P[:, 2]                   # per-block implied coeff
        w = P[:, 2] ** 2                            # weight by vby^2 (signal strength)
        X = np.column_stack([np.ones(len(P)), np.minimum(P[:, 0], VCLIP)])
        WD = w[:, None]
        sol = np.linalg.lstsq(X * np.sqrt(WD), g_obs * np.sqrt(w), rcond=None)[0]
        fit[nm] = {"a": float(sol[0]), "b": float(sol[1]), "n": int(len(P))}
    return fit, sig_rows


def turn_band_rmse(dat, A, B, p):
    W, C, vb, tt, Q = dat["W"], dat["C"], dat["vb"], dat["tt"], dat["Q"]
    tilt = np.degrees(np.arccos(np.clip([quat_to_R(qfix(q))[2, 2] for q in Q], -1, 1)))
    pred = W[:-1] @ A.T + C[:-1] @ B.T + wv_term(vb[:-1], p)
    e = pred - W[1:]
    m = (np.diff(tt) < 1.4 * DT) & (tilt[:-1] > 15) & (tilt[:-1] < 30)
    return np.sqrt(np.mean(e[m] ** 2, axis=0)) if m.sum() > 20 else np.full(3, np.nan)


def brake_rollout(dat, A, B, p, K=72):
    W, C, vb = dat["W"], dat["C"], dat["vb"]
    err2 = np.zeros(3); cnt = 0
    om = W[0].copy()
    wv = wv_term(vb, p)
    for k in range(len(W) - 1):
        if k % K == 0:
            om = W[k].copy()
        om = A @ om + B @ C[k] + wv[k]
        err2 += (om - W[k + 1]) ** 2; cnt += 1
    return err2, cnt


root = str(data_root())
AB_RUNS = (sorted(glob.glob(root + "/*_collect_vq")) + sorted(glob.glob(root + "/*_corner_speed"))
           + sorted(glob.glob(root + "/*_vq_deploy*")) + sorted(glob.glob(root + "/*_collect_vq_hr"))
           + sorted(glob.glob(root + "/*_collect_vq_brake")))
G_RUNS = (sorted(glob.glob(root + "/*_collect_vq_brake")) + sorted(glob.glob(root + "/*_collect_vq_crab"))
          + sorted(glob.glob(root + "/*_corner_speed")) + sorted(glob.glob(root + "/*_collect_vq"))
          + sorted(glob.glob(root + "/*_collect_vq_ramp")) + sorted(glob.glob(root + "/*_collect_vq_lateral")))
AB_DAT = [x for x in (load_clean(r) for r in AB_RUNS) if x]
G_DAT = [x for x in (load_clean(r) for r in G_RUNS) if x]
print(f"A,B pool {len(AB_DAT)} runs; g(|v|) pool {len(G_DAT)} runs; ridge={RID}", flush=True)

A, B, p, n, pwv = fit_joint(AB_DAT)
print(f"joint fit: pitch-wv cols (discarded, should be ~0): a={pwv[0]:+.3f} b={pwv[1]:+.3f}", flush=True)
fitres, sig_rows = block_g(G_DAT, A, B)
print(f"block-mean implied g (should ~= wv2 params): roll a{fitres['roll']['a']:+.3f} b{fitres['roll']['b']:+.3f} "
      f"(fit {p['roll']['a']:+.3f},{p['roll']['b']:+.3f}) | yaw a{fitres['yaw']['a']:+.3f} b{fitres['yaw']['b']:+.3f} "
      f"(fit {p['yaw']['a']:+.3f},{p['yaw']['b']:+.3f})", flush=True)
eig = np.abs(np.linalg.eigvals(A))
print(f"\n|eig(A)|={np.round(np.sort(eig)[::-1], 4)}  n_trans={n}", flush=True)
print(f"wv2 final: roll coeff(|v|) = {p['roll']['a']:+.3f} {p['roll']['b']:+.3f}*min(|v|,{VCLIP})  "
      f"(0m/s: {p['roll']['a']:+.2f}, 8m/s: {p['roll']['a'] + 8 * p['roll']['b']:+.2f})", flush=True)
print(f"           yaw  coeff(|v|) = {p['yaw']['a']:+.3f} {p['yaw']['b']:+.3f}*min(|v|,{VCLIP})  "
      f"(0m/s: {p['yaw']['a']:+.2f}, 8m/s: {p['yaw']['a'] + 8 * p['yaw']['b']:+.2f})", flush=True)

# --- validation gates ---
OLD = {"roll": {"a": -0.105, "b": 0.0}, "yaw": {"a": model["weathervane"]["wv_coeff"], "b": 0.0}}
A0 = np.array(model["rate_loop_mimo"]["A"], float); B0 = np.array(model["rate_loop_mimo"]["B"], float)
print("\n== turn-band 1-step RMSE (corner validation runs; gate <= ~[0.06,0.04,0.06]) ==", flush=True)
for rid in ("135925", "125746"):
    dat = next((x for x in G_DAT if rid in x["run"]), None)
    if dat:
        print(f"  {dat['run']}: old={np.round(turn_band_rmse(dat, A0, B0, OLD), 3)} "
              f"new={np.round(turn_band_rmse(dat, A, B, p), 3)}", flush=True)
print("== brake-run anchored 1s rollout omega RMSE (old vs new) ==", flush=True)
e_o = np.zeros(3); e_n = np.zeros(3); c = 0
for dat in G_DAT:
    if "brake" not in dat["run"]:
        continue
    eo, n1 = brake_rollout(dat, A0, B0, OLD); en, _ = brake_rollout(dat, A, B, p)
    e_o += eo; e_n += en; c += n1
print(f"  pooled brake runs: old={np.round(np.sqrt(e_o / c), 3)} new={np.round(np.sqrt(e_n / c), 3)}", flush=True)

# --- DR disturbance spec from final residuals ---
R = np.array([r for r, _, _, _ in sig_rows]); V = np.array([v for _, v, _, _ in sig_rows])
SPD = np.array([s for _, _, s, _ in sig_rows]); TL = np.array([t for _, _, _, t in sig_rows])
spdc = np.minimum(SPD, VCLIP)
R2 = R.copy()
R2[:, 0] -= (p["roll"]["a"] + p["roll"]["b"] * spdc) * V[:, 1]
R2[:, 2] -= (p["yaw"]["a"] + p["yaw"]["b"] * spdc) * V[:, 1]
dr = []
print("\n== dr_rate_disturbance (tilt-binned RMS of post-fit block-mean residual, rad/s^2) ==", flush=True)
for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 45), (45, 90)]:
    m = (TL >= lo) & (TL < hi)
    if m.sum() < 10:
        continue
    rms = np.sqrt(np.mean(R2[m] ** 2, axis=0))
    dr.append({"tilt_deg": [lo, hi], "sigma": [float(x) for x in rms], "n": int(m.sum())})
    print(f"  tilt[{lo:2d},{hi:2d}): {np.round(rms, 2)} (n={int(m.sum())})", flush=True)

model["rate_loop_mimo"] = {"A": A.tolist(), "B": B.tolist(),
                           "note": "om[k+1]=A@om+B@wcmd+weathervane_v2; joint refit w/ speed-dep wv (06-10); incl brake runs; symmetrized; dt=1/72",
                           "n_transitions": int(n), "ridge": RID}
model["weathervane_v2"] = {
    "form": "om_ax += (a + b*min(|v|,vclip)) * v_body_y * dt, ax in {roll,yaw}",
    "roll": {"a": p["roll"]["a"], "b": p["roll"]["b"]},
    "yaw": {"a": p["yaw"]["a"], "b": p["yaw"]["b"]},
    "vclip": VCLIP,
    "note": "speed-dependent wv, unifies directional-roll fit + WV-DYNAMIC flip + const yaw (06-10 brake-data refit); REFUTES vbx~0 braking-wv blowup",
}
model["dr_rate_disturbance"] = {
    "bins": dr,
    "block_s": 0.5,
    "note": "measured sustained unmodeled-moment field (block-mean residual RMS by TRUE tilt) -- DR retrain should inject per-axis bias noise of this magnitude, ~0.5-1s correlation",
}
json.dump(model, open("sysid/vq_model.json", "w"), indent=2)
print("\nwrote rate_loop_mimo + weathervane_v2 + dr_rate_disturbance -> sysid/vq_model.json", flush=True)
