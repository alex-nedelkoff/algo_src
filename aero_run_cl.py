"""Closed-loop fit/validate (T7): identify lateral drag D_y, thrust-coupled drag (D=D0+D_T*T),
the weathervane moment + damping, from the 2-1-1 doublet battery using the MEASURED control
allocation (tau_motor reconstructed from motor outputs). The control effort IS the measurement.

Key idea for the unknown inertia scale kappa (mass-normalized pitch inertia): the moment balance
   tau_motor = kappa*(I_ratio*wd + w x (I_ratio*w)) - moment_features(v,w) @ theta_M
is LINEAR in [kappa, theta_M], so we fit them jointly by least-squares with tau_motor as the target.
Force (horizontal, thrust-independent): a_aero_{x,y} = -(D0+D_T*T)*v - C*v|v| per axis.

Usage: python aero_run_cl.py [norefine]
"""
import sys, glob, json, time, os
import numpy as np, pandas as pd
from aigp.aero_dataset import build_targets_cl
from aigp.aero_model import moment_features_cl, MOMENT_COLS_CL
from aigp.aero_rollout import refine_nuisance

I_RATIO = np.array([3.7, 1.0, 1.0])


def load_mp():
    p = json.load(open("docs/motor_model.json"))
    for k in ("k_f", "k_q", "L"):
        p[k] = float(p[k])
    return p


SPD_MAX = 4.0          # pre-registered envelope filter: drop destabilized samples (>~2x trim)
TILT_W_MAX = 6.0       # rad/s body-rate cap (spin-up / tumble guard)


def run_ingredients(df, mp, r, lag):
    """build_targets_cl with kappa=1; recover tau_motor = inertia - tau_aero(kappa=1).
    Applies a uniform, physics-based sample filter (NOT residual-based): horizontal speed < SPD_MAX
    and |omega| < TILT_W_MAX — excludes the destabilized/tumbling regime the near-trim model doesn't cover."""
    out = build_targets_cl(df, I_RATIO, 1.0, mp, r=r, lag=int(lag))
    v, w, wd, T = out["v_body"], out["omega"], out["omega_dot"], out["T"]
    inertia = I_RATIO * wd + np.cross(w, I_RATIO * w)        # (N,3)
    tau_motor = inertia - out["tau_aero"]                    # since tau_aero(1)=inertia-tau_motor
    a = out["a_aero"]
    keep = (np.linalg.norm(v[:, :2], axis=1) < SPD_MAX) & (np.linalg.norm(w, axis=1) < TILT_W_MAX)
    return dict(v=v[keep], w=w[keep], wd=wd[keep], T=T[keep], a=a[keep],
                inertia=inertia[keep], tau_motor=tau_motor[keep], kept=int(keep.sum()), total=len(v))


def fit_axis_force(v_ax, T, a_ax):
    Phi = np.stack([-v_ax, -T * v_ax, -v_ax * np.abs(v_ax)], axis=1)   # D0, D_T, C
    th, *_ = np.linalg.lstsq(Phi, a_ax, rcond=None)
    pred = Phi @ th
    ss = np.sum((a_ax - pred) ** 2); tot = np.sum((a_ax - a_ax.mean()) ** 2)
    return th, (1 - ss / tot if tot > 0 else 0.0)


def fit_moment(ing_list):
    """Joint [kappa, theta_M(6)] via tau_motor = kappa*inertia - moment_features@theta_M."""
    rows, b = [], []
    for ing in ing_list:
        v, w, inertia, tau_motor = ing["v"], ing["w"], ing["inertia"], ing["tau_motor"]
        for i in range(len(v)):
            Phi_m = moment_features_cl(v[i], w[i])            # (3,6)
            A_i = np.hstack([inertia[i][:, None], -Phi_m])    # (3,7): [kappa | -features]
            rows.append(A_i); b.append(tau_motor[i])
    A = np.vstack(rows); b = np.concatenate(b)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    pred = A @ x; ss = np.sum((b - pred) ** 2); tot = np.sum((b - b.mean()) ** 2)
    return x[0], x[1:], (1 - ss / tot if tot > 0 else 0.0)      # kappa, theta_M, r2(on tau_motor)


def moment_predict_effort(ing, kappa, thM):
    """Predict the control effort tau_motor the model implies; return RMSE/R2 vs measured (held-out)."""
    v, w, inertia, tau_motor = ing["v"], ing["w"], ing["inertia"], ing["tau_motor"]
    pred = np.array([kappa * inertia[i] - moment_features_cl(v[i], w[i]) @ thM for i in range(len(v))])
    rmse = float(np.sqrt(np.mean((tau_motor - pred) ** 2)))
    ss = np.sum((tau_motor - pred) ** 2); tot = np.sum((tau_motor - tau_motor.mean(0)) ** 2)
    return rmse, (1 - ss / tot if tot > 0 else 0.0)


def main():
    norefine = len(sys.argv) > 1 and sys.argv[1] == "norefine"
    SKIP = ("195243", "195437")   # 195243 = aborted tilt-99 tumble; 195437 = early smoke (pre-registered exclusions)
    files = [f for f in sorted(glob.glob("aero_data/d211_*.parquet")) if not any(s in f for s in SKIP)]
    dfs = {f: pd.read_parquet(f) for f in files}
    mp = load_mp()
    lat = [f for f in files if "d211_lat" in f]; yaw = [f for f in files if "d211_yaw" in f]
    held = [lat[-1], yaw[-1]]; train = [f for f in files if f not in held]
    print(f"train {len(train)} runs, held-out {[os.path.basename(h) for h in held]}")

    if norefine:
        nz = json.load(open("aero_data/cl_nuisance.json"))   # cached refine result
        r = np.asarray(nz["r"]); lag = int(nz["lag"])
    else:
        print("refining lever-arm r + latency lag (force-only)...", flush=True)
        out = refine_nuisance([dfs[f] for f in train], I_RATIO, 1.0, mp, restarts=2, w_moment=0.0)
        r = np.asarray(out["r"]); lag = int(out["lag"])
        json.dump({"r": [float(x) for x in r], "lag": lag}, open("aero_data/cl_nuisance.json", "w"))
    print(f"r={r.round(3)}  lag={lag}")

    ing = {f: run_ingredients(dfs[f], mp, r, lag) for f in files}
    for f in files:
        g = ing[f]
        print(f"  {os.path.basename(f):30s} kept {g['kept']}/{g['total']} samples")
    tr = [ing[f] for f in train]
    V = np.vstack([g["v"] for g in tr]); T = np.concatenate([g["T"] for g in tr])
    A = np.vstack([g["a"] for g in tr])

    thx, r2x = fit_axis_force(V[:, 0], T, A[:, 0])
    thy, r2y = fit_axis_force(V[:, 1], T, A[:, 1])
    kappa, thM, r2m = fit_moment(tr)
    print(f"\nFORCE x: D0={thx[0]:+.3f} D_T={thx[1]:+.4f} C={thx[2]:+.4f}  R2={r2x:.3f}")
    print(f"FORCE y: D0={thy[0]:+.3f} D_T={thy[1]:+.4f} C={thy[2]:+.4f}  R2={r2y:.3f}   <- D_y")
    print(f"MOMENT: kappa={kappa:.4f}  R2(tau_motor)={r2m:.3f}")
    for c, val in zip(MOMENT_COLS_CL, thM):
        print(f"   {c:5s} {val:+.4f}")

    # held-out validation
    print("\nHELD-OUT:")
    for h in held:
        g = ing[h]
        # force y single-step
        Phi = np.stack([-g["v"][:, 1], -g["T"] * g["v"][:, 1], -g["v"][:, 1] * np.abs(g["v"][:, 1])], axis=1)
        py = Phi @ thy; rmse_fy = float(np.sqrt(np.mean((g["a"][:, 1] - py) ** 2)))
        rmse_e, r2_e = moment_predict_effort(g, kappa, thM)
        sb = np.abs(g["v"][:, 1]).max()
        print(f"  {os.path.basename(h):28s} sideslip|max={sb:.2f}  force_y RMSE={rmse_fy:.3f}  "
              f"control-effort RMSE={rmse_e:.3f} R2={r2_e:.3f}")

    res = dict(D0=[thx[0], thy[0]], D_T=[thx[1], thy[1]], C=[thx[2], thy[2]],
               r2_force_xy=[r2x, r2y], kappa=float(kappa), moment_cols=MOMENT_COLS_CL,
               theta_M=[float(x) for x in thM], r2_moment=float(r2m),
               r=[float(x) for x in r], lag=int(lag),
               wv_z=float(thM[MOMENT_COLS_CL.index("wv_z")]),
               d=[float(thM[MOMENT_COLS_CL.index(f"d_{a}")]) for a in "xyz"],
               note="closed-loop 2-1-1 fit; D_y from powered horizontal force (thrust-independent); "
                    "weathervane+damping from joint kappa/theta_M fit on measured tau_motor")
    json.dump(res, open("aero_data/sim_aero_cl.json", "w"), indent=2)
    print("\nwrote aero_data/sim_aero_cl.json")


if __name__ == "__main__":
    main()
