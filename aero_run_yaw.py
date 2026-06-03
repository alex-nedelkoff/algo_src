"""Per-axis yaw weathervane fit (wv_z) — decoupled from the joint kappa fit.

Loads the gentle yaw-doublet runs (d211_yaw_*), reconstructs tau_motor_z from the
measured motor allocation (run_ingredients), and regresses the yaw moment balance
    tau_motor_z = kappa*inertia_z + d_z*w_z - wv_z*v_y
in ISOLATION (aigp.aero_fit.fit_yaw_axis), breaking the kappa/damping collinearity
that washes wv_z out of the joint 3-axis fit. Validates by predicting the held-out
run's control effort (tau_motor_z). Pre-registered filter (speed<4, |w|<6) is applied
inside run_ingredients; the aborted run 195243 + smoke 195437 are excluded.

Usage: python aero_run_yaw.py [norefine]
  norefine: read cached aero_data/cl_nuisance.json (r, lag) instead of re-refining.
"""
import sys, glob, json, os
import numpy as np, pandas as pd
from aero_run_cl import load_mp, run_ingredients, I_RATIO
from aigp.aero_fit import fit_yaw_axis
from aigp.aero_rollout import refine_nuisance

# Pre-registered exclusions: the 2026-06-02 yaw batch was flown at amp 1.0-1.2 and
# DESTABILIZED (spun to ~11 m/s) — excluded at the run level (not on residuals).
# Gentle re-collected runs (amp~0.4) carry 2026-06-03+ timestamps and are kept.
SKIP = ("195243", "195437",
        "d211_yaw_20260602_235646", "d211_yaw_20260602_235714", "d211_yaw_20260602_235752")


def _held_out_effort(g, res, kappa):
    """Predict held-out yaw control effort tau_motor_z from the fitted model.
    Returns (rmse, r2, r2_demeaned). r2_demeaned removes the per-run constant
    yaw-trim offset (a calibration nuisance) so it scores the sideslip-driven
    VARIATION — the weathervane — which is what we are actually identifying."""
    W, WD, V, TZ = g["w"], g["wd"], g["v"], g["tau_motor"][:, 2]
    inertia_z = WD[:, 2] + (I_RATIO[1] - I_RATIO[0]) * W[:, 0] * W[:, 1]
    pred = res.get("bias", 0.0) + kappa * inertia_z + res["d_z"] * W[:, 2] - res["wv_z"] * V[:, 1]
    rmse = float(np.sqrt(np.mean((TZ - pred) ** 2)))
    ss = np.sum((TZ - pred) ** 2); tot = np.sum((TZ - TZ.mean()) ** 2)
    r2 = 1 - ss / tot if tot > 0 else 0.0
    tzd = TZ - TZ.mean(); pdd = pred - pred.mean()
    ssd = np.sum((tzd - pdd) ** 2); totd = np.sum(tzd ** 2)
    r2_dm = 1 - ssd / totd if totd > 0 else 0.0
    return rmse, r2, r2_dm


def main():
    norefine = "norefine" in sys.argv[1:]
    files = [f for f in sorted(glob.glob("aero_data/d211_yaw_*.parquet")) if not any(s in f for s in SKIP)]
    if len(files) < 2:
        print(f"need >=2 yaw runs, found {len(files)}: {[os.path.basename(f) for f in files]}")
        return
    dfs = {f: pd.read_parquet(f) for f in files}
    mp = load_mp()
    held = files[-1]; train = files[:-1]
    print(f"train {len(train)} yaw runs, held-out {os.path.basename(held)}")

    if norefine:
        nz = json.load(open("aero_data/cl_nuisance.json"))
        r = np.asarray(nz["r"]); lag = int(nz["lag"])
    else:
        print("refining lever-arm r + latency lag (force-only)...", flush=True)
        out = refine_nuisance([dfs[f] for f in train], I_RATIO, 1.0, mp, restarts=2, w_moment=0.0)
        r = np.asarray(out["r"]); lag = int(out["lag"])
    print(f"r={r.round(3)} lag={lag}")

    ing = {f: run_ingredients(dfs[f], mp, r, lag) for f in files}
    for f in files:
        g = ing[f]
        sb = np.abs(g["v"][:, 1]).max() if g["kept"] else 0.0
        wz = np.abs(g["w"][:, 2]).max() if g["kept"] else 0.0
        print(f"  {os.path.basename(f):30s} kept {g['kept']:4d}/{g['total']:<4d}  |v_y|max={sb:4.2f}  |w_z|max={wz:4.2f}")

    tr = [ing[f] for f in train if ing[f]["kept"] > 0]
    W = np.vstack([g["w"] for g in tr]); WD = np.vstack([g["wd"] for g in tr])
    V = np.vstack([g["v"] for g in tr]); TZ = np.concatenate([g["tau_motor"][:, 2] for g in tr])
    print(f"\npooled train samples: {len(TZ)}")

    # (1) FREE per-axis fit — kappa local to the yaw axis (decoupled from roll/pitch).
    # fit_intercept absorbs the constant yaw-trim torque (mixer-imbalance residual).
    free = fit_yaw_axis(W, WD, V, TZ, I_RATIO, kappa=None)
    print(f"FREE bias={free['bias']:+.4f} kappa_z={free['kappa']:+.4f} d_z={free['d_z']:+.4f} "
          f"wv_z={free['wv_z']:+.5f} R2={free['r2']:.3f} cond={free['cond']:.1f}")

    # (2) FIXED-kappa fit at kappa=0 — sanity check. The free fit finds the yaw
    # inertia term negligible (kappa_z~0): these heading-hold maneuvers barely
    # excite yaw acceleration, and spline omega_dot_z is noise-dominated. Pinning
    # kappa=0 should reproduce wv_z, confirming it does not lean on the inertia term.
    kfix = 0.0
    fix = fit_yaw_axis(W, WD, V, TZ, I_RATIO, kappa=kfix)
    print(f"FIX  bias={fix['bias']:+.4f} kappa={kfix:+.4f}(fixed) d_z={fix['d_z']:+.4f} "
          f"wv_z={fix['wv_z']:+.5f} R2={fix['r2']:.3f} cond={fix['cond']:.1f}")

    # held-out control-effort prediction (the validation)
    print("\nHELD-OUT control-effort prediction:")
    g = ing[held]
    if g["kept"] > 0:
        for name, res, kap in (("free", free, free["kappa"]), ("fix", fix, kfix)):
            rmse, r2, r2_dm = _held_out_effort(g, res, kap)
            print(f"  [{name}] tau_z RMSE={rmse:.4f} R2={r2:.3f} R2_demeaned={r2_dm:.3f}  "
                  f"(|v_y|max={np.abs(g['v'][:, 1]).max():.2f}, n={g['kept']})")
    else:
        print("  held-out run had 0 samples after the envelope filter")

    # persist
    try:
        d = json.load(open("aero_data/sim_aero_cl.json"))
    except Exception:
        d = {}
    d["wv_z_yawfit"] = {"free": free, "fixed": fix, "kappa_fixed": kfix,
                        "held_out": os.path.basename(held), "n_train": int(len(TZ)),
                        "r": [float(x) for x in r], "lag": int(lag)}
    json.dump(d, open("aero_data/sim_aero_cl.json", "w"), indent=2)
    print("\nupdated aero_data/sim_aero_cl.json (wv_z_yawfit block)")


if __name__ == "__main__":
    main()
