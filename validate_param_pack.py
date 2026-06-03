"""Validate the parameter pack's drag model against HELD-OUT coast flight.
Layer 2 (single-step): predict aero force from the pack's force model on coast samples (motors off =>
  IMU specific force IS aero force), report per-axis RMSE/R2.
Layer 3 (rollout): forward-integrate body velocity under the predicted aero force over each coast burst,
  compare to actual measured velocity, report drift.
The pack cites sim_aero.json for D_x (lit-validated); this re-confirms it carries through end-to-end."""
import glob, json
import numpy as np, pandas as pd
from aigp.aero_dataset import build_targets
from aigp.aero_model import force_features

I_RATIO = np.array([3.7, 1.0, 1.0])
thF = np.array(json.load(open("docs/sim_aero.json"))["theta_F"])   # [D_x,D_y,D_z,C_x,C_y,C_z]
HELD = "aero_data/coast_20260602_171627.parquet"                    # pre-registered held-out (report)


def metrics(y, p):
    y = y.reshape(-1, 3); p = p.reshape(-1, 3)
    rmse = np.sqrt(((y - p) ** 2).mean(0))
    ss = ((y - p) ** 2).sum(0); tot = ((y - y.mean(0)) ** 2).sum(0)
    r2 = 1 - ss / np.where(tot > 0, tot, np.nan)
    return rmse, r2


def predict_force(V):
    return np.array([force_features(V[i]) @ thF for i in range(len(V))])


# ---- Layer 2: held-out single-step ----
df = pd.read_parquet(HELD)
out = build_targets(df, I_RATIO)
m = out["coast"]
V = out["v_body"][m]; A = out["a_aero"][m]
P = predict_force(V)
rmse, r2 = metrics(A, P)
print(f"thF (D_x,D_y,D_z,C_x,C_y,C_z) = {thF.round(3)}")
print(f"\nLAYER 2 — held-out single-step on {HELD.split('/')[-1]} ({m.sum()} coast samples)")
for i, ax in enumerate("xyz"):
    print(f"  axis {ax}: RMSE={rmse[i]:.3f} m/s^2   R2={r2[i]:+.3f}")
print(f"  speed range: {np.linalg.norm(V,axis=1).min():.1f}-{np.linalg.norm(V,axis=1).max():.1f} m/s")

# ---- Layer 3: rollout over each coast burst (x-axis drag is the validated term) ----
print(f"\nLAYER 3 — velocity rollout vs actual (per coast burst, x-axis)")
errs = []
for f in sorted(glob.glob("aero_data/coast_*.parquet")):
    df = pd.read_parquet(f); out = build_targets(df, I_RATIO)
    m = out["coast"]
    if m.sum() < 8:
        continue
    t = df["t"].to_numpy()[m]; V = out["v_body"][m]
    vx = V[:, 0].copy(); vx_pred = np.empty_like(vx); vx_pred[0] = vx[0]
    for k in range(1, len(vx)):
        dt = t[k] - t[k - 1]
        a = force_features(np.array([vx_pred[k-1], V[k-1, 1], V[k-1, 2]])) @ thF
        vx_pred[k] = vx_pred[k - 1] + a[0] * dt
    e = abs(vx_pred[-1] - vx[-1]); errs.append(e)
    print(f"  {f.split('/')[-1]:34s} v_x {vx[0]:+.1f}->{vx[-1]:+.1f} m/s, pred end {vx_pred[-1]:+.1f}  |err|={e:.2f}")
print(f"\n  mean final-v_x rollout error: {np.mean(errs):.2f} m/s over {len(errs)} bursts")
