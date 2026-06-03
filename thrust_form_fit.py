"""Resolve the thrust-form conflict: fit T(u) from the collective sweep.
thrust_accel = -f_z (body) since thrust is always along body -z (tilt-independent). Use collective-
dominant samples (4 motors ~equal) at low speed (aero negligible). Fit power p in T = k * mean_u^p
(log-log slope) and compare explicit linear (g=u) vs quadratic (g=u^2) fits."""
import glob
import numpy as np, pandas as pd

rows_u, rows_T = [], []
for f in sorted(glob.glob("aero_data/collective_*.parquet")):
    df = pd.read_parquet(f)
    u = df[["u0", "u1", "u2", "u3"]].to_numpy()
    fz = df["fz"].to_numpy()
    v = df[["vx", "vy", "vz"]].to_numpy()
    spd = np.linalg.norm(v, axis=1)
    umean = u.mean(1); ustd = u.std(1)
    T = -fz                                          # thrust accel (body -z), tilt-independent
    keep = (ustd < 0.03) & (umean > 0.08) & (T > 1.0) & (spd < 3.0)   # collective, above noise, low aero
    rows_u.append(umean[keep]); rows_T.append(T[keep])
U = np.concatenate(rows_u); Th = np.concatenate(rows_T)
print(f"n={len(U)} clean collective samples; u range {U.min():.3f}-{U.max():.3f}, T range {Th.min():.1f}-{Th.max():.1f} m/s^2")

# 1) log-log power
lp = np.polyfit(np.log(U), np.log(Th), 1)
print(f"\nlog-log fit: T ~ u^{lp[0]:.2f}   (1.0=linear, 2.0=quadratic)")

# 2) explicit form comparison (through-origin and affine)
def fit(Phi, y, name):
    th, *_ = np.linalg.lstsq(Phi, y, rcond=None); pred = Phi @ th
    ss = np.sum((y-pred)**2); tot = np.sum((y-y.mean())**2); r2 = 1-ss/tot
    rmse = np.sqrt(np.mean((y-pred)**2))
    print(f"  {name:28s} R2={r2:.4f} RMSE={rmse:.3f}  coef={np.round(th,3)}")
    return r2
one = np.ones(len(U))
print("explicit fits (T vs mean_u):")
fit(np.stack([U], 1), Th, "linear  T=a*u")
fit(np.stack([U*U], 1), Th, "quadratic T=a*u^2")
fit(np.stack([U, one], 1), Th, "affine-linear T=a*u+b")
fit(np.stack([U*U, one], 1), Th, "affine-quad  T=a*u^2+b")
fit(np.stack([U*U, U, one], 1), Th, "full T=a*u^2+b*u+c")
print(f"\nverdict: power ~ {lp[0]:.1f} -> {'QUADRATIC (motor_model correct; sim_dynamics power=1 was the rough fit)' if lp[0] > 1.5 else 'LINEAR' if lp[0] < 1.3 else 'BETWEEN linear and quadratic'}")
