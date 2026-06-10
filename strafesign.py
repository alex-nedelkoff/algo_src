"""Pin strafe sign: during attempt-23 lock (t8.5-13) gate drifted RIGHT in image = drone moving
camera-LEFT. Sign of vb[1] in that window => vb[1] sign for camera-left motion."""
import numpy as np
from aigp.recorder import load_run
import glob
runs = sorted(glob.glob(r"C:\Users\alexj\Documents\vq_data\*_fly_gate2"))
d = load_run(runs[-1])
tu = d["t_us"]; _, u = np.unique(tu, return_index=True); u = np.sort(u)
t = tu[u]/1e6; t = t - t[0]; V = d["vel"][u]
m = (t > 8.5) & (t < 13.0)
print(f"run {runs[-1].split(chr(92))[-1]} lock window: vb1 mean={V[m,1].mean():+.3f} (std {V[m,1].std():.3f}) "
      f"vb0 mean={V[m,0].mean():+.3f}")
print("=> camera-LEFT motion has vb[1] sign:", "+" if V[m,1].mean() > 0 else "-")
