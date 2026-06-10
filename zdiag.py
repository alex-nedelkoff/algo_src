"""Altitude during the approach: morning pass run vs afternoon fail runs."""
import numpy as np
from aigp.recorder import load_run
for rid, label in [("20260610T123458_fly_gate2", "MORNING PASS"),
                   ("20260610T144807_fly_gate2", "AFTERNOON FAIL")]:
    d = load_run(rf"C:\Users\alexj\Documents\vq_data\{rid}")
    tu = d["t_us"]; _, u = np.unique(tu, return_index=True); u = np.sort(u)
    t = tu[u]/1e6; t = t - t[0]; P = d["pos"][u]
    print(f"\n{label} ({rid}): z_ned (down+) over approach")
    for tq in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12]:
        i = np.argmin(np.abs(t - tq))
        print(f"  t={tq:2d} z={P[i,2]:+6.2f} (alt change {-(P[i,2]-P[0,2]):+5.2f} m)")
