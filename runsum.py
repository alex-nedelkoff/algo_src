import numpy as np
from aigp.recorder import load_run
d = load_run(r"C:\Users\alexj\Documents\vq_data\20260610T153253_fly_gate2")
tu = d["t_us"]; _, u = np.unique(tu, return_index=True); u = np.sort(u)
t = tu[u]/1e6; t = t - t[0]
gi = d["gate_idx"][u]; cs = d["coll_seq"][u]
print(f"dur={t[-1]:.1f}s rows={len(u)} gate_idx: start={gi[0]} max={gi.max()} end={gi[-1]}")
for g in range(int(gi.max())+1):
    i = np.argmax(gi >= g)
    print(f"  gi>={g} at t={t[i]:.1f}")
print("collisions:", int(cs[-1]-cs[0]))
