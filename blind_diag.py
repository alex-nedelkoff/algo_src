"""What happened during the blind cruise after gate 1 (run 123458)?"""
import numpy as np
from aigp.recorder import load_run
from aigp.geometry import quat_to_R
from fit_model import qfix
d = load_run(r"C:\Users\alexj\Documents\vq_data\20260610T123458_fly_gate2")
tu = d["t_us"]; _, u = np.unique(tu, return_index=True); u = np.sort(u)
t = tu[u]/1e6; t = t - t[0]
P = d["pos"][u]; V = d["vel"][u]; Q = d["quat"][u]
for tq in [10, 15, 20, 25, 30, 35, 40, 45, 48, 50, 52, 54]:
    i = np.argmin(np.abs(t - tq))
    tilt_true = np.degrees(np.arccos(np.clip(quat_to_R(qfix(Q[i]))[2,2], -1, 1)))
    tilt_live = np.degrees(np.arccos(np.clip(quat_to_R(Q[i])[2,2], -1, 1)))
    yaw = np.degrees(np.arctan2(quat_to_R(Q[i])[1,0], quat_to_R(Q[i])[0,0]))
    print(f"t={tq:3d} pos=[{P[i,0]:+7.1f} {P[i,1]:+7.1f} {P[i,2]:+6.1f}] vb=[{V[i,0]:+5.2f} {V[i,1]:+5.2f} {V[i,2]:+5.2f}] "
          f"yaw={yaw:+6.1f} tilt_true={tilt_true:3.0f} tilt_live={tilt_live:3.0f}")
