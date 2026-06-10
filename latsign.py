"""Determine sign of camera-right world direction vs -vb[1] from fly_gate2 run (lateral slide phase)."""
import numpy as np
from aigp.recorder import load_run
from aigp.geometry import quat_to_R
from fit_model import qfix
d = load_run(r"C:\Users\alexj\Documents\vq_data\20260610T123213_fly_gate2")
tu = d["t_us"]; _, u = np.unique(tu, return_index=True); u = np.sort(u)
t = tu[u]/1e6; P = d["pos"][u]; V = d["vel"][u]; Q = d["quat"][u]
m = (t > t[0]+55) & (t < t[0]+72) & ~np.isnan(P).any(1) & ~np.isnan(Q).any(1)
Pw = P[m]; Vb = V[m]; Qs = Q[m]; ts = t[m]
dP = np.gradient(Pw, ts, axis=0)                    # world velocity (pos is world NED)
# live-frame yaw (same arctan2 the controllers use, on the RAW live quat)
yaws = np.array([np.arctan2(quat_to_R(q)[1,0], quat_to_R(q)[0,0]) for q in Qs])
fwd = -np.column_stack([np.cos(yaws), np.sin(yaws)])             # camera dir (controller convention)
right_w = np.column_stack([-fwd[:,1], fwd[:,0]])                 # candidate camera-right world
v_right_world = np.einsum("ij,ij->i", dP[:,:2], right_w)
v_cam_right_body = -Vb[:,1]
c = np.corrcoef(v_right_world, v_cam_right_body)[0,1]
print(f"corr(world-right-vel, -vb_y) = {c:+.3f}  (n={m.sum()})")
print(f"  -> lat_world = [-fwd_y, fwd_x] is camera-{'RIGHT' if c>0 else 'LEFT'}; damping v_lat = {'-vb[1]' if c>0 else '+vb[1]'}")
print(f"  mean |v_right_world|={np.abs(v_right_world).mean():.2f} |vb_y|={np.abs(Vb[:,1]).mean():.2f}")
