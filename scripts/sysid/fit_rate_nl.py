"""Fit nonlinear rate-loop residual terms on tumble-excitation data.
Residual r_k = om_true[k+1] - MIMO(om_true[k], wcmd_k). Candidates per axis:
quadratic self-damping |om_ax|om_ax, |om| cross-damping, gyro coupling (om x om terms)."""
import json, sys, glob
import numpy as np
sys.path.insert(0, ".")
from sim.dynamics.vq_matched import VQMatchedDynamics, POS, VEL, QUAT, OMEGA
from scripts.sysid.closed_loop_policy import quat_to_R

DT = 1.0 / 72.0
model = json.load(open("sysid/vq_model.json"))
dyn = VQMatchedDynamics(model, dt=DT, frame="NED")
SGN = np.array([1.0, -1.0, 1.0])

OM0, OM1, RES, W = [], [], [], []
for p in sorted(glob.glob("/tmp/vq_replay/*_tumble.npz")):
    d = np.load(p)
    t = d["t_us"] / 1e6
    _, uidx = np.unique(d["t_us"], return_index=True); uidx = np.sort(uidx)
    t = t[uidx]; cmd = d["cmd"][uidx]; vel_b = d["vel"][uidx]
    quat_live = d["quat"][uidx]; om_live = d["omega"][uidx]
    n = len(t) - 1
    for k in range(n):
        dtk = t[k+1] - t[k]
        if not (0.5*DT < dtk < 2.5*DT): continue
        q0 = quat_live[k][[1,2,3,0]]; nq = np.linalg.norm(q0)
        if nq < 1e-6 or np.isnan(nq): continue
        s = dyn.reset(1)
        s[0, QUAT] = q0/nq; s[0, VEL] = quat_to_R(s[0, QUAT]) @ vel_b[k]
        om0 = om_live[k]*SGN
        s[0, OMEGA] = om0; s[0, POS] = 0.0
        a = np.array([[cmd[k,3], cmd[k,0], cmd[k,1], cmd[k,2]]])
        s2 = dyn.step(s, a, dt=dtk)
        OM0.append(om0); W.append(cmd[k,0:3])
        RES.append((om_live[k+1]*SGN - s2[0, OMEGA]) / dtk)   # residual ang-accel
om0 = np.array(OM0); res = np.array(RES); w = np.array(W)
wm = np.linalg.norm(om0, axis=1)
print(f"transitions {len(om0)}, |om|>1.5: {(wm>1.5).sum()}")
m = wm > 1.0
gyro = np.stack([om0[:,1]*om0[:,2], om0[:,0]*om0[:,2], om0[:,0]*om0[:,1]], axis=1)
for ax, name in enumerate("roll pitch yaw".split()):
    # basis: quad self-damp, cross |om| damp, gyro
    X = np.stack([-np.abs(om0[m,ax])*om0[m,ax], -wm[m]*om0[m,ax], gyro[m,ax]], axis=1)
    k_, *_ = np.linalg.lstsq(X, res[m,ax], rcond=None)
    pred = X @ k_
    r0 = np.sqrt(np.mean(res[m,ax]**2)); r1 = np.sqrt(np.mean((res[m,ax]-pred)**2))
    print(f"{name}: kq={k_[0]:+.4f} kx={k_[1]:+.4f} kg={k_[2]:+.4f}  resid {r0:.1f} -> {r1:.1f} rad/s^2 ({100*(1-r1/r0):.0f}%)")
