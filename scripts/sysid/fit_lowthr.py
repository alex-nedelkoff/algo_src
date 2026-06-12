"""Fit the low-thr z-force on the DEDICATED clean recording (open-loop thr cuts, no doublets).
Characterize first (residual vs thr/vbz/vpl on tight blocks), then fit candidates."""
import json
import numpy as np

M = json.load(open("sysid/vq_model.json"))
F0, DF = M["thrust"]["f0"], M["thrust"]["df_dthr"]
FLOOR = M["thrust"]["thr_floor"]
q = M["drag_quadratic_body"]; qz, Dz = q["qz"], q["Dz"]
BLOCK = 0.2

d = np.load("/tmp/vq_replay/20260612T105637_lowthr.npz")
t = d["t_wall"] - d["t_wall"][0]

def blocks(arr):
    out, t0 = [], t[0]
    while t0 < t[-1]:
        m = (t >= t0) & (t < t0 + BLOCK)
        if m.sum() >= 3: out.append(arr[m].mean(axis=0))
        t0 += BLOCK
    return np.array(out)

qt = d["quat"][:, [1, 2, 3, 0]]
tilt_r = np.degrees(np.arccos(np.clip(1 - 2*(qt[:,1]**2 + qt[:,2]**2), -1, 1)))
vb = blocks(d["vel"]); acc = blocks(d["acc"]); thr = blocks(d["cmd"][:, 3:4])[:, 0]
om = blocks(np.abs(d["omega"])); tilt = blocks(tilt_r[:, None])[:, 0]
vmag = np.linalg.norm(vb, axis=1); vpl = np.linalg.norm(vb[:, :2], axis=1)
ctrl = (om.max(axis=1) < 1.5) & np.isfinite(acc).all(axis=1)
cut = ctrl & (thr < 0.22)
print(f"blocks {len(vb)}, controlled {ctrl.sum()}, low-thr {cut.sum()}")

c = -(F0 + DF * np.maximum(thr, FLOOR))
rz = acc[:, 2] - (-(Dz + qz*vmag)*vb[:, 2] - c)
rx = acc[:, 0] - (-(0.1 + 0.032*vmag)*vb[:, 0])

print("\nlow-thr blocks: rz vs (thr, vbz, vpl):")
for lo, hi in ((0.0, 0.03), (0.03, 0.12), (0.12, 0.22)):
    m = cut & (thr >= lo) & (thr < hi)
    if m.sum() < 5: continue
    print(f" thr[{lo:.2f},{hi:.2f}): n={m.sum():3d}  vbz {np.median(vb[m,2]):+5.1f}  vpl {np.median(vpl[m]):4.1f}  rz med {np.median(rz[m]):+6.2f}  rx med {np.median(rx[m]):+5.2f}")
print("\nrz vs vbz (thr<0.12, |om|<1.5):")
mm = cut & (thr < 0.12)
for zlo, zhi in ((-8,-3),(-3,-0.5),(-0.5,0.5),(0.5,3),(3,6),(6,9),(9,13)):
    m = mm & (vb[:,2] >= zlo) & (vb[:,2] < zhi)
    if m.sum() < 4: continue
    print(f"  vbz[{zlo:+4.1f},{zhi:+4.1f}): n={m.sum():3d}  rz med {np.median(rz[m]):+6.2f}  vpl med {np.median(vpl[m]):4.1f}  rz/vbz {np.median(rz[m])/np.median(vb[m,2]) if abs(np.median(vb[m,2]))>0.3 else float('nan'):+.2f}")
# candidate fits on the cut blocks
print("\nfits (low-thr blocks):")
for name, col in {"-vbz": -vb[:,2], "-|vbz|vbz": -np.abs(vb[:,2])*vb[:,2], "-vmag*vbz": -vmag*vb[:,2]}.items():
    k, *_ = np.linalg.lstsq(col[cut][:,None], rz[cut], rcond=None)
    rms = np.sqrt(np.mean((rz[cut]-col[cut]*k[0])**2))
    print(f"  {name:10s} k={k[0]:+.4f}  rms {np.sqrt(np.mean(rz[cut]**2)):5.2f} -> {rms:5.2f}")
X = np.stack([-vb[cut,2], -vmag[cut]*vb[cut,2]], axis=1)
k, *_ = np.linalg.lstsq(X, rz[cut], rcond=None)
print(f"  joint Dz={k[0]:+.4f} qz2={k[1]:+.4f}  rms -> {np.sqrt(np.mean((rz[cut]-X@k)**2)):5.2f}")
