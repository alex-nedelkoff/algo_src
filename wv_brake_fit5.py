"""wv_brake_fit5.py -- phase 5 (decisive): from clean CONTROLLED data only (brake/crab/corner/cvq/
ramp/lat block means), quantify
  (1) g_axis(|v|): weathervane coefficient (resid = g*vby) binned by TOTAL airspeed -- tests the
      unification: prior fits (low-v strong ~-0.22, v7 ~0) + new brake data (vbx~0 at speed, g~0)
      are ONE decaying-coefficient curve g(|v|), NOT a vbx-blowup ("braking wv") -- the HANDOFF
      premise. Also pitch-vs-features in the decel regime.
  (2) sigma(tilt): RMS of the block-mean residual left after the wv fit, binned by tilt -- the
      measured magnitude of the unmodeled-moment field, i.e. the DR disturbance spec the retrain
      should randomize over (policy must tolerate biases of THIS size, not exploit model precision).
Usage: python wv_brake_fit5.py
"""
import glob
import json

import numpy as np

from aigp.geometry import quat_to_R
from aigp.recorder import load_run, data_root
from fit_model import WFIX, qfix

DT = 1.0 / 72.0
BLK = 36
model = json.load(open("sysid/vq_model.json"))
A = np.array(model["rate_loop_mimo"]["A"], float)
B = np.array(model["rate_loop_mimo"]["B"], float)


def blocks(run_dir, tag):
    d = load_run(run_dir)
    tu = d["t_us"]; t = d["t_wall"]; cs = d["coll_seq"]
    _, u = np.unique(tu, return_index=True); u = np.sort(u)
    t, cs = t[u], cs[u]
    W = d["omega"][u] * WFIX; C4 = d["cmd"][u]; C = C4[:, :3]; thr = C4[:, 3]
    vb = d["vel"][u]; Q = d["quat"][u]
    tt = tu[u] / 1e6
    ok = (t > 3.5) & ~np.isnan(vb).any(1) & ~np.isnan(W).any(1) & ~np.isnan(C).any(1) & ~np.isnan(Q).any(1)
    idx = np.flatnonzero(ok)
    if len(idx) < BLK + 1:
        return []
    cs0 = cs[idx[0]]
    bad = np.flatnonzero(cs[idx] != cs0)
    if len(bad):
        idx = idx[: bad[0]]
    if len(idx) < BLK + 1:
        return []
    W, C, thr, vb, Q, tt = W[idx], C[idx], thr[idx], vb[idx], Q[idx], tt[idx]
    resid = (W[1:] - (W[:-1] @ A.T + C[:-1] @ B.T)) / DT
    dt_ok = np.diff(tt) < 1.4 * DT
    calm = (np.abs(W[:-1]) < 6).all(1)
    out = []
    for s0 in range(0, len(resid) - BLK, BLK):
        sl = slice(s0, s0 + BLK)
        if not (dt_ok[sl].all() and calm[sl].all()):
            continue
        tilt = np.degrees(np.arccos(np.clip(quat_to_R(qfix(Q[s0 + BLK // 2]))[2, 2], -1, 1)))
        out.append(dict(r=resid[sl].mean(0), vb=vb[sl].mean(0), thr=float(thr[sl].mean()),
                        spd=float(np.linalg.norm(vb[sl], axis=1).mean()), tilt=float(tilt), tag=tag))
    return out


root = str(data_root())
BL = []
for pat, tag in (("/*_collect_vq_brake", "brake"), ("/*_collect_vq_crab", "crab"),
                 ("/*_corner_speed", "corner"), ("/*_collect_vq", "cvq"),
                 ("/*_collect_vq_ramp", "ramp"), ("/*_collect_vq_lateral", "lat")):
    for r in sorted(glob.glob(root + pat)):
        BL += blocks(r, tag)
R = np.array([b["r"] for b in BL]); VB = np.array([b["vb"] for b in BL])
SP = np.array([b["spd"] for b in BL]); TL = np.array([b["tilt"] for b in BL])
vby = VB[:, 1]
print(f"controlled blocks: {len(BL)}", flush=True)

print("\n== (1) g(|v|): resid_ax = g * vby per total-airspeed bin (|vby|>0.5 blocks) ==", flush=True)
print(f"   {'|v| bin':>10} {'n':>5} {'g_roll':>8} {'r2':>5} {'g_yaw':>8} {'r2':>5}", flush=True)
gv = {}
for lo, hi in [(0, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 4.5), (4.5, 5.5), (5.5, 7), (7, 12)]:
    m = (SP >= lo) & (SP < hi) & (np.abs(vby) > 0.5)
    n = int(m.sum())
    if n < 12:
        print(f"   [{lo:4.1f},{hi:4.1f}) {n:>5}  (skip)", flush=True); continue
    row = f"   [{lo:4.1f},{hi:4.1f}) {n:>5}"
    for ax in (0, 2):
        g = float(R[m, ax] @ vby[m] / (vby[m] @ vby[m]))
        r2 = 1 - np.var(R[m, ax] - g * vby[m]) / max(np.var(R[m, ax]), 1e-12)
        row += f" {g:>+8.3f} {r2:>5.2f}"
        gv.setdefault(ax, []).append((0.5 * (lo + hi), g, n))
    print(row, flush=True)
for ax, nm in ((0, "roll"), (2, "yaw")):
    pts = np.array([(v, g) for v, g, _ in gv.get(ax, [])])
    if len(pts) >= 4:
        # exponential-decay fit g(v) = g0*exp(-v/v0) via log on |g| where sign consistent w/ g0
        g0 = pts[0, 1]
        ok = pts[:, 1] / g0 > 0.05
        if ok.sum() >= 3:
            ln = np.log(np.abs(pts[ok, 1]))
            sl, off = np.polyfit(pts[ok, 0], ln, 1)
            print(f"   {nm}: g(|v|) ~ {np.sign(g0) * np.exp(off):+.3f} * exp(-|v|/{-1 / sl:.1f})", flush=True)

print("\n== pitch in decel regime: corr(block resid_pitch, feature) on brake-run blocks ==", flush=True)
mb = np.array([b["tag"] == "brake" for b in BL])
rp = R[mb, 1]; vbb = VB[mb]; thb = np.array([b["thr"] for b in BL])[mb]; spb = SP[mb]
for nm, f in (("vbx", vbb[:, 0]), ("vbz", vbb[:, 2]), ("vbx*thr", vbb[:, 0] * thb),
              ("|vby|", np.abs(vbb[:, 1])), ("|v|", spb)):
    print(f"   corr(resid_pitch, {nm:8s}) = {np.corrcoef(rp, f)[0, 1]:+.3f}", flush=True)
g = float(rp @ vbb[:, 0] / (vbb[:, 0] @ vbb[:, 0]))
print(f"   lsq resid_pitch = {g:+.3f}*vbx on brake blocks (r2 {1 - np.var(rp - g * vbb[:, 0]) / np.var(rp):.2f})", flush=True)

print("\n== (2) DR disturbance spec: per-axis RMS of block-mean residual AFTER current-model wv, by tilt ==", flush=True)
yw = model["weathervane"]["wv_coeff"]; rw0, rw1 = -0.105, -0.019
Rwv = R.copy()
Rwv[:, 0] -= (rw0 + rw1 * VB[:, 0]) * vby
Rwv[:, 2] -= yw * vby
print(f"   {'tilt bin':>10} {'n':>5} {'roll':>6} {'pitch':>6} {'yaw':>6}  (rad/s^2, sustained 0.5s biases)", flush=True)
for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 45), (45, 90)]:
    m = (TL >= lo) & (TL < hi)
    if m.sum() < 10:
        continue
    rms = np.sqrt(np.mean(Rwv[m] ** 2, axis=0))
    print(f"   [{lo:3d},{hi:3d}) {int(m.sum()):>5} {rms[0]:>6.2f} {rms[1]:>6.2f} {rms[2]:>6.2f}", flush=True)
