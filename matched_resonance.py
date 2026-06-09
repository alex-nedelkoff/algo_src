"""matched_resonance.py -- does vq_matched reproduce the corner over-tilt angular dynamics?
1-step omega prediction (rate loop + weathervane, frame per wv_validate: WFIX omega, odom-vel=body)
vs actual omega[k+1], DECOMPOSED by actual tilt band (gentle vs over-tilt) and with the MODEL gain G
vs the CONTROLLER gain RG (sim_response) -- to separate a gain-mismatch confound from a true
resonance the first-order rate loop would miss. Omega-only + tilt-from-quat => no qfix/attitude-integ
frame risk. Usage: python matched_resonance.py <corner_run_dir>"""
import sys, json
import numpy as np
from aigp.recorder import load_run
from fit_model import qfix, WFIX
from aigp.geometry import quat_to_R

run = sys.argv[1]
model = json.load(open("sysid/vq_model.json"))
rl = model["rate_loop"]
G = np.array([rl[k]["gain_G"] for k in ("roll", "pitch", "yaw")])
TAU = np.array([rl[k]["tau_ms"] for k in ("roll", "pitch", "yaw")]) / 1e3
sr = json.load(open("sysid/sim_response.json"))
RG = np.array([sr["rate_gain_axes"][k] for k in ("roll", "pitch", "yaw")])
DT = 1.0 / 72.0
rw0, rw1, yw = -0.105, -0.019, model["weathervane"]["wv_coeff"]

d = load_run(run)
_, uidx = np.unique(d["t_us"], return_index=True); uidx = np.sort(uidx)
Q = d["quat"][uidx]; W = d["omega"][uidx] * WFIX; C = d["cmd"][uidx]; vb = d["vel"][uidx]
tilt = np.degrees(np.arccos(np.clip([quat_to_R(q)[2, 2] for q in Q], -1, 1)))   # actual tilt from recorded quat
spd = np.linalg.norm(vb[:, :2], axis=1)                                          # body horizontal speed


def pred(gain):
    a = np.exp(-DT / TAU)
    p = a * W[:-1] + (1 - a) * gain * C[:-1, :3]
    p[:, 0] += (rw0 + rw1 * vb[:-1, 0]) * vb[:-1, 1] * DT
    p[:, 2] += yw * vb[:-1, 1] * DT
    return p


def rmse(p, mask):
    e = (p - W[1:])[mask[:-1]]
    return np.sqrt(np.mean(e ** 2, axis=0)) if e.size else np.full(3, np.nan)


mimo = model.get("rate_loop_mimo")
A = np.array(mimo["A"]) if mimo else None; Bm = np.array(mimo["B"]) if mimo else None


def pred_mimo():
    p = W[:-1] @ A.T + C[:-1, :3] @ Bm.T
    p[:, 0] += (rw0 + rw1 * vb[:-1, 0]) * vb[:-1, 1] * DT
    p[:, 2] += yw * vb[:-1, 1] * DT
    return p


pg, pr = pred(G), pred(RG)
pm = pred_mimo() if mimo else None
print(f"run={run.split(chr(92))[-1]}  rows={len(W)}  tilt[min/med/max]={tilt.min():.0f}/{np.median(tilt):.0f}/{tilt.max():.0f}  spd_max={spd.max():.1f}", flush=True)
print("                       1-step omega RMSE [roll,pitch,yaw] rad/s        | mean|W|", flush=True)
for lab, m in [("tilt<15 (gentle)", tilt < 15), ("15-30 (turn)", (tilt >= 15) & (tilt < 30)),
               ("30-60 (over-tilt)", (tilt >= 30) & (tilt < 60)), ("60+  (tumble)", tilt >= 60)]:
    n = int(m[:-1].sum())
    if n < 5:
        print(f"  {lab:18s} n={n:5d}  (too few)", flush=True); continue
    wmag = np.abs(W[1:][m[:-1]]).mean(0)
    mm = f"  MIMO:{np.round(rmse(pm,m),3)}" if mimo else ""
    print(f"  {lab:18s} n={n:5d}  diagG:{np.round(rmse(pg,m),3)}{mm}  |W|:{np.round(wmag,2)}", flush=True)

# --- multi-step OMEGA rollout (RL-relevant: stability + tracking over the turn; no attitude integ) ---
if mimo:
    turn = np.where(tilt > 15)[0]
    if len(turn) > 40:
        k0 = int(turn[0]); H = min(72, len(W) - 1 - k0)        # ~1 s open-loop omega rollout from turn onset
        om = W[k0].copy(); errs = []
        for j in range(H):
            wv = np.array([(rw0 + rw1 * vb[k0 + j, 0]) * vb[k0 + j, 1] * DT, 0.0, yw * vb[k0 + j, 1] * DT])
            om = A @ om + Bm @ C[k0 + j, :3] + wv
            errs.append(np.abs(om - W[k0 + j + 1]))
        errs = np.array(errs)
        print(f"  MIMO {H}-step omega rollout from turn onset: final|err|={np.round(errs[-1],3)} "
              f"mean|err|={np.round(errs.mean(0),3)} (vs |W|~{np.round(np.abs(W[k0:k0+H]).mean(0),2)}); stable={bool(np.all(np.isfinite(om)) and errs.max()<5)}", flush=True)
print("\nInterpretation: if RMSE stays small in 'turn'/'over-tilt' AND ~RG<<G error => rate loop is faithful (gain-only gap).", flush=True)
print("If RMSE blows up vs |W| even with RG => first-order model MISSES the dynamics there (resonance) => fix before RL.", flush=True)
