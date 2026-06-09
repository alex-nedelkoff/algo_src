"""wv_validate.py -- does the matched model reproduce the live high-sideslip angular dynamics? 1-step
omega prediction (rate loop + weathervane) vs actual omega[k+1] on a held-out crab run, with CURRENT
vs REFIT weathervane coefficients. PASS = refit roll+yaw RMSE <= current on the dynamic-regime run.
Usage: python wv_validate.py <crab_run_dir> [--roll_wv0 -0.105 --roll_wv1 -0.019 --yaw_wv -0.149]
"""
import sys, json
import numpy as np
from aigp.recorder import load_run
from fit_model import qfix, WFIX
from aigp.geometry import quat_to_R


def argf(flag, d): return float(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else d


run = sys.argv[1]
model = json.load(open("sysid/vq_model.json"))
rl = model["rate_loop"]; G = np.array([rl[k]["gain_G"] for k in ("roll", "pitch", "yaw")])
TAU = np.array([rl[k]["tau_ms"] for k in ("roll", "pitch", "yaw")]) / 1e3
DT = 1.0 / 72.0
d = load_run(run)
_, uidx = np.unique(d["t_us"], return_index=True); uidx = np.sort(uidx)
Q = d["quat"][uidx]; W = d["omega"][uidx] * WFIX; C = d["cmd"][uidx]; vb = d["vel"][uidx]  # odom vel = body


def rollout_omega(roll_wv0, roll_wv1, yaw_wv):
    a = np.exp(-DT / TAU)
    pred = a * W[:-1] + (1 - a) * G * C[:-1, :3]
    pred[:, 0] += (roll_wv0 + roll_wv1 * vb[:-1, 0]) * vb[:-1, 1] * DT
    pred[:, 2] += yaw_wv * vb[:-1, 1] * DT
    return np.sqrt(np.mean((pred - W[1:]) ** 2, axis=0))


cur = rollout_omega(-0.105, -0.019, model["weathervane"]["wv_coeff"])
new = rollout_omega(argf("--roll_wv0", -0.105), argf("--roll_wv1", -0.019), argf("--yaw_wv", -0.149))
print(f"run={run.split(chr(92))[-1]}  rows={len(W)}", flush=True)
print(f"omega RMSE [roll,pitch,yaw]  current={np.round(cur,4)}  refit={np.round(new,4)}", flush=True)
print(f"refit better on roll+yaw: {bool(new[0] <= cur[0] and new[2] <= cur[2])}", flush=True)
