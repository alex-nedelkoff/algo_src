"""yaw_envelope.py -- OFFLINE yaw-agility envelope on the matched plant (MIMO + weathervane_v2).

Question (COR-125/fly_gate2 + general high-turn maneuvering): what sustained flat-turn yaw rate is
stable, as a function of forward speed, and how much do the two known stabilizers buy:
  - KD rate damping on roll/pitch (corner_speed --kd 0.3, live-validated)
  - ZVD shaping of the yaw-rate setpoint (EXP-20's ring killer; shapes the ONSET + reversals)
Protocol per cell: spawn at speed, fly the fly_gate2-style law (level attitude + speed hold + yaw_ref
advancing at wz_cmd; yaw_ref REVERSES at t=5 = scan-reversal stress), 10 s. Survive = true tilt < 60
throughout. Reuses closed_loop_corner's solved plant<->live-frame adapter + math (do NOT re-derive).

CAVEAT: the hover yaw-spin lethality (deploy3) was never fit -- the plant may UNDER-model the v=0
column; treat v=0 results as optimistic until validated live.
Usage: python scripts/sysid/yaw_envelope.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "sysid"))

import closed_loop_corner as clc                      # noqa: E402  (main-guarded; adapter + math)
from closed_loop_corner import (                      # noqa: E402
    quat_to_R, tilt_deg, desired_attitude, mat_to_quat, attitude_error_quat,
    collective_accel, accel_to_thrust_norm)
from sim.dynamics.vq_matched import VQMatchedDynamics  # noqa: E402

import json                                            # noqa: E402

DT = 1.0 / 72.0
KP_ATT = np.array([0.7, 1.6, 1.0]); KP_YAW = 3.0; KD_YAW = 0.3
KP_Z = 1.8; KD_Z = 3.0; WMAX = 4.0
TILTMAX = np.tan(np.radians(15)) * 9.81
T_END = 10.0; T_REV = 5.0

model = json.load(open(ROOT / "sysid" / "vq_model.json"))
r = json.load(open(ROOT / "sysid" / "sim_response.json"))
RG = np.array([r["rate_gain_axes"]["roll"], r["rate_gain_axes"]["pitch"], r["rate_gain_axes"]["yaw"]])
HOVER = r["hover_thrust"]; KA = r["k_a"]

# ZVD 3-impulse shaper on the yaw-rate setpoint (corner_speed constants: zeta=0.14, Td/2=55 ms)
TD2 = 0.055; _K = np.exp(-0.14 * np.pi / np.sqrt(1 - 0.14 ** 2)); _D = 1 + 2 * _K + _K * _K
ZVD_A = [1 / _D, 2 * _K / _D, _K * _K / _D]; ZVD_T = [0.0, TD2, 2 * TD2]


def run_cell(v_tgt, wz_cmd, kd, zvd):
    dyn = VQMatchedDynamics(model, dt=DT, frame="NED")
    s = dyn.reset(1)
    s[0, clc.QUAT] = np.array([0.0, 0.0, 0.0, 1.0])    # live-frame identity spawn
    if v_tgt:
        s[0, clc.VEL] = quat_to_R(s[0, clc.QUAT]) @ np.array([v_tgt, 0.0, 0.0])
    yaw_ref = 0.0
    ybuf = []                                          # (t, yr_raw) for the ZVD shaper
    peak_tilt = 0.0; yaw_int = 0.0
    n = int(T_END / DT)
    for k in range(n):
        t = k * DT
        q_true = s[0, clc.QUAT]; om_true = s[0, clc.OMEGA]
        vel_body = quat_to_R(q_true).T @ s[0, clc.VEL]
        quat = q_true[[3, 0, 1, 2]]                    # live io frame (adapter, closed_loop_corner)
        om = om_true * np.array([1.0, -1.0, 1.0])
        tilt = tilt_deg(q_true); peak_tilt = max(peak_tilt, tilt)
        if tilt > 60.0:
            return dict(ok=False, t=t, peak=peak_tilt, eff=yaw_int / max(wz_cmd * t, 1e-9))
        wz_now = wz_cmd if t < T_REV else -wz_cmd      # reversal stress (scan/high-turn)
        yaw_ref += wz_now * DT
        Rl = quat_to_R(quat)
        yaw_cur = float(np.arctan2(Rl[1, 0], Rl[0, 0]))
        fwd = -np.array([np.cos(yaw_cur), np.sin(yaw_cur)]); lat = np.array([-fwd[1], fwd[0]])
        v_al = float(vel_body[0]); v_lat = float(vel_body[1])
        a_al = float(np.clip(1.2 * (v_tgt - v_al), -0.8, 0.5))
        a_lat = float(np.clip(1.4 * (0.0 - v_lat), -2.0, 2.0))
        a = np.zeros(3); a[:2] = a_al * fwd + a_lat * lat
        a[2] = KP_Z * (0.0 - s[0, 2]) + KD_Z * (0.0 - s[0, 5])
        nh = float(np.linalg.norm(a[:2]))
        if nh > TILTMAX:
            a[:2] *= TILTMAX / nh
        q_des = mat_to_quat(desired_attitude(a, yaw_ref))
        w = KP_ATT * attitude_error_quat(quat, q_des)
        w[0] -= kd * om[0]; w[1] -= kd * om[1]
        yr_raw = KP_YAW * ((yaw_ref - yaw_cur + np.pi) % (2 * np.pi) - np.pi) - KD_YAW * om[2]
        if zvd:
            ybuf.append((t, yr_raw))
            while ybuf and t - ybuf[0][0] > 0.3:
                ybuf.pop(0)
            out = 0.0
            for aa, dd in zip(ZVD_A, ZVD_T):
                tgt = t - dd; val = yr_raw; best = 1e9
                for tb, vb in ybuf:
                    e = abs(tb - tgt)
                    if e < best:
                        best = e; val = vb
                out += aa * val
            yr_raw = out
        w[2] = yr_raw
        thr = accel_to_thrust_norm(collective_accel(a, quat), HOVER, KA)
        action = np.array([thr, *np.clip(w / RG, -WMAX, WMAX)])
        yaw_int += abs(om[2]) * DT
        s = dyn.step(s, action[None, :])
    return dict(ok=True, t=T_END, peak=peak_tilt, eff=yaw_int / (wz_cmd * T_END))


SPEEDS = [0.0, 1.0, 2.0, 3.0, 5.0]
RATES = [0.3, 0.6, 1.0, 1.5, 2.0, 3.0]
VARIANTS = [("base", 0.0, False), ("kd.3", 0.3, False), ("zvd", 0.0, True), ("kd+zvd", 0.3, True)]

for vn, kd, zvd in VARIANTS:
    print(f"\n== {vn} ==  cell: OK(peakTilt)/X(t=fail)   rows=v_fwd, cols=wz_cmd rad/s", flush=True)
    print("  v\\wz " + "".join(f"{wz:>10.1f}" for wz in RATES), flush=True)
    for v in SPEEDS:
        row = f"  {v:4.1f} "
        for wz in RATES:
            o = run_cell(v, wz, kd, zvd)
            row += (f"  OK({o['peak']:3.0f})" if o["ok"] else f"  X(t={o['t']:3.1f})").rjust(10)
        print(row, flush=True)
print("\n(eff = integrated |yaw rate| / commanded; reversal at t=5 s stresses the ring. "
      "v=0 column optimistic: hover-spin lethality never fit.)", flush=True)
