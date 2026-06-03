"""Anchor the inertia scale kappa from SHARP-PULSE data (pulse_{roll,pitch}_*).

Pulses decouple omega_dot from omega (unlike oscillatory doublets), so the inertia term separates
from damping. Motor torque tau_motor_{x,y} = L*k_f*(s . u^2) is KNOWN via the hover-calibrated k_f,
so this fixes kappa in mass-normalized units (propagating the gravity anchor to the rotational axes).

Roll:  tau_motor_x = kappa*3.7*wd_x        + d_x*w_x - wv_x*v_y + bias   (gyro_x = 0)   -> kappa = c/3.7
Pitch: tau_motor_y = kappa*(wd_y+2.7*wx*wz)+ d_y*w_y - wv_y*v_x + bias                   -> kappa = c

Contact guard: drop samples with |specific force| > FMAG_MAX (gate contact, not aero). Smoothed
finite-diff omega_dot (spline is too noisy). Writes aero_data/kappa.json."""
import glob, json
import numpy as np, pandas as pd
from aigp.aero_dataset import finite_diff_filtered, world_to_body_vel
from aigp.geometry import quat_to_R
from aigp.motor_model import motor_outputs_to_wrench
from aero_run_cl import load_mp, I_RATIO

mp = load_mp()
FMAG_MAX = 30.0   # |specific force| above this = contact, not aero (hover~10, max aero+thrust<~20)
WIN = 5           # finite-diff smoothing on the DEDUPED (~94 Hz telemetry) series


def corr(a, b):
    a = a-a.mean(); b = b-b.mean(); d = np.sqrt((a@a)*(b@b)); return float(a@b/d) if d > 0 else 0


def ingredients(files):
    out = []
    for f in files:
        df = pd.read_parquet(f)
        omega_all = df[["wx", "wy", "wz"]].to_numpy()
        # DEDUP: the loop logs at ~250 Hz but telemetry updates at ~94 Hz, so each gyro value
        # repeats ~2.6x then jumps -> finite-diff aliases into huge noise. Keep only fresh samples.
        fresh = np.ones(len(df), bool)
        fresh[1:] = np.any(np.diff(omega_all, axis=0) != 0, axis=1)
        df = df.iloc[fresh].reset_index(drop=True)
        n = len(df); t = df["t"].to_numpy()
        omega = df[["wx", "wy", "wz"]].to_numpy()
        u = df[["u0", "u1", "u2", "u3"]].to_numpy()
        fmag = np.linalg.norm(df[["fx", "fy", "fz"]].to_numpy(), axis=1)
        vb = np.zeros((n, 3))
        for i in range(n):
            R = quat_to_R(df[["qw", "qx", "qy", "qz"]].iloc[i].to_numpy())
            vb[i] = world_to_body_vel(df[["vx", "vy", "vz"]].iloc[i].to_numpy(), R)
        wd = finite_diff_filtered(t, omega, win=WIN)
        tau = np.array([motor_outputs_to_wrench(u[i], mp)[1] for i in range(n)])
        keep = (np.linalg.norm(vb[:, :2], axis=1) < 4.0) & (np.linalg.norm(omega, axis=1) < 6.0) & (fmag < FMAG_MAX)
        out.append(dict(tau=tau[keep], w=omega[keep], wd=wd[keep], v=vb[keep],
                        kept=int(keep.sum()), total=n, contact=int((fmag >= FMAG_MAX).sum())))
    return out


def fit(A, y, names):
    th, *_ = np.linalg.lstsq(A, y, rcond=None); pred = A@th
    ss = np.sum((y-pred)**2); tot = np.sum((y-y.mean())**2); r2 = 1-ss/tot if tot > 0 else 0
    print("    " + "  ".join(f"{n}={v:+.5f}" for n, v in zip(names, th)) + f"  R2={r2:+.3f}")
    return th


def axis_kappa(ings, idx, gyro_fn, vcol, scale, label):
    tau = np.concatenate([g["tau"][:, idx] for g in ings])
    w = np.concatenate([g["w"] for g in ings]); wd = np.concatenate([g["wd"] for g in ings])
    v = np.concatenate([g["v"] for g in ings])
    inertia = scale*wd[:, idx] + gyro_fn(w)
    print(f"  {label}: n={len(tau)} std(tau)={tau.std():.4f} std(inertia)={inertia.std():.3f} "
          f"corr(tau,inertia)={corr(tau,inertia):+.3f}")
    th = fit(np.stack([inertia, w[:, idx], -v[:, vcol], np.ones(len(tau))], 1), tau,
             ["kap_term", "d", "wv", "bias"])
    return th[0] / (scale if label == "ROLL" else 1.0)   # roll: scale=3.7 -> kappa; pitch: scale=1


roll_f = sorted(glob.glob("aero_data/pulse_roll_*.parquet"))
pitch_f = sorted(glob.glob("aero_data/pulse_pitch_*.parquet"))
print(f"roll runs {len(roll_f)}, pitch runs {len(pitch_f)}; k_f={mp['k_f']:.3f} L={mp['L']} FMAG_MAX={FMAG_MAX}")
ri = ingredients(roll_f); pi = ingredients(pitch_f)
for g, f in list(zip(ri, roll_f)) + list(zip(pi, pitch_f)):
    print(f"  {f.split('/')[-1]:34s} kept {g['kept']}/{g['total']}  contact-dropped {g['contact']}")

kap_roll = axis_kappa(ri, 0, lambda w: 0.0*w[:, 0], 1, 3.7, "ROLL")
kap_pitch = axis_kappa(pi, 1, lambda w: 2.7*w[:, 0]*w[:, 2], 0, 1.0, "PITCH")
print(f"\nkappa(roll)/I_ratio = {kap_roll:+.5f}   kappa(pitch=I_z) = {kap_pitch:+.5f}")
print("(roll has 3.7x inertia -> small real omega_dot -> noise-dominated -> unreliable; pitch=I_z is the anchor)")
kappa = kap_pitch   # pitch = I_y = I_z directly (the yaw anchor)

# --- pin k_q (yaw torque coeff) from sharp YAW pulses, with kappa fixed ---
# yaw Euler: kappa*inertia_z = k_q*(sz.u^2) - d_z*w_z + wv_z*v_y + bias  ; inertia_z = wd_z - 2.7*wx*wy
kq_seed = mp["k_q"]
yaw_f = sorted(glob.glob("aero_data/pulse_yaw_*.parquet"))
yi = ingredients(yaw_f)
print(f"\nyaw pulse runs {len(yaw_f)}:")
for g, f in zip(yi, yaw_f):
    print(f"  {f.split('/')[-1]:34s} kept {g['kept']}/{g['total']}  contact-dropped {g['contact']}")
tau = np.concatenate([g["tau"] for g in yi]); w = np.concatenate([g["w"] for g in yi])
wd = np.concatenate([g["wd"] for g in yi]); v = np.concatenate([g["v"] for g in yi])
szu2 = tau[:, 2] / kq_seed                                  # sz . u^2  (= tau_z / seeded k_q)
inertia_z = wd[:, 2] - 2.7 * w[:, 0] * w[:, 1]
target = kappa * inertia_z
print(f"  YAW: n={len(target)} corr(kappa*inertia_z, sz.u^2)={corr(target, szu2):+.3f}  std(target)={target.std():.4f}")
th = fit(np.stack([szu2, w[:, 2], -v[:, 1], np.ones(len(target))], 1), target, ["k_q", "d_z", "wv_z", "bias"])
k_q = float(th[0])
print(f"\n  pinned k_q = {k_q:+.4f}  (seeded {kq_seed:.4f};  k_q/k_f = {k_q/mp['k_f']:.4f})")
print(f"  -> wv_z magnitude scale factor = k_q_pinned/k_q_seed = {k_q/kq_seed:+.3f}")

json.dump({"kappa": float(kappa), "kappa_roll": float(kap_roll), "kappa_pitch": float(kap_pitch),
           "k_q": k_q, "k_q_seed": float(kq_seed), "k_q_scale": float(k_q/kq_seed),
           "method": "sharp-pulse; kappa from pitch via hover-calibrated k_f; k_q from yaw pulses with kappa fixed",
           "fmag_max": FMAG_MAX, "win": WIN},
          open("aero_data/kappa.json", "w"), indent=2)
print(f"\nwrote aero_data/kappa.json  kappa={kappa:+.5f}  k_q={k_q:+.4f}")
