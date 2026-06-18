"""fit_model.py -- Phase-1 grey-box fit of the VQ drone from collected data.

Loads all clean collect_vq rows and fits the cleanly-identifiable dynamics from the BODY-frame IMU
specific force (gravity-excluded) + the rate-interface commands:

  THRUST    f_body_z = -c(thr_cmd)                       (thrust-cmd -> specific-force, c=T/m)
  DRAG      f_body_x = -Dx*v_body_x,  f_body_y = -Dy*v_body_y   (linear rotor drag, body frame)
            (thrust is along body-z, so body-x/y specific force IS the aero drag -- clean)
  RATE LOOP domega/dt = a*w_cmd - b*omega  ->  gain G=a/b, lag tau=1/b   (per axis; the sim's
            rate-interface response, incl. the "explosive" tracking)

Validates DRAG+THRUST accel prediction on a held-out run; writes sysid/vq_model.json (with the
measured loop delays) for the RL training sim to point at. Weathervane (a YAW MOMENT vs sideslip)
shows in the yaw-rate residual -> flagged as a rotational follow-up.
"""
import json
import numpy as np
from aigp.recorder import load_index, load_run
from aigp.geometry import quat_to_R

G = 9.81
PRIOR_DRAG = {"x": 0.338, "y": 0.234}      # from the prior aero sysID (vq_drone_params)


def qfix(q):
    """Correct quaternion convention (found by frame search, corr +1.00): stored[w,x,y,z] slots are
    actually [z,w,x,y] -> true (w,x,y,z) = stored[1,2,3,0]. io_layer reordered an already-wxyz sim q."""
    return np.array([q[1], q[2], q[3], q[0]])


WFIX = np.array([1.0, -1.0, 1.0])    # omega pitch-axis sign matches qfix frame (att search -> 0.46 deg)


def prep(run):
    """DEDUP by odometry t_us (recorder oversamples 170Hz vs 72Hz telemetry -> 55% stale dup rows
    corrupt finite diffs). Then derive accel from KINEMATICS (dV/dt, body frame) -- IMU `acc` is a
    separate-message channel misaligned with attitude, so we drop it. True ~72Hz cadence."""
    d = load_run(run["path"]); t = d["t_wall"]; cs = d["coll_seq"]; tu = d["t_us"]
    if len(t) < 80:
        return None
    _, uidx = np.unique(tu, return_index=True); uidx = np.sort(uidx)    # one row per odometry frame
    for k in d:
        if k != "meta" and hasattr(d[k], "__len__") and len(d[k]) == len(t):
            d[k] = d[k][uidx]
    t = t[uidx]; cs = cs[uidx]; tu = tu[uidx]
    base = cs[(np.abs(t - 4)).argmin()]
    m = (t > 3.5) & (cs == base)
    m &= ~np.isnan(d["vel"]).any(1) & ~np.isnan(d["quat"]).any(1)
    Q, V, W, C = d["quat"][m], d["vel"][m], d["omega"][m] * WFIX, d["cmd"][m]
    tt = tu[m] / 1e6                                                 # SIM time (s), true cadence
    if len(tt) < 50:
        return None
    Rs = np.array([quat_to_R(qfix(q)) for q in Q])                  # corrected attitude
    vb = V                                                          # ODOMETRY velocity is BODY frame already
    Vw = np.einsum("nij,nj->ni", Rs, V)                            # world velocity = R @ v_body
    dVw = np.gradient(Vw, tt, axis=0)                              # world accel (kinematic)
    A = np.einsum("nij,nj->ni", np.transpose(Rs, (0, 2, 1)), dVw - np.array([0, 0, G]))  # body specific force
    Aimu = d["acc"][m]                                              # IMU body specific force (vertical thrust)
    dW = np.gradient(W, tt, axis=0)                                  # domega/dt
    sched = d["meta"].get("schedule") or []

    def _ph(tv):
        for sg in sched:
            if sg["t"][0] <= tv < sg["t"][1]:
                return sg["phase"]
        return "decel"
    ph = np.array([_ph(float(x)) for x in tt])                      # per-run phase (handles schedule changes)
    return dict(vb=vb, A=A, Aimu=Aimu, W=W, C=C, dW=dW, tt=tt, ph=ph, n=len(tt),
                Wk=W[:-1], Wk1=W[1:], Ck=C[:-1], vbyk=vb[:-1, 1], dtk=np.diff(tt))


def fit_drag(vb, A):
    out = {}
    for ax, nm in [(0, "x"), (1, "y")]:
        x = vb[:, ax]; y = A[:, ax]
        M = np.vstack([np.ones_like(x), x]).T
        (off, slope), *_ = np.linalg.lstsq(M, y, rcond=None)
        r2 = 1 - np.var(y - M @ [off, slope]) / np.var(y)
        out[nm] = {"D": float(-slope), "offset": float(off), "r2": float(r2)}
    return out


def fit_thrust(C, A):
    thr = C[:, 3]; fz = A[:, 2]
    M = np.vstack([np.ones_like(thr), thr]).T
    (off, slope), *_ = np.linalg.lstsq(M, fz, rcond=None)
    r2 = 1 - np.var(fz - M @ [off, slope]) / np.var(fz)
    # f_z = -c  ->  c = -(off + slope*thr);  hover thr where c=g
    thr_hover = (-G - off) / slope
    return {"f0": float(off), "df_dthr": float(slope), "dCacc_dthr": float(-slope),
            "thr_hover_pred": float(thr_hover), "r2": float(r2)}


def fit_rate(Wk, Wk1, Ck, vby, dt):
    """Discrete one-step fit  omega[k+1] = a*omega[k] + b*wcmd[k] (+g*v_body_y on yaw) -- clean, no
    gradient. Derive continuous (tau, gain) so the simulator is dt-agnostic; yaw g -> weathervane g/dt."""
    out = {}; wv = None
    for ax, nm in [(0, "roll"), (1, "pitch"), (2, "yaw")]:
        if nm == "yaw":
            M = np.vstack([Wk[:, 2], Ck[:, 2], vby]).T
            sol, *_ = np.linalg.lstsq(M, Wk1[:, 2], rcond=None)
            a, b, g = sol; wv = float(g / dt)
        else:
            M = np.vstack([Wk[:, ax], Ck[:, ax]]).T
            sol, *_ = np.linalg.lstsq(M, Wk1[:, ax], rcond=None)
            a, b = sol
        r2 = 1 - np.var(Wk1[:, ax] - M @ sol) / np.var(Wk1[:, ax])
        tau = float(-dt / np.log(a)) if 0 < a < 1 else float(dt / max(1 - a, 1e-3))
        gain = float(b / (1 - a)) if abs(1 - a) > 1e-6 else None
        out[nm] = {"alpha": float(a), "tau_ms": tau * 1e3, "gain_G": gain, "r2": float(r2)}
    return out, wv


def phase_of(t):
    return np.select([t < 3, t < 10, t < 17, t < 23, t < 31],
                     ["entry", "speed_sweep", "doublets", "lateral_steps", "slalom"], "decel")


def _lin(x, y):
    M = np.vstack([np.ones_like(x), x]).T
    (off, slope), *_ = np.linalg.lstsq(M, y, rcond=None)
    r2 = 1 - np.var(y - M @ [off, slope]) / np.var(y)
    return {"D": float(-slope), "offset": float(off), "r2": float(r2), "n": int(len(x))}


def fit_drag_phase(vb, A, ph):
    """D_x from steady forward (speed_sweep), D_y from sustained lateral weave (slalom)."""
    return {"x": _lin(vb[ph == "speed_sweep", 0], A[ph == "speed_sweep", 0]),
            "y": _lin(vb[ph == "slalom", 1], A[ph == "slalom", 1])}


def fit_weathervane(W, C, dW, vb, ph):
    """Yaw-rate residual vs lateral velocity: domega_z = a*wcmd_z - b*omega_z + wv*v_body_y."""
    m = (ph == "lateral_steps") | (ph == "slalom")
    M = np.vstack([C[m, 2], -W[m, 2], vb[m, 1]]).T
    (a, b, wv), *_ = np.linalg.lstsq(M, dW[m, 2], rcond=None)
    r2 = 1 - np.var(dW[m, 2] - M @ [a, b, wv]) / np.var(dW[m, 2])
    return {"wv_coeff": float(wv), "tau_yaw_ms": float(1e3 / b) if b > 1e-6 else None,
            "r2": float(r2), "n": int(m.sum())}


def pred_accel(p, drag, thr):
    """Predicted body-frame specific force from drag+thrust fit (x,y,z)."""
    ax = -drag["x"]["D"] * p["vb"][:, 0] + drag["x"]["offset"]
    ay = -drag["y"]["D"] * p["vb"][:, 1] + drag["y"]["offset"]
    az = thr["f0"] + thr["df_dthr"] * p["C"][:, 3]
    return np.column_stack([ax, ay, az])


def main():
    runs = [r for r in load_index() if r["script"] == "collect_vq"]
    P = [prep(r) for r in runs]
    P = [(r, p) for r, p in zip(runs, P) if p is not None]
    print(f"loaded {len(P)} runs; holding out the last for validation", flush=True)
    train = P[:-1]; hold_r, hold = P[-1]
    vb = np.concatenate([p["vb"] for _, p in train]); A = np.concatenate([p["A"] for _, p in train])
    Aimu = np.concatenate([p["Aimu"] for _, p in train])
    W = np.concatenate([p["W"] for _, p in train]); C = np.concatenate([p["C"] for _, p in train])
    tt = np.concatenate([p["tt"] for _, p in train])
    Wk = np.concatenate([p["Wk"] for _, p in train]); Wk1 = np.concatenate([p["Wk1"] for _, p in train])
    Ck = np.concatenate([p["Ck"] for _, p in train]); vbyk = np.concatenate([p["vbyk"] for _, p in train])
    dt_med = float(np.median(np.concatenate([p["dtk"] for _, p in train])))
    ph = np.concatenate([p["ph"] for _, p in train])                # per-run phase labels
    print(f"train rows: {len(A)}  dt_med={dt_med*1e3:.1f}ms  speed_sweep rows={int((ph=='speed_sweep').sum())}", flush=True)

    drag = fit_drag_phase(vb, A, ph); thr = fit_thrust(C, Aimu)      # thrust from IMU acc_z (vertical, robust)
    if not (0 < drag["y"]["D"] < 1.0):                              # slalom-y polluted by roll/weathervane -> guard
        drag["y"] = {**drag["y"], "D": drag["x"]["D"], "guarded": "neg/oob -> isotropic = D_x"}
    rate, wv_coeff = fit_rate(Wk, Wk1, Ck, vbyk, dt_med)
    wv = {"wv_coeff": wv_coeff, "method": "discrete yaw fit (g/dt)"}

    print("\n=== DRAG (linear rotor drag, body frame; steady phases) ===", flush=True)
    print(f"  D_x = {drag['x']['D']:.3f} /s (prior {PRIOR_DRAG['x']:.3f}) R^2={drag['x']['r2']:.2f} [speed_sweep n={drag['x']['n']}]", flush=True)
    print(f"  D_y = {drag['y']['D']:.3f} /s (prior {PRIOR_DRAG['y']:.3f}) R^2={drag['y']['r2']:.2f} [slalom n={drag['y']['n']}]", flush=True)
    print("=== THRUST ===", flush=True)
    print(f"  f_z = {thr['f0']:.2f} + {thr['df_dthr']:.1f}*thr   dCacc/dthr={thr['dCacc_dthr']:.1f}   "
          f"hover_thr~{thr['thr_hover_pred']:.3f} (prior 0.23)   R^2={thr['r2']:.2f}", flush=True)
    print("=== RATE LOOP (discrete one-step: omega[k+1]=a*omega+b*wcmd) ===", flush=True)
    for nm in ("roll", "pitch", "yaw"):
        g = rate[nm]["gain_G"]; ta = rate[nm]["tau_ms"]
        print(f"  {nm:5s} gain={g:.2f}  tau={ta:.0f} ms  alpha={rate[nm]['alpha']:.3f}  R^2={rate[nm]['r2']:.3f}", flush=True)
    print(f"=== WEATHERVANE ===  wv_coeff = {wv['wv_coeff']:+.4f}  (prior ~ -0.0033)", flush=True)

    # held-out validation (drag+thrust accel)
    pa = pred_accel(hold, drag, thr)
    err = hold["A"] - pa
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    print(f"\n=== HELD-OUT VALIDATION ({hold_r['run_id'][9:]}, n={hold['n']}) ===", flush=True)
    print(f"  body-accel RMSE  x={rmse[0]:.2f} y={rmse[1]:.2f} z={rmse[2]:.2f} m/s^2  "
          f"(signal std {np.std(hold['A'],axis=0).round(2)})", flush=True)

    model = {"source": "fit_model.py", "n_train_rows": int(len(A)), "n_runs": len(P),
             "drag_linear_body": {"Dx": drag["x"]["D"], "Dy": drag["y"]["D"], "Dz": 0.0,
                                  "r2_x": drag["x"]["r2"], "r2_y": drag["y"]["r2"]},
             "thrust": thr, "rate_loop": rate, "weathervane": wv,
             "frame_conventions": {"qfix": "true_wxyz = stored[1,2,3,0]", "wfix": [1, -1, 1],
                                   "velocity": "ODOMETRY vel is BODY frame (world = R@v_body)",
                                   "note": "data-side only; live io_layer/controllers unchanged"},
             "delays_measured": {"comms_roundtrip_ms": 19.0, "thrust_lag_ms_range": [40, 90],
                                 "state_cadence_hz": {"odometry": 72, "imu": 144, "actuator": 73}},
             "holdout_rmse_body_accel": {"x": float(rmse[0]), "y": float(rmse[1]), "z": float(rmse[2])},
             "notes": "rate interface; drag body-frame; weathervane (yaw moment vs sideslip) TODO (rotational)."}
    import os
    out = "sysid/vq_model.json"
    os.makedirs("sysid", exist_ok=True)
    json.dump(model, open(out, "w"), indent=2)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
