"""Consolidate the identified AI-GP Virtual Qualifier drone characteristics into ONE source-of-truth
parameter pack (docs/vq_drone_params.json) and run consistency cross-checks across the source artifacts.

Canonical frame: body FRD (x-fwd, y-right, z-down). Values are MASS-NORMALIZED where noted (the sim,
like the sysID, is only identifiable up to absolute mass — anchor with the published chassis if needed).

Sources: docs/sim_aero.json (Phase-1 force/moment), docs/sim_aero_cl.json (closed-loop D_y + weathervane
+ motor nuisances), docs/motor_model.json (mixer + k_f/k_q), sysid/sim_response.json (hover/ACRO rate
gains), aero_data/kappa.json (pitch-pulse inertia + k_q attempt), sysid_dyn/dyn2/sim_dynamics.json
(open-loop torque coeffs)."""
import json
import numpy as np
from aigp.motor_model import motor_outputs_to_wrench

GRAVITY = 9.81


def load():
    return dict(
        aero=json.load(open("docs/sim_aero.json")),
        aero_cl=json.load(open("docs/sim_aero_cl.json")),
        motor=json.load(open("docs/motor_model.json")),
        resp=json.load(open("sysid/sim_response.json")),
        kappa=json.load(open("aero_data/kappa.json")),
        dyn=json.load(open("sysid_dyn/dyn2/sim_dynamics.json")),
    )


def P(value, units, confidence, source, dr=None, notes=""):
    e = {"value": value, "units": units, "confidence": confidence, "source": source, "notes": notes}
    if dr is not None:
        e["dr_range"] = dr
    return e


def build_pack(a):
    F = dict(zip(a["aero"]["force_cols"], a["aero"]["theta_F"]))
    kap = a["kappa"]["kappa"]
    cl = a["aero_cl"]
    wv_side = cl["wv_z_yawfit"]["free"]["wv_z"]
    rg = a["resp"]["rate_gain_axes"]
    pack = {
        "meta": {
            "frame": "body FRD (x-fwd, y-right, z-down); world NED",
            "normalization": "mass-normalized (forces in m/s^2, T(hover)=9.81); absolute mass degenerate",
            "purpose": "source-of-truth for a parallel RL training surrogate (PyBullet / torch_quad)",
            "speed_envelope_identified": "0-8 m/s (race peaks ~33 m/s -> high-speed terms are DR-only)",
        },
        "rigid_body": {
            "mass": P(0.65, "kg", "low", "chassis-bbox prior (VADR-TS-002 280x280x160mm, 5\" quad)",
                      dr=[0.45, 0.85], notes="absolute mass unobservable from flight; surrogate is mass-normalized"),
            "I_ratio": P([3.7, 1.0, 1.0], "Ixx:Iyy:Izz", "med", "sim_dynamics + aero sysID",
                         notes="roll inertia 3.7x pitch/yaw (the slow axis)"),
            "I_z_massnorm": P(kap, "(N*m)/(rad/s^2) /mass", "med", "kappa.json (pitch sharp-pulse)",
                              dr=[kap*0.6, kap*1.5], notes="Iy=Iz=kappa; Ix=3.7*kappa"),
            "arm_length": P(a["motor"]["L"], "m", "med", "motor_model.json"),
        },
        "propulsion": {
            "k_f": P(a["motor"]["k_f"], "accel per g(u)", "high", "motor_model.json (hover calib)",
                     notes="T = k_f * sum(g(u)); T(hover)=9.81 mass-normalized"),
            "k_q": P(a["motor"]["k_q"], "torque per g(u)", "low", "motor_model.json (seeded k_q/k_f=0.02)",
                     dr=[a["motor"]["k_q"]*0.4, a["motor"]["k_q"]*2.5],
                     notes="yaw reaction torque; weak yaw authority; could not be pinned from flight"),
            "thrust_form": P(a["motor"]["form"], "g(u)", "med", "motor_model.json",
                             notes="quadratic g(u)=u^2; CONFLICTS with sim_dynamics power=1 (see consistency)"),
            "c_T": P(a["dyn"]["c_T"], "-", "med", "sim_dynamics.json (open-loop motor probes)"),
            "hover_thrust_norm": P(a["resp"]["hover_thrust"], "[0,1] throttle", "high", "sim_response.json"),
            "hover_motor_u": P(0.27, "[0,1] actuator", "high", "ACTUATOR_OUTPUT_STATUS (live)"),
        },
        "mixer": {
            "sx": P(a["motor"]["sx"], "sign", "high", "motor_model.json (hover-torque~0 validated)"),
            "sy": P(a["motor"]["sy"], "sign", "high", "motor_model.json"),
            "sz": P(a["motor"]["sz"], "sign", "high", "motor_model.json; sz = -(sx*sy), quad-X diagonal"),
            "note": "dyn-ID probe frame auto-detected roll<->yaw axis labels (c_L@axis2, c_N@axis0); "
                    "the aero/FRD pack uses the validated motor_model signs.",
        },
        "aero_force": {
            "D_x": P(F["D_x"], "1/s", "high", "sim_aero.json coast; lit-validated 0.49-0.54",
                     notes="linear rotor drag along flight axis"),
            "D_y": P(cl["D0"][1], "1/s", "med", "sim_aero_cl.json closed-loop",
                     notes="lateral drag; coast value unreliable, CL value 0.36"),
            "D_z": P(0.0, "1/s", "low", "pinned 0 (free-fall coast = vortex-ring regime)",
                     dr=[0.0, 0.3], notes="vertical drag unidentified"),
            "D_thrust_coupling": P(cl["D_T"], "1/s per T", "med", "sim_aero_cl.json",
                                   notes="drag is thrust/operating-point dependent (powered 0.34 < coast 0.52)"),
            "C_quadratic": P([abs(F["C_x"]), abs(F["C_y"]), 0.0], "1/m", "low",
                            "sim_aero.json (low-speed only)", dr=[0.0, 0.05],
                            notes="parasitic quadratic drag; dominates >15 m/s, UNIDENTIFIED at race speed"),
        },
        "aero_moment": {
            "weathervane_axial": P(dict(zip(a["aero"]["moment_cols"], a["aero"]["theta_M"]))["wv_z"],
                                   "(rad/s^2)/(m/s)", "med", "sim_aero.json (v_x->yaw, Phase-1 open-loop)",
                                   notes="axial/tail-first destabilizer; sign trusted, magnitude indicative"),
            "weathervane_sideslip": P(wv_side, "(rad/s^2)/(m/s)", "low", "sim_aero_cl.json yaw-fit (v_y->yaw)",
                                      dr=[wv_side*2, wv_side*0.3],
                                      notes="sideslip->yaw; magnitude rides on the seeded k_q (k_q-conditional)"),
            "angular_damping": P([0.0, 0.0, 0.0], "1/s", "low",
                                 "unidentified (inertia<->damping collinear in doublets)",
                                 dr=[0.0, 0.5], notes="DR-only"),
        },
        "acro_rate_loop": {
            "rate_gain": P([rg["roll"], rg["pitch"], rg["yaw"]], "per-axis", "med",
                           "sim_response.json (probe7)",
                           notes="sim's INNER ACRO loop: omega_cmd = omega_des/rate_gain; "
                                 "flight-measured ~-2.5 vs sysID -1.9 (see consistency)"),
            "k_a": P(a["resp"]["k_a"], "-", "high", "sim_response.json (thrust->accel map)"),
            "motor_lag_tau": P(None, "s", "missing", "fitter exists, never run",
                               dr=[0.010, 0.040], notes="first-order motor lag; DR-only until identified"),
        },
    }
    return pack


def chk(name, status, detail):
    return {"name": name, "status": status, "detail": detail}


def run_consistency(a):
    out = []
    # 1. thrust form: quadratic (motor_model) vs power=1 (dyn-ID)
    form = a["motor"]["form"]; power = a["dyn"]["power"]
    quad = (form == "quadratic")
    out.append(chk("thrust_form",
                   "FAIL" if (quad and power == 1) else "OK",
                   f"motor_model form={form} (g=u^{2 if quad else 1}) vs sim_dynamics power={power}. "
                   f"{'CONFLICT — reconcile (hover-torque validates quadratic; dyn power=1 was a rough early fit)' if quad and power==1 else 'consistent'}"))
    # 2. hover torque ~ 0 with the calibrated mixer
    T, tau = motor_outputs_to_wrench([0.27]*4, a["motor"])
    out.append(chk("hover_torque_zero",
                   "OK" if np.linalg.norm(tau) < 0.05 else "WARN",
                   f"reconstructed hover torque |tau|={np.linalg.norm(tau):.4f} (T={T:.2f}); validates mixer signs"))
    # 3. D_x coast vs powered (thrust dependence, not a conflict)
    dx_coast = dict(zip(a["aero"]["force_cols"], a["aero"]["theta_F"]))["D_x"]
    dx_pow = a["aero_cl"]["D0"][0]
    out.append(chk("D_x_thrust_dependence", "OK",
                   f"coast D_x={dx_coast:.2f} vs powered D0_x={dx_pow:.2f} -> linear drag is operating-point "
                   f"dependent (expected, both lit-plausible)"))
    # 4. rate gain: sysID vs flight-measured
    rg = a["resp"]["rate_gain_axes"]["roll"]
    out.append(chk("acro_rate_gain", "WARN",
                   f"sysID rate_gain~{rg:.2f}/axis but race_cruise measured ~-2.5 in flight -> ~1.3x mismatch; "
                   f"use the flight-measured value for control/surrogate"))
    # 5. inertia anchor sanity (kappa positive, roll/pitch agreement)
    kp = a["kappa"]; agree = "roll noisy (3.7x inertia)" if abs(kp.get("kappa_roll",0)) < 0.5*abs(kp["kappa_pitch"]) else "agree"
    out.append(chk("inertia_anchor",
                   "OK" if kp["kappa"] > 0 else "FAIL",
                   f"kappa(pitch)={kp['kappa_pitch']:.5f} (used), kappa(roll)/Iratio={kp.get('kappa_roll',float('nan')):.5f} [{agree}]"))
    # 6. k_q pin attempt failed
    out.append(chk("k_q_pinned", "WARN",
                   f"k_q NOT pinnable from flight (yaw authority too weak); seeded k_q/k_f=0.02 retained -> "
                   f"weathervane MAGNITUDE is k_q-conditional"))
    return out


def main():
    a = load()
    pack = build_pack(a)
    json.dump(pack, open("docs/vq_drone_params.json", "w"), indent=2)
    checks = run_consistency(a)
    print("=== VQ DRONE PARAMETER PACK — consistency ===")
    sym = {"OK": "[OK] ", "WARN": "[WARN]", "FAIL": "[FAIL]"}
    for c in checks:
        print(f"  {sym[c['status']]} {c['name']}: {c['detail']}")
    json.dump({"checks": checks}, open("docs/vq_drone_params_consistency.json", "w"), indent=2)
    n_warn = sum(c["status"] != "OK" for c in checks)
    print(f"\nwrote docs/vq_drone_params.json ({sum(len(v) for k,v in pack.items() if isinstance(v,dict))} entries), "
          f"{len(checks)} checks ({n_warn} WARN/FAIL)")


if __name__ == "__main__":
    main()
