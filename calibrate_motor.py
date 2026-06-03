"""T2 step 3: live motor-model calibration. Hover -> k_f and u_hover; single-axis rate inputs ->
the mixer SIGN pattern (sx,sy,sz). Saves aero_data/motor_model.json. Absolute torque<->angular-accel
scale is resolved later in the fit/rollout; here we pin the structural signs + thrust scale."""
import time, json, numpy as np
import aero_capture as ac
from aigp.motor_model import calibrate, motor_outputs_to_wrench

ds0, yaw0, z_sp = ac.setup()


def avg_u(extra, secs, skip=0.4):
    t0 = time.time(); us = []
    while time.time() - t0 < secs:
        ds = ac.s.get_drone()
        if ds is not None:
            wcmd, thr, ta, tilt = ac.control(ds, np.zeros(3), z_sp, yaw0, extra_w=extra)
            ac.c.send_attitude_target(wcmd, thr)
            a = ac.s.get_actuators()
            if a is not None and (time.time() - t0) > secs * skip:
                us.append(a[0].copy())
        time.sleep(0.004)
    return np.mean(us, axis=0)


u_hover = avg_u(None, 3.0)
u_roll = avg_u(np.array([1.2, 0, 0]), 1.0); avg_u(None, 1.0)   # re-settle between axes
u_pitch = avg_u(np.array([0, 1.2, 0]), 1.0); avg_u(None, 1.0)
u_yaw = avg_u(np.array([0, 0, 1.2]), 1.0)
ac.idle()

def differential(u_axis):
    d = u_axis - u_hover
    return d - d.mean()              # remove the collective (altitude-loop) component -> pure differential

dm_roll = differential(u_roll); dm_pitch = differential(u_pitch); dm_yaw = differential(u_yaw)
sx = np.sign(dm_roll).astype(int); sy = np.sign(dm_pitch).astype(int); sz = np.sign(dm_yaw).astype(int)
print("u_hover    :", u_hover.round(3))
print("roll  diff :", dm_roll.round(3), "-> sx", sx.tolist())
print("pitch diff :", dm_pitch.round(3), "-> sy", sy.tolist())
print("yaw   diff :", dm_yaw.round(3), "-> sz", sz.tolist())
# quad-X cross-check: yaw pairs should be the DIAGONALS of the roll/pitch corners
diag = (sx * sy)                      # +1 on one diagonal, -1 on the other
print("diagonal(sx*sy):", diag.tolist(), " (yaw sz should match one of +/-diag)")

params = calibrate(u_hover, 9.81, geom=dict(sx=sx.tolist(), sy=sy.tolist(), sz=sz.tolist(), L=0.14))
T_h, tau_h = motor_outputs_to_wrench(u_hover, params)
T_r, tau_r = motor_outputs_to_wrench(u_roll, params)
print(f"\ncheck hover : T={T_h:.2f} (want ~9.81)  tau={tau_h.round(3)} (want ~0)")
print(f"check roll  : tau={tau_r.round(3)} (want +roll dominant)")
json.dump(params, open("aero_data/motor_model.json", "w"), indent=2)
print("saved aero_data/motor_model.json:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in params.items()})
