"""Fly straight via the ACRO rate controller. Velocity tracking (KD only, no
position windup): vel_sp = SPEED north. Forward motion is on the strong/fast pitch
axis; roll only nulls lateral drift. Yaw actively held. Altitude held. 30 s."""
import json, time
import numpy as np
from pymavlink import mavutil
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store
from aigp.commander import Commander
from aigp.geometry import quat_to_R
from aigp.control_math import (desired_attitude, mat_to_quat, attitude_error_quat,
                               collective_accel, accel_to_thrust_norm)

r = json.load(open("sysid/sim_response.json"))
HOVER = r["hover_thrust"]; KA = r["k_a"]
RG = np.array([r["rate_gain_axes"]["roll"], r["rate_gain_axes"]["pitch"], r["rate_gain_axes"]["yaw"]])
KP_ATT = 2.5; KP_YAW = 2.0; WMAX = 4.0; DURATION = 30.0; LOOP_DT = 0.004
KD_H = 1.2; KP_Z = 1.8; KD_Z = 3.0; SPEED = 2.0; TILT_MAX_ACC = np.tan(np.radians(12)) * 9.81
IDLE = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE

s = Store(); m = MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); VisionIO(s).start()
boot = int(time.time() * 1000); c = Commander(m.conn, boot)


def idle():
    m.conn.mav.set_attitude_target_send(int(time.time() * 1000) - boot, m.conn.target_system,
                                        m.conn.target_component, IDLE, [1.0, 0, 0, 0], 0, 0, 0, 0)


def fresh_start():
    t = time.time()
    while time.time() - t < 1.0:
        idle(); time.sleep(0.02)
    prev = s.get_race(); pb = prev["boot_ms"] if prev else None
    c.sim_reset(); t = time.time()
    while time.time() - t < 30:
        idle(); r2 = s.get_race()
        if r2 and (not r2["race_live"] or (pb and r2["boot_ms"] < pb)):
            break
        time.sleep(0.02)
    while time.time() - t < 30:
        idle(); d = s.get_drone()
        if s.get_race_live() and d is not None and np.linalg.norm(d.vel_ned) < 3.0:
            return True
        time.sleep(0.02)
    return False


assert fresh_start(), "not live"
ds0 = s.get_drone(); spawn = ds0.pos_ned.copy(); z_sp = spawn[2]
yaw0 = float(np.arctan2(quat_to_R(ds0.quat_wxyz)[1, 0], quat_to_R(ds0.quat_wxyz)[0, 0]))
fwd = -np.array([np.cos(yaw0), np.sin(yaw0)])    # camera-forward = OPPOSITE our body-x heading
c.arm()
t0 = time.time(); last = -1; max_tilt = 0.0
while time.time() - t0 < DURATION:
    ds = s.get_drone()
    if ds is not None:
        vel_sp = SPEED * fwd
        a = np.zeros(3)
        a[:2] = KD_H * (vel_sp - ds.vel_ned[:2])
        a[2] = KP_Z * (z_sp - ds.pos_ned[2]) + KD_Z * (0.0 - ds.vel_ned[2])
        ah = a[:2]; n = float(np.linalg.norm(ah))
        if n > TILT_MAX_ACC:
            a[:2] = ah / n * TILT_MAX_ACC
        Rc = quat_to_R(ds.quat_wxyz); yaw_cur = float(np.arctan2(Rc[1, 0], Rc[0, 0]))
        q_des = mat_to_quat(desired_attitude(a, yaw_cur))
        w_des = KP_ATT * attitude_error_quat(ds.quat_wxyz, q_des)
        w_des[2] = KP_YAW * ((yaw0 - yaw_cur + np.pi) % (2 * np.pi) - np.pi)
        thr = accel_to_thrust_norm(collective_accel(a, ds.quat_wxyz), HOVER, KA)
        c.send_attitude_target(np.clip(w_des / RG, -WMAX, WMAX), thr)
        zb = Rc[:, 2]; tilt = np.degrees(np.arccos(max(-1, min(1, zb[2])))); max_tilt = max(max_tilt, tilt)
        k = int((time.time() - t0) / 3.0)
        if k != last:
            last = k
            d = ds.pos_ned - spawn
            fdist = float(d[:2] @ fwd); lat = float(d[0] * -fwd[1] + d[1] * fwd[0])
            print(f"t={time.time()-t0:4.1f} fwd={fdist:7.1f} lat={lat:+6.1f} alt_err={d[2]:+5.1f} "
                  f"spd={np.linalg.norm(ds.vel_ned[:2]):4.1f} tilt={tilt:3.0f}", flush=True)
    time.sleep(LOOP_DT)
d = s.get_drone().pos_ned - spawn
fdist = float(d[:2] @ fwd); lat = float(d[0] * -fwd[1] + d[1] * fwd[0])
print(f"\nRESULT: fwd={fdist:.0f}m lateral={lat:+.0f}m alt_err={d[2]:+.0f}m max_tilt={max_tilt:.0f} over {DURATION:.0f}s", flush=True)
