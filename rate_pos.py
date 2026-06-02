"""ACRO rate-control POSITION HOLD (15 s). Leveling worked; yaw oscillated, so
hold yaw passively (yaw-rate command = 0 -> sim rate-damps heading). Full position
loop: pos/vel error -> desired accel -> desired attitude (tilt) -> attitude error
-> body rate / rate_gain. No quaternion remap: rate feedback closes in our own
convention. Hold spawn position + altitude."""
import json, time
import numpy as np
from pymavlink import mavutil
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store
from aigp.commander import Commander
from aigp.geometry import quat_to_R
from aigp.control_math import (desired_accel, desired_attitude, mat_to_quat,
                               attitude_error_quat, collective_accel, accel_to_thrust_norm)

r = json.load(open("sysid/sim_response.json"))
HOVER = r["hover_thrust"]; KA = r["k_a"]
RG = np.array([r["rate_gain_axes"]["roll"], r["rate_gain_axes"]["pitch"], r["rate_gain_axes"]["yaw"]])
KP_ATT = 2.5; KP_YAW = 2.0; WMAX = 4.0; DURATION = 15.0; LOOP_DT = 0.004   # ~250 Hz like the example
KP = np.array([0.0, 0.0, 1.8]); KD = np.array([1.0, 1.0, 3.0]); TILT_MAX_ACC = np.tan(np.radians(10)) * 9.81
IDLE = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE

s = Store(); m = MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); VisionIO(s).start()
boot = int(time.time() * 1000); c = Commander(m.conn, boot)


def idle():
    m.conn.mav.set_attitude_target_send(int(time.time() * 1000) - boot, m.conn.target_system,
                                        m.conn.target_component, IDLE, [1.0, 0, 0, 0], 0, 0, 0, 0)


def fresh_start():
    """Clear latched throttle, reset, wait for the countdown to RECYCLE (live
    False->True) with the drone at rest -> guaranteed fresh spawn."""
    t = time.time()
    while time.time() - t < 1.0:               # idle thrust=0 clears latched throttle
        idle(); time.sleep(0.02)
    prev = s.get_race(); prev_boot = prev["boot_ms"] if prev else None
    c.sim_reset()
    t = time.time()
    while time.time() - t < 30:                 # phase 1: detect reset (not-live or boot drop)
        idle()
        r = s.get_race()
        if r and (not r["race_live"] or (prev_boot and r["boot_ms"] < prev_boot)):
            break
        time.sleep(0.02)
    while time.time() - t < 30:                 # phase 2: fresh live + drone at rest
        idle()
        d = s.get_drone()
        if s.get_race_live() and d is not None and np.linalg.norm(d.vel_ned) < 3.0:
            return True
        time.sleep(0.02)
    return False


assert fresh_start(), "never went fresh-live"
print("FRESH LIVE", flush=True)

ds0 = s.get_drone(); spawn = ds0.pos_ned.copy()
yaw0 = float(np.arctan2(quat_to_R(ds0.quat_wxyz)[1, 0], quat_to_R(ds0.quat_wxyz)[0, 0]))
c.arm()
t0 = time.time(); last = -1; max_tilt = 0.0
while time.time() - t0 < DURATION:
    ds = s.get_drone()
    if ds is not None:
        a = desired_accel(ds.pos_ned, ds.vel_ned, spawn, np.zeros(3), KP, KD)
        ah = a[:2]; n = float(np.linalg.norm(ah))
        if n > TILT_MAX_ACC:
            a[:2] = ah / n * TILT_MAX_ACC
        Rc = quat_to_R(ds.quat_wxyz)
        yaw_cur = float(np.arctan2(Rc[1, 0], Rc[0, 0]))
        q_des = mat_to_quat(desired_attitude(a, yaw_cur))   # heading=current -> no yaw error in roll/pitch
        w_des = KP_ATT * attitude_error_quat(ds.quat_wxyz, q_des)
        yaw_err = (yaw0 - yaw_cur + np.pi) % (2 * np.pi) - np.pi   # wrap to [-pi,pi]
        w_des[2] = KP_YAW * yaw_err                      # actively hold spawn heading
        w_cmd = np.clip(w_des / RG, -WMAX, WMAX)
        thr = accel_to_thrust_norm(collective_accel(a, ds.quat_wxyz), HOVER, KA)
        c.send_attitude_target(w_cmd, thr)
        zb = quat_to_R(ds.quat_wxyz)[:, 2]
        tilt = np.degrees(np.arccos(max(-1, min(1, zb[2]))))
        max_tilt = max(max_tilt, tilt)
        k = int((time.time() - t0) / 1.5)
        if k != last:
            last = k
            d = ds.pos_ned - spawn
            print(f"t={time.time()-t0:4.1f} tilt={tilt:3.0f} dz={d[2]:+5.1f} dxy=({d[0]:+5.1f},{d[1]:+5.1f}) "
                  f"spd={np.linalg.norm(ds.vel_ned[:2]):4.1f} thr={thr:.2f}", flush=True)
    time.sleep(LOOP_DT)
d = s.get_drone().pos_ned - spawn
print(f"\nRESULT: dz={d[2]:+.1f} dxy=({d[0]:+.1f},{d[1]:+.1f}) max_tilt={max_tilt:.0f} over {DURATION:.0f}s", flush=True)
