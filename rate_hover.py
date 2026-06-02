"""ACRO rate-control leveling hover. The sim is in ACRO: it tracks BODY RATES +
thrust (not attitude, not velocity). Outer loop: attitude error -> desired body
rate; then command = desired_rate / rate_gain (rate_gain ~ -1.9 per axis, the
sim's inverted/scaled rate response from probe7). Holds level at spawn heading +
spawn altitude. Idle (thrust 0) until race-live to respect the countdown gate."""
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
KP_ATT = 5.0; WMAX = 3.0; KP_Z = 2.0; KD_Z = 3.0; DURATION = 10.0
IDLE = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE

s = Store(); m = MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); VisionIO(s).start()
boot = int(time.time() * 1000); c = Commander(m.conn, boot)


def idle():
    m.conn.mav.set_attitude_target_send(
        int(time.time() * 1000) - boot, m.conn.target_system, m.conn.target_component,
        IDLE, [1.0, 0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0)


# start: idle until live; if stuck, one reset
print("waiting for race-live (idle thrust=0)...", flush=True)
t0 = time.time(); did_reset = False
while not s.get_race_live():
    idle()
    if not did_reset and time.time() - t0 > 6:
        c.sim_reset(); did_reset = True; print("sent sim_reset", flush=True)
    if time.time() - t0 > 30:
        print("FAILED to go live"); raise SystemExit
    time.sleep(0.02)
print(f"LIVE after {time.time()-t0:.1f}s", flush=True)

ds0 = s.get_drone(); z_sp = ds0.pos_ned[2]; spawn = ds0.pos_ned.copy()
yaw0 = float(np.arctan2(quat_to_R(ds0.quat_wxyz)[1, 0], quat_to_R(ds0.quat_wxyz)[0, 0]))
c.arm()
t0 = time.time(); last = -1
while time.time() - t0 < DURATION:
    ds = s.get_drone()
    if ds is not None:
        q_des = mat_to_quat(desired_attitude([0, 0, 0], yaw0))     # level, hold spawn heading
        w_des = KP_ATT * attitude_error_quat(ds.quat_wxyz, q_des)
        w_cmd = np.clip(w_des / RG, -WMAX, WMAX)                    # invert sim rate gain
        az = KP_Z * (z_sp - ds.pos_ned[2]) + KD_Z * (0.0 - ds.vel_ned[2])
        thr = accel_to_thrust_norm(collective_accel([0, 0, az], ds.quat_wxyz), HOVER, KA)
        c.send_attitude_target(w_cmd, thr)
        n = int((time.time() - t0) / 1.0)
        if n != last:
            last = n
            zb = quat_to_R(ds.quat_wxyz)[:, 2]
            tilt = np.degrees(np.arccos(max(-1, min(1, zb[2]))))
            d = ds.pos_ned - spawn
            print(f"t={time.time()-t0:4.1f} tilt={tilt:3.0f} dz={d[2]:+5.1f} dxy=({d[0]:+5.1f},{d[1]:+5.1f}) "
                  f"w_cmd=({w_cmd[0]:+.1f},{w_cmd[1]:+.1f},{w_cmd[2]:+.1f}) thr={thr:.2f}", flush=True)
    time.sleep(0.02)
d = s.get_drone().pos_ned - spawn
print(f"\nRESULT: dz={d[2]:+.1f} dxy=({d[0]:+.1f},{d[1]:+.1f}) final tilt logged above", flush=True)
