"""Direction test in the RATE path. Leveling proved attitude is achieved in our
convention, but leveling is direction-agnostic. Here: command a FIXED tilt toward
N/E/S/W (via the rate loop) and measure which way the drone accelerates. Pins
whether desired_attitude's horizontal tilt direction matches the sim's world frame."""
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
KP_ATT = 4.0; WMAX = 4.0; IDLE = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE

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


def go(a_des, label, T=2.5):
    a_des = np.asarray(a_des, float)
    assert fresh_start(), "not live"
    p0 = s.get_drone().pos_ned.copy()
    c.arm(); t0 = time.time()
    while time.time() - t0 < T:
        ds = s.get_drone()
        if ds is not None:
            Rc = quat_to_R(ds.quat_wxyz); yaw = float(np.arctan2(Rc[1, 0], Rc[0, 0]))
            q_des = mat_to_quat(desired_attitude(a_des, yaw))
            w = KP_ATT * attitude_error_quat(ds.quat_wxyz, q_des); w[2] = 0.0
            thr = accel_to_thrust_norm(collective_accel(a_des, ds.quat_wxyz), HOVER, KA)
            c.send_attitude_target(np.clip(w / RG, -WMAX, WMAX), thr)
        time.sleep(0.004)
    d = s.get_drone().pos_ned - p0
    ok = {"NORTH": d[0] > 1, "EAST": d[1] > 1, "SOUTH": d[0] < -1, "WEST": d[1] < -1}[label]
    print(f"{label:6s} a_des={a_des} -> dpos N={d[0]:+5.1f} E={d[1]:+5.1f} D={d[2]:+5.1f}  {'OK' if ok else 'WRONG'}", flush=True)


go([3, 0, 0], "NORTH")
go([0, 3, 0], "EAST")
go([-3, 0, 0], "SOUTH")
go([0, -3, 0], "WEST")
print("done", flush=True)
