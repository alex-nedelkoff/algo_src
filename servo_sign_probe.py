"""Bring-up: hover (position hold) at spawn, command a small +yaw step, report how the
detected gate's u moves in-image -> pins ServoCfg.sign_x. Camera looks -body-x (empirical)."""
import json, time
import numpy as np
from pymavlink import mavutil
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store
from aigp.commander import Commander
from aigp.geometry import quat_to_R
from aigp.control_math import (desired_accel, desired_attitude, mat_to_quat,
                               attitude_error_quat, collective_accel, accel_to_thrust_norm)
from aigp.gate_detect import detect_gate, load_params

r = json.load(open("sysid/sim_response.json"))
HOVER = r["hover_thrust"]; KA = r["k_a"]
RG = np.array([r["rate_gain_axes"]["roll"], r["rate_gain_axes"]["pitch"], r["rate_gain_axes"]["yaw"]])
KP_ATT = np.array([0.7, 1.6, 1.0]); WMAX = 4.0; LOOP_DT = 0.004
KP = np.array([0.0, 0.0, 1.8]); KD = np.array([1.0, 1.0, 3.0]); TILT = np.tan(np.radians(10)) * 9.81
IDLE = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
P = load_params()

s = Store(); m = MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); VisionIO(s).start()
boot = int(time.time() * 1000); c = Commander(m.conn, boot)


def idle():
    m.conn.mav.set_attitude_target_send(int(time.time()*1000)-boot, m.conn.target_system,
                                        m.conn.target_component, IDLE, [1, 0, 0, 0], 0, 0, 0, 0)


def fresh_start():
    t = time.time()
    while time.time()-t < 1.0: idle(); time.sleep(0.02)
    prev = s.get_race(); pb = prev["boot_ms"] if prev else None
    c.sim_reset(); t = time.time()
    while time.time()-t < 30:
        idle(); r2 = s.get_race()
        if r2 and (not r2["race_live"] or (pb and r2["boot_ms"] < pb)): break
        time.sleep(0.02)
    while time.time()-t < 30:
        idle(); d = s.get_drone()
        if s.get_race_live() and d is not None and np.linalg.norm(d.vel_ned) < 3.0: return True
        time.sleep(0.02)
    return False


def gate_u():
    fr, _ = s.get_frame()
    if fr is None: return None
    det = detect_gate(fr[0], P)
    return det.u if det else None


assert fresh_start(), "not live"
ds0 = s.get_drone(); spawn = ds0.pos_ned.copy()
yaw0 = float(np.arctan2(quat_to_R(ds0.quat_wxyz)[1, 0], quat_to_R(ds0.quat_wxyz)[0, 0]))
c.arm()


def hold(yaw_target, secs):
    us = []; t0 = time.time()
    while time.time()-t0 < secs:
        ds = s.get_drone()
        if ds is not None:
            a = desired_accel(ds.pos_ned, ds.vel_ned, spawn, np.zeros(3), KP, KD)
            n = float(np.linalg.norm(a[:2]))
            if n > TILT: a[:2] = a[:2]/n*TILT
            Rc = quat_to_R(ds.quat_wxyz); yc = float(np.arctan2(Rc[1, 0], Rc[0, 0]))
            q = mat_to_quat(desired_attitude(a, yc))
            w = KP_ATT * attitude_error_quat(ds.quat_wxyz, q)
            w[2] = 2.0 * ((yaw_target - yc + np.pi) % (2*np.pi) - np.pi)
            thr = accel_to_thrust_norm(collective_accel(a, ds.quat_wxyz), HOVER, KA)
            c.send_attitude_target(np.clip(w/RG, -WMAX, WMAX), thr)
            if time.time()-t0 > secs*0.5:
                u = gate_u()
                if u is not None: us.append(u)
        time.sleep(LOOP_DT)
    return float(np.mean(us)) if us else None


u0 = hold(yaw0, 4.0)
u1 = hold(yaw0 + np.radians(8), 4.0)
print(f"gate u at yaw0={u0}, at yaw0+8deg={u1}", flush=True)
if u0 is not None and u1 is not None:
    sign_x = -1.0 if (u1 - u0) > 0 else 1.0
    print(f"du/d(+yaw) = {u1-u0:+.1f}px  -> ServoCfg.sign_x = {sign_x:+.0f}", flush=True)
else:
    print("gate not detected during hold — check detector/orientation.", flush=True)
