"""vel_probe.py -- pin the live io vel frame ONCE: push known world directions, compare vb to world
velocity (pos gradient). Phases: settle -> push CAMERA dir -> push world-LEFT of camera. Prints the
mapping. (canonical rule: probe, don't guess signs)"""
import json, sys, time
import numpy as np
from pymavlink import mavutil
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store
from aigp.commander import Commander
from aigp.geometry import quat_to_R
from aigp.control_math import desired_attitude, mat_to_quat, attitude_error_quat, collective_accel, accel_to_thrust_norm

r = json.load(open("sysid/sim_response.json")); HOVER = r["hover_thrust"]; KA = r["k_a"]
RG = np.array([r["rate_gain_axes"]["roll"], r["rate_gain_axes"]["pitch"], r["rate_gain_axes"]["yaw"]])
KP_ATT = np.array([0.7, 1.6, 1.0]); KP_YAW = 3.0; KD_YAW = 0.3; KP_Z = 1.8; KD_Z = 3.0
IDLE = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
s = Store(); m = MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); VisionIO(s).start()
boot = int(time.time()*1000); c = Commander(m.conn, boot)

def idle():
    m.conn.mav.set_attitude_target_send(int(time.time()*1000)-boot, m.conn.target_system,
                                        m.conn.target_component, IDLE, [1,0,0,0], 0,0,0,0)

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

assert fresh_start(), "not live"
ds0 = s.get_drone(); spawn = ds0.pos_ned.copy(); z_sp = float(spawn[2])
yaw0 = float(np.arctan2(quat_to_R(ds0.quat_wxyz)[1,0], quat_to_R(ds0.quat_wxyz)[0,0]))
cam = -np.array([np.cos(yaw0), np.sin(yaw0)]); left = np.array([-cam[1], cam[0]])
c.arm()
print(f"yaw0={np.degrees(yaw0):+.0f} cam_world={cam.round(2)} leftcand_world={left.round(2)}", flush=True)
t0 = time.time(); rows = []
prev_p = None; prev_t = None
while time.time()-t0 < 14.0:
    ds = s.get_drone()
    if ds is None: time.sleep(0.004); continue
    t = time.time()-t0
    if t < 4.0:   a2d = np.zeros(2); ph = "settle"
    elif t < 8.0: a2d = 1.5*cam;     ph = "cam+"
    else:         a2d = 1.5*left;    ph = "left+"
    a = np.array([a2d[0], a2d[1], KP_Z*(z_sp-ds.pos_ned[2]) + KD_Z*(0.0-ds.vel_ned[2])])
    q = mat_to_quat(desired_attitude(a, yaw0))
    w = KP_ATT*attitude_error_quat(ds.quat_wxyz, q)
    yaw_cur = float(np.arctan2(quat_to_R(ds.quat_wxyz)[1,0], quat_to_R(ds.quat_wxyz)[0,0]))
    w[2] = KP_YAW*((yaw0-yaw_cur+np.pi)%(2*np.pi)-np.pi) - KD_YAW*float(ds.omega[2])
    thr = accel_to_thrust_norm(collective_accel(a, ds.quat_wxyz), HOVER, KA)
    c.send_attitude_target(np.clip(w/RG, -4, 4), thr)
    now = time.time()
    if prev_p is not None and now-prev_t > 0.45:
        vw = (ds.pos_ned - prev_p)/(now-prev_t)
        print(f"t={t:4.1f} {ph:6s} vb=[{ds.vel_ned[0]:+5.2f} {ds.vel_ned[1]:+5.2f} {ds.vel_ned[2]:+5.2f}] "
              f"vw_cam={float(vw[:2]@cam):+5.2f} vw_left={float(vw[:2]@left):+5.2f}", flush=True)
        prev_p = ds.pos_ned.copy(); prev_t = now
    elif prev_p is None:
        prev_p = ds.pos_ned.copy(); prev_t = now
    time.sleep(0.004)
idle()
print("done", flush=True)
