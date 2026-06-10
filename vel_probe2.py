"""vel_probe2.py -- calibrate TRUE-FRAME control for fly_gate2: (0) hover-settle with true-frame
attitude hold (verify: stable, level, no ground drag), (1) push along camera heading, (2) push
along left candidate. Readout pins fwd/lat signs + confirms hover viability. One flight, decisive."""
import json, sys, time
import numpy as np
from pymavlink import mavutil
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store
from aigp.commander import Commander
from aigp.geometry import quat_to_R
from aigp.control_math import desired_attitude, mat_to_quat, attitude_error_quat, collective_accel, accel_to_thrust_norm
from fit_model import qfix

r = json.load(open("sysid/sim_response.json")); HOVER = r["hover_thrust"]; KA = r["k_a"]
RG = np.array([r["rate_gain_axes"]["roll"], r["rate_gain_axes"]["pitch"], r["rate_gain_axes"]["yaw"]])
KP_ATT = np.array([0.7, 1.6, 1.0]); KP_YAW = 3.0; KD_YAW = 0.3; KD_ATT = 0.3
KP_Z = 1.8; KD_Z = 3.0
WFIX = np.array([1.0, -1.0, 1.0])
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
ds0 = s.get_drone(); spawn = ds0.pos_ned.copy(); z_sp = float(spawn[2]) - 1.0   # hover 1 m up
yaw0_live = float(np.arctan2(quat_to_R(ds0.quat_wxyz)[1,0], quat_to_R(ds0.quat_wxyz)[0,0]))
cam_live = -np.array([np.cos(yaw0_live), np.sin(yaw0_live)])
q_t0 = qfix(ds0.quat_wxyz); R_t0 = quat_to_R(q_t0)
s_cam = 1.0 if float(R_t0[:2,0] @ cam_live) > 0 else -1.0
yaw_cam0 = float(np.arctan2(s_cam*R_t0[1,0], s_cam*R_t0[0,0]))
cam_t = np.array([np.cos(yaw_cam0), np.sin(yaw_cam0)])
left_t = np.array([-cam_t[1], cam_t[0]])
c.arm()
print(f"s_cam={s_cam:+.0f} cam_live={cam_live.round(2)} cam_true={cam_t.round(2)}", flush=True)
_, coll0 = s.get_collision()
t0 = time.time(); prev_p = None; prev_t = None
while time.time()-t0 < 18.0:
    ds = s.get_drone()
    if ds is None: time.sleep(0.004); continue
    t = time.time()-t0
    if t < 5.0:   a2d = np.zeros(2); ph = "hover"
    elif t < 9.0: a2d = 1.2*left_t; ph = "left+"
    elif t < 10.5: a2d = np.zeros(2); ph = "pause"
    else:          a2d = np.zeros(2); ph = "yaw+"
    a = np.array([a2d[0], a2d[1], np.clip(KP_Z*(z_sp-ds.pos_ned[2]) + KD_Z*(0.0-ds.vel_ned[2]), -3, 3)])
    q_t = qfix(ds.quat_wxyz); R_t = quat_to_R(q_t)
    yaw_body = float(np.arctan2(R_t[1,0], R_t[0,0]))
    # LATERAL INVERSION FIX (probe run 1: push along left_t moved the drone along -left_t):
    # flip the component of a perpendicular to the body heading before desired_attitude.
    hd = np.array([np.cos(yaw_body), np.sin(yaw_body)])
    a_par = float(a[:2] @ hd); a_perp = float(a[:2] @ np.array([-hd[1], hd[0]]))
    a[:2] = a_par * hd - a_perp * np.array([-hd[1], hd[0]])
    if ph == "yaw+":
        yaw_tgt = getattr(sys.modules[__name__], "_yawtgt", None)
        if yaw_tgt is None:
            sys.modules[__name__]._yawtgt = yaw_body; yaw_tgt = yaw_body
        yaw_tgt = sys.modules[__name__]._yawtgt = yaw_tgt + 0.25*0.004
    else:
        yaw_tgt = yaw_body
    q_des_t = mat_to_quat(desired_attitude(a, yaw_tgt))
    om_t = np.asarray(ds.omega, float) * WFIX
    w_t = KP_ATT * attitude_error_quat(q_t, q_des_t)
    w_t[0] -= KD_ATT*om_t[0]; w_t[1] -= KD_ATT*om_t[1]
    w_t[2] = KP_YAW*((yaw_tgt - yaw_body + np.pi)%(2*np.pi)-np.pi) - KD_YAW*om_t[2]
    w = np.clip(w_t * WFIX / RG, -4, 4)
    thr = accel_to_thrust_norm(min(collective_accel(a, q_t), 16.0), HOVER, KA)
    c.send_attitude_target(w, thr)
    now = time.time()
    if prev_p is None or now-prev_t > 0.6:
        if prev_p is not None:
            vw = (ds.pos_ned-prev_p)/(now-prev_t)
            tilt_t = float(np.degrees(np.arccos(np.clip(R_t[2,2],-1,1))))
            _, coll = s.get_collision()
            print(f"t={t:4.1f} {ph:5s} vb=[{ds.vel_ned[0]:+5.2f} {ds.vel_ned[1]:+5.2f}] "
                  f"vw_cam={float(vw[:2]@cam_t):+5.2f} vw_left={float(vw[:2]@left_t):+5.2f} "
                  f"yawB={np.degrees(yaw_body):+6.0f} z={ds.pos_ned[2]-spawn[2]:+5.2f} tiltT={tilt_t:3.0f} coll={int(coll-coll0)}", flush=True)
        prev_p = ds.pos_ned.copy(); prev_t = now
    time.sleep(0.004)
idle()
print("done", flush=True)
