"""chirality_probe.py -- THE handedness test for the qfix 'true frame' (findings doc 06-10).
L-flight: takeoff -> forward leg -> stop -> commanded +90 deg CHART yaw -> forward leg.
Witnesses: (1) recorded world-track turn direction (chart's claim, computed at end);
(2) FPV frames during the yaw (camera pan direction); (3) the user watching the sim window.
Chart-left == screen-left -> proper chart (mirrors = remap residue). Opposite -> REFLECTED chart.
True-frame control per fly_gate3 (qfix in, wfix out, world-y mirror fix, takeoff phase)."""
import json, os, sys, time
import cv2
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
ds0 = s.get_drone(); spawn = ds0.pos_ned.copy()
yaw0_live = float(np.arctan2(quat_to_R(ds0.quat_wxyz)[1,0], quat_to_R(ds0.quat_wxyz)[0,0]))
cam_live = -np.array([np.cos(yaw0_live), np.sin(yaw0_live)])
q_t0 = qfix(ds0.quat_wxyz); R_t0 = quat_to_R(q_t0)
s_cam = 1.0 if float(R_t0[:2,0] @ cam_live) > 0 else -1.0
yaw_ref = float(np.arctan2(s_cam*R_t0[1,0], s_cam*R_t0[0,0]))
c.arm()
os.makedirs("chiral_frames", exist_ok=True)
print(f"chirality probe: s_cam={s_cam:+.0f}; commanding +90deg CHART yaw between legs", flush=True)
print(">>> USER: watch the sim window — note which way the drone turns (its left or its right) <<<", flush=True)
t0 = time.time(); track = []
last_dump = 0.0; z_ref = float(spawn[2]) - 1.2; t_prev = time.time()
while time.time()-t0 < 26.0:
    ds = s.get_drone()
    if ds is None: time.sleep(0.004); continue
    now = time.time(); dt = min(now - t_prev, 0.05); t_prev = now; t = now-t0
    q_t = qfix(ds.quat_wxyz); R_t = quat_to_R(q_t)
    yaw_cur = float(np.arctan2(s_cam*R_t[1,0], s_cam*R_t[0,0]))
    fwd = np.array([np.cos(yaw_cur), np.sin(yaw_cur)])
    v_al = float(ds.vel_ned[0]); v_lat = float(ds.vel_ned[1])
    lat = np.array([-fwd[1], fwd[0]])
    if t < 3.0:    ph, a_al = "takeoff", 0.0
    elif t < 9.0:  ph, a_al = "leg1", float(np.clip(1.2*(1.5 - v_al), -0.8, 0.5))
    elif t < 12.0: ph, a_al = "stop1", float(np.clip(1.2*(0.0 - v_al), -1.2, 0.3))
    elif t < 17.5: ph, a_al = "YAW+90", 0.0
    elif t < 25.0: ph, a_al = "leg2", float(np.clip(1.2*(1.5 - v_al), -0.8, 0.5))
    else:          ph, a_al = "end", 0.0
    if ph == "YAW+90":
        yaw_ref += (np.pi/2) / 5.5 * dt                     # +90 deg chart-yaw over the 5.5 s phase (wall dt)
    a_lat = float(np.clip(1.4*(0.0 - v_lat), -2.0, 2.0))
    a = np.zeros(3); a[:2] = a_al*fwd + a_lat*lat
    a[2] = float(np.clip(KP_Z*(z_ref - ds.pos_ned[2]) + KD_Z*(0.0 - ds.vel_ned[2]), -3, 3))
    yaw_body = yaw_ref if s_cam > 0 else yaw_ref + np.pi
    hd = np.array([np.cos(yaw_body), np.sin(yaw_body)]); pp = np.array([-hd[1], hd[0]])
    a_par = float(a[:2] @ hd); a_perp = float(a[:2] @ pp)
    a[:2] = a_par*hd - a_perp*pp
    q_des = mat_to_quat(desired_attitude(a, yaw_body))
    om_t = np.asarray(ds.omega, float) * WFIX
    w_t = KP_ATT * attitude_error_quat(q_t, q_des)
    w_t[0] -= KD_ATT*om_t[0]; w_t[1] -= KD_ATT*om_t[1]
    yaw_cur_b = float(np.arctan2(R_t[1,0], R_t[0,0]))
    w_t[2] = float(np.clip(KP_YAW*((yaw_body - yaw_cur_b + np.pi)%(2*np.pi)-np.pi) - KD_YAW*om_t[2], -1.0, 1.0))
    w = np.clip(w_t * WFIX / RG, -4, 4)
    thr = accel_to_thrust_norm(min(collective_accel(a, q_t), 16.0), HOVER, KA)
    c.send_attitude_target(w, thr)
    track.append((t, float(ds.pos_ned[0]), float(ds.pos_ned[1]), ph))
    if ph == "YAW+90" and t - last_dump > 0.8:
        last_dump = t
        fr, _ = s.get_frame()
        if fr: cv2.imwrite(f"chiral_frames/yaw_t{t:05.1f}.jpg", fr[0])
    time.sleep(0.004)
idle()
T = np.array([(a,b,c) for a,b,c,_ in track]); PH = [p for *_, p in track]
def leg_dir(name):
    idx = [i for i,p in enumerate(PH) if p == name]
    i0, i1 = idx[len(idx)//3], idx[-1]
    d = T[i1,1:3] - T[i0,1:3]
    return d/ (np.linalg.norm(d)+1e-9)
d1, d2 = leg_dir("leg1"), leg_dir("leg2")
crossz = d1[0]*d2[1] - d1[1]*d2[0]
ang = np.degrees(np.arctan2(crossz, float(d1@d2)))
print(f"\nCHART verdict: leg1 dir={d1.round(2)} leg2 dir={d2.round(2)} turn={ang:+.0f} deg "
      f"(cross-z {crossz:+.2f}) -> chart says turn = {'CCW(+)' if crossz>0 else 'CW(-)'} in recorded x-y", flush=True)
print("Compare with: camera pan in chiral_frames/, and the USER's screen observation.", flush=True)
