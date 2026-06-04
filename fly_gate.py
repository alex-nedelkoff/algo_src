"""Fly through ONE gate, camera-only.
- Horizontal: fixed heading (hold spawn yaw) + visual strafe (gate ex -> cross-track velocity).
- Vertical: pitch-invariant world-elevation climb (back-project gate pixel through the camera
  mount [axis=-body-x, 20deg UP, right=-body_y] + live attitude -> world elevation).
- Terminal: once the gate is close AND centered (sz>SZ_LOCK, |ex|<EX_LOCK), LOCK its world
  elevation and fly a straight BALLISTIC line (hold heading, climb at v_fwd*tan(elev)) through
  it, ignoring the now-unreliable close-range detection.
Success = active_gate_index increments.  argv: SIGN_S K_VZ RRD_PATH"""
import json, sys, time
import numpy as np
from pymavlink import mavutil
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store
from aigp.commander import Commander
from aigp.geometry import quat_to_R
from aigp.control_math import (desired_attitude, mat_to_quat, attitude_error_quat,
                               collective_accel, accel_to_thrust_norm)
from aigp.gate_detect import detect_gate, red_mask, draw_overlay, load_params
from aigp import viz

SIGN_S = float(sys.argv[1]) if len(sys.argv) > 1 else -1.0
K_VZ = float(sys.argv[2]) if len(sys.argv) > 2 else 7.0
RRD = sys.argv[3] if len(sys.argv) > 3 else None

TILT_DEG = 20.0; C20 = np.cos(np.radians(TILT_DEG)); S20 = np.sin(np.radians(TILT_DEG))
R_OPT_BODY = np.array([[0.0, -S20, -C20], [-1.0, 0.0, 0.0], [0.0, C20, -S20]])
CX = 320.0; FX = 320.0; FY = 320.0; CY = 180.0
r = json.load(open("sysid/sim_response.json"))
HOVER = r["hover_thrust"]; KA = r["k_a"]
RG = np.array([r["rate_gain_axes"]["roll"], r["rate_gain_axes"]["pitch"], r["rate_gain_axes"]["yaw"]])
KP_ATT = np.array([0.7, 1.6, 1.0]); KP_YAW = 3.0; KD_YAW = 0.3
KD_AL = 1.2; AL_MAX = 0.5; KD_LAT = 1.4; K_STRAFE = 4.0; VLAT_MAX = 1.2
VZ_MAX = 2.2; FWD = 1.2; KP_Z = 1.8; KD_Z = 3.0
SZ_LOCK = 100.0; EX_LOCK = 0.20
WMAX = 4.0; LOOP_DT = 0.004; DURATION = 45.0; N_GATES = 2   # chain N gates (heading held throughout)
TILTMAX = np.tan(np.radians(15)) * 9.81
IDLE = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
P = load_params()

s = Store(); m = MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); VisionIO(s).start()
boot = int(time.time()*1000); c = Commander(m.conn, boot); viz.init(RRD)


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


def gate_world_elev(u, v, R_bw):
    d_opt = np.array([(u - CX) / FX, (v - CY) / FY, 1.0]); d_opt /= np.linalg.norm(d_opt)
    d_world = R_bw @ (R_OPT_BODY @ d_opt)
    return float(-np.arcsin(np.clip(d_world[2], -1.0, 1.0)))


assert fresh_start(), "not live"
ds0 = s.get_drone(); spawn = ds0.pos_ned.copy(); z_ref = float(spawn[2]); gi0 = s.get_gate_idx()
yaw0 = float(np.arctan2(quat_to_R(ds0.quat_wxyz)[1, 0], quat_to_R(ds0.quat_wxyz)[0, 0]))
fwd = -np.array([np.cos(yaw0), np.sin(yaw0)]); lat = np.array([-fwd[1], fwd[0]])
c.arm()
print(f"SIGN_S={SIGN_S} K_VZ={K_VZ} SZ_LOCK={SZ_LOCK:.0f} rrd={RRD}", flush=True)
t0 = time.time(); passed = False; logn = 0; trail = []; lastlog = -1; locked = False; tan_lock = 0.0
prev_gi = gi0; target = gi0 + N_GATES
while time.time()-t0 < DURATION:
    ds = s.get_drone()
    if ds is not None:
        Rc = quat_to_R(ds.quat_wxyz)
        fr, _ = s.get_frame(); bgr = fr[0] if fr else None
        det = detect_gate(bgr, P) if bgr is not None else None
        v_al = float(ds.vel_ned[:2] @ fwd); v_lat = float(ds.vel_ned[:2] @ lat)
        a_al = float(np.clip(KD_AL * (FWD - v_al), -4.0, AL_MAX))
        elev = None
        if det is not None and not locked and det.w_px > SZ_LOCK and abs((det.u - CX) / FX) < EX_LOCK:
            phi = gate_world_elev(det.u, det.v, Rc); tan_lock = float(np.tan(phi)); locked = True
            print(f"LOCK t={time.time()-t0:.1f}s phi={np.degrees(phi):+.0f}deg", flush=True)
        if locked:
            v_lat_sp = 0.0                                      # hold heading (u unreliable at the edge)
            if det is not None:
                elev = gate_world_elev(CX, det.v, Rc)          # track vertical by v only (assume centered)
                vz_sp = float(np.clip(-K_VZ * elev, -VZ_MAX, VZ_MAX))
                z_ref += vz_sp * LOOP_DT
                z_ref = float(np.clip(z_ref, spawn[2] - 2.0, spawn[2] + 5.0))
            a2 = float(np.clip(KP_Z * (z_ref - ds.pos_ned[2]) + KD_Z * (0.0 - ds.vel_ned[2]), -4.0, 4.0))
        elif det is not None:
            ex = (det.u - CX) / FX
            v_lat_sp = float(np.clip(K_STRAFE * SIGN_S * ex, -VLAT_MAX, VLAT_MAX))
            elev = gate_world_elev(det.u, det.v, Rc)
            vz_sp = float(np.clip(-K_VZ * elev, -VZ_MAX, VZ_MAX))
            z_ref += vz_sp * LOOP_DT
            z_ref = float(np.clip(z_ref, spawn[2] - 2.0, spawn[2] + 5.0))
            a2 = float(np.clip(KP_Z * (z_ref - ds.pos_ned[2]) + KD_Z * (0.0 - ds.vel_ned[2]), -4.0, 4.0))
        else:
            v_lat_sp = 0.0
            a2 = KP_Z * (z_ref - ds.pos_ned[2]) + KD_Z * (0.0 - ds.vel_ned[2])
        a_lat = KD_LAT * (v_lat_sp - v_lat)
        a = np.zeros(3); a[:2] = a_al * fwd + a_lat * lat; a[2] = a2
        nrm = float(np.linalg.norm(a[:2]))
        if nrm > TILTMAX: a[:2] = a[:2]/nrm*TILTMAX
        yaw_cur = float(np.arctan2(Rc[1, 0], Rc[0, 0]))
        q = mat_to_quat(desired_attitude(a, yaw_cur))
        w = KP_ATT * attitude_error_quat(ds.quat_wxyz, q)
        w[2] = KP_YAW * ((yaw0 - yaw_cur + np.pi) % (2*np.pi) - np.pi) - KD_YAW * float(ds.omega[2])
        thr = accel_to_thrust_norm(collective_accel(a, ds.quat_wxyz), HOVER, KA)
        c.send_attitude_target(np.clip(w/RG, -WMAX, WMAX), thr)
        gi = s.get_gate_idx()
        if gi > prev_gi:
            if bgr is not None:
                trail.append((ds.pos_ned - spawn).copy())
                viz.log_step(time.time()-t0, ds.pos_ned - spawn, ds.vel_ned, bgr,
                             red_mask(bgr, P), draw_overlay(bgr, det), det, None, trail=trail)
            print(f"*** GATE PASSED at t={time.time()-t0:.1f}s (idx {prev_gi}->{gi}) ***", flush=True)
            prev_gi = gi; locked = False                     # reset lock -> re-acquire the next gate
            if gi >= target:
                passed = True; break
        logn += 1
        if logn % 10 == 0 and bgr is not None:
            trail.append((ds.pos_ned - spawn).copy())
            viz.log_step(time.time()-t0, ds.pos_ned - spawn, ds.vel_ned, bgr,
                         red_mask(bgr, P), draw_overlay(bgr, det), det, None, trail=trail)
        k = int((time.time()-t0)/2.0)
        if k != lastlog:
            lastlog = k; d = ds.pos_ned - spawn
            uu = f"{det.u:.0f}" if det else "--"; vv = f"{det.v:.0f}" if det else "--"; zz = f"{det.w_px:.0f}" if det else "--"
            print(f"t={time.time()-t0:4.1f} u={uu} v={vv} sz={zz} spd={np.linalg.norm(ds.vel_ned[:2]):4.1f} "
                  f"dDown={d[2]:+5.1f} zref={z_ref-spawn[2]:+4.1f} lk={int(locked)} fwd={d[:2]@fwd:+5.1f} gi={gi}", flush=True)
    time.sleep(LOOP_DT)
d = s.get_drone().pos_ned - spawn
n_passed = s.get_gate_idx() - gi0
print(f"\nRESULT: passed {n_passed}/{N_GATES} gate(s) | fwd={d[:2]@fwd:+.0f}m dDown={d[2]:+.0f}m over {time.time()-t0:.0f}s", flush=True)
