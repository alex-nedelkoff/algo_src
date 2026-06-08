"""vq_waypoint.py -- FF waypoint nav on the LIVE VQ sim through a SYNTHETIC gate corridor.

Tests "can our control pseudo-waypoint-navigate the real VQ?" (transfer of the matched-sim FF).
Uses the proven aigp NED-native controller (race_cruise/coord_turn primitives) aimed at a synthetic
gate sequence laid out ahead of the spawn (camera-forward = -body_x, same geometry as the matched-sim
corridor). Nose-to-velocity (weathervane-stable). Advances gate on proximity; logs gates reached +
sideslip. Dashboard ON (HARD RULE). Usage: python vq_waypoint.py [--v 4] [--gates 6] [--space 10]
"""
import sys, json, time
import numpy as np
from pymavlink import mavutil
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store
from aigp.commander import Commander
from aigp.geometry import quat_to_R
from aigp.control_math import (desired_attitude, mat_to_quat, attitude_error_quat,
                               collective_accel, accel_to_thrust_norm)
from aigp.flight_telemetry import sideslip_deg, tilt_deg
import aigp.flight_telemetry as ftm

def argf(f, d): return float(sys.argv[sys.argv.index(f)+1]) if f in sys.argv else d
VDES = argf("--v", 4.0); NG = int(argf("--gates", 6)); SPACE = argf("--space", 10.0)
GATE_R = 1.5; LOOP_DT = 0.004; MAX_T = 60.0; ABORT_TILT = 80.0
KV = 1.5; KP_POS_Z = 1.8; KD_Z = 3.0; KP_ATT = np.array([0.5, 1.6, 1.0]); KP_YAW = 3.0; KD_YAW = 0.3
WMAX = 4.0; TILT_MAX_ACC = np.tan(np.radians(25)) * 9.81
r = json.load(open("sysid/sim_response.json")); HOVER = r["hover_thrust"]; KA = r["k_a"]
RG = np.array([r["rate_gain_axes"]["roll"], r["rate_gain_axes"]["pitch"], r["rate_gain_axes"]["yaw"]])
IDLE = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
s = Store(); m = MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); VisionIO(s).start()
boot = int(time.time()*1000); c = Commander(m.conn, boot)


def idle():
    m.conn.mav.set_attitude_target_send(int(time.time()*1000)-boot, m.conn.target_system,
                                        m.conn.target_component, IDLE, [1.0, 0, 0, 0], 0, 0, 0, 0)


def fresh_start():
    t = time.time()
    while time.time()-t < 1.0:
        idle(); time.sleep(0.02)
    prev = s.get_race(); pb = prev["boot_ms"] if prev else None
    c.sim_reset(); t = time.time()
    while time.time()-t < 30:
        idle(); r2 = s.get_race()
        if r2 and (not r2["race_live"] or (pb and r2["boot_ms"] < pb)):
            break
        time.sleep(0.02)
    while time.time()-t < 30:
        idle(); d = s.get_drone()
        if s.get_race_live() and d is not None and np.linalg.norm(d.vel_ned) < 3.0:
            return True
        time.sleep(0.02)
    return False


def main():
    print(f"VQ WAYPOINT: v={VDES} gates={NG} spacing={SPACE}", flush=True)
    assert fresh_start(), "not live"
    ds0 = s.get_drone(); spawn = ds0.pos_ned.copy()
    yaw0 = float(np.arctan2(quat_to_R(ds0.quat_wxyz)[1, 0], quat_to_R(ds0.quat_wxyz)[0, 0]))
    travel = -np.array([np.cos(yaw0), np.sin(yaw0)])         # camera-forward
    lat = np.array([-travel[1], travel[0]])
    gates = []
    for i in range(NG):
        gxy = spawn[:2] + travel*SPACE*(i+1) + lat*2.0*np.sin(i*0.9)
        gates.append(np.array([gxy[0], gxy[1], spawn[2]]))   # same altitude (z) as spawn
    gates = np.array(gates)
    c.arm(); flog = ftm.from_args(sys.argv, RG, run_name="vq_waypoint", store=s)
    t0 = time.time(); gi = 0; betas = []; last = -1
    while time.time()-t0 < MAX_T and gi < NG:
        ds = s.get_drone()
        if ds is None:
            time.sleep(LOOP_DT); continue
        t = time.time()-t0; pos = ds.pos_ned; vel = ds.vel_ned
        gate = gates[gi]; to = gate[:2]-pos[:2]; dist = np.linalg.norm(to)+1e-6
        vdir = to/dist
        a = np.zeros(3)
        a[:2] = KV*(VDES*vdir - vel[:2])
        n = np.linalg.norm(a[:2])
        if n > TILT_MAX_ACC:
            a[:2] = a[:2]/n*TILT_MAX_ACC
        a[2] = KP_POS_Z*(gate[2]-pos[2]) + KD_Z*(0.0-vel[2])
        vh = vel[:2]; vmag = np.linalg.norm(vh)
        yaw_sp = float(np.arctan2(vh[1], vh[0])) if vmag > 0.5 else float(np.arctan2(vdir[1], vdir[0]))
        Rc = quat_to_R(ds.quat_wxyz); yaw_cur = float(np.arctan2(Rc[1, 0], Rc[0, 0]))
        q_des = mat_to_quat(desired_attitude(a, yaw_sp))
        w_des = KP_ATT*attitude_error_quat(ds.quat_wxyz, q_des)
        w_des[2] = KP_YAW*((yaw_sp-yaw_cur+np.pi) % (2*np.pi)-np.pi) - KD_YAW*float(ds.omega[2])
        thr = accel_to_thrust_norm(collective_accel(a, ds.quat_wxyz), HOVER, KA)
        w_cmd = np.clip(w_des/RG, -WMAX, WMAX)
        c.send_attitude_target(w_cmd, thr)
        beta = sideslip_deg(ds.quat_wxyz, ds.vel_ned)
        if vmag > 0.5:
            betas.append(abs(beta))
        if dist < GATE_R or (gi < NG and np.linalg.norm(gate-pos) < GATE_R):
            gi += 1; print(f"  GATE {gi}/{NG} reached t={t:.1f} v={vmag:.1f} |beta|={abs(beta):.0f}", flush=True)
        tilt = tilt_deg(ds.quat_wxyz)
        if flog is not None:
            flog.push(t, ds, {"a": a, "w_des": w_des, "q_des": q_des, "thr": thr}, tangent=travel, cruise=VDES, running=s.get_race_live(), armed=True)
        if tilt > ABORT_TILT:
            print(f"  ABORT tilt={tilt:.0f} t={t:.1f} at gate {gi}/{NG}", flush=True); break
        if int(t) != last:
            last = int(t)
            print(f"  t={t:4.1f} gate {gi}/{NG} dist={dist:4.1f} v={vmag:4.1f} tilt={tilt:3.0f} |beta|={abs(beta):3.0f}", flush=True)
        time.sleep(LOOP_DT)
    idle()
    if flog is not None:
        flog.close()
    print(f"\nVQ WAYPOINT DONE: reached {gi}/{NG} gates | mean|beta|={np.mean(betas) if betas else 0:.0f} "
          f"| {'TRANSFERS — FF pseudo-waypoint-navs the real VQ' if gi >= NG*0.7 else 'partial/failed'}", flush=True)


if __name__ == "__main__":
    main()
