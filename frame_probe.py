"""frame_probe.py -- DIAGNOSTIC (COR-127 cornering debug): is live ds.vel_ned BODY or WORLD frame?
Flies straight nose-first briefly; prints raw vel_ned vs R^T@vel_ned. In straight nose-first flight the
BODY-frame velocity is [~+-speed, ~0, ~0] (v_y ~ 0). Whichever interpretation has v_y~0 is body frame.
No turn, no weathervane FF. Read-only re: the frame question."""
import sys, json, time
import numpy as np
from pymavlink import mavutil
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store
from aigp.commander import Commander
from aigp.geometry import quat_to_R
from aigp.control_math import (desired_attitude, mat_to_quat, attitude_error_quat,
                               collective_accel, accel_to_thrust_norm)
from aigp.flight_telemetry import sideslip_deg
import aigp.flight_telemetry as ftm

G = 9.81
KP_ATT = np.array([0.5, 1.6, 1.0]); KP_YAW = 3.0; KD_YAW = 0.3; KP_Z = 1.8; KD_Z = 3.0
WMAX = 4.0; LOOP_DT = 0.004; ABORT_TILT = 80.0; MAX_T = 14.0
r = json.load(open("sysid/sim_response.json")); HOVER = r["hover_thrust"]; KA = r["k_a"]
RG = np.array([r["rate_gain_axes"]["roll"], r["rate_gain_axes"]["pitch"], r["rate_gain_axes"]["yaw"]])
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


def main():
    assert fresh_start(), "not live"
    ds0 = s.get_drone(); z_sp = float(ds0.pos_ned[2])
    yaw0 = float(np.arctan2(quat_to_R(ds0.quat_wxyz)[1, 0], quat_to_R(ds0.quat_wxyz)[0, 0]))
    push = -np.array([-np.cos(yaw0), -np.sin(yaw0)])   # +body_x-forward push (same as corner_speed entry)
    c.arm()
    flog = ftm.from_args(sys.argv, RG, "frame_probe", store=s)   # dashboard hard rule
    t0 = time.time(); last = -1
    print(f"frame_probe yaw0={np.degrees(yaw0):.0f}deg  (cols: raw=ds.vel_ned, body=R^T@vel_ned)", flush=True)
    while time.time() - t0 < MAX_T:
        ds = s.get_drone()
        if ds is not None:
            t = time.time() - t0
            vel = ds.vel_ned
            Rc = quat_to_R(ds.quat_wxyz)
            body = Rc.T @ np.asarray(vel, float)
            a = np.zeros(3)
            a[:2] = float(np.clip(1.0 * (5.0 - float(vel[:2] @ push)), -1.0, 0.8)) * push
            a[2] = KP_Z * (z_sp - ds.pos_ned[2]) + KD_Z * (0.0 - ds.vel_ned[2])
            yaw_cur = float(np.arctan2(Rc[1, 0], Rc[0, 0]))
            q_des = mat_to_quat(desired_attitude(a, yaw0))
            w_des = KP_ATT * attitude_error_quat(ds.quat_wxyz, q_des)
            w_des[2] = KP_YAW * ((yaw0 - yaw_cur + np.pi) % (2 * np.pi) - np.pi) - KD_YAW * float(ds.omega[2])
            thr = accel_to_thrust_norm(collective_accel(a, ds.quat_wxyz), HOVER, KA)
            c.send_attitude_target(np.clip(w_des / RG, -WMAX, WMAX), thr)
            if flog is not None:
                flog.push(t, ds, {"a": a, "w_des": w_des, "q_des": q_des, "thr": thr}, running=s.get_race_live(), armed=True)
            tilt = float(np.degrees(np.arccos(max(-1, min(1, Rc[2, 2])))))
            spd = float(np.linalg.norm(vel[:2]))
            beta = sideslip_deg(ds.quat_wxyz, ds.vel_ned)
            if int(t / 0.5) != last:
                last = int(t / 0.5)
                print(f"t={t:4.1f} spd={spd:4.1f} tilt={tilt:3.0f} yaw={np.degrees(yaw_cur):+4.0f} beta={beta:+4.0f} "
                      f"raw=[{vel[0]:+5.2f},{vel[1]:+5.2f},{vel[2]:+5.2f}] body=[{body[0]:+5.2f},{body[1]:+5.2f},{body[2]:+5.2f}]",
                      flush=True)
            if tilt > ABORT_TILT:
                print(f"ABORT tilt={tilt:.0f} t={t:.1f}", flush=True); break
        time.sleep(LOOP_DT)
    print("done", flush=True)


if __name__ == "__main__":
    main()
