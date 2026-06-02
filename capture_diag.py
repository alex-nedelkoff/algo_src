"""Capture full-rate telemetry for the diagnostic report. Same ACRO rate
controller, three setpoints: hover (vel_sp=0), stable (+fwd), unstable (-fwd).
Logs every control step to CSV: pos/vel NED, euler+tilt, measured body rates,
commanded accel, commanded rates, thrust."""
import json, time, csv
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
KP_ATT = 2.5; KP_YAW = 2.0; WMAX = 4.0; LOOP_DT = 0.004
KD_H = 1.2; KP_Z = 1.8; KD_Z = 3.0; SPEED = 2.0; TILT_MAX_ACC = np.tan(np.radians(12)) * 9.81
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


def euler_deg(R):
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    pitch = np.degrees(-np.arcsin(max(-1, min(1, R[2, 0]))))
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return roll, pitch, yaw


def run(mode, fname, T):
    assert fresh_start(), "not live"
    ds0 = s.get_drone(); spawn = ds0.pos_ned.copy(); z_sp = spawn[2]
    yaw0 = float(np.arctan2(quat_to_R(ds0.quat_wxyz)[1, 0], quat_to_R(ds0.quat_wxyz)[0, 0]))
    base = np.array([np.cos(yaw0), np.sin(yaw0)])
    fwd = {"hover": np.zeros(2), "stable": base, "unstable": -base}[mode]
    rows = []
    c.arm(); t0 = time.time()
    while time.time() - t0 < T:
        ds = s.get_drone()
        if ds is not None:
            vel_sp = SPEED * fwd
            a = np.zeros(3)
            a[:2] = KD_H * (vel_sp - ds.vel_ned[:2])
            a[2] = KP_Z * (z_sp - ds.pos_ned[2]) + KD_Z * (0.0 - ds.vel_ned[2])
            ah = a[:2]; n = float(np.linalg.norm(ah))
            if n > TILT_MAX_ACC:
                a[:2] = ah / n * TILT_MAX_ACC
            Rc = quat_to_R(ds.quat_wxyz); yaw_cur = float(np.arctan2(Rc[1, 0], Rc[0, 0]))
            q_des = mat_to_quat(desired_attitude(a, yaw_cur))
            w_des = KP_ATT * attitude_error_quat(ds.quat_wxyz, q_des)
            w_des[2] = KP_YAW * ((yaw0 - yaw_cur + np.pi) % (2 * np.pi) - np.pi)
            w_cmd = np.clip(w_des / RG, -WMAX, WMAX)
            thr = accel_to_thrust_norm(collective_accel(a, ds.quat_wxyz), HOVER, KA)
            c.send_attitude_target(w_cmd, thr)
            roll, pitch, yaw = euler_deg(Rc)
            tilt = np.degrees(np.arccos(max(-1, min(1, Rc[2, 2]))))
            d = ds.pos_ned - spawn
            om = ds.omega
            rows.append([time.time() - t0, d[0], d[1], d[2], ds.vel_ned[0], ds.vel_ned[1], ds.vel_ned[2],
                         roll, pitch, yaw, tilt, om[0], om[1], om[2], a[0], a[1], a[2],
                         w_cmd[0], w_cmd[1], w_cmd[2], thr])
        time.sleep(LOOP_DT)
    with open(fname, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t", "N", "E", "D", "vN", "vE", "vD", "roll", "pitch", "yaw", "tilt",
                     "p", "q", "rr", "aN", "aE", "aD", "wr", "wp", "wy", "thr"])
        wr.writerows(rows)
    print(f"{mode}: {len(rows)} rows -> {fname}  (yaw0={np.degrees(yaw0):.0f}deg)", flush=True)


run("hover", "diag_hover.csv", 12)
run("stable", "diag_stable.csv", 25)
run("unstable", "diag_unstable.csv", 25)
print("CAPTURE DONE", flush=True)
