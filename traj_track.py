"""TRAJ_TRACK -- generalizable rate-interface trajectory tracker.

race_cruise's proven loop (fixed-heading straight cruise) generalized to track a
moving reference along an OPEN GateTrajectory through arbitrary gates. The planner
(gate_traj.GateTrajectory) owns ALL geometry; this controller is geometry-agnostic,
so the SAME code flies a same-orientation 'translated' course and a gently curving
corridor -- only the gates differ. Reuses race_cruise gains + the ZVD yaw prefilter
(EXP-20a, 9.1 Hz / zeta 0.14).

Reference is DRONE-LOCKED: s_ref = nearest_s(drone) + LEAD, clamped to the path end,
so the reference can never run away from the vehicle (the over-speed failure that
sank waypoint_nav). Conservative first cut: feedback tracking + velocity-match +
yaw FF; NO centripetal FF yet (added once the base flies clean).

Usage: python traj_track.py [translated|curve] [--dry] [--no-viz]
"""
import sys, json, time, collections
import numpy as np
from pymavlink import mavutil
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store
from aigp.commander import Commander
from aigp.geometry import quat_to_R
from aigp.control_math import (desired_attitude, mat_to_quat, attitude_error_quat,
                               collective_accel, accel_to_thrust_norm)
from aigp.flight_telemetry import sideslip_deg
from aigp.recorder import Recorder
import aigp.flight_telemetry as ftm
from gate_traj import GateTrajectory, G

# --- gains (race_cruise winning config) ---
KP_ATT = np.array([0.5, 1.6, 1.0]); KP_YAW = 3.0; KD_YAW = 0.3
KP_CT = 0.5; KD_CT = 1.2; KD_AL = 1.2; KP_AL = 0.6
KP_Z = 1.8; KD_Z = 3.0
WMAX = 4.0; LOOP_DT = 0.004
# --- tilt budget (single source of truth, shared with the planner speed law) ---
TILT_BUDGET_DEG = 25.0; C_DRAG = 0.057; MARGIN = 0.6
TILT_MAX_ACC = np.tan(np.radians(TILT_BUDGET_DEG)) * G
AL_MAX = TILT_MAX_ACC                         # forward governor = tilt budget (was 0.6); planner caps v
LEAD = 2.5; ARRIVE = 1.5; CRUISE = 8.0; MAX_T = 90.0
# ZVD yaw prefilter (EXP-20a)
TD2 = 0.055; _K = np.exp(-0.14 * np.pi / np.sqrt(1 - 0.14 ** 2)); _D = 1 + 2 * _K + _K * _K
ZVD_A = [1 / _D, 2 * _K / _D, _K * _K / _D]; ZVD_T = [0.0, TD2, 2 * TD2]
YR_CAP = 1.5; _YBUF = collections.deque()

COURSES = {
    "straight":   [(12, 0.0, 0.0), (24, 0.0, 0.0), (36, 0.0, 0.0), (48, 0.0, 0.0)],
    "translated": [(12, 1.5, 0.3), (24, -1.0, -0.2), (36, 2.0, 0.5), (48, 0.0, 0.0)],
    "curve":      [(12, 1.0, 0.0), (23, 4.0, 0.5), (33, 9.0, 0.0), (41, 16.0, -1.0)],
}

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


def shape_yaw(yr_raw, now):
    _YBUF.append((now, yr_raw))
    while _YBUF and now - _YBUF[0][0] > 0.3:
        _YBUF.popleft()
    out = 0.0
    for a, d in zip(ZVD_A, ZVD_T):
        tgt = now - d; val = yr_raw; best = 1e9
        for tb, vb in _YBUF:
            e = abs(tb - tgt)
            if e < best:
                best = e; val = vb
        out += a * val
    return out


def main():
    argv = ftm.strip_viz_args(sys.argv[1:])
    pos = [a for a in argv if not a.startswith("--")]
    course = pos[0] if pos else "translated"
    body = COURSES.get(course, COURSES["translated"])

    assert fresh_start(), "not live"
    ds0 = s.get_drone(); spawn = ds0.pos_ned.copy()
    yaw0 = float(np.arctan2(quat_to_R(ds0.quat_wxyz)[1, 0], quat_to_R(ds0.quat_wxyz)[0, 0]))
    cy, sy = np.cos(yaw0), np.sin(yaw0)
    fwd = np.array([-cy, -sy]); right = np.array([-sy, cy])     # camera-forward = -body_x
    gates = [spawn.copy()]
    for (f, rg, up) in body:
        xy = spawn[:2] + f * fwd + rg * right
        gates.append(np.array([xy[0], xy[1], spawn[2] - up]))
    traj = GateTrajectory(np.array(gates), v_cruise=CRUISE, tilt_budget_deg=TILT_BUDGET_DEG,
                          c_drag=C_DRAG, margin=MARGIN)
    print(f"TRAJ_TRACK course={course} gates={len(gates)} len={traj.s_max:.1f}m CRUISE={CRUISE}", flush=True)

    if "--dry" in sys.argv:
        for frac in np.linspace(0, 1, 11):
            rr = traj.sample(frac * traj.s_max)
            print(f"  s={frac:4.2f} pos={np.round(rr['pos'], 1)} v={rr['v']:.2f} "
                  f"bank={np.degrees(np.arctan2(abs(rr['a_lat']), G)):4.1f} "
                  f"yaw={np.degrees(rr['yaw']):+6.1f} yawrate={np.degrees(rr['yaw_rate']):+6.1f}", flush=True)
        return

    c.arm()
    flog = ftm.from_args(sys.argv, RG, "traj_track")
    if flog is not None:
        flog.set_path(traj._P)
    rec = Recorder(s, script="traj_track", mode="rate",
                   notes=f"tilt-budget speed-scheduled tracker, course={course}",
                   extra_meta={"cmd_layout": ["wx", "wy", "wz", "thrust"], "base_controller": "race_cruise",
                               "tilt_budget_deg": TILT_BUDGET_DEG, "c_drag": C_DRAG, "margin": MARGIN,
                               "course": course})
    t0 = time.time(); last = -1; max_tilt = 0.0; peak_ct = 0.0; peak_spd = 0.0; bad = False
    while time.time() - t0 < MAX_T:
        ds = s.get_drone()
        if ds is not None:
            t = time.time() - t0
            # race_cruise's loop, but the frame follows the spline at the drone's projection
            s_drone = traj.nearest_s(ds.pos_ned)
            ref = traj.sample(s_drone)                          # local tangent line at the projection
            tang = ref["tang"][:2]; travel = tang / max(np.linalg.norm(tang), 1e-9)
            lat_hat = np.array([-travel[1], travel[0]])
            d = ds.pos_ned - ref["pos"]
            v_al = float(ds.vel_ned[:2] @ travel); v_ct = float(ds.vel_ned[:2] @ lat_hat)
            p_ct = float(d[:2] @ lat_hat)
            a_al = float(np.clip(KD_AL * (ref["v"] - v_al), -4.0, AL_MAX))     # pure speed governor
            # budget-aware cross-track: clamp lateral accel to the tilt budget LEFT after holding speed
            # (drag = C_DRAG*v_al^2). Limits the transient line-acquisition that triggered CRUISE-04.
            a_ct_max = max(0.0, TILT_MAX_ACC * MARGIN - C_DRAG * v_al * v_al)
            a_ct = float(np.clip(-KP_CT * p_ct - KD_CT * v_ct, -a_ct_max, a_ct_max))
            a = np.zeros(3); a[:2] = a_al * travel + a_ct * lat_hat
            a[2] = KP_Z * (ref["pos"][2] - ds.pos_ned[2]) + KD_Z * (0.0 - ds.vel_ned[2])
            ah = a[:2]; n = float(np.linalg.norm(ah))
            if n > TILT_MAX_ACC:
                a[:2] = ah / n * TILT_MAX_ACC
            Rc = quat_to_R(ds.quat_wxyz); yaw_cur = float(np.arctan2(Rc[1, 0], Rc[0, 0]))
            q_des = mat_to_quat(desired_attitude(a, ref["yaw"]))
            w_des = KP_ATT * attitude_error_quat(ds.quat_wxyz, q_des)
            w_des[2] = (KP_YAW * ((ref["yaw"] - yaw_cur + np.pi) % (2 * np.pi) - np.pi)
                        - KD_YAW * float(ds.omega[2]))           # rigid heading-hold (race_cruise)
            thr = accel_to_thrust_norm(collective_accel(a, ds.quat_wxyz), HOVER, KA)
            w_cmd = np.clip(w_des / RG, -WMAX, WMAX)
            c.send_attitude_target(w_cmd, thr)
            rec.log([float(w_cmd[0]), float(w_cmd[1]), float(w_cmd[2]), float(thr)])
            zb = Rc[:, 2]; tilt = np.degrees(np.arccos(max(-1, min(1, zb[2]))))
            max_tilt = max(max_tilt, tilt); peak_ct = max(peak_ct, abs(p_ct))
            peak_spd = max(peak_spd, float(np.linalg.norm(ds.vel_ned[:2])))
            beta = sideslip_deg(ds.quat_wxyz, ds.vel_ned)
            if flog is not None:
                flog.push(t, ds, {"a": a, "w_des": w_des, "q_des": q_des, "thr": thr},
                          nearest=ref["pos"], tangent=travel, cruise=ref["v"],
                          running=s.get_race_live(), armed=True)
            if tilt > 75:
                bad = True
            if s_drone >= traj.s_max - ARRIVE:
                print(f"REACHED END t={t:.1f}s (peak_spd={peak_spd:.1f} peak_ct={peak_ct:.1f})", flush=True)
                break
            if tilt > 80:
                print(f"ABORT tilt={tilt:.0f} t={t:.1f} s={s_drone:.1f}/{traj.s_max:.1f}", flush=True)
                break
            k = int(t / 1.0)
            if k != last:
                last = k
                tilt_cmd = np.degrees(np.arccos(max(-1, min(1, float(quat_to_R(q_des)[2, 2])))))
                print(f"t={t:4.1f} s={s_drone:5.1f}/{traj.s_max:.0f} "
                      f"v={float(np.linalg.norm(ds.vel_ned[:2])):4.1f}/{ref['v']:.1f} "
                      f"ct={p_ct:+5.1f} tiltC/A={tilt_cmd:3.0f}/{tilt:3.0f} beta={beta:+4.0f} "
                      f"yawC/A={np.degrees(ref['yaw']):+5.0f}/{np.degrees(yaw_cur):+5.0f}", flush=True)
        time.sleep(LOOP_DT)
    rec.close()
    if flog is not None:
        flog.close()
    df = s.get_drone().pos_ned
    print(f"DONE course={course} reached_s={traj.nearest_s(df):.1f}/{traj.s_max:.1f} "
          f"peak_spd={peak_spd:.1f} peak_ct={peak_ct:.1f} max_tilt={max_tilt:.0f} "
          f"{'TUMBLED' if bad else 'OK'}", flush=True)


if __name__ == "__main__":
    main()
