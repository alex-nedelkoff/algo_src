"""vq_track_wp.py -- fly the VQ course through TRACK_INFO gate poses (VQ1 qualification track).

TRACK-WP-01 discovery: the sim broadcasts ENCAP_TRACK_INFO (gate id/pos_ned/quat/size) after a
race reset -- io_layer has parsed it all along (store.get_gates()). Vision estimates (vq_gate_wp
arc-calibrated) match gate 0 to 0.1 m horizontally, so positions are cross-validated; this script
flies the gates DIRECTLY as waypoints through vq_waypoint2's proven guidance (true-frame attitude,
s_lat probe, M2 velocity-vector loop). Each gate becomes pre/post through-points along the local
course direction. Pass truth = active_gate_index increments.

Same documented-interface caveat as ODOMETRY (TRACK_INFO is not in Table 4.3) -- vision pipeline
remains the contingency; this is the reliability-first primary for VQ1.

Usage: python vq_track_wp.py [--v 2.2] [--ngates 8] [--thru 2.0] [--dur 300] [--tilt 15] [--no-viz]
"""
import json, sys, time
import numpy as np
from pymavlink import mavutil
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store
from aigp.commander import Commander
from aigp.geometry import quat_to_R
from fit_model import qfix
from aigp.control_math import (desired_attitude, mat_to_quat, attitude_error_quat,
                               collective_accel, accel_to_thrust_norm)
import aigp.flight_telemetry as ftm
from aigp.recorder import Recorder


def argf(flag, d): return float(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else d


VMAX = argf("--v", 2.2); N_GATES = int(argf("--ngates", 8)); DURATION = argf("--dur", 300.0)
AMAX = argf("--amax", 0.6); THRU = argf("--thru", 2.0)
WP_R = argf("--wpr", 1.2); WP_TIMEOUT = 30.0
KP_ATT = np.array([0.7, 1.6, 1.0]); KP_YAW = 3.0; KD_YAW = 0.3; KD_ATT = 0.3
KD_AL = 1.2; KD_LAT = 1.4
KP_Z = 1.8; KD_Z = 3.0
YAWRATE_MAX = 0.35
WMAX = 4.0; LOOP_DT = 0.004
TILTMAX = np.tan(np.radians(argf("--tilt", 15.0))) * 9.81; ABORT_TILT = 60.0
IDLE = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE

r = json.load(open("sysid/sim_response.json"))
HOVER = r["hover_thrust"]; KA = r["k_a"]
RG = np.array([r["rate_gain_axes"]["roll"], r["rate_gain_axes"]["pitch"], r["rate_gain_axes"]["yaw"]])
s = Store(); m = MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); VisionIO(s).start()
boot = int(time.time() * 1000); c = Commander(m.conn, boot)


def idle():
    m.conn.mav.set_attitude_target_send(int(time.time() * 1000) - boot, m.conn.target_system,
                                        m.conn.target_component, IDLE, [1, 0, 0, 0], 0, 0, 0, 0)


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


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    assert fresh_start(), "not live"
    gates = None
    t_g = time.time()
    while time.time() - t_g < 10 and gates is None:
        gates = s.get_gates(); time.sleep(0.1)
    assert gates is not None, "no TRACK_INFO after reset"
    ds0 = s.get_drone(); spawn = ds0.pos_ned.copy(); gi0 = s.get_gate_idx()
    yaw0 = float(np.arctan2(quat_to_R(ds0.quat_wxyz)[1, 0], quat_to_R(ds0.quat_wxyz)[0, 0]))
    q_t0 = qfix(ds0.quat_wxyz); R_t0 = quat_to_R(q_t0)
    cam_live = -np.array([np.cos(yaw0), np.sin(yaw0)])
    s_cam = 1.0 if float(R_t0[:2, 0] @ cam_live) > 0 else -1.0
    yaw0_t = float(np.arctan2(s_cam * R_t0[1, 0], s_cam * R_t0[0, 0]))
    yaw_ref = yaw0_t
    WFIX = np.array([1.0, -1.0, 1.0])

    # course: per gate, pre/post points THRU m before/after the gate center along the local
    # course direction (prev point -> gate); z = gate z (gate pos is the aperture center)
    ng = min(N_GATES, len(gates))
    wps = []; wp_r = []; wp_center = []; prev_pt = spawn.copy()
    for g in gates[:ng]:
        gp = np.asarray(g.pos_ned, float)
        d = gp[:2] - prev_pt[:2]; d /= max(np.linalg.norm(d), 0.1)
        wps.append(np.array([gp[0] - THRU * d[0], gp[1] - THRU * d[1], gp[2]]))
        wp_r.append(WP_R); wp_center.append(False)
        wps.append(gp.copy())                       # exact aperture center
        wp_r.append(0.8); wp_center.append(True)
        wps.append(np.array([gp[0] + THRU * d[0], gp[1] + THRU * d[1], gp[2]]))
        wp_r.append(WP_R); wp_center.append(False)
        prev_pt = gp
    wps = np.array(wps)
    print(f"track: {ng} gates -> {len(wps)} wps; gate0 {np.round(gates[0].pos_ned, 1).tolist()} "
          f"size {gates[0].width:.2f}", flush=True)
    print(f"true-frame ctl: s_cam={s_cam:+.0f} yaw0_true_cam={np.degrees(yaw0_t):+.0f}", flush=True)

    c.arm()
    flog = ftm.from_args(sys.argv, RG, run_name="vq_track_wp", store=s)
    rec = Recorder(s, script="vq_track_wp", mode="rate",
                   notes="TRACK_INFO gates flown as waypoints via vq_waypoint2 guidance (VQ1)",
                   extra_meta={"cmd_layout": ["wx", "wy", "wz", "thrust"], "vmax": VMAX,
                               "n_gates": ng, "thru": THRU,
                               "gates_ned": [np.asarray(g.pos_ned).tolist() for g in gates[:ng]]})
    t0 = time.time(); t_prev = t0; wi = 0; wp_t0 = t0; prev_gi = gi0
    _, coll0 = s.get_collision(); max_tilt = 0.0; lastlog = -1
    s_lat = 0.0; probe_t0 = None; probe_v0 = 0.0
    pos_prev = spawn.copy(); vel_w = np.zeros(2)
    lat_course = np.array([-cam_live[1], cam_live[0]])
    z_ref = float(spawn[2])
    yaw_base = yaw0_t
    try:
        while time.time() - t0 < DURATION and wi < len(wps):
            ds = s.get_drone()
            if ds is None:
                time.sleep(LOOP_DT); continue
            now = time.time(); dt = min(now - t_prev, 0.05); t_prev = now; t = now - t0
            q_t = qfix(ds.quat_wxyz); R_t = quat_to_R(q_t)
            tilt_true = float(np.degrees(np.arccos(np.clip(R_t[2, 2], -1, 1))))
            yaw_cur = float(np.arctan2(s_cam * R_t[1, 0], s_cam * R_t[0, 0]))
            pos = ds.pos_ned.copy()
            if dt > 1e-4:
                vw_raw = (pos[:2] - pos_prev[:2]) / dt
                vel_w = 0.85 * vel_w + 0.15 * vw_raw
            pos_prev = pos.copy()
            v_lat_w = float(vel_w @ lat_course)

            wp = wps[wi]
            err = wp[:2] - pos[:2]
            dist = float(np.linalg.norm(err))

            takeoff = t < 2.5
            if takeoff:
                z_ref = float(spawn[2]) - 1.2
            else:
                z_ref = float(wp[2])

            a_al = 0.0; a_lat = 0.0
            if s_lat == 0.0 and not takeoff:
                if probe_t0 is None:
                    probe_t0 = now; probe_v0 = v_lat_w
                a_lat = 1.2
                if now - probe_t0 > 1.5:
                    s_lat = 1.0 if (v_lat_w - probe_v0) > 0 else -1.0
                    print(f"s_lat locked: {s_lat:+.0f} (dv_lat {v_lat_w - probe_v0:+.2f})", flush=True)
            elif not takeoff:
                u0 = np.array([np.cos(yaw0_t), np.sin(yaw0_t)]); p0 = np.array([-u0[1], u0[0]])
                pc = np.array([-cam_live[1], cam_live[0]])
                M2 = np.column_stack([cam_live, s_lat * pc]) @ np.column_stack([u0, p0]).T
                fwd_live = M2 @ np.array([np.cos(yaw_cur), np.sin(yaw_cur)])
                lat_live = s_lat * np.array([-fwd_live[1], fwd_live[0]])
                v_w_sp = 0.7 * err
                nv = float(np.linalg.norm(v_w_sp))
                if nv > VMAX:
                    v_w_sp *= VMAX / nv
                a_w = np.clip(KD_AL * (v_w_sp - vel_w), -TILTMAX, TILTMAX)
                a_al = float(np.clip(a_w @ fwd_live, -0.5 * TILTMAX, AMAX))
                a_lat = float(np.clip(a_w @ lat_live, -TILTMAX, TILTMAX))
                beta = float(np.arctan2(float(err @ (np.array([-fwd_live[1], fwd_live[0]]))),
                                        float(err @ fwd_live)))
                if abs(wrap(yaw_ref - yaw_cur)) < 0.12 and dist > WP_R:
                    yaw_base += float(np.clip(1.0 * s_lat * beta, -YAWRATE_MAX, YAWRATE_MAX)) * dt
                    yaw_ref = yaw_base

            a2 = float(np.clip(KP_Z * (z_ref - pos[2]) + KD_Z * (0.0 - ds.vel_ned[2]), -4.0, 4.0))
            fwd = np.array([np.cos(yaw_cur), np.sin(yaw_cur)])
            lat = np.array([-fwd[1], fwd[0]])
            a = np.zeros(3); a[:2] = a_al * fwd + a_lat * lat; a[2] = a2
            nrm = float(np.linalg.norm(a[:2]))
            if nrm > TILTMAX:
                a[:2] = a[:2] / nrm * TILTMAX
            yaw_body_t = yaw_ref if s_cam > 0 else yaw_ref + np.pi
            a[1] = -a[1]
            q_des_t = mat_to_quat(desired_attitude(a, yaw_body_t))
            om_t = np.asarray(ds.omega, float) * WFIX
            w_t = KP_ATT * attitude_error_quat(q_t, q_des_t)
            w_t[0] = float(np.clip(w_t[0] - KD_ATT * om_t[0], -2.0, 2.0))
            w_t[1] = float(np.clip(w_t[1] - KD_ATT * om_t[1], -2.0, 2.0))
            yaw_cur_b = float(np.arctan2(R_t[1, 0], R_t[0, 0]))
            w_t[2] = float(np.clip(KP_YAW * wrap(yaw_body_t - yaw_cur_b) - KD_YAW * om_t[2], -1.5, 1.5))
            w = w_t * WFIX
            c_max = 10.0 if tilt_true > 40.0 else 18.0
            thr = accel_to_thrust_norm(min(collective_accel(a, q_t), c_max), HOVER, KA)
            w_cmd = np.clip(w / RG, -WMAX, WMAX)
            c.send_attitude_target(w_cmd, thr)
            rec.log([float(w_cmd[0]), float(w_cmd[1]), float(w_cmd[2]), float(thr)])
            if flog is not None:
                flog.push(t, ds, {"a": a, "w_des": w, "q_des": q_des_t, "thr": thr},
                          tangent=cam_live, cruise=VMAX, running=s.get_race_live(), armed=True)

            gi = s.get_gate_idx()
            if gi > prev_gi:
                print(f"*** GATE PASSED t={t:.1f}s (idx {prev_gi}->{gi}) ***", flush=True)
                prev_gi = gi

            # explicit plane-crossing detector: sign flip of (pos - gate)*course_dir while inside
            # the aperture = a TRUE physical pass; print the race dict at that instant
            g_i = wi // 3
            if g_i < ng:
                gp = np.asarray(gates[g_i].pos_ned, float)
                dh = wps[3 * g_i + 2][:2] - wps[3 * g_i][:2]; dh /= max(np.linalg.norm(dh), 1e-6)
                s_now = float(np.sign((pos[:2] - gp[:2]) @ dh)) or 1.0
                key = f"_s{g_i}"
                s_old = getattr(main, key, None)
                if s_old is not None and s_now != s_old:
                    perp = float(np.hypot((pos[:2] - gp[:2]) @ np.array([-dh[1], dh[0]]),
                                          pos[2] - gp[2]))
                    half = gates[g_i].width / 2.0
                    print(f"PLANE-CROSS gate {g_i} t={t:.1f} offset={perp:.2f} "
                          f"({'INSIDE' if perp < half else 'OUTSIDE'} half={half:.2f}) "
                          f"race={s.get_race()}", flush=True)
                setattr(main, key, s_now)
            zr = 0.8 if wp_center[wi] else 1.5
            if dist < wp_r[wi] and abs(pos[2] - wp[2]) < zr:
                if wp_center[wi]:
                    print(f"CENTER {wi // 3} t={t:.1f} race={s.get_race()}", flush=True)
                wi += 1; wp_t0 = now
                print(f"wp {wi}/{len(wps)} t={t:.1f}", flush=True)
            elif now - wp_t0 > WP_TIMEOUT + dist / 1.0:
                print(f"WP {wi + 1} TIMEOUT t={t:.1f} dist={dist:.1f} -> skip", flush=True)
                wi += 1; wp_t0 = now

            max_tilt = max(max_tilt, tilt_true)
            if t < 3.5:
                _, coll0 = s.get_collision()
            else:
                _, coll = s.get_collision()
                if coll != coll0:
                    print(f"COLLISION t={t:.1f} gi={gi} -> stop", flush=True); break
            if tilt_true > ABORT_TILT:
                print(f"TUMBLE tilt={tilt_true:.0f} t={t:.1f} -> stop", flush=True); break
            k = int(t / 2.0)
            if k != lastlog:
                lastlog = k
                print(f"t={t:5.1f} wp={wi + 1}/{len(wps)} dist={dist:4.1f} "
                      f"spd={np.linalg.norm(vel_w):4.1f} z={pos[2] - spawn[2]:+5.1f} "
                      f"yerr={np.degrees(wrap(yaw_ref - yaw_cur)):+4.0f} tilt={tilt_true:3.0f} "
                      f"gi={gi}", flush=True)
            time.sleep(LOOP_DT)
    finally:
        idle(); rec.close()
        if flog is not None:
            flog.close()
    n = s.get_gate_idx() - gi0
    print(f"\nRESULT: passed {n}/{N_GATES} gate(s) in {time.time() - t0:.0f}s, max_tilt={max_tilt:.0f} "
          f"(run {rec.run_id})", flush=True)


if __name__ == "__main__":
    main()
