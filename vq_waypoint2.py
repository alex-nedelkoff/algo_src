"""vq_waypoint2.py -- WAYPOINT traversal on the live VQ, true-frame control (VQ1 qualification track).

First actual flight of the waypoint idea (vq_waypoint.py was written 06-08 but never flown — no
exp-log row, no run dirs). Reuses fly_gate3's PROVEN true-frame core verbatim (qfix attitude,
s_cam spawn calibration, world-y mirror fix, WFIX rates, YR_CAP, tilt-conditional collective
clamp); replaces the visual target with waypoints in the LIVE world frame (ODOMETRY pos_ned).

Frame-risk policy (don't re-derive frames): every sign that is not already proven is
AUTO-CALIBRATED from data in the first seconds of flight and printed:
  - s_lat: does +vel_ned[1] move pos along +lat_course or -lat_course?  (live world is y-mirrored)
  - s_yawb (mission 2): which yaw_ref slew direction reduces the live-world bearing error?

Missions:
  --mission 1 (default): course along the spawn camera axis with +-3 m lateral offsets —
      reachable by strafe at fixed heading. Validates position-waypoint guidance with ZERO new
      frame risk (heading never leaves spawn yaw).
  --mission 2: same legs but the course BENDS (--turn rad/leg, default 0.25) — yaw_ref slews
      toward the waypoint bearing (rate-capped, yaw-error governor).

Usage: python vq_waypoint2.py [--mission 1] [--v 2.2] [--leg 12] [--turn 0.25] [--nwp 5] [--dur 120] [--no-viz]
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


MISSION = int(argf("--mission", 1))
VMAX = argf("--v", 2.2); LEG = argf("--leg", 12.0); TURN = argf("--turn", 0.25)
NWP = int(argf("--nwp", 5)); DURATION = argf("--dur", 120.0)
SQUARE = argf("--square", 0.0)   # >0: fly a fixed-heading square of this side length (m)
AMAX = argf("--amax", 0.6)   # fwd accel cap: 0.6 -> terminal ~3 m/s vs live drag; raise to go faster
DZ = argf("--dz", 0.0)       # alternate wp z by +-DZ (vertical-channel stress)
YAWSWING = argf("--yawswing", 0.0)  # WV-FRAME-01: sinusoidal yaw_ref swing (rad) about the course
                                    # bearing at speed -- aero-wv lethality probe in the TRUE frame
FRAME = "live" if "--frame" in sys.argv and sys.argv[sys.argv.index("--frame") + 1] == "live" else "true"
# --frame live (WV-FRAME-01 B-side): IDENTICAL guidance, but the attitude/yaw chain runs in the
# LIVE chart exactly like race_cruise/fly_gate2 (desired_attitude on the raw quat, no world-y
# mirror, yaw_ref = spawn live yaw + swing, NO bearing steer). Warp prediction: commanded real
# tilt ~ 1 deg per deg of yaw-from-spawn -> dies under swing at speeds the true frame shrugs off.
WP_R = argf("--wpr", 1.5); WP_TIMEOUT = 30.0
KP_ATT = np.array([0.7, 1.6, 1.0]); KP_YAW = 3.0; KD_YAW = 0.3; KD_ATT = 0.3
KD_AL = 1.2; KD_LAT = 1.4; VLAT_MAX = 1.5
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
    global NWP
    assert fresh_start(), "not live"
    ds0 = s.get_drone(); spawn = ds0.pos_ned.copy(); gi0 = s.get_gate_idx()
    yaw0 = float(np.arctan2(quat_to_R(ds0.quat_wxyz)[1, 0], quat_to_R(ds0.quat_wxyz)[0, 0]))
    # --- true-frame control init (fly_gate3, verbatim) ---
    q_t0 = qfix(ds0.quat_wxyz); R_t0 = quat_to_R(q_t0)
    cam_live = -np.array([np.cos(yaw0), np.sin(yaw0)])           # camera world dir at spawn (proven)
    s_cam = 1.0 if float(R_t0[:2, 0] @ cam_live) > 0 else -1.0
    yaw0_t = float(np.arctan2(s_cam * R_t0[1, 0], s_cam * R_t0[0, 0]))
    yaw_ref = yaw0_t
    WFIX = np.array([1.0, -1.0, 1.0])

    # --- course in the LIVE world frame (pos_ned chart) ---
    lat_course = np.array([-cam_live[1], cam_live[0]])           # course-perp, live world
    wps = []
    if SQUARE > 0:
        # fixed-heading square (camera/forward = cam_live, right = lat_course): fwd, fwd+right,
        # right, back to spawn. Two strafe legs (right/left) exercise the lateral chain.
        S = SQUARE
        corners = [cam_live * S, cam_live * S + lat_course * S, lat_course * S, np.zeros(2)]
        for cxy in corners:
            p = spawn[:2] + cxy
            wps.append(np.array([p[0], p[1], spawn[2] - 1.5]))
    elif MISSION == 1:
        offs = [0.0, +3.0, 0.0, -3.0, 0.0, +3.0, 0.0, -3.0]
        for i in range(NWP):
            p = spawn[:2] + cam_live * LEG * (i + 1) + lat_course * offs[i % len(offs)]
            wps.append(np.array([p[0], p[1], spawn[2] - 1.5]))
    else:
        hd = 0.0; p = spawn[:2].copy()
        for i in range(NWP):
            hd += TURN
            d = np.cos(hd) * cam_live + np.sin(hd) * lat_course   # heading measured from the spawn camera axis
            p = p + d * LEG
            zoff = DZ * (1.0 if i % 2 == 0 else -1.0)
            wps.append(np.array([p[0], p[1], spawn[2] - 1.5 - zoff]))
    wps = np.array(wps)
    if SQUARE > 0:
        NWP = len(wps)
    print(f"mission {MISSION}: {NWP} wps, leg {LEG}, turn {TURN if MISSION == 2 else 0}, vmax {VMAX}"
          f"{' SQUARE %.0fm' % SQUARE if SQUARE > 0 else ''}", flush=True)
    print(f"true-frame ctl: s_cam={s_cam:+.0f} yaw0_true_cam={np.degrees(yaw0_t):+.0f}", flush=True)

    c.arm()
    flog = ftm.from_args(sys.argv, RG, run_name="vq_waypoint2", store=s)
    rec = Recorder(s, script="vq_waypoint2", mode="rate",
                   notes=f"waypoint traversal mission {MISSION} (true-frame ctl; VQ1 track)",
                   extra_meta={"cmd_layout": ["wx", "wy", "wz", "thrust"], "mission": MISSION,
                               "vmax": VMAX, "leg": LEG, "turn": TURN, "nwp": NWP,
                               "wps_live_world": wps.tolist()})
    t0 = time.time(); t_prev = t0; wi = 0; wp_t0 = t0
    _, coll0 = s.get_collision(); max_tilt = 0.0; reached = 0; lastlog = -1
    # auto-calibrated signs (printed once locked)
    s_lat = 0.0; probe_t0 = None; probe_v0 = 0.0
    s_yawb = 0.0; yawb_probe = None
    pos_prev = spawn.copy(); vel_w = np.zeros(2)
    z_ref = float(spawn[2])
    try:
        while time.time() - t0 < DURATION and wi < NWP:
            ds = s.get_drone()
            if ds is None:
                time.sleep(LOOP_DT); continue
            now = time.time(); dt = min(now - t_prev, 0.05); t_prev = now; t = now - t0
            q_t = qfix(ds.quat_wxyz); R_t = quat_to_R(q_t)
            tilt_true = float(np.degrees(np.arccos(np.clip(R_t[2, 2], -1, 1))))
            yaw_cur = float(np.arctan2(s_cam * R_t[1, 0], s_cam * R_t[0, 0]))
            pos = ds.pos_ned.copy()
            v_al = float(ds.vel_ned[0])

            # live-world velocity measured DIRECTLY from position (frame-free; run-1 lesson:
            # vb[1]'s sign convention fights the push chain — close the loop on what we control)
            if dt > 1e-4:
                vw_raw = (pos[:2] - pos_prev[:2]) / dt
                vel_w = 0.85 * vel_w + 0.15 * vw_raw
            pos_prev = pos.copy()
            v_al_w = float(vel_w @ cam_live)                     # forward speed, live world
            v_lat_w = float(vel_w @ lat_course)                  # lateral speed, live world

            wp = wps[wi]
            err = wp[:2] - pos[:2]
            dist = float(np.linalg.norm(err))
            e_al = float(err @ cam_live)                         # FIXED spawn axes (mission 1
            e_lat = float(err @ lat_course)                      # heading never changes)

            takeoff = t < 2.5
            if takeoff:
                z_ref = float(spawn[2]) - 1.2
            else:
                z_ref = float(wp[2])

            # forward: bidirectional at low speed (run 1: overshot wp became unreachable)
            v_al_sp = float(np.clip(0.7 * e_al, -0.8, VMAX))
            a_al = float(np.clip(KD_AL * (v_al_sp - v_al_w), -1.0, 0.0 if takeoff else AMAX))

            # lateral: PUSH-SIGN PROBE then world-velocity loop. t in [2.5, 4.0): command a
            # fixed lateral push, watch the achieved world lateral velocity, lock the sign.
            v_lat_sp = float(np.clip(0.9 * e_lat, -VLAT_MAX, VLAT_MAX))
            if s_lat == 0.0 and not takeoff:
                if probe_t0 is None:
                    probe_t0 = now; probe_v0 = v_lat_w
                a_lat = 1.2
                if now - probe_t0 > 1.5:
                    s_lat = 1.0 if (v_lat_w - probe_v0) > 0 else -1.0
                    print(f"s_lat locked: {s_lat:+.0f} (dv_lat {v_lat_w - probe_v0:+.2f})", flush=True)
            else:
                a_lat = float(np.clip(KD_LAT * (v_lat_sp - v_lat_w), -TILTMAX, TILTMAX)) * (s_lat if s_lat else 1.0)

            # mission 2: once s_lat is locked, the live-world <-> push-frame map M2 is fully
            # determined (rotation if s_lat=+1, reflection if -1) -- columns from the spawn facts.
            # Generalized guidance: world-velocity loop decomposed onto the CURRENT camera axes;
            # yaw steered toward the waypoint bearing with sign = s_lat (mirror flips angles).
            if MISSION == 2 and s_lat != 0.0 and not takeoff:
                u0 = np.array([np.cos(yaw0_t), np.sin(yaw0_t)]); p0 = np.array([-u0[1], u0[0]])
                pc = np.array([-cam_live[1], cam_live[0]])
                M2 = np.column_stack([cam_live, s_lat * pc]) @ np.column_stack([u0, p0]).T
                fwd_live = M2 @ np.array([np.cos(yaw_cur), np.sin(yaw_cur)])
                lat_live = s_lat * np.array([-fwd_live[1], fwd_live[0]])
                v_w_sp = np.clip(0.7, 0.0, 1.0) * err
                nv = float(np.linalg.norm(v_w_sp))
                if nv > VMAX:
                    v_w_sp *= VMAX / nv
                a_w = np.clip(KD_AL * (v_w_sp - vel_w), -TILTMAX, TILTMAX)
                main._a_w = a_w.copy()
                a_al = float(np.clip(a_w @ fwd_live, -0.5 * TILTMAX, AMAX))
                a_lat = float(np.clip(a_w @ lat_live, -TILTMAX, TILTMAX))
                beta = float(np.arctan2(float(err @ (np.array([-fwd_live[1], fwd_live[0]]))),
                                        float(err @ fwd_live)))
                if abs(wrap(yaw_ref - yaw_cur)) < (0.45 if YAWSWING > 0 else 0.12) and dist > WP_R:
                    main._yaw_base = getattr(main, "_yaw_base", yaw_ref) + \
                        float(np.clip(1.0 * s_lat * beta, -YAWRATE_MAX, YAWRATE_MAX)) * dt
                    swing = YAWSWING * np.sin(2 * np.pi * 0.15 * t)
                    yaw_ref = main._yaw_base + swing

            a2 = float(np.clip(KP_Z * (z_ref - pos[2]) + KD_Z * (0.0 - ds.vel_ned[2]), -4.0, 4.0))
            fwd = np.array([np.cos(yaw_cur), np.sin(yaw_cur)])   # TRUE-frame heading coords
            lat = np.array([-fwd[1], fwd[0]])
            a = np.zeros(3); a[:2] = a_al * fwd + a_lat * lat; a[2] = a2
            nrm = float(np.linalg.norm(a[:2]))
            if nrm > TILTMAX:
                a[:2] = a[:2] / nrm * TILTMAX

            if FRAME == "live":
                # ---- B-side: live-chart attitude law (race_cruise/fly_gate2 family) ----
                # a must be LIVE-world coords: pos_ned's chart IS the live world, so use the
                # vector-loop accel directly (no true-frame decomposition, no mirror).
                if getattr(main, "_a_w", None) is not None and not takeoff:
                    a = np.array([main._a_w[0], main._a_w[1], a2])
                    nrm = float(np.linalg.norm(a[:2]))
                    if nrm > TILTMAX:
                        a[:2] = a[:2] / nrm * TILTMAX
                swing = YAWSWING * np.sin(2 * np.pi * 0.15 * t) if not takeoff else 0.0
                yaw_ref_live = yaw0 + swing
                Rc = quat_to_R(ds.quat_wxyz)
                yaw_cur_live = float(np.arctan2(Rc[1, 0], Rc[0, 0]))
                q_des = mat_to_quat(desired_attitude(a, yaw_ref_live))
                w = KP_ATT * attitude_error_quat(ds.quat_wxyz, q_des)
                w[0] = float(np.clip(w[0] - KD_ATT * float(ds.omega[0]), -2.0, 2.0))
                w[1] = float(np.clip(w[1] - KD_ATT * float(ds.omega[1]), -2.0, 2.0))
                w[2] = float(np.clip(KP_YAW * wrap(yaw_ref_live - yaw_cur_live) - KD_YAW * float(ds.omega[2]), -1.5, 1.5))
                q_des_t = q_des
            else:
                # desired attitude in the TRUE frame; world-y mirror fix (fly_gate3 v2, verbatim)
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

            if dist < WP_R and abs(pos[2] - wp[2]) < 2.0:
                reached += 1; wi += 1; wp_t0 = now
                print(f"*** WP {reached}/{NWP} reached t={t:.1f}s ***", flush=True)
            elif now - wp_t0 > WP_TIMEOUT:
                print(f"WP {wi + 1} TIMEOUT t={t:.1f} dist={dist:.1f} -> skip", flush=True)
                wi += 1; wp_t0 = now

            max_tilt = max(max_tilt, tilt_true)
            if t < 3.5:
                _, coll0 = s.get_collision()
            else:
                _, coll = s.get_collision()
                if coll != coll0:
                    print(f"COLLISION t={t:.1f} -> stop", flush=True); break
            if tilt_true > ABORT_TILT:
                print(f"TUMBLE tilt={tilt_true:.0f} t={t:.1f} -> stop", flush=True); break
            k = int(t / 2.0)
            if k != lastlog:
                lastlog = k
                print(f"t={t:5.1f} wp={wi + 1} dist={dist:4.1f} e_al={e_al:+4.1f} e_lat={e_lat:+4.1f} "
                      f"vfw={v_al_w:+4.1f} vlw={v_lat_w:+4.1f} z={pos[2] - spawn[2]:+4.1f} "
                      f"yerr={np.degrees(wrap(yaw_ref - yaw_cur)):+4.0f} tilt={tilt_true:3.0f}", flush=True)
            time.sleep(LOOP_DT)
    finally:
        idle(); rec.close()
        if flog is not None:
            flog.close()
    print(f"\nRESULT: {reached}/{NWP} waypoints in {time.time() - t0:.0f}s, max_tilt={max_tilt:.0f} "
          f"(run {rec.run_id})", flush=True)


if __name__ == "__main__":
    main()
