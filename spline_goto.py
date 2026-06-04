"""SPLINE_GOTO — smooth multi-waypoint navigation by FOLLOWING A SPLINE through the waypoints.

Unlike goto.py (straight segments + stop-and-turn at each corner, which trips the weathervane on
sharp turns), this fits a cubic spline through [spawn, wp1, wp2, ...] (sim.spline.GateSpline, the same
racing-line construct the RL spline rewards use) and CRUISES along it continuously: command velocity
along the spline tangent at the nearest point + cross-track pull onto the spline, with the NOSE
TRACKING THE TANGENT (slew-limited) so the drone always flies nose-first. The weathervane is stabilizing
nose-first (relative wind from the front) and destabilizing sideways/backward, so a fixed heading tumbles
on the return legs; tracking the tangent keeps every part of the loop in the flowing-forward regime.

STATUS (live-tested on the VQ sim, 4 instrumented runs): the SPLINE MATH is correct -- `--dry` shows a
smooth tangent field (yaw sweeps the loop continuously, mild ~7.8 m overshoot on a 6x5 box). But the
hand-tuned ACRO PID CANNOT FLY A SUSTAINED TURN: every turning attempt diverges via an UNCOMMANDED TILT
(while the cross/along accel is clamped to <=2.6 m/s^2, achieved tilt runs to 30->80 deg and speed to
>9 m/s -> tumble). Root cause is consistent across fixed-heading, nose-tracking, hover-rotate, and
slew-limited variants: the validated flight envelope of this controller is roughly-STRAIGHT, small-yaw-
error forward flight (race_cruise: 47 m straight, tilt 2 deg; goto.py single-waypoint: reached, tilt 2
deg). The moment the body must follow a rotating tangent, small misalignment feeds the weathervane and
the attitude loop loses it. Path representation was never the blocker -- TURNING AUTHORITY is. Smooth
curved/looping nav needs the learned RL policy (training, learning) or a turn-coordinated cascaded
attitude controller with feedforward, NOT more PID tuning. Kept as a documented artifact + the finding.

Usage:
  python spline_goto.py                    # safe default body box -> flown as a smooth closed loop
  python spline_goto.py body 6 0 0  6 5 0  0 5 0   # body (fwd,right,down) rel spawn heading
  python spline_goto.py world 8 0 0  0 8 0         # world NED offsets
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
from aigp.flight_telemetry import FlightLog
from sim.spline import GateSpline

KP_ATT = np.array([0.5, 1.6, 1.0]); KP_YAW = 3.0; KD_YAW = 0.3
KP_CT = 0.6; KD_CT = 1.2; AL_MAX = 0.5; KD_AL = 1.2; KP_Z = 1.8; KD_Z = 3.0
WMAX = 4.0; LOOP_DT = 0.004; TILT_MAX_ACC = np.tan(np.radians(15)) * 9.81; ABORT_TILT = 80.0
CRUISE = 1.5          # constant cruise along the spline (no stopping); modest = weathervane-safe
YAW_SLEW = 0.45       # rad/s cap on how fast the heading setpoint chases the tangent -> small yaw
                      # error at ALL times (this PID tumbles on large yaw cmds); path radius must be
                      # gentle enough that tangent-rate (CRUISE/R) stays under this.
LEAVE = 3.0; RETURN = 1.6; MAX_T = 90.0   # lap done when we leave spawn then return near it

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


def parse_args(argv):
    argv = [a for a in argv if not a.startswith("--")]
    body = False
    if argv and argv[0] in ("body", "world"):
        body = (argv[0] == "body"); argv = argv[1:]
    if len(argv) >= 3:
        f = [float(x) for x in argv]
        return body, [tuple(f[i:i + 3]) for i in range(0, len(f) - len(f) % 3, 3)]
    return True, [(6.0, 0.0, 0.0), (6.0, 5.0, 0.0), (0.0, 5.0, 0.0)]  # body box (closed-looped by spline)


def cmd(ds, a2, z_sp, yaw0):
    a = np.zeros(3); a[:2] = a2
    n = float(np.linalg.norm(a[:2]))
    if n > TILT_MAX_ACC:
        a[:2] = a[:2] / n * TILT_MAX_ACC
    a[2] = KP_Z * (z_sp - ds.pos_ned[2]) + KD_Z * (0.0 - ds.vel_ned[2])
    Rc = quat_to_R(ds.quat_wxyz); yaw_cur = float(np.arctan2(Rc[1, 0], Rc[0, 0]))
    q_des = mat_to_quat(desired_attitude(a, yaw_cur))
    w_des = KP_ATT * attitude_error_quat(ds.quat_wxyz, q_des)
    w_des[2] = KP_YAW * ((yaw0 - yaw_cur + np.pi) % (2 * np.pi) - np.pi) - KD_YAW * float(ds.omega[2])
    thr = accel_to_thrust_norm(collective_accel(a, ds.quat_wxyz), HOVER, KA)
    c.send_attitude_target(np.clip(w_des / RG, -WMAX, WMAX), thr)   # command sent BEFORE any telemetry
    zb = Rc[:, 2]
    tilt = float(np.degrees(np.arccos(max(-1, min(1, zb[2])))))
    return tilt, {"a": a, "w_des": w_des, "q_des": q_des, "thr": thr}


def circle_offsets(R, n=8):
    """n body-frame (fwd,right,down) offsets on a radius-R circle through spawn, tangent=fwd at spawn,
    curving right. Gentle (large R) -> tangent rotates slowly -> trackable by the slew-limited yaw."""
    a = np.radians(-90 + 360.0 / n * np.arange(1, n))   # spawn is point 0; these are the other n-1
    return [(float(R * np.cos(t)), float(R + R * np.sin(t)), 0.0) for t in a]


def main():
    argv = sys.argv[1:]
    rrd = None
    if "--rrd" in argv:                       # --rrd <path> saves a shareable recording instead of live
        i = argv.index("--rrd"); rrd = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    if any(a == "circle" for a in argv):
        nf = [a for a in argv if not a.startswith("--")]
        R = float(nf[nf.index("circle") + 1]) if len(nf) > nf.index("circle") + 1 else 6.0
        body, wps = True, circle_offsets(R)
    else:
        body, wps = parse_args(argv)
    assert fresh_start(), "not live"
    ds0 = s.get_drone(); spawn = ds0.pos_ned.copy()
    yaw0 = float(np.arctan2(quat_to_R(ds0.quat_wxyz)[1, 0], quat_to_R(ds0.quat_wxyz)[0, 0]))
    if body:
        cy, sy = np.cos(yaw0), np.sin(yaw0)
        fwd = np.array([-cy, -sy]); right = np.array([-sy, cy])
        offs = [np.array([o[0] * fwd[0] + o[1] * right[0], o[0] * fwd[1] + o[1] * right[1], o[2]]) for o in wps]
    else:
        offs = [np.array(o, float) for o in wps]
    pts = np.vstack([spawn] + [spawn + o for o in offs])       # spline through spawn + waypoints
    spline = GateSpline(pts, samples_per_segment=60)
    print(f"SPLINE_GOTO through {len(pts)} points ({'body' if body else 'world'}): {wps}", flush=True)
    if "--dry" in sys.argv:
        print(f"  yaw0(spawn heading)={np.degrees(yaw0):6.1f}deg", flush=True)
        n0, t0 = spline.nearest_point_and_tangent(spawn)
        ys0 = np.degrees(np.arctan2(-t0[1], -t0[0]))
        print(f"  @spawn: tangent={t0[:2].round(2)} yaw_sp={ys0:6.1f}deg  err_vs_yaw0={((ys0-np.degrees(yaw0)+180)%360)-180:6.1f}deg", flush=True)
        # sample tangent heading + offset-from-spline along the whole loop
        samp = spline._samples; tans = spline._tangents
        prev = np.degrees(yaw0); mx = 0.0
        for i in range(0, len(samp), max(1, len(samp)//16)):
            ys = np.degrees(np.arctan2(-tans[i, 1], -tans[i, 0]))
            d = ((ys - prev + 180) % 360) - 180; mx = max(mx, abs(d)); prev = ys
            off = np.linalg.norm((samp[i] - spawn)[:2])
            print(f"   s[{i:3d}] pos_off={off:5.1f} yaw_sp={ys:7.1f} dyaw={d:6.1f}", flush=True)
        print(f"  max |dyaw| between samples = {mx:.1f}deg  (overshoot/wiggle if large)", flush=True)
        return
    c.arm()
    flog = FlightLog(rate_gain=RG, decimate=5, rrd_path=rrd) if ("--viz" in sys.argv or rrd) else None
    if flog is not None:
        flog.set_path(spline._samples)
    t0 = time.time(); last = -1; left = False
    yaw_cmd = yaw0; t_prev = t0    # heading setpoint, slewed toward the tangent (starts at spawn heading)
    viz_acc = 0.0; viz_max = 0.0; viz_n = 0
    while time.time() - t0 < MAX_T:
        ds = s.get_drone()
        if ds is not None:
            d_spawn = float(np.linalg.norm((spawn - ds.pos_ned)[:2]))
            if d_spawn > LEAVE:
                left = True
            if left and d_spawn < RETURN:
                print(f"LAP COMPLETE (returned to spawn, {time.time()-t0:.0f}s)", flush=True); break
            nearest, tan = spline.nearest_point_and_tangent(ds.pos_ned)
            th = tan[:2]; nt = float(np.linalg.norm(th))
            th = th / nt if nt > 1e-6 else np.array([1.0, 0.0])    # forward tangent (unit)
            lat = np.array([-th[1], th[0]])
            v_al = float(ds.vel_ned[:2] @ th); v_ct = float(ds.vel_ned[:2] @ lat)
            p_ct = float((ds.pos_ned - nearest)[:2] @ lat)
            a_al = float(np.clip(KD_AL * (CRUISE - v_al), -2.0, AL_MAX))   # hold cruise along spline
            a2 = a_al * th + (-KP_CT * p_ct - KD_CT * v_ct) * lat
            # Nose tracks the tangent -> always fly NOSE-FIRST = weathervane-stable. Camera-forward in NED
            # is -body-x (fwd=[-cos psi,-sin psi]), so point body-x opposite the tangent: psi=atan2(-thE,-thN).
            # SLEW-limit the setpoint so the yaw error (and thus the yaw-rate cmd) stays small at all times.
            yaw_tgt = float(np.arctan2(-th[1], -th[0]))
            now = time.time(); dt = now - t_prev; t_prev = now
            dyaw = ((yaw_tgt - yaw_cmd + np.pi) % (2 * np.pi)) - np.pi
            yaw_cmd += float(np.clip(dyaw, -YAW_SLEW * dt, YAW_SLEW * dt))
            tilt, dbg = cmd(ds, a2, float(nearest[2]), yaw_cmd)
            if flog is not None:                      # log AFTER the command is sent (never delays it)
                _tb = time.perf_counter()
                flog.step(time.time() - t0, ds, dbg, nearest=nearest, tangent=th,
                          cruise=CRUISE, running=s.get_race_live(), armed=True)
                _d = time.perf_counter() - _tb; viz_acc += _d; viz_max = max(viz_max, _d); viz_n += 1
            if tilt > ABORT_TILT:
                print(f"ABORT tilt={tilt:.0f}", flush=True); return
            k = int((time.time() - t0) / 2.0)
            if k != last:
                last = k
                ctd = spline.distance_to_nearest(ds.pos_ned)
                print(f"t={time.time()-t0:4.1f} d_spawn={d_spawn:5.1f} xtrack={ctd:4.1f} "
                      f"spd={np.linalg.norm(ds.vel_ned[:2]):4.1f} tilt={tilt:3.0f}", flush=True)
        time.sleep(LOOP_DT)
    else:
        print("MAX_T reached", flush=True)
    if viz_n:
        print(f"viz overhead: avg {1e3*viz_acc/viz_n:.3f} ms, max {1e3*viz_max:.3f} ms over {viz_n} logs "
              f"(loop budget {1e3*LOOP_DT:.0f} ms; decimate={flog.decimate})", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
