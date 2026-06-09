"""collect_vq_crab.py -- collect CONTROLLED high-sideslip data to validate/refit the weathervane in
the dynamic (forward-racing) regime. The trap prior sysID hit: closed-loop slalom suppresses sideslip
(can't ID the weathervane). Fix: decouple "make sideslip" from "stay upright" -- hold forward speed +
FIXED heading, command a LATERAL-VELOCITY crab (not line-hold) so the drone slides sideways relative
to its nose -> sustained, controlled v_body_y. The attitude loop keeps it level; the crab supplies the
sideslip. Alternating-sign, growing-magnitude velocity steps keep position bounded while sweeping
sideslip up to the wall. Reuses race_cruise's control law VERBATIM + Recorder + dashboard + tilt gate.

Forward speed is set by --almax (drag-balance), NOT --cruise (the v_al-sign quirk makes the setpoint
non-binding). Run at several --almax for v~3/5/7. RUN ON A CLEAN SIM.
  python collect_vq_crab.py [--almax 1.5] [--vlat 4] [--step 4] [--dur 44] [--tiltcap 30] [--no-viz]
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
from aigp.flight_telemetry import tilt_deg
import aigp.flight_telemetry as ftm
from aigp.recorder import Recorder


def argf(flag, d): return float(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else d


KP_ATT = np.array([0.5, 1.6, 1.0]); KP_YAW = 3.0; KD_YAW = 0.3
KP_CT = 0.5; KD_CT = 1.2; KD_AL = 1.2
KP_Z = 1.8; KD_Z = 3.0
WMAX = 4.0; LOOP_DT = 0.004
AL_MAX = argf("--almax", 1.5); VLAT = argf("--vlat", 4.0); STEP = argf("--step", 4.0)
DUR = argf("--dur", 44.0); SETTLE = 6.0
TILT_CAP_DEG = argf("--tiltcap", 30.0); TILT_MAX_ACC = np.tan(np.radians(TILT_CAP_DEG)) * 9.81
ABORT_TILT = 75.0

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


def crab_vref(t):
    """Alternating-sign, growing-magnitude lateral-velocity setpoint (m/s). 0 during settle."""
    if t < SETTLE:
        return 0.0, 0.0
    te = t - SETTLE
    level = min(VLAT, 0.5 + (VLAT - 0.5) * te / max(DUR - SETTLE, 1e-6))
    sign = 1.0 if int(te / STEP) % 2 == 0 else -1.0
    return sign * level, level


def main():
    print(f"collect_vq_crab AL_MAX={AL_MAX} VLAT={VLAT} STEP={STEP} DUR={DUR} tiltcap={TILT_CAP_DEG}", flush=True)
    assert fresh_start(), "not live"
    ds0 = s.get_drone(); spawn = ds0.pos_ned.copy(); z_sp = float(spawn[2])
    yaw0 = float(np.arctan2(quat_to_R(ds0.quat_wxyz)[1, 0], quat_to_R(ds0.quat_wxyz)[0, 0]))
    travel = -np.array([np.cos(yaw0), np.sin(yaw0)])          # camera-forward = -body_x
    lat_hat = np.array([-travel[1], travel[0]])
    c.arm()
    flog = ftm.from_args(sys.argv, RG, run_name="collect_vq_crab", store=s)
    rec = Recorder(s, script="collect_vq_crab", mode="rate",
                   notes="controlled high-sideslip crab sweep for weathervane refit (dynamic regime)",
                   extra_meta={"cmd_layout": ["wx", "wy", "wz", "thrust"], "base_controller": "race_cruise",
                               "al_max": AL_MAX, "vlat": VLAT, "step": STEP, "tilt_cap_deg": TILT_CAP_DEG})
    t0 = time.time(); last = -1; _, coll0 = s.get_collision(); max_tilt = 0.0; max_vct = 0.0
    try:
        while time.time() - t0 < DUR + 2.0:
            ds = s.get_drone()
            if ds is None:
                time.sleep(LOOP_DT); continue
            t = time.time() - t0
            v_ref, level = crab_vref(t)
            d = ds.pos_ned - spawn
            v_al = float(ds.vel_ned[:2] @ travel); v_ct = float(ds.vel_ned[:2] @ lat_hat)
            a_al = float(np.clip(KD_AL * (12.0 - v_al), -4.0, AL_MAX))     # always push fwd; AL_MAX governs speed
            a_ct = float(np.clip(KD_CT * (v_ref - v_ct), -TILT_MAX_ACC, TILT_MAX_ACC))  # lateral-VELOCITY crab
            a = np.zeros(3); a[:2] = a_al * travel + a_ct * lat_hat
            a[2] = KP_Z * (z_sp - ds.pos_ned[2]) + KD_Z * (0.0 - ds.vel_ned[2])
            ah = a[:2]; n = float(np.linalg.norm(ah))
            if n > TILT_MAX_ACC:
                a[:2] = ah / n * TILT_MAX_ACC
            Rc = quat_to_R(ds.quat_wxyz); yaw_cur = float(np.arctan2(Rc[1, 0], Rc[0, 0]))
            q_des = mat_to_quat(desired_attitude(a, yaw0))            # FIXED heading (crab), not nose-follows-tangent
            w_des = KP_ATT * attitude_error_quat(ds.quat_wxyz, q_des)
            w_des[2] = KP_YAW * ((yaw0 - yaw_cur + np.pi) % (2 * np.pi) - np.pi) - KD_YAW * float(ds.omega[2])
            thr = accel_to_thrust_norm(collective_accel(a, ds.quat_wxyz), HOVER, KA)
            w_cmd = np.clip(w_des / RG, -WMAX, WMAX)
            c.send_attitude_target(w_cmd, thr)
            rec.log([float(w_cmd[0]), float(w_cmd[1]), float(w_cmd[2]), float(thr)])
            if flog is not None:
                flog.push(t, ds, {"a": a, "w_des": w_des, "q_des": q_des, "thr": thr},
                          tangent=travel, cruise=AL_MAX, running=s.get_race_live(), armed=True)
            tilt = tilt_deg(ds.quat_wxyz); max_tilt = max(max_tilt, tilt)
            if t >= SETTLE and tilt < 40:
                max_vct = max(max_vct, abs(v_ct))
            if t < 3.5:
                _, coll0 = s.get_collision()
            else:
                _, coll = s.get_collision()
                if coll != coll0:
                    print(f"  COLLISION at t={t:.1f} v_ct={v_ct:+.1f} -> stop", flush=True); break
            if tilt > ABORT_TILT:
                print(f"  TUMBLE tilt={tilt:.0f} at t={t:.1f} v_ct={v_ct:+.1f} -> stop", flush=True); break
            k = int(t)
            if k != last:
                last = k
                print(f"t={t:4.1f} fwd={v_al:+4.1f} v_ct={v_ct:+4.1f} vref={v_ref:+4.1f} tilt={tilt:3.0f}", flush=True)
            time.sleep(LOOP_DT)
    finally:
        idle(); rec.close()
        if flog is not None:
            flog.close()
    print(f"DONE collect_vq_crab almax={AL_MAX}: max |v_ct| held (tilt<40) = {max_vct:.1f} m/s, "
          f"max_tilt={max_tilt:.0f} (run {rec.run_id})", flush=True)


if __name__ == "__main__":
    main()
