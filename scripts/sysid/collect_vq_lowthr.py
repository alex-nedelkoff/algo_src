"""collect_vq_lowthr.py -- collect CLEAN low-throttle force-ID data (COR-127 CAP-02 follow-up:
the last live-sim gap is the thr<0.2 regime; the pooled deploy/brake recordings cover it only
with tumble-contaminated blocks, n~30, and the fitted sign is ambiguous).

Profile per event: ACCEL (race_cruise law verbatim, brake-collector lineage) to a target speed
-> HOLD -> THRCUT: thrust OPEN-LOOP at a fixed low value while the attitude loop holds a fixed
commanded tilt (level, or pitched into the velocity = the policy's braking attitude), ~1.8 s or
until the altitude guard trips -> RECOVER (full law, climb back to spawn altitude, settle)
-> next event. Hover-drop events (target_v 0) give pure-vertical descent rows (vbz>0, vpl~0).

Coverage goal: thr {0, 0.05, 0.10, 0.16} x vpl {0, 5, 8, 10} x commanded tilt {0, 30, 45}
with |omega| small throughout (no doublets -- this is a FORCE fit, not a moment fit).

  python collect_vq_lowthr.py [--cutdur 1.8] [--maxdrop 12] [--no-viz]
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
KD_AL = 1.2; KP_Z = 1.8; KD_Z = 3.0
WMAX = 4.0; LOOP_DT = 0.004
SETTLE = 6.0
TILT_CAP_DEG = argf("--tiltcap", 28.0); TILT_MAX_ACC = np.tan(np.radians(TILT_CAP_DEG)) * 9.81
CUT_DUR = argf("--cutdur", 1.8)            # s of open-loop low thrust per event
MAX_DROP = argf("--maxdrop", 12.0)         # m altitude-loss guard during a cut
# (target |v| at cut, thr during cut, commanded tilt deg INTO -travel during cut)
EVENTS = [(0.0, 0.00, 0.0), (0.0, 0.10, 0.0),
          (5.0, 0.00, 0.0), (5.0, 0.10, 0.0),
          (8.0, 0.00, 30.0), (8.0, 0.10, 30.0), (8.0, 0.16, 30.0),
          (10.0, 0.05, 45.0)]
ACCEL_TIMEOUT = 14.0; HOLD_T = 1.0; RECOVER_T = 6.0
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


def main():
    print(f"collect_vq_lowthr events={EVENTS} cutdur={CUT_DUR} maxdrop={MAX_DROP}", flush=True)
    assert fresh_start(), "not live"
    ds0 = s.get_drone(); spawn = ds0.pos_ned.copy(); z_sp = float(spawn[2])
    yaw0 = float(np.arctan2(quat_to_R(ds0.quat_wxyz)[1, 0], quat_to_R(ds0.quat_wxyz)[0, 0]))
    travel = -np.array([np.cos(yaw0), np.sin(yaw0)])          # camera-forward = -body_x
    lat_hat = np.array([-travel[1], travel[0]])
    c.arm()
    flog = ftm.from_args(sys.argv, RG, run_name="collect_vq_lowthr", store=s)
    rec = Recorder(s, script="collect_vq_lowthr", mode="rate",
                   notes="open-loop low-thr cuts at swept speed/tilt + hover drops for the low-thr z-force refit (COR-127 CAP-02)",
                   extra_meta={"cmd_layout": ["wx", "wy", "wz", "thrust"], "base_controller": "race_cruise",
                               "events": EVENTS, "cut_dur": CUT_DUR, "tilt_cap_deg": TILT_CAP_DEG})
    t0 = time.time(); last = -1; _, coll0 = s.get_collision(); max_tilt = 0.0
    ev = 0; phase = "SETTLE"; ph_t0 = 0.0; branch_checked = False; cuts_done = 0
    try:
        while True:
            ds = s.get_drone()
            if ds is None:
                time.sleep(LOOP_DT); continue
            t = time.time() - t0
            v_al = float(ds.vel_ned[:2] @ travel); v_ct = float(ds.vel_ned[:2] @ lat_hat)
            tgt_v, thr_cut, cut_tilt = EVENTS[ev]
            drop = float(ds.pos_ned[2] - z_sp)               # NED: + = below spawn

            # --- phase machine ---
            if phase == "SETTLE" and t >= SETTLE:
                phase, ph_t0 = ("HOLD" if tgt_v < 0.5 else "ACCEL"), t
            elif phase == "ACCEL":
                if abs(v_al) >= tgt_v or (t - ph_t0) > ACCEL_TIMEOUT:
                    print(f"  ev{ev} ACCEL->HOLD v={v_al:.1f} t={t:.1f}", flush=True)
                    phase, ph_t0 = "HOLD", t
            elif phase == "HOLD" and (t - ph_t0) > HOLD_T:
                print(f"  ev{ev} HOLD->THRCUT thr={thr_cut} tilt={cut_tilt} v={v_al:.1f} t={t:.1f}", flush=True)
                phase, ph_t0 = "THRCUT", t
            elif phase == "THRCUT" and ((t - ph_t0) > CUT_DUR or drop > MAX_DROP):
                print(f"  ev{ev} THRCUT done t={t:.1f} drop={drop:.1f}", flush=True)
                cuts_done += 1
                phase, ph_t0 = "RECOVER", t
            elif phase == "RECOVER" and ((t - ph_t0) > RECOVER_T or
                                         (abs(drop) < 3.0 and abs(v_al) < 1.5 and (t - ph_t0) > 2.0)):
                ev += 1
                if ev >= len(EVENTS):
                    phase = "END"
                else:
                    phase, ph_t0 = ("HOLD" if EVENTS[ev][0] < 0.5 else "ACCEL"), t
                    print(f"  ev{ev} start t={t:.1f}", flush=True)
            if phase == "END":
                break

            if phase == "ACCEL" and not branch_checked and abs(v_al) > 1.5:
                vbx_now = float(ds.vel_ned[0])               # odometry vel IS body frame
                if vbx_now > 0:
                    travel = -travel; lat_hat = -lat_hat
                    v_al = -v_al; v_ct = -v_ct
                    print(f"  BRANCH FLIP: vbx={vbx_now:+.1f} -> push reversed", flush=True)
                branch_checked = True

            if phase == "THRCUT":
                # attitude loop active at a FIXED commanded tilt; thrust OPEN-LOOP at thr_cut
                a_att = np.zeros(3)
                a_att[:2] = 9.81 * np.tan(np.radians(cut_tilt)) * (-travel)   # pitch into velocity (brake attitude)
                Rc = quat_to_R(ds.quat_wxyz); yaw_cur = float(np.arctan2(Rc[1, 0], Rc[0, 0]))
                q_des = mat_to_quat(desired_attitude(a_att, yaw0))
                w_des = KP_ATT * attitude_error_quat(ds.quat_wxyz, q_des)
                w_des[2] = KP_YAW * ((yaw0 - yaw_cur + np.pi) % (2 * np.pi) - np.pi) - KD_YAW * float(ds.omega[2])
                thr = float(thr_cut)
            else:
                if phase in ("SETTLE",):
                    a_al = 0.0
                elif phase in ("ACCEL", "HOLD"):
                    a_al = float(np.clip(KD_AL * (tgt_v - v_al), -3.0, 2.5)) if tgt_v >= 0.5 else 0.0
                else:                                        # RECOVER: kill speed, climb home
                    a_al = float(np.clip(KD_AL * (0.0 - v_al), -3.0, 3.0))
                a_ct = float(np.clip(0.5 * (0.0 - v_ct), -1.0, 1.0))
                a = np.zeros(3); a[:2] = a_al * travel + a_ct * lat_hat
                a[2] = KP_Z * (z_sp - ds.pos_ned[2]) + KD_Z * (0.0 - ds.vel_ned[2])
                ah = a[:2]; n = float(np.linalg.norm(ah))
                if n > TILT_MAX_ACC:
                    a[:2] = ah / n * TILT_MAX_ACC
                Rc = quat_to_R(ds.quat_wxyz); yaw_cur = float(np.arctan2(Rc[1, 0], Rc[0, 0]))
                q_des = mat_to_quat(desired_attitude(a, yaw0))
                w_des = KP_ATT * attitude_error_quat(ds.quat_wxyz, q_des)
                w_des[2] = KP_YAW * ((yaw0 - yaw_cur + np.pi) % (2 * np.pi) - np.pi) - KD_YAW * float(ds.omega[2])
                thr = accel_to_thrust_norm(collective_accel(a, ds.quat_wxyz), HOVER, KA)
            w_cmd = np.clip(w_des / RG, -WMAX, WMAX)
            c.send_attitude_target(w_cmd, thr)
            rec.log([float(w_cmd[0]), float(w_cmd[1]), float(w_cmd[2]), float(thr)])
            if flog is not None:
                flog.push(t, ds, {"w_des": w_des, "q_des": q_des, "thr": thr},
                          tangent=travel, cruise=tgt_v, running=s.get_race_live(), armed=True)
            tilt = tilt_deg(ds.quat_wxyz); max_tilt = max(max_tilt, tilt)
            if t < 3.5:
                _, coll0 = s.get_collision()
            else:
                _, coll = s.get_collision()
                if coll != coll0:
                    print(f"  COLLISION t={t:.1f} v={v_al:+.1f} -> stop", flush=True); break
            if tilt > ABORT_TILT:
                print(f"  TUMBLE tilt={tilt:.0f} t={t:.1f} phase={phase} ev{ev} -> stop", flush=True)
                break
            k = int(t)
            if k != last:
                last = k
                print(f"t={t:4.1f} {phase:7s} ev{ev} fwd={v_al:+4.1f} vz={float(ds.vel_ned[2]):+4.1f} "
                      f"drop={drop:+4.1f} tilt={tilt:3.0f} thr={thr:.2f}", flush=True)
            time.sleep(LOOP_DT)
    finally:
        idle(); rec.close()
        if flog is not None:
            flog.close()
    print(f"DONE collect_vq_lowthr: cuts={cuts_done}/{len(EVENTS)} max_tilt={max_tilt:.0f} (run {rec.run_id})", flush=True)


if __name__ == "__main__":
    main()
