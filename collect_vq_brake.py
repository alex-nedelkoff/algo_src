"""collect_vq_brake.py -- collect CONTROLLED braking / velocity-reversal data for the braking-
weathervane refit (DEPLOY-02: live nose snaps beta 170->13 during the policy's braking approach;
the regime v_body_x sweeping zero AT SPEED is absent from every prior dataset -- collect_vq decels
are from v<3, ramp/crab/lateral runs tumbled before any controlled decel).

Profile per event: ACCEL (race_cruise law, AL_MAX-governed) to a target speed -> HOLD -> hard BRAKE
(speed target 0, decel tilt-capped) with small alternating ROLL/YAW rate-DOUBLETS superimposed
during the brake (REFIT-01 lesson: the wv moment is unidentifiable unexcited). Heading FIXED
(camera-forward) so the nose stays anti-velocity while v_body_x sweeps to zero = the snap regime,
under a controller whose commands are smooth + recorded. 3 events per flight (v~3.5 / 5 / 6.5).
Reuses race_cruise's control law VERBATIM (KP/KD, desired_attitude, mixer) + Recorder + dashboard
+ tilt gate. RUN ON A CLEAN SIM.
  python collect_vq_brake.py [--brakeacc 4.5] [--damp 1.2] [--tiltcap 28] [--no-viz]
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
BRAKE_ACC = argf("--brakeacc", 4.5)        # decel cap during BRAKE (m/s^2, still tilt-capped overall)
DAMP = argf("--damp", 1.2)                 # doublet amplitude on w_des during BRAKE (rad/s)
DPER = argf("--dper", 0.5)                 # doublet period (s); axis alternates roll/yaw per period
EVENTS = [(1.2, 3.0), (2.0, 4.5), (3.0, 6.0)]   # (AL_MAX accel cap, brake-entry |speed|)
# NOTE the v_al-sign quirk (known, race_cruise lineage): flying camera-forward reads v_al ~ -|v|.
# All speed thresholds use abs(v_al); BRAKE pushes constant REVERSE accel (taper near stop).
ACCEL_TIMEOUT = 14.0; HOLD_T = 1.5; BRAKE_TIMEOUT = 7.0
# BRANCH GUARD: run-2..5 data came out vbx>0 = NOSE-first (beta~0, stable branch) -- the SETTLE
# v->0 loop is unstable under the v_al-sign quirk and picked the wrong branch. The snap regime is
# CAMERA-forward (vbx<0). So: SETTLE = no horizontal push; once |v_al|>1.5 in the first ACCEL,
# check odometry vbx (BODY frame) and flip the push direction if vbx>0.
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
    print(f"collect_vq_brake events={EVENTS} brakeacc={BRAKE_ACC} damp={DAMP} tiltcap={TILT_CAP_DEG}", flush=True)
    assert fresh_start(), "not live"
    ds0 = s.get_drone(); spawn = ds0.pos_ned.copy(); z_sp = float(spawn[2])
    yaw0 = float(np.arctan2(quat_to_R(ds0.quat_wxyz)[1, 0], quat_to_R(ds0.quat_wxyz)[0, 0]))
    travel = -np.array([np.cos(yaw0), np.sin(yaw0)])          # camera-forward = -body_x
    lat_hat = np.array([-travel[1], travel[0]])
    c.arm()
    flog = ftm.from_args(sys.argv, RG, run_name="collect_vq_brake", store=s)
    rec = Recorder(s, script="collect_vq_brake", mode="rate",
                   notes="controlled hard-brake events + roll/yaw doublets for braking-weathervane refit (COR-127 DEPLOY-02 snap regime)",
                   extra_meta={"cmd_layout": ["wx", "wy", "wz", "thrust"], "base_controller": "race_cruise",
                               "events": EVENTS, "brake_acc": BRAKE_ACC, "doublet_amp": DAMP,
                               "doublet_period": DPER, "tilt_cap_deg": TILT_CAP_DEG})
    t0 = time.time(); last = -1; _, coll0 = s.get_collision(); max_tilt = 0.0
    ev = 0; phase = "SETTLE"; ph_t0 = 0.0; ph_log = []; branch_checked = False
    try:
        while True:
            ds = s.get_drone()
            if ds is None:
                time.sleep(LOOP_DT); continue
            t = time.time() - t0
            v_al = float(ds.vel_ned[:2] @ travel); v_ct = float(ds.vel_ned[:2] @ lat_hat)

            # --- phase machine ---
            if phase == "SETTLE" and t >= SETTLE:
                phase, ph_t0 = "ACCEL", t
            elif phase == "ACCEL":
                if abs(v_al) >= EVENTS[ev][1] or (t - ph_t0) > ACCEL_TIMEOUT:
                    print(f"  ev{ev} ACCEL->HOLD at v={v_al:.1f} t={t:.1f}", flush=True)
                    phase, ph_t0 = "HOLD", t
            elif phase == "HOLD" and (t - ph_t0) > HOLD_T:
                print(f"  ev{ev} HOLD->BRAKE at v={v_al:.1f} t={t:.1f}", flush=True)
                phase, ph_t0 = "BRAKE", t
                ph_log.append({"phase": f"brake{ev}", "t": [t, None]})
            elif phase == "BRAKE" and (abs(v_al) < 0.4 or (t - ph_t0) > BRAKE_TIMEOUT):
                print(f"  ev{ev} BRAKE done at v={v_al:.1f} t={t:.1f}", flush=True)
                if ph_log:
                    ph_log[-1]["t"][1] = t
                ev += 1
                if ev >= len(EVENTS):
                    phase = "END"
                else:
                    phase, ph_t0 = "ACCEL", t
            if phase == "END":
                break

            al_cap = EVENTS[ev][0]
            if phase == "ACCEL" and not branch_checked and abs(v_al) > 1.5:
                vbx_now = float(ds.vel_ned[0])               # odometry vel IS body frame
                if vbx_now > 0:                               # nose-first -> flip to camera-forward branch
                    travel = -travel; lat_hat = -lat_hat
                    v_al = -v_al; v_ct = -v_ct
                    print(f"  BRANCH FLIP at t={t:.1f}: vbx={vbx_now:+.1f} (nose-first) -> push reversed", flush=True)
                branch_checked = True
            if phase in ("SETTLE",):
                a_al = 0.0
            elif phase in ("ACCEL", "HOLD"):
                a_al = float(np.clip(KD_AL * (12.0 - v_al), -4.0, al_cap))   # AL_MAX governs speed
            else:                                                            # BRAKE: constant reverse accel, tapered near stop
                a_al = -min(BRAKE_ACC, 2.0 * abs(v_al) + 0.5)
            a_ct = float(np.clip(0.5 * (0.0 - v_ct), -1.0, 1.0))             # keep lateral bounded (loose)
            a = np.zeros(3); a[:2] = a_al * travel + a_ct * lat_hat
            a[2] = KP_Z * (z_sp - ds.pos_ned[2]) + KD_Z * (0.0 - ds.vel_ned[2])
            ah = a[:2]; n = float(np.linalg.norm(ah))
            if n > TILT_MAX_ACC:
                a[:2] = ah / n * TILT_MAX_ACC
            Rc = quat_to_R(ds.quat_wxyz); yaw_cur = float(np.arctan2(Rc[1, 0], Rc[0, 0]))
            q_des = mat_to_quat(desired_attitude(a, yaw0))                   # FIXED heading
            w_des = KP_ATT * attitude_error_quat(ds.quat_wxyz, q_des)
            w_des[2] = KP_YAW * ((yaw0 - yaw_cur + np.pi) % (2 * np.pi) - np.pi) - KD_YAW * float(ds.omega[2])
            if phase == "BRAKE":                                             # wv excitation doublets
                tb = t - ph_t0
                k = int(tb / DPER)
                sgn = 1.0 if int(tb / (DPER / 2)) % 2 == 0 else -1.0
                w_des[0 if k % 2 == 0 else 2] += sgn * DAMP
            thr = accel_to_thrust_norm(collective_accel(a, ds.quat_wxyz), HOVER, KA)
            w_cmd = np.clip(w_des / RG, -WMAX, WMAX)
            c.send_attitude_target(w_cmd, thr)
            rec.log([float(w_cmd[0]), float(w_cmd[1]), float(w_cmd[2]), float(thr)])
            if flog is not None:
                flog.push(t, ds, {"a": a, "w_des": w_des, "q_des": q_des, "thr": thr},
                          tangent=travel, cruise=al_cap, running=s.get_race_live(), armed=True)
            tilt = tilt_deg(ds.quat_wxyz); max_tilt = max(max_tilt, tilt)
            if t < 3.5:
                _, coll0 = s.get_collision()
            else:
                _, coll = s.get_collision()
                if coll != coll0:
                    print(f"  COLLISION at t={t:.1f} v={v_al:+.1f} -> stop", flush=True); break
            if tilt > ABORT_TILT:
                print(f"  TUMBLE tilt={tilt:.0f} at t={t:.1f} v={v_al:+.1f} phase={phase} ev{ev} -> stop", flush=True)
                break
            k = int(t)
            if k != last:
                last = k
                print(f"t={t:4.1f} {phase:6s} ev{ev} fwd={v_al:+4.1f} ct={v_ct:+4.1f} vbx={float(ds.vel_ned[0]):+4.1f} tilt={tilt:3.0f}", flush=True)
            time.sleep(LOOP_DT)
    finally:
        idle(); rec.close()
        if flog is not None:
            flog.close()
    print(f"DONE collect_vq_brake: events completed={ev}/{len(EVENTS)} max_tilt={max_tilt:.0f} (run {rec.run_id})", flush=True)


if __name__ == "__main__":
    main()
