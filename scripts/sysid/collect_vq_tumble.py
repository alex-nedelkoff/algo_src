"""collect_vq_tumble.py -- TUMBLE-EXCITATION high-|omega| rate-loop ID (COR-127, MonoRace recipe).

The rate-loop model has ~40% one-step omega error at |omega| 2-5 rad/s -- the regime every
capped policy enters at gate-entry and the regime controlled-flight collection cannot reach
(HR collection capped at 42 deg tilt). In the sim, crashes are FREE: command open-loop
rate-cmd CHIRPS at high amplitude, accept the tumble, recover analytically between events.

Profile: CLIMB +25 m -> per event: CHIRP (one axis or mixed; wire amp 1.5/3/5; freq sweep
0.5->3 Hz over 2.5 s; thr fixed at hover) -> RECOVER (strong attitude law, low thrust while
inverted, climb back) -> next event. Yaw events LAST (hover yaw-spin = known kill regime --
if the flight dies there, roll/pitch data is already on disk). No tilt abort. Altitude floor
ends the flight before ground impact.

  python collect_vq_tumble.py [--cdur 2.5] [--alt 25] [--no-viz]
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


KP_Z = 1.8; KD_Z = 3.0
KP_REC = np.array([3.0, 3.0, 1.5]); KD_REC = 0.4   # strong recovery attitude gains
WMAX = 4.0; LOOP_DT = 0.004
SETTLE = 5.0
CHIRP_DUR = argf("--cdur", 2.5)
ALT = argf("--alt", 40.0)                  # m above spawn for the playground
F0_HZ, F1_HZ = 0.5, 3.0                    # chirp frequency sweep
THR_CHIRP = 0.27                           # hover collective during excitation
RECOVER_T = 12.0
# (label, wire amps [roll, pitch, yaw], -1 marks chirped axis sign alternation handled by sin)
EVENTS = [("yaw1.5", [0, 0, 1.5]), ("yaw3", [0, 0, 3.0]), ("yaw5", [0, 0, 5.0]),
          ("pitch3", [0, 3.0, 0]), ("mix3", [3.0, 3.0, 0]), ("roll5", [5.0, 0, 0])]

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
    print(f"collect_vq_tumble events={[e[0] for e in EVENTS]} cdur={CHIRP_DUR} alt={ALT}", flush=True)
    assert fresh_start(), "not live"
    ds0 = s.get_drone(); spawn = ds0.pos_ned.copy(); z_sp = float(spawn[2])
    z_play = z_sp - ALT                      # NED: up = -z
    z_floor = z_sp - 5.0                     # below this (closer to spawn alt) -> end flight
    yaw0 = float(np.arctan2(quat_to_R(ds0.quat_wxyz)[1, 0], quat_to_R(ds0.quat_wxyz)[0, 0]))
    c.arm()
    flog = ftm.from_args(sys.argv, RG, run_name="collect_vq_tumble", store=s)
    rec = Recorder(s, script="collect_vq_tumble", mode="rate",
                   notes="open-loop high-amplitude rate chirps (tumble-excitation, MonoRace recipe) for the high-|omega| rate-loop refit (COR-127)",
                   extra_meta={"cmd_layout": ["wx", "wy", "wz", "thrust"], "events": [e[0] for e in EVENTS],
                               "chirp_dur": CHIRP_DUR, "f_hz": [F0_HZ, F1_HZ], "thr_chirp": THR_CHIRP})
    t0 = time.time(); last = -1; _, coll0 = s.get_collision()
    ev = 0; phase = "SETTLE"; ph_t0 = 0.0; done_ev = []
    try:
        while True:
            ds = s.get_drone()
            if ds is None:
                time.sleep(LOOP_DT); continue
            t = time.time() - t0
            z = float(ds.pos_ned[2]); tilt = tilt_deg(ds.quat_wxyz)

            # --- phase machine ---
            if phase == "SETTLE" and t >= SETTLE:
                phase, ph_t0 = "CLIMB", t
            elif phase == "CLIMB" and abs(z - z_play) < 3.0:
                print(f"  ev{ev} {EVENTS[ev][0]} CHIRP start t={t:.1f}", flush=True)
                phase, ph_t0 = "CHIRP", t
            elif phase == "CHIRP" and ((t - ph_t0) > CHIRP_DUR or z > z_sp - ALT * 0.6):
                print(f"  ev{ev} {EVENTS[ev][0]} done t={t:.1f} tilt={tilt:.0f} alt={z_sp - z:+.1f}", flush=True)
                done_ev.append(EVENTS[ev][0]); phase, ph_t0 = "RECOVER", t
            elif phase == "RECOVER" and ((tilt < 20 and abs(z - z_play) < 4.0 and
                                          np.linalg.norm(ds.vel_ned) < 2.0 and (t - ph_t0) > 1.5)
                                         or (t - ph_t0) > RECOVER_T):
                if (t - ph_t0) > RECOVER_T and tilt > 45:
                    print(f"  RECOVER FAILED (tilt {tilt:.0f}) -> end flight, data kept", flush=True)
                    break
                ev += 1
                if ev >= len(EVENTS):
                    phase = "END"
                else:
                    phase, ph_t0 = "CLIMB", t
            if phase == "END":
                break
            if z > z_floor and phase in ("CHIRP", "RECOVER"):
                print(f"  ALT FLOOR t={t:.1f} (alt {z_sp - z:+.1f}) -> end flight", flush=True)
                break

            # --- commands ---
            if phase == "CHIRP":
                tc = t - ph_t0
                f = F0_HZ + (F1_HZ - F0_HZ) * tc / CHIRP_DUR
                amp = np.array(EVENTS[ev][1], float)
                w_cmd = amp * np.sin(2 * np.pi * f * tc)
                thr = THR_CHIRP
                c.send_attitude_target(w_cmd, thr)
            else:
                Rc = quat_to_R(ds.quat_wxyz); yaw_cur = float(np.arctan2(Rc[1, 0], Rc[0, 0]))
                if Rc[2, 2] < 0.3:                       # inverted-ish: low thrust, pure righting
                    q_des = mat_to_quat(desired_attitude(np.zeros(3), yaw_cur))
                    w_des = KP_REC * attitude_error_quat(ds.quat_wxyz, q_des) - KD_REC * np.asarray(ds.omega)
                    thr = 0.02
                else:
                    ztgt = z_sp - 2.0 if phase == "SETTLE" else z_play
                    a = np.zeros(3)
                    a[2] = KP_Z * (ztgt - z) + KD_Z * (0.0 - float(ds.vel_ned[2]))
                    q_des = mat_to_quat(desired_attitude(a, yaw0))
                    kp = KP_REC if phase == "RECOVER" else np.array([0.5, 1.6, 1.0])
                    w_des = kp * attitude_error_quat(ds.quat_wxyz, q_des)
                    w_des[2] = 3.0 * ((yaw0 - yaw_cur + np.pi) % (2 * np.pi) - np.pi) - 0.3 * float(ds.omega[2])
                    thr = accel_to_thrust_norm(collective_accel(a, ds.quat_wxyz), HOVER, KA)
                w_cmd = np.clip(w_des / RG, -WMAX, WMAX)
                c.send_attitude_target(w_cmd, thr)
            rec.log([float(w_cmd[0]), float(w_cmd[1]), float(w_cmd[2]), float(thr)])
            if flog is not None:
                flog.push(t, ds, {"thr": thr}, cruise=0.0, running=s.get_race_live(), armed=True)
            if t < 3.5:
                _, coll0 = s.get_collision()
            else:
                _, coll = s.get_collision()
                if coll != coll0:
                    print(f"  COLLISION t={t:.1f} -> stop", flush=True); break
            k = int(t)
            if k != last:
                last = k
                print(f"t={t:4.1f} {phase:7s} ev{ev} alt={z_sp - z:+5.1f} tilt={tilt:3.0f} "
                      f"|om|={float(np.linalg.norm(ds.omega)):4.1f} thr={thr:.2f}", flush=True)
            time.sleep(LOOP_DT)
    finally:
        idle(); rec.close()
        if flog is not None:
            flog.close()
    print(f"DONE collect_vq_tumble: events={len(done_ev)}/{len(EVENTS)} {done_ev} (run {rec.run_id})", flush=True)


if __name__ == "__main__":
    main()
