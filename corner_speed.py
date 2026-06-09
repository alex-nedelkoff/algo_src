"""EXP-28 -- coordinated-turn controller: the THREE analytic FF terms, scheduled on (v,R).

Converged diagnosis (with the other agent): the turn needs three feedforwards, all closed-form in (v,R):
  1. BANK    phi = atan(v^2/(R g))  -- commanded explicitly (kills the roll over-bank), RAMPED onset
             (a roll step would re-ring the ~9 Hz zeta~0.14 mode, same as yaw).
  2. YAW-RATE psi_dot = v/R          -- nose leads with the geometry (ZVD-shaped on the yaw channel).
  3. PITCH-COMP theta_dot = sin(phi)*psi_dot  -- cancels the -sin(phi)*psi_dot KINEMATIC pitch coupling
             (bank x yaw-rate -> world pitch-rate) that leans thrust FORWARD = the speed pump. This is a
             physics term of the turn STATE, not the command path, so it survives any re-architecture and
             must be fed forward. THIS is the new term under test.
Nose tracks velocity (beta->0); bank turns the velocity (no heading ramp, no deadlock since a_cent=v^2/R
is the real centripetal). Speed held at CRUISE. Logs vel + actual/cmd quats for the a_along decomposition.

Usage: python coord_turn.py [R] [CRUISE]   (default 10 2.5 -> bank 3.6deg, psi_dot 14deg/s). Dashboard ON.
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
from aigp.flight_telemetry import sideslip_deg, tilt_deg
import aigp.flight_telemetry as ftm
from corner_ff import weathervane_ff

G = 9.81
# --- weathervane FF (WV-DYNAMIC-validated coeffs; sign/gain verified live) ---
ROLL_WV0, ROLL_WV1, YAW_WV = -0.105, -0.019, -0.149
WVFF = "--no-wvff" not in sys.argv
WVFF_SIGN = float(sys.argv[sys.argv.index("--wvff_sign") + 1]) if "--wvff_sign" in sys.argv else 1.0
WVFF_GAIN = float(sys.argv[sys.argv.index("--wvff_gain") + 1]) if "--wvff_gain" in sys.argv else 1.0
WVFF_CLIP = 1.0     # rad/s: clamp the FF so a bad sideslip estimate can't command a huge rate
KP_ATT = np.array([0.5, 1.6, 1.0]); KP_YAW = 3.0; KD_YAW = 0.3; KP_Z = 1.8; KD_Z = 3.0
KV = 1.5; ACCEL_MAX = 0.6; DECEL_MAX = 2.0
WMAX = 4.0; LOOP_DT = 0.004; ABORT_TILT = 80.0; T_ENTRY = 5.0; MAX_T = 60.0
RAMP = 4.0                                     # s: smooth onset of ALL three FF terms (bank/yaw/pitch)
PITCH_FF_SIGN = float(sys.argv[sys.argv.index("--pitch_sign") + 1]) if "--pitch_sign" in sys.argv else +1.0  # pitch-comp sign (flip if speed pumps)
MAX_BANK_DEG = 70.0; TILT_MAX_ACC = np.tan(np.radians(MAX_BANK_DEG)) * G   # EXP-33: thrust allows ~80deg (TWR 4.3)
TD2 = 0.055; _K = np.exp(-0.14*np.pi/np.sqrt(1-0.14**2)); _D = 1+2*_K+_K*_K
ZVD_A = [1/_D, 2*_K/_D, _K*_K/_D]; ZVD_T = [0.0, TD2, 2*TD2]
YR_CAP = 1.5; _YBUF = collections.deque()

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


def cmd(ds, a2, z_sp, yaw_sp, yaw_ff, pitch_ff=0.0):
    a = np.zeros(3); a[:2] = np.asarray(a2, float)
    n = float(np.linalg.norm(a[:2]))
    if n > TILT_MAX_ACC:
        a[:2] = a[:2] / n * TILT_MAX_ACC
    a[2] = KP_Z * (z_sp - ds.pos_ned[2]) + KD_Z * (0.0 - ds.vel_ned[2])
    Rc = quat_to_R(ds.quat_wxyz); yaw_cur = float(np.arctan2(Rc[1, 0], Rc[0, 0]))
    q_des = mat_to_quat(desired_attitude(a, yaw_sp))
    w_des = KP_ATT * attitude_error_quat(ds.quat_wxyz, q_des)
    w_des[1] += pitch_ff                                              # PITCH-COUPLING COMPENSATION (the new term)
    yr_raw = KP_YAW * ((yaw_sp - yaw_cur + np.pi) % (2*np.pi) - np.pi) - KD_YAW * float(ds.omega[2]) + yaw_ff
    w_des[2] = float(np.clip(shape_yaw(yr_raw, time.time()), -YR_CAP, YR_CAP))
    wcmd = w_des / RG
    if WVFF:                                               # weathervane FF: cancel wv*v_body_y
        vb = np.asarray(ds.vel_ned, float)                 # ds.vel_ned IS body-frame (raw odometry; vq_model frame_conventions + frame_probe). Do NOT R^T it.
        roll_ff, yaw_ff_wv = weathervane_ff(float(vb[0]), float(vb[1]), ROLL_WV0, ROLL_WV1, YAW_WV,
                                            RG[0], RG[2], WVFF_GAIN, WVFF_SIGN)
        wcmd[0] += float(np.clip(roll_ff, -WVFF_CLIP, WVFF_CLIP))
        wcmd[2] += float(np.clip(yaw_ff_wv, -WVFF_CLIP, WVFF_CLIP))
    thr = accel_to_thrust_norm(collective_accel(a, ds.quat_wxyz), HOVER, KA)
    c.send_attitude_target(np.clip(wcmd, -WMAX, WMAX), thr)
    return float(np.degrees(np.arccos(max(-1, min(1, Rc[2, 2]))))), {"a": a, "w_des": w_des, "q_des": q_des, "thr": thr}


def main():
    argv = ftm.strip_viz_args(sys.argv[1:])
    R = float(argv[0]) if len(argv) > 0 else 10.0
    CRUISE = float(argv[1]) if len(argv) > 1 else 2.5
    SDIR = -1.0                                  # turn direction (left); inside = SDIR*[-vhat_y, vhat_x]
    PHI_SAFE = np.radians(40.0)                  # coordinated bank envelope (well within ~80deg thrust limit)
    A_CENT_MAX = G * np.tan(PHI_SAFE)            # cap centripetal at the safe bank -> no over-bank
    C_DRAG = 0.057                                     # REFIT-02 quadratic drag
    V_MAX = float(np.sqrt(G * np.tan(PHI_SAFE) / (C_DRAG + 1.0 / R)))   # tilt budget: drag + centripetal <= budget
    assert fresh_start(), "not live"
    ds0 = s.get_drone(); spawn = ds0.pos_ned.copy(); z_sp = float(spawn[2])
    yaw0 = float(np.arctan2(quat_to_R(ds0.quat_wxyz)[1, 0], quat_to_R(ds0.quat_wxyz)[0, 0]))
    fwd = -np.array([np.cos(yaw0), np.sin(yaw0)]); push = -fwd; latp = np.array([-push[1], push[0]])
    phi0 = np.degrees(np.arctan2(CRUISE**2/(R*G), 1.0))
    print(f"corner_speed wvff={WVFF} sign={WVFF_SIGN} R={R} CRUISE={CRUISE}: bank={phi0:.1f}deg psi_dot={np.degrees(CRUISE/R):.1f}deg/s "
          f"pitchFF={np.degrees(np.sin(np.radians(phi0))*CRUISE/R):.1f}deg/s  (3 FF terms, ramp {RAMP}s)", flush=True)
    c.arm()
    flog = ftm.from_args(sys.argv, RG, "corner_speed", store=s)   # store=s -> collision flag streamed (dashboard hard rule)
    from aigp.recorder import Recorder
    rec_d = Recorder(s, script="corner_speed", mode="rate",
                     notes=f"coordinated corner + weathervane-FF (wvff={WVFF} sign={WVFF_SIGN} gain={WVFF_GAIN})",
                     extra_meta={"cmd_layout": ["wx", "wy", "wz", "thrust"], "R": R, "wvff": WVFF,
                                 "wvff_sign": WVFF_SIGN, "wvff_gain": WVFF_GAIN})
    t0 = time.time(); last = -1; phase = "ENTRY"; t_eng = None; turned = 0.0; yaw_prev = None
    peak_beta = 0.0; peak_spd = 0.0; rec = []; last_rec = -1; yaw_ramp = None; t_prev = t0
    while time.time() - t0 < MAX_T:
        ds = s.get_drone()
        if ds is not None:
            t = time.time() - t0
            vel = ds.vel_ned[:2]; v = float(np.linalg.norm(vel)); beta = sideslip_deg(ds.quat_wxyz, ds.vel_ned)
            if phase == "ENTRY":
                v_al = float(vel @ push); v_ct = float(vel @ latp); p_ct = float((ds.pos_ned[:2]-spawn[:2]) @ latp)
                a2 = float(np.clip(1.2*(CRUISE - v_al), -2.0, 0.5)) * push + (-0.6*p_ct - 1.2*v_ct) * latp
                yaw_sp = yaw0; yaw_ff = 0.0; pitch_ff = 0.0
                if t > T_ENTRY and abs(beta) < 25 and v > 0.6*CRUISE:
                    phase = "TURN"; t_eng = t; print(f"  --> ENTRY done t={t:.1f}s v={v:.1f} beta={beta:+.0f}; coord turn engaged", flush=True)
            else:
                now = time.time(); dt = min(now - t_prev, 0.05); t_prev = now
                rr = min(1.0, (t - t_eng) / RAMP)             # smooth onset of all three FF terms
                vhat = vel / max(v, 1e-6); inside = SDIR * np.array([-vhat[1], vhat[0]])
                a_cent = rr * min(v*v / R, A_CENT_MAX)        # CAP centripetal at the safe-bank envelope (no over-bank)
                phi = np.arctan2(a_cent, G)                   # bank from the (capped) centripetal
                psidot = a_cent / max(v, 0.5)                 # ACHIEVABLE turn rate (matches bank) -- breaks the v^2->bank->v loop
                a_along = float(np.clip(KV*(min(CRUISE, V_MAX) - v), -DECEL_MAX, ACCEL_MAX))  # govern speed to ACHIEVABLE
                turn_a2 = a_along * vhat + a_cent * inside
                entry_a2 = float(np.clip(1.2*(CRUISE - float(vel @ push)), -2.0, 0.5)) * push  # +body_x (entry form)
                a2 = (1.0 - rr) * entry_a2 + rr * turn_a2        # BLEND entry->turn over the ramp (no command-flip step)
                yaw_sp = float(np.arctan2(-vhat[1], -vhat[0]))   # velocity-tracking (closed-loop on beta, robust)
                yaw_ff = SDIR * psidot
                pitch_ff = PITCH_FF_SIGN * SDIR * np.sin(phi) * psidot   # KINEMATIC PITCH-COMP, from live ramped phi,psidot
                vh_hdg = float(np.arctan2(vhat[1], vhat[0]))
                if yaw_prev is not None:
                    turned += ((vh_hdg - yaw_prev + np.pi) % (2*np.pi)) - np.pi
                yaw_prev = vh_hdg; peak_beta = max(peak_beta, abs(beta)); peak_spd = max(peak_spd, v)
            tilt, dbg = cmd(ds, a2, z_sp, yaw_sp, yaw_ff, pitch_ff if phase == "TURN" else 0.0)
            rec_d.log([float(dbg["w_des"][0] / RG[0]), float(dbg["w_des"][1] / RG[1]),
                       float(dbg["w_des"][2] / RG[2]), float(dbg["thr"])])
            if int(t/0.04) != last_rec:
                last_rec = int(t/0.04); vn = ds.vel_ned; qa = ds.quat_wxyz; qd = dbg['q_des']
                rec.append((t, float(vn[0]),float(vn[1]),float(vn[2]), float(qa[0]),float(qa[1]),float(qa[2]),float(qa[3]),
                            float(qd[0]),float(qd[1]),float(qd[2]),float(qd[3]), float(dbg['thr']), float(beta), 1 if phase=="TURN" else 0))
            if flog is not None:
                flog.push(t, ds, dbg, cruise=CRUISE, running=s.get_race_live(), armed=True)
            if tilt > ABORT_TILT:
                print(f"ABORT tilt={tilt:.0f} t={t:.1f} {phase} (turned={np.degrees(turned):.0f} peak_spd={peak_spd:.1f} peak|beta|={peak_beta:.0f})", flush=True); break
            if phase == "TURN" and abs(turned) >= 4*np.pi:
                print(f"SUCCESS: 2 full turns ({np.degrees(turned):.0f}deg, peak|beta|={peak_beta:.0f}, peak_spd={peak_spd:.1f})", flush=True); break
            k = int(t/0.5)
            if k != last:
                last = k
                print(f"t={t:4.1f} {phase[0]} v={v:4.1f} tiltC/A={tilt_deg(dbg['q_des']):3.0f}/{tilt:3.0f} "
                      f"beta={beta:+4.0f} turned={np.degrees(turned):+5.0f}", flush=True)
        time.sleep(LOOP_DT)
    else:
        print(f"MAX_T reached (turned={np.degrees(turned):.0f} peak_spd={peak_spd:.1f} peak|beta|={peak_beta:.0f})", flush=True)
    if flog is not None:
        flog.close()
    print(f"done turned={np.degrees(turned):.0f}deg peak_spd={peak_spd:.1f} peak|beta|={peak_beta:.0f}", flush=True)
    rec_d.close()
    print("REC " + " ".join(",".join(f"{x:.4f}" for x in row[:-1]) + f",{row[-1]}" for row in rec), flush=True)


if __name__ == "__main__":
    main()
