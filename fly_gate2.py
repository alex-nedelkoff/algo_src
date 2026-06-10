"""fly_gate2.py -- fly_gate.py + YAW CONTROL: progress through ALL VQ gates slowly, camera-only.

fly_gate (COR-125) held the spawn heading and strafed -> only gates near the spawn axis reachable
(stopped at ~2). This adds a rate-capped visual yaw reference so the nose (camera) steers toward
the detected gate and the course can turn:
    yaw_ref += clip(K_YAWVIS * SIGN_Y * ex, +-YAWRATE_MAX) * dt     (only while moving forward)
    w_z      = KP_YAW * wrap(yaw_ref - yaw_cur) - KD_YAW * omega_z
Forward push rotates with yaw_ref (fly_gate hardcoded the spawn axis). Moving-yaw with these
primitives is proven live by corner_speed (139-160 deg turned). YAWRATE_MAX is load-bearing:
hover yaw-spins are LETHAL (deploy3) and uncapped yaw-error commands resonance-tumble (YR_CAP).
Keeps fly_gate's strafe, pitch-invariant elevation law, terminal ballistic lock. Adds Recorder +
dashboard (hard rule), gate-lost slow scan toward the last-seen side, tilt/collision aborts.
  python fly_gate2.py [--signs -1] [--signy -1] [--kyaw 1.2] [--fwd 1.2] [--ngates 8]
                      [--dur 150] [--no-viz]
"""
import json, os, sys, time
import cv2
import numpy as np
from pymavlink import mavutil
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store
from aigp.commander import Commander
from aigp.geometry import quat_to_R
from fit_model import qfix
from aigp.control_math import (desired_attitude, mat_to_quat, attitude_error_quat,
                               collective_accel, accel_to_thrust_norm)
from aigp.gate_detect import GateDetection, detect_gate, load_params, red_mask


def detect_tracked(bgr, p, u_track):
    """All gate candidates (same filters as detect_gate), pick NEAREST to the tracked u -- the
    largest-area pick bounces between similar-size gates at range (attempt-17: u 241<->402<->143).
    Falls back to largest when no track."""
    H, W = bgr.shape[:2]
    cnts, hier = cv2.findContours(red_mask(bgr, p), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for i, c in enumerate(cnts):
        if hier[0][i][3] != -1:
            continue                                    # children handled via their parent
        area = float(cv2.contourArea(c))
        if area < p["min_area_px"]:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0 or abs(w / float(h) - 1.0) > p["square_tol"]:
            continue
        if (y + h / 2.0) > p["max_v_frac"] * H:
            continue
        # AIM AT THE APERTURE, not the ring centroid: this gate's wide top banner drags the
        # centroid into solid structure (attempt-25 flew into the banner). Largest child contour
        # of the red ring = the dark hole; use its center when meaningful.
        u_t, v_t = x + w / 2.0, y + h / 2.0
        best_hole = 0.0
        j = hier[0][i][2]                               # first child
        while j != -1:
            ha = float(cv2.contourArea(cnts[j]))
            if ha > best_hole and ha > 0.05 * area:
                hx, hy, hw, hh = cv2.boundingRect(cnts[j])
                u_t, v_t = hx + hw / 2.0, hy + hh / 2.0
                best_hole = ha
            j = hier[0][j][0]                           # next sibling
        cands.append(GateDetection(u_t, v_t, float(w), float(h), area, (x, y, w, h)))
    if not cands:
        return None
    if u_track is None:
        return max(cands, key=lambda d: d.area)
    u_tr, sz_tr = u_track
    ok = [d for d in cands if d.w_px > 0.5 * sz_tr]     # never hop to a much-smaller (farther) gate
    if not ok:
        return None
    return min(ok, key=lambda d: abs(d.u - u_tr) + 3.0 * abs(d.w_px - sz_tr))


def detect_beam(bgr):
    """Course-line fallback: the gates are strung along a bright CYAN spline beam (FPV frames
    06-10). Centroid of the cyan mask in the lower 2/3 of the frame -> steer along the course when
    no gate is detected (replaces blind wandering). Returns u or None."""
    import cv2
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([80, 80, 120]), np.array([110, 255, 255]))
    H = mask.shape[0]
    mask[: H // 3] = 0                                  # beam overhead segment misleads; use lower 2/3
    msum = float(mask.sum()) / 255.0
    if msum < 150.0:
        return None
    xs = np.nonzero(mask.any(0))[0]
    cols = mask.sum(0).astype(float)
    return float((cols * np.arange(mask.shape[1])).sum() / cols.sum())
from aigp.flight_telemetry import tilt_deg
import aigp.flight_telemetry as ftm
from aigp.recorder import Recorder


def argf(flag, d): return float(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else d


DUMP = "--dump" in sys.argv                # save FPV frames every 0.4 s to frames_dump/
SIGN_S = argf("--signs", -1.0); SIGN_Y = argf("--signy", -1.0)   # -1 CONFIRMED by frames (attempt-16: +yaw_ref = camera LEFT)
K_YAWVIS = argf("--kyaw", 1.2)            # visual yaw gain: ex (norm. pixel) -> yaw_ref rate
FWD = argf("--fwd", 1.2); N_GATES = int(argf("--ngates", 8)); DURATION = argf("--dur", 150.0)
YAWRATE_MAX = 0.4                          # rad/s cap on yaw_ref slew (hover-spin = lethal regime)
SCAN_RATE = 0.25; SCAN_AFTER = 5.0         # gate-lost: slow scan toward last-seen side (while MOVING only)
TILT_DEG = 20.0; C20 = np.cos(np.radians(TILT_DEG)); S20 = np.sin(np.radians(TILT_DEG))
R_OPT_BODY = np.array([[0.0, -S20, -C20], [-1.0, 0.0, 0.0], [0.0, C20, -S20]])
CX = 320.0; FX = 320.0; FY = 320.0; CY = 180.0
KP_ATT = np.array([0.7, 1.6, 1.0]); KP_YAW = 3.0; KD_YAW = 0.3; KD_ATT = 0.3
KD_AL = 1.2; AL_MAX = 0.4; KD_LAT = 1.4; K_STRAFE = 4.0; VLAT_MAX = 1.2
K_VZ = 7.0; VZ_MAX = 2.2; KP_Z = 1.8; KD_Z = 3.0
SZ_LOCK = 100.0; EX_LOCK = 0.12
WMAX = 4.0; LOOP_DT = 0.004
TILTMAX = np.tan(np.radians(15)) * 9.81; ABORT_TILT = 60.0
IDLE = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
P = load_params()

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


def yaw_cur_of(R_bw):
    return float(np.arctan2(R_bw[1, 0], R_bw[0, 0]))


def gate_world_elev(u, v, R_bw):
    d_opt = np.array([(u - CX) / FX, (v - CY) / FY, 1.0]); d_opt /= np.linalg.norm(d_opt)
    d_world = R_bw @ (R_OPT_BODY @ d_opt)
    return float(-np.arcsin(np.clip(d_world[2], -1.0, 1.0)))


def main():
    assert fresh_start(), "not live"
    ds0 = s.get_drone(); spawn = ds0.pos_ned.copy(); z_ref = float(spawn[2]); gi0 = s.get_gate_idx()
    yaw0 = float(np.arctan2(quat_to_R(ds0.quat_wxyz)[1, 0], quat_to_R(ds0.quat_wxyz)[0, 0]))
    yaw_ref = yaw0
    c.arm()
    flog = ftm.from_args(sys.argv, RG, run_name="fly_gate2", store=s)
    rec = Recorder(s, script="fly_gate2", mode="rate",
                   notes="camera-only multi-gate slow pass w/ rate-capped visual yaw steering (fly_gate + yaw control)",
                   extra_meta={"cmd_layout": ["wx", "wy", "wz", "thrust"], "base_controller": "fly_gate",
                               "sign_s": SIGN_S, "sign_y": SIGN_Y, "k_yawvis": K_YAWVIS, "fwd": FWD,
                               "yawrate_max": YAWRATE_MAX, "n_gates": N_GATES})
    print(f"fly_gate2 SIGN_S={SIGN_S} SIGN_Y={SIGN_Y} K_YAWVIS={K_YAWVIS} FWD={FWD} N_GATES={N_GATES}", flush=True)
    t0 = time.time(); prev_gi = gi0; target = gi0 + N_GATES
    locked = False; last_det_t = time.time(); last_ex_sign = 0.0; lastlog = -1
    _, coll0 = s.get_collision(); max_tilt = 0.0; t_prev = time.time()
    try:
        while time.time() - t0 < DURATION:
            ds = s.get_drone()
            if ds is None:
                time.sleep(LOOP_DT); continue
            now = time.time(); dt = min(now - t_prev, 0.05); t_prev = now; t = now - t0
            Rc = quat_to_R(ds.quat_wxyz)
            yaw_cur = yaw_cur_of(Rc)
            fr, _ = s.get_frame(); bgr = fr[0] if fr else None
            u_tr = getattr(main, "_u_track", None)
            if u_tr is not None and (now - last_det_t) > 2.5:
                u_tr = None; main._u_track = None               # stale track -> free acquisition
            det = detect_tracked(bgr, P, u_tr) if bgr is not None else None
            if det is not None:
                if u_tr is not None and abs(det.u - u_tr[0]) > 100.0:
                    det = None                                  # nothing near the track this frame
                else:
                    main._u_track = (det.u, det.w_px)
            # camera-forward speed in body terms. EMPIRICAL (run 123458): flying camera-forward
            # (gate visible, sz growing) reads vb[0] = +2.8 -> camera-forward speed = +vb[0] in this
            # io frame. (The -vb[0] guess silently blocked the scan + left speed at the AL_MAX
            # equilibrium.) Rotation-proof, unlike the attempt-1 world-mix (runaway 1.4->13).
            v_al = float(ds.vel_ned[0])
            fwd = -np.array([np.cos(yaw_cur), np.sin(yaw_cur)])             # push along ACTUAL camera dir
            blind = (now - last_det_t) > 2.5 and not locked
            # race_cruise speed law: PUSH-ONLY, AL_MAX sets the equilibrium (drag is the brake).
            # A bidirectional speed loop oscillates against the attitude resonance (attempt-9 pump
            # 4<->8.5 m/s); the proven stable cruise NEVER applies negative along-track accel.
            # TURN-THEN-GO: speed scheduled on alignment -- crawl while the gate is off-center
            # (attempt-12: off-axis gate exited the frame because speed built before centering).
            ex_cur = abs((det.u - CX) / FX) if det is not None else (0.0 if locked else 0.6)
            ev_cur = abs((det.v - CY) / FY) if det is not None else 0.0
            align = max(0.15, 1.0 - ex_cur / 0.35 - max(0.0, ev_cur - 0.15) / 0.3)
            al_cap = (0.15 if blind else AL_MAX * align)
            a_al = float(np.clip(KD_AL * (12.0 - v_al), 0.0, al_cap))
            moving = v_al > 0.3

            if det is not None and not locked and det.w_px > SZ_LOCK and abs((det.u - CX) / FX) < EX_LOCK:
                locked = True; main._lock_t = now
                print(f"LOCK t={t:.1f}s gi={s.get_gate_idx()}", flush=True)
            if locked and det is None and (now - last_det_t) > 2.5:
                locked = False                                   # lock timeout: missed pass -> resume search
                print(f"UNLOCK (timeout) t={t:.1f}s", flush=True)
            v_lat_sp = 0.0
            if locked:                                           # ballistic heading + elevation + u-STRAFE
                if det is not None:
                    last_det_t = now
                    ex = (det.u - CX) / FX
                    # strafe hard while there is range; FREEZE lateral for the final crossing
                    # (attempt-24 clipped the frame: lateral motion at the gate plane)
                    v_lat_sp = float(np.clip(2.0 * ex, -1.0, 1.0)) if det.w_px < 250 else 0.0
                    elev = gate_world_elev(CX, det.v, Rc)
                    z_ref += float(np.clip(-K_VZ * elev, -VZ_MAX, VZ_MAX)) * LOOP_DT
                    z_ref = float(np.clip(z_ref, spawn[2] - 8.0, spawn[2] + 5.0))   # NED: -8 = 8 m climb (gate 1 sits HIGH on this course; the old 2 m cap flew under it)
            elif det is not None:
                ex = (det.u - CX) / FX
                last_det_t = now; last_ex_sign = np.sign(ex) if abs(ex) > 0.05 else last_ex_sign
                v_lat_sp = float(np.clip(2.0 * ex, -0.8, 0.8))   # strafe toward gate (lat=camera-RIGHT, pinned)
                if moving:                                       # visual yaw steering (rate-capped)
                    yaw_ref += float(np.clip(K_YAWVIS * SIGN_Y * ex, -YAWRATE_MAX, YAWRATE_MAX)) * dt
                elev = gate_world_elev(det.u, det.v, Rc)
                z_ref += float(np.clip(-K_VZ * elev, -VZ_MAX, VZ_MAX)) * LOOP_DT
                z_ref = float(np.clip(z_ref, spawn[2] - 8.0, spawn[2] + 5.0))   # NED: -8 = 8 m climb (gate 1 sits HIGH on this course; the old 2 m cap flew under it)
            else:
                ub = detect_beam(bgr) if bgr is not None else None
                if ub is not None and moving:
                    exb = (ub - CX) / FX                # follow the course BEAM when no gate det
                    yaw_ref += float(np.clip(0.5 * K_YAWVIS * SIGN_Y * exb, -YAWRATE_MAX, YAWRATE_MAX)) * dt
                    last_det_t = now - 1.0              # beam counts as half-seen: keep speed up, no scan
                elif moving and t > 10.0 and (now - last_det_t) > SCAN_AFTER:
                    sd = last_ex_sign if last_ex_sign != 0.0 else 1.0   # default scan dir when lost centered
                    if int((now - last_det_t - SCAN_AFTER) / 6.0) % 2 == 1:
                        sd = -sd                                        # alternate every 6 s -> sweep both sides
                    yaw_ref += SIGN_Y * sd * SCAN_RATE * dt
            a2 = float(np.clip(KP_Z * (z_ref - ds.pos_ned[2]) + KD_Z * (0.0 - ds.vel_ned[2]), -4.0, 4.0))
            # lateral DAMPING, sign data-determined (latsign.py on run 123213, corr -0.88):
            # lat_world = [-fwd_y, fwd_x] pairs with body measure v_lat = +vb[1]. Without this the
            # uncorrected sideslip built to 6.6 m/s in attempt 2 (speed loop is blind to it).
            lat = np.array([-fwd[1], fwd[0]])
            v_lat = float(ds.vel_ned[1])
            spd_h = float(np.hypot(v_al, v_lat))
            in_safe = (blind and spd_h > 2.5) or now < getattr(main, "_recover_until", 0.0)
            main._in_safe = in_safe
            if in_safe:
                # SAFE-SLOW: blind+fast = above the drag-tilt wall, unstabilizable wandering.
                # Velocity damping on the pinned axes, vz-damping only (no z-position chase), yaw
                # frozen; hold until slow for a full second; rebase z_ref (no post-recovery yank).
                if spd_h > 1.0:
                    main._recover_until = now + 1.0
                vw = v_al * fwd + v_lat * lat
                a = np.zeros(3); a[:2] = -1.2 * vw
                nrm = float(np.linalg.norm(a[:2]))
                if nrm > 2.0:
                    a[:2] *= 2.0 / nrm
                a[2] = float(np.clip(KD_Z * (0.0 - ds.vel_ned[2]), -2.0, 2.0))
                z_ref = float(ds.pos_ned[2])
                yaw_ref = yaw_cur
            else:
                a_lat = float(np.clip(KD_LAT * (v_lat_sp - v_lat), -2.0, 2.0))
                a = np.zeros(3); a[:2] = a_al * fwd + a_lat * lat; a[2] = a2
                nrm = float(np.linalg.norm(a[:2]))
                if nrm > TILTMAX:
                    a[:2] = a[:2] / nrm * TILTMAX
            q = mat_to_quat(desired_attitude(a, yaw_ref))
            w = KP_ATT * attitude_error_quat(ds.quat_wxyz, q)
            w[0] -= KD_ATT * float(ds.omega[0])                 # roll/pitch rate damping, corner_speed
            w[1] -= KD_ATT * float(ds.omega[1])                 # --kd 0.3 (live-validated, halves the pump)
            wcap = 1.0 if in_safe else 2.0
            w[0] = float(np.clip(w[0], -wcap, wcap)); w[1] = float(np.clip(w[1], -wcap, wcap))
            w[2] = KP_YAW * ((yaw_ref - yaw_cur + np.pi) % (2 * np.pi) - np.pi) - KD_YAW * float(ds.omega[2])
            w[2] = float(np.clip(w[2], -1.5, 1.5))              # YR_CAP: uncapped yaw err = resonance tumble
            # CLAMP collective: collective_accel divides by cos(tilt) (altitude hold), which at
            # tilt>45 becomes a 20+ m/s^2 HORIZONTAL accelerator = the wandering-pump energy source
            # (race_cruise never tilts enough to meet it). Altitude sags during upsets instead.
            # tilt-CONDITIONAL: the flat clamp 11.5 starved normal z-recovery (needs ~14) and the
            # drone sank 2.5 m/s^2 the whole approach (attempt-22 zdiag). Only clamp when tilted.
            c_max = 10.0 if tilt_deg(ds.quat_wxyz) > 40.0 else 18.0
            thr = accel_to_thrust_norm(min(collective_accel(a, ds.quat_wxyz), c_max), HOVER, KA)
            if in_safe and spd_h > 4.0:
                # BALLISTIC BRAKE: blind+fast = above the drag-tilt wall, unstabilizable (REFIT-02
                # ramp-wall limit cycle); any altitude-holding thrust at the resulting tilt re-feeds
                # it. Cut thrust, let quadratic drag brake (~6 m/s^2 @ v10); TWR 4.3 recovers alt.
                thr = 0.12
            w_cmd = np.clip(w / RG, -WMAX, WMAX)
            c.send_attitude_target(w_cmd, thr)
            rec.log([float(w_cmd[0]), float(w_cmd[1]), float(w_cmd[2]), float(thr)])
            if flog is not None:
                flog.push(t, ds, {"a": a, "w_des": w, "q_des": q, "thr": thr},
                          tangent=fwd, cruise=FWD, running=s.get_race_live(), armed=True)
            if DUMP and bgr is not None and now - getattr(main, "_dump_t", 0) > 0.4:
                main._dump_t = now
                os.makedirs("frames_dump", exist_ok=True)
                cv2.imwrite(f"frames_dump/f{t:06.1f}_gi{s.get_gate_idx()}_det{int(det is not None)}.jpg", bgr)
            gi = s.get_gate_idx()
            if gi > prev_gi:
                print(f"*** GATE PASSED t={t:.1f}s (idx {prev_gi}->{gi}) ***", flush=True)
                prev_gi = gi; locked = False; last_det_t = now
                if gi >= target:
                    break
            # TRUE tilt (qfix) -- the live-frame reading warps in turns (canonical); attempt-3
            # "tumble" was a false abort at true tilt ~16.
            tilt = float(np.degrees(np.arccos(np.clip(quat_to_R(qfix(ds.quat_wxyz))[2, 2], -1, 1))))
            max_tilt = max(max_tilt, tilt)
            if t < 3.5:
                _, coll0 = s.get_collision()
            else:
                _, coll = s.get_collision()
                if coll != coll0:
                    print(f"  COLLISION t={t:.1f} gi={gi} -> stop", flush=True); break
            if tilt > ABORT_TILT:
                print(f"  TUMBLE tilt={tilt:.0f} t={t:.1f} -> stop", flush=True); break
            k = int(t / 2.0)
            if k != lastlog:
                lastlog = k
                uu = f"{det.u:.0f}" if det else "--"; zz = f"{det.w_px:.0f}" if det else "--"
                print(f"t={t:5.1f} u={uu} sz={zz} spd={np.linalg.norm(ds.vel_ned[:2]):4.1f} "
                      f"yawref={np.degrees(yaw_ref - yaw0):+5.0f} tilt={tilt:3.0f} lk={int(locked)} gi={gi}", flush=True)
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
