"""vq_gate_wp.py -- GATE-DETECT -> WORLD WAYPOINT -> waypoint flier (VQ1 qualification track). v6.

Pipeline: fly_gate3's tracked detector (aperture-aimed) -> range from aperture size (spec 1.5 m,
fx=320; ring/aperture px ratio auto-calibrated to fill frames where the hole contour is missed)
-> ray to world (frustum-fix matrix R_true @ R_OPT(s_cam)) -> world waypoint -> vq_waypoint2's
proven velocity-vector guidance. Yaw steers on PIXEL error with an IN-FLIGHT PROBED sign.

Frame-risk policy (probe, don't assume -- every sign in this script is measured in flight):
  s_lat   lateral push sign (1.5 s push probe, vq_waypoint2 verbatim)
  s_yawu  yaw->pixel sign (0.12 rad slew on the first stable detection, watch du)
  H       the live-chart world map for camera rays. Runs 1-5: a fixed world-y flip made
          consistent in-run projections but estimates 3 m apart BETWEEN runs (the reflection
          axis of the live chart is not proven to be world-x). v6 runs an ARC CALIBRATION:
          strafe +-ARC_A m at standoff while the pixel-yaw loop holds the gate centered, then
          pick H from 9 hypotheses (identity + reflections about 8 horizontal axes) by minimum
          triangulation scatter of the SAME physical gate.

Per gate: ACQUIRE (hover scan) -> [gate 1 only: ARC-CALIB] -> GOTO (fly at the median estimate,
refining all the way; close range trusts the current frame only) -> THRU (target gate + THRU_M
along a latched axis; estimate latched on detection loss) -> gate_idx++ -> next gate.

Usage: python vq_gate_wp.py [--v 2.0] [--ngates 8] [--dur 300] [--no-viz] [--dump]
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
from aigp.gate_detect import GateDetection, load_params, red_mask
import aigp.flight_telemetry as ftm
from aigp.recorder import Recorder


def argf(flag, d): return float(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else d


DUMP = "--dump" in sys.argv
VMAX = argf("--v", 2.0); N_GATES = int(argf("--ngates", 8)); DURATION = argf("--dur", 300.0)
AMAX = argf("--amax", 0.6)
GATE_AP = 1.5                            # spec aperture (m); env trains 2.0 -- spec wins live
CX = 320.0; FX = 320.0; FY = 320.0; CY = 180.0
THRU_TRIG = 3.0; THRU_M = 2.5; GATE_TIMEOUT = 45.0; THRU_DWELL = 2.5
EST_MIN = 6; EST_KEEP = 14
RATIO_MIN = 5
ARC_A = 1.0                              # arc-calib lateral speed (m/s); ~ +-2 m sweep
ARC_BASE = 2.5                           # required lateral baseline (m) before voting H
KP_ATT = np.array([0.7, 1.6, 1.0]); KP_YAW = 3.0; KD_YAW = 0.3; KD_ATT = 0.3
KD_AL = 1.2; KD_LAT = 1.4
KP_Z = 1.8; KD_Z = 3.0
YAWRATE_MAX = 0.35; SCAN_RATE = 0.25; YPROBE_SLEW = 0.12
WMAX = 4.0; LOOP_DT = 0.004
TILTMAX = np.tan(np.radians(argf("--tilt", 15.0))) * 9.81; ABORT_TILT = 60.0
IDLE = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
P = load_params()
T20 = np.radians(20.0)

r = json.load(open("sysid/sim_response.json"))
HOVER = r["hover_thrust"]; KA = r["k_a"]
RG = np.array([r["rate_gain_axes"]["roll"], r["rate_gain_axes"]["pitch"], r["rate_gain_axes"]["yaw"]])
s = Store(); m = MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); VisionIO(s).start()
boot = int(time.time() * 1000); c = Commander(m.conn, boot)


def r_opt_body_true(s_cam):
    S, C = np.sin(T20), np.cos(T20)
    return np.array([[0.0, s_cam * S, s_cam * C],
                     [s_cam, 0.0, 0.0],
                     [0.0, C, -S]])


def world_maps():
    """Candidate live-chart maps for camera rays: identity + reflections about 8 horizontal
    axes (theta = axis angle). Reflection about world-x (theta=0) = the old y-flip."""
    Hs = [("ident", np.eye(3))]
    for k in range(8):
        th = np.pi * k / 8.0
        c2, s2 = np.cos(2 * th), np.sin(2 * th)
        Hm = np.array([[c2, s2, 0.0], [s2, -c2, 0.0], [0.0, 0.0, 1.0]])
        Hs.append((f"refl{np.degrees(th):.0f}", Hm))
    return Hs


H_CANDS = world_maps()


def detect_tracked(bgr, p, u_track, require_hole=False):
    H, W = bgr.shape[:2]
    cnts, hier = cv2.findContours(red_mask(bgr, p), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for i, ct in enumerate(cnts):
        if hier[0][i][3] != -1:
            continue
        area = float(cv2.contourArea(ct))
        if area < p["min_area_px"]:
            continue
        x, y, w, h = cv2.boundingRect(ct)
        if w == 0 or h == 0 or abs(w / float(h) - 1.0) > p["square_tol"]:
            continue
        if (y + h / 2.0) > p["max_v_frac"] * H:
            continue
        u_t, v_t = x + w / 2.0, y + h / 2.0
        best_hole = 0.0; hole_wh = None
        j = hier[0][i][2]
        while j != -1:
            ha = float(cv2.contourArea(cnts[j]))
            if ha > best_hole and ha > 0.05 * area:
                hx, hy, hw, hh = cv2.boundingRect(cnts[j])
                u_t, v_t = hx + hw / 2.0, hy + hh / 2.0
                best_hole = ha; hole_wh = (float(hw), float(hh))
            j = hier[0][j][0]
        if require_hole and best_hole <= 0.0:
            continue
        cands.append((GateDetection(u_t, v_t, float(w), float(h), area, (x, y, w, h)), hole_wh))
    if not cands:
        return None, None
    if u_track is None:
        return max(cands, key=lambda d: d[0].area)
    u_tr, sz_tr = u_track
    ok = [d for d in cands if d[0].w_px > 0.5 * sz_tr]
    if not ok:
        return None, None
    return min(ok, key=lambda d: abs(d[0].u - u_tr) + 3.0 * abs(d[0].w_px - sz_tr))


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


class GateEstimator:
    """Stores RAW (pos, true-frame ray, Z) samples; the world map H is chosen by arc-calib
    triangulation scatter and applied at readout, so calibration retroactively fixes the cloud."""

    def __init__(self):
        self.samples = []                  # (pos3, ray3, Z)
        self.H = None; self.Hname = None

    def add(self, pos, ray, Z):
        self.samples.append((pos.copy(), ray.copy(), float(Z)))
        if len(self.samples) > 200:
            self.samples.pop(0)

    def baseline(self):
        if len(self.samples) < 2:
            return 0.0
        ps = np.array([s_[0][:2] for s_ in self.samples])
        return float(np.linalg.norm(ps.max(0) - ps.min(0)))

    def calibrate(self):
        """Pick H minimizing triangulation scatter over ALL samples. Returns True when locked."""
        if len(self.samples) < 20 or self.baseline() < ARC_BASE:
            return False
        best = None
        for name, Hm in H_CANDS:
            pts = np.array([p + Z * (Hm @ ray) for p, ray, Z in self.samples])
            sc = float(np.std(pts[:, :2], 0).sum())   # horizontal scatter: z unaffected by H
            if best is None or sc < best[0]:
                best = (sc, name, Hm)
        self.H = best[2]; self.Hname = best[1]
        scs = {n: float(np.std(np.array([p + Z * (Hm @ ray) for p, ray, Z in self.samples])[:, :2], 0).sum())
               for n, Hm in H_CANDS}
        print(f"H locked: {best[1]} (scatter {best[0]:.2f}); all: " +
              " ".join(f"{n}={v:.2f}" for n, v in scs.items()), flush=True)
        return True

    def point(self, pos, ray, Z):
        Hm = self.H if self.H is not None else H_CANDS[1][1]   # default old y-flip until calib
        return pos + Z * (Hm @ ray)

    def wp(self):
        if len(self.samples) < EST_MIN:
            return None
        Hm = self.H if self.H is not None else H_CANDS[1][1]
        pts = np.array([p + Z * (Hm @ ray) for p, ray, Z in self.samples[-EST_KEEP:]])
        return np.median(pts, 0)

    def reset(self):
        self.samples = []                  # H persists across gates


def main():
    assert fresh_start(), "not live"
    ds0 = s.get_drone(); spawn = ds0.pos_ned.copy(); gi0 = s.get_gate_idx()
    yaw0 = float(np.arctan2(quat_to_R(ds0.quat_wxyz)[1, 0], quat_to_R(ds0.quat_wxyz)[0, 0]))
    q_t0 = qfix(ds0.quat_wxyz); R_t0 = quat_to_R(q_t0)
    cam_live = -np.array([np.cos(yaw0), np.sin(yaw0)])
    s_cam = 1.0 if float(R_t0[:2, 0] @ cam_live) > 0 else -1.0
    yaw0_t = float(np.arctan2(s_cam * R_t0[1, 0], s_cam * R_t0[0, 0]))
    yaw_ref = yaw0_t
    WFIX = np.array([1.0, -1.0, 1.0])
    R_OPT_T = r_opt_body_true(s_cam)
    lat_course = np.array([-cam_live[1], cam_live[0]])
    print(f"true-frame ctl: s_cam={s_cam:+.0f} yaw0_true_cam={np.degrees(yaw0_t):+.0f}", flush=True)

    c.arm()
    flog = ftm.from_args(sys.argv, RG, run_name="vq_gate_wp", store=s)
    rec = Recorder(s, script="vq_gate_wp", mode="rate",
                   notes="gate-detect -> world waypoint flier, arc-calibrated world map (VQ1, v6)",
                   extra_meta={"cmd_layout": ["wx", "wy", "wz", "thrust"], "vmax": VMAX,
                               "gate_ap": GATE_AP, "n_gates": N_GATES})
    t0 = time.time(); t_prev = t0; target = gi0 + N_GATES; prev_gi = gi0
    _, coll0 = s.get_collision(); max_tilt = 0.0; lastlog = -1
    s_lat = 0.0; probe_t0 = None; probe_v0 = 0.0
    s_yawu = 0.0; yprobe = None
    pos_prev = spawn.copy(); vel_w = np.zeros(2)
    z_ref = float(spawn[2])
    est = GateEstimator(); gate_t0 = t0
    ratio_samples = []; ratio = None
    thru_dir = None; thru_reach_t = None; gate_latch = None
    arc_phase = 0.0; arc_dir = 1.0; arc_t0 = None   # arc-calib state
    u_track = None; sz_mark = 0.0; last_det_t = t0; last_ex_sign = 1.0
    yaw_base = yaw0_t; scan_sd = 1.0
    try:
        while time.time() - t0 < DURATION:
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

            takeoff = t < 2.5
            if takeoff:
                z_ref = float(spawn[2]) - 1.2

            # --- detection + raw ray sample ---
            fr, _ = s.get_frame(); bgr = fr[0] if fr else None
            if u_track is not None and (now - last_det_t) > 2.5:
                u_track = None
            det, hole_wh = (detect_tracked(bgr, P, u_track, require_hole=(u_track is None))
                            if (bgr is not None and not takeoff) else (None, None))
            if det is not None:
                if u_track is not None and abs(det.u - u_track[0]) > 100.0:
                    det = None
                elif det.w_px < 0.5 * sz_mark:
                    det = None; u_track = None
                else:
                    u_track = (det.u, det.w_px); sz_mark = max(sz_mark, det.w_px)
                    last_det_t = now
                    ex = (det.u - CX) / FX
                    last_ex_sign = np.sign(ex) if abs(ex) > 0.05 else last_ex_sign
            ray_t = None; Z_now = None
            if det is not None:
                d_opt = np.array([(det.u - CX) / FX, (det.v - CY) / FY, 1.0])
                d_opt /= np.linalg.norm(d_opt)
                ray_t = R_t @ (R_OPT_T @ d_opt)
                ap_px = None
                if hole_wh is not None:
                    ap_px = max(hole_wh)
                    if ap_px >= 6.0:
                        ratio_samples.append(det.w_px / ap_px)
                        if len(ratio_samples) >= RATIO_MIN and ratio is None:
                            ratio = float(np.median(ratio_samples))
                            print(f"ring/aperture ratio locked: {ratio:.2f}", flush=True)
                elif ratio is not None:
                    ap_px = det.w_px / ratio
                if ap_px is not None and ap_px >= 6.0:
                    Z_now = FX * GATE_AP / ap_px
                    if Z_now < 40.0 and det.w_px < 260.0:
                        est.add(pos, ray_t, Z_now)

            # --- arc calibration (gate 1, once): strafe while pixel-yaw holds the gate ---
            calibrating = (est.H is None and s_yawu != 0.0 and not takeoff)
            if calibrating and det is not None:
                if arc_t0 is None:
                    arc_t0 = now
                    print(f"ARC-CALIB start t={t:.1f}", flush=True)
                if now - arc_t0 > 4.0:
                    arc_t0 = now; arc_dir = -arc_dir
                if est.calibrate():
                    calibrating = False; arc_t0 = None

            # --- target selection ---
            gate_wp = est.wp()
            if det is not None and det.w_px >= 120.0 and ray_t is not None and Z_now is not None:
                gate_wp = est.point(pos, ray_t, Z_now)   # close range: current frame only
                gate_latch = gate_wp.copy()
            elif det is None and thru_dir is not None and gate_latch is not None:
                gate_wp = gate_latch
            dist_g = float(np.linalg.norm(gate_wp[:2] - pos[:2])) if gate_wp is not None else 99.0
            if (gate_wp is not None and thru_dir is None and dist_g < THRU_TRIG
                    and est.H is not None):
                ap_v = gate_wp[:2] - pos[:2]
                n = float(np.linalg.norm(ap_v))
                thru_dir = ap_v / n if n > 0.1 else np.array([np.cos(yaw_cur), np.sin(yaw_cur)])
                gate_t0 = now
                print(f"THRU t={t:.1f} gate at [{gate_wp[0]:.1f},{gate_wp[1]:.1f},{gate_wp[2]:.1f}]",
                      flush=True)
            if thru_dir is not None and gate_wp is not None:
                wp = gate_wp.copy(); wp[:2] += thru_dir * THRU_M
            else:
                wp = gate_wp

            # --- guidance ---
            a_al = 0.0; a_lat = 0.0
            v_lat_w = float(vel_w @ lat_course)
            if wp is not None:
                err = wp[:2] - pos[:2]
                dist = float(np.linalg.norm(err))
                z_ref = float(np.clip(wp[2], spawn[2] - 8.0, spawn[2] + 5.0))
            else:
                err = np.zeros(2); dist = 0.0

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
                if calibrating and det is not None:
                    v_w_sp = ARC_A * arc_dir * lat_live    # arc: pure lateral sweep
                elif wp is not None and est.H is not None:
                    if det is not None:
                        align = max(0.2, 1.0 - abs((det.u - CX) / FX) / 0.35)
                    else:
                        align = 0.6
                    vcap = VMAX * align if thru_dir is None else min(VMAX, 1.6)
                    v_w_sp = 0.7 * err
                    nv = float(np.linalg.norm(v_w_sp))
                    if nv > vcap:
                        v_w_sp *= vcap / nv
                else:
                    v_w_sp = np.zeros(2)
                a_w = np.clip(KD_AL * (v_w_sp - vel_w), -TILTMAX, TILTMAX)
                a_al = float(np.clip(a_w @ fwd_live, -0.5 * TILTMAX, AMAX))
                a_lat = float(np.clip(a_w @ lat_live, -TILTMAX, TILTMAX))

                # --- yaw: probe then pixel steering ---
                if det is not None and s_yawu == 0.0:
                    if yprobe is None:
                        yprobe = (now, det.u, yaw_base)
                    yaw_base = yprobe[2] + YPROBE_SLEW * min(1.0, (now - yprobe[0]) / 0.7)
                    if now - yprobe[0] > 0.9:
                        du = det.u - yprobe[1]
                        s_yawu = 1.0 if du > 0 else -1.0
                        print(f"s_yawu locked: {s_yawu:+.0f} (du {du:+.0f} px)", flush=True)
                    yaw_ref = yaw_base
                elif abs(wrap(yaw_ref - yaw_cur)) < 0.12:
                    if det is not None and s_yawu != 0.0:
                        ex = (det.u - CX) / FX
                        yaw_base += float(np.clip(-s_yawu * 1.2 * ex, -YAWRATE_MAX, YAWRATE_MAX)) * dt
                        scan_sd = last_ex_sign
                    elif gate_wp is not None and dist_g > 1.0 and s_yawu != 0.0:
                        # blind with estimate: hold yaw (pixel servo resumes on reacquire)
                        pass
                    elif (now - last_det_t) > 2.0 and gate_wp is None:
                        yaw_base += -s_lat * scan_sd * SCAN_RATE * dt
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
            if DUMP and bgr is not None and now - getattr(main, "_dump_t", 0) > 0.4:
                main._dump_t = now
                os.makedirs("frames_dump", exist_ok=True)
                cv2.imwrite(f"frames_dump/g{t:06.1f}_gi{s.get_gate_idx()}_det{int(det is not None)}.jpg", bgr)

            gi = s.get_gate_idx()
            if thru_dir is not None and wp is not None and dist < 0.6 and thru_reach_t is None:
                thru_reach_t = now
            if gi > prev_gi:
                print(f"*** GATE PASSED t={t:.1f}s (idx {prev_gi}->{gi}) ***", flush=True)
                prev_gi = gi; est.reset(); thru_dir = None; thru_reach_t = None; gate_latch = None
                u_track = None; sz_mark = 0.0; gate_t0 = now; last_det_t = now
                z_ref = float(pos[2])
                if gi >= target:
                    break
            elif ((thru_reach_t is not None and now - thru_reach_t > THRU_DWELL
                   and (now - last_det_t) > 1.5) or now - gate_t0 > GATE_TIMEOUT):
                why = "THRU dead-end" if thru_reach_t is not None else "GATE TIMEOUT"
                print(f"{why} t={t:.1f} -> re-acquire", flush=True)
                est.reset(); thru_dir = None; thru_reach_t = None; gate_latch = None
                u_track = None; sz_mark = 0.0; gate_t0 = now
                z_ref = float(pos[2])

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
                uu = f"{det.u:.0f}" if det else "--"
                zz = f"{det.w_px:.0f}" if det else "--"
                wps = f"[{wp[0]:.0f},{wp[1]:.0f}]" if wp is not None else "--"
                up = "--"
                if gate_wp is not None and est.H is not None:
                    d_w = np.linalg.inv(est.H) @ (gate_wp - pos)
                    d_o = R_OPT_T.T @ (R_t.T @ d_w)
                    up = f"{CX + FX * d_o[0] / d_o[2]:.0f}" if d_o[2] > 0.2 else "X"
                print(f"t={t:5.1f} u={uu} upred={up} ring={zz} wp={wps} dist={dist:4.1f} "
                      f"spd={np.linalg.norm(vel_w):4.1f} z={pos[2] - spawn[2]:+4.1f} "
                      f"yerr={np.degrees(wrap(yaw_ref - yaw_cur)):+4.0f} tilt={tilt_true:3.0f} "
                      f"gi={gi} thru={int(thru_dir is not None)} cal={int(est.H is not None)} "
                      f"n={len(est.samples)}", flush=True)
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
