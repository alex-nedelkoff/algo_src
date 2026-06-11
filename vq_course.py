"""vq_course.py -- FULL VQ1 PIPELINE: vision-anchored track course (gate-detect -> world anchor
-> TRACK_INFO relative layout -> waypoint course -> proven flier).

Why anchored: TRACK_INFO gate poses are ABSOLUTE sim-world; the ODOMETRY chart origin jumps per
session (observed offsets up to 290 m / 27 m vertical). Gate-to-gate vectors are translation-
invariant, and the vision pipeline locates gate 0 in the ODOMETRY chart to 0.1 m horizontally
(vq_gate_wp v6, arc-calibrated world map). So:

    course_wp_i = gate0_vision + (track_gate_i - track_gate_0)   [+ z banner-bias correction]

Phase A (vision lock, vq_gate_wp v6 verbatim): s_lat probe -> s_yawu probe -> arc-calib world
map H (9-hypothesis triangulation-scatter vote) -> approach to ~6 m collecting close-range
samples -> anchor = median estimate. Vertical banner bias (bbox center sits ~1 m above the
aperture center at the gate-0 truth check) corrected via --zbias (default +0.95 NED).

Phase B (course, vq_track_wp): pre/center/post waypoints per gate along the local course
direction; M2 velocity-vector guidance; explicit PLANE-CROSS detector self-judges passes
(active_gate_index has never been observed to increment -- printed for the record at each
crossing; organizer-side scoring semantics = open question).

Usage: python vq_course.py [--v 3.0] [--ngates 8] [--dur 360] [--zbias 0.95] [--no-viz] [--dump]
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
VMAX = argf("--v", 3.0); N_GATES = int(argf("--ngates", 8)); DURATION = argf("--dur", 360.0)
AMAX = argf("--amax", 0.6); THRU = argf("--thru", 2.0); ZBIAS = argf("--zbias", 0.95)
GATE_AP = 1.5
CX = 320.0; FX = 320.0; FY = 320.0; CY = 180.0
EST_MIN = 6; EST_KEEP = 14; RATIO_MIN = 5
ARC_A = 1.0; ARC_BASE = 2.5
ANCHOR_RING = 180.0                      # come in until ring >= this, then anchor on the median
                                         # (110 anchored 1.3 m off; close samples gave 0.1 m)
WP_R = 1.2; WP_TIMEOUT = 30.0
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
    Hs = [("ident", np.eye(3))]
    for k in range(8):
        th = np.pi * k / 8.0
        c2, s2 = np.cos(2 * th), np.sin(2 * th)
        Hs.append((f"refl{np.degrees(th):.0f}",
                   np.array([[c2, s2, 0.0], [s2, -c2, 0.0], [0.0, 0.0, 1.0]])))
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
    def __init__(self):
        self.samples = []
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
        if len(self.samples) < 20 or self.baseline() < ARC_BASE:
            return False
        best = None
        for name, Hm in H_CANDS:
            pts = np.array([p + Z * (Hm @ ray) for p, ray, Z in self.samples])
            sc = float(np.std(pts[:, :2], 0).sum())
            if best is None or sc < best[0]:
                best = (sc, name, Hm)
        self.H = best[2]; self.Hname = best[1]
        print(f"H locked: {best[1]} (scatter {best[0]:.2f}, n={len(self.samples)})", flush=True)
        return True

    def point(self, pos, ray, Z):
        Hm = self.H if self.H is not None else H_CANDS[1][1]
        return pos + Z * (Hm @ ray)

    def wp(self, last=EST_KEEP):
        if len(self.samples) < EST_MIN:
            return None
        Hm = self.H if self.H is not None else H_CANDS[1][1]
        pts = np.array([p + Z * (Hm @ ray) for p, ray, Z in self.samples[-last:]])
        return np.median(pts, 0)


def main():
    assert fresh_start(), "not live"
    gates = None
    t_g = time.time()
    while time.time() - t_g < 10 and gates is None:
        gates = s.get_gates(); time.sleep(0.1)
    assert gates is not None, "no TRACK_INFO after reset"
    ng = min(N_GATES, len(gates))
    track = np.array([np.asarray(g.pos_ned, float) for g in gates[:ng]])
    half_ap = gates[0].width / 2.0
    print(f"track: {ng} gates, rel spans {np.round(track[-1] - track[0], 1).tolist()}, "
          f"size {gates[0].width:.2f}", flush=True)

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
    flog = ftm.from_args(sys.argv, RG, run_name="vq_course", store=s)
    rec = Recorder(s, script="vq_course", mode="rate",
                   notes="VQ1 pipeline: vision anchor (gate 0) + TRACK_INFO relative layout + waypoint course",
                   extra_meta={"cmd_layout": ["wx", "wy", "wz", "thrust"], "vmax": VMAX,
                               "n_gates": ng, "thru": THRU, "zbias": ZBIAS,
                               "track_rel": (track - track[0]).tolist()})
    t0 = time.time(); t_prev = t0; prev_gi = gi0
    _, coll0 = s.get_collision(); max_tilt = 0.0; lastlog = -1
    s_lat = 0.0; probe_t0 = None; probe_v0 = 0.0
    s_yawu = 0.0; yprobe = None
    pos_prev = spawn.copy(); vel_w = np.zeros(2)
    z_ref = float(spawn[2])
    est = GateEstimator()
    ratio_samples = []; ratio = None
    u_track = None; sz_mark = 0.0; last_det_t = t0; last_ex_sign = 1.0
    yaw_base = yaw0_t; scan_sd = 1.0
    arc_t0 = None; arc_dir = 1.0
    main._vis = {}; main._vfixed = {}
    # phase: A = vision anchor, B = course
    phase = "A"; anchor = None; wps = None; wp_r = None; wp_center = None
    wi = 0; wp_t0 = t0; crossings = 0
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
            v_lat_w = float(vel_w @ lat_course)
            takeoff = t < 2.5
            if takeoff:
                z_ref = float(spawn[2]) - 1.2

            # ---------- detection (phase A only) ----------
            det = None; hole_wh = None; ray_t = None; Z_now = None
            if not takeoff:
                fr, _ = s.get_frame(); bgr = fr[0] if fr else None
                if u_track is not None and (now - last_det_t) > 2.5:
                    u_track = None
                if bgr is not None:
                    det, hole_wh = detect_tracked(bgr, P, u_track, require_hole=(u_track is None))
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
                # phase-B per-gate vision refinement: translation-only anchoring leaves a
                # chart-rotation error that GROWS with distance (crossing offsets 1.15->1.35
                # over gates 0-2; gate 3 at 88 m = frame strike x4). The track shape is the
                # prior; live vision measurements correct each gate as it comes into view.
                if (phase == "B" and det is not None and ray_t is not None and Z_now is not None
                        and est.H is not None and 60.0 <= det.w_px <= 260.0):
                    # ring>=60 only: at ring 25 (~30 m) +-1 px of aperture quantization = +-1.7 m
                    # of range error -- a long-range VIS-FIX moved gate 1 by 3 m and made the
                    # crossing WORSE; <=12 m the same pixel is <=0.3 m
                    g_t = wi // 2
                    if g_t < ng:
                        p_v = est.point(pos, ray_t, Z_now)
                        buf = main._vis.setdefault(g_t, [])
                        buf.append(p_v)
                        spread = float(np.std(np.array(buf), 0).sum()) if len(buf) >= 5 else 9.9
                        if len(buf) >= 5 and spread < 1.2:
                            off = (np.median(np.array(buf), 0) + np.array([0.0, 0.0, ZBIAS])
                                   - main._course_pts[g_t])   # same banner z-correction as the anchor
                            n_off = float(np.linalg.norm(off))
                            if n_off < 3.0:                   # larger = probably the wrong object
                                if n_off > 2.0:
                                    off *= 2.0 / n_off        # rate-limit a single fix
                                main._course_pts[g_t] = main._course_pts[g_t] + off
                                wps[2 * g_t] = wps[2 * g_t] + off
                                wps[2 * g_t + 1] = wps[2 * g_t + 1] + off
                                if np.linalg.norm(off) > 0.3:
                                    print(f"VIS-FIX gate {g_t} t={t:.1f} off=[{off[0]:+.1f},{off[1]:+.1f},{off[2]:+.1f}]",
                                          flush=True)
                            main._vis[g_t] = []               # fresh window for the next (closer) fix
                # arc calib
                calibrating = est.H is None and s_yawu != 0.0
                if calibrating and det is not None:
                    if arc_t0 is None:
                        arc_t0 = now; print(f"ARC-CALIB start t={t:.1f}", flush=True)
                    if now - arc_t0 > 4.0:
                        arc_t0 = now; arc_dir = -arc_dir
                    est.calibrate()
                # ANCHOR: calibrated + close enough (ring big) -> lock the course
                if (phase == "A" and est.H is not None and det is not None
                        and det.w_px >= ANCHOR_RING):
                    g0 = est.wp(last=10)
                    if g0 is not None:
                        anchor = g0.copy(); anchor[2] += ZBIAS
                        rel = track - track[0]
                        pts = anchor[None, :] + rel
                        # pre/post only -- the center wp caused a ~1 m limit-cycle dwell + 30 s
                        # timeouts; the plane-cross detector judges the pass, not wp arrival
                        wps = []; wp_r = []
                        # pre/post along the CENTRAL-DIFFERENCE course tangent (the racing-line
                        # direction through the aperture) -- the straight chord from the previous
                        # gate clipped structure ~6 m short of gate 3 on runs 2-4
                        for i, gp in enumerate(pts):
                            a_ = pts[max(i - 1, 0)] if i > 0 else np.append(pos[:2], gp[2])
                            b_ = pts[min(i + 1, len(pts) - 1)]
                            d = b_[:2] - a_[:2]
                            n = np.linalg.norm(d)
                            if n < 0.1:
                                d = gp[:2] - pos[:2]; n = max(np.linalg.norm(d), 0.1)
                            d = d / n
                            wps.append(np.array([gp[0] - THRU * d[0], gp[1] - THRU * d[1], gp[2]]))
                            wp_r.append(1.0)
                            wps.append(np.array([gp[0] + THRU * d[0], gp[1] + THRU * d[1], gp[2]]))
                            wp_r.append(WP_R)
                        wps = np.array(wps)
                        main._course_pts = pts
                        phase = "B"; wi = 0; wp_t0 = now
                        print(f"ANCHOR t={t:.1f} gate0 [{anchor[0]:.1f},{anchor[1]:.1f},{anchor[2]:.1f}] "
                              f"-> course {len(wps)} wps", flush=True)

            # ---------- guidance ----------
            a_al = 0.0; a_lat = 0.0
            wp = None; dist = 0.0
            if phase == "B":
                wp = wps[wi]
                err = wp[:2] - pos[:2]
                dist = float(np.linalg.norm(err))
                # transit altitude: hold the HIGHER of (leg-start, target) until 15 m out, then
                # blend down, settled 4 m before the wp. Run 1: step-descend 25 m out = terrain;
                # run 3: linear blend along the leg = structure below the high line, also terrain;
                # run 2: hold-then-drop-in-8-m cleared the transit but hit the gate frame sinking.
                leg_z = getattr(main, "_leg_z", wp[2])
                z_hold = min(leg_z, wp[2])               # NED: min = higher altitude
                frac = np.clip((dist - 4.0) / 11.0, 0.0, 1.0)
                z_ref = float(wp[2] + (z_hold - wp[2]) * frac)
                z_err = abs(pos[2] - wp[2])
            elif est.wp() is not None and not takeoff:
                gw = est.wp()
                wp = gw
                err = gw[:2] - pos[:2]
                dist = float(np.linalg.norm(err))
                z_ref = float(np.clip(gw[2], spawn[2] - 8.0, spawn[2] + 5.0))
            else:
                err = np.zeros(2)
                if not takeoff:
                    z_ref = float(spawn[2]) - 1.5

            if s_lat == 0.0 and not takeoff:
                if probe_t0 is None:
                    probe_t0 = now; probe_v0 = v_lat_w
                a_lat = 1.2
                if now - probe_t0 > 1.5:
                    s_lat = 1.0 if (v_lat_w - probe_v0) > 0 else -1.0
                    print(f"s_lat locked: {s_lat:+.0f}", flush=True)
            elif not takeoff:
                u0 = np.array([np.cos(yaw0_t), np.sin(yaw0_t)]); p0 = np.array([-u0[1], u0[0]])
                pc = np.array([-cam_live[1], cam_live[0]])
                M2 = np.column_stack([cam_live, s_lat * pc]) @ np.column_stack([u0, p0]).T
                fwd_live = M2 @ np.array([np.cos(yaw_cur), np.sin(yaw_cur)])
                lat_live = s_lat * np.array([-fwd_live[1], fwd_live[0]])
                calibrating = phase == "A" and est.H is None and s_yawu != 0.0
                if calibrating and det is not None:
                    v_w_sp = ARC_A * arc_dir * lat_live
                elif wp is not None:
                    stop_short = phase == "A" and det is not None and det.w_px >= 200.0
                    vcap = VMAX if phase == "B" else (0.0 if stop_short else 0.8 * VMAX)
                    if phase == "B" and dist < 15.0:
                        ze = abs(pos[2] - wp[2])
                        if ze > 1.5:
                            vcap = max(1.0, vcap * 1.5 / ze)   # let altitude converge first
                    v_w_sp = 0.7 * err
                    nv = float(np.linalg.norm(v_w_sp))
                    if nv > max(vcap, 0.01):
                        v_w_sp *= max(vcap, 0.01) / nv
                else:
                    v_w_sp = np.zeros(2)
                a_w = np.clip(KD_AL * (v_w_sp - vel_w), -TILTMAX, TILTMAX)
                a_al = float(np.clip(a_w @ fwd_live, -0.5 * TILTMAX, AMAX))
                a_lat = float(np.clip(a_w @ lat_live, -TILTMAX, TILTMAX))

                if det is not None and s_yawu == 0.0 and s_lat != 0.0:
                    if yprobe is None:
                        yprobe = (now, det.u, yaw_base)
                    yaw_base = yprobe[2] + YPROBE_SLEW * min(1.0, (now - yprobe[0]) / 0.7)
                    if now - yprobe[0] > 0.9:
                        s_yawu = 1.0 if (det.u - yprobe[1]) > 0 else -1.0
                        print(f"s_yawu locked: {s_yawu:+.0f}", flush=True)
                    yaw_ref = yaw_base
                elif abs(wrap(yaw_ref - yaw_cur)) < 0.12:
                    if phase == "A" and det is not None and s_yawu != 0.0:
                        ex = (det.u - CX) / FX
                        yaw_base += float(np.clip(-s_yawu * 1.2 * ex, -YAWRATE_MAX, YAWRATE_MAX)) * dt
                    elif phase == "B" and wp is not None and dist > WP_R:
                        beta = float(np.arctan2(float(err @ (np.array([-fwd_live[1], fwd_live[0]]))),
                                                float(err @ fwd_live)))
                        yaw_base += float(np.clip(1.0 * s_lat * beta, -YAWRATE_MAX, YAWRATE_MAX)) * dt
                    elif phase == "A" and (now - last_det_t) > 2.0:
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

            # ---------- phase B bookkeeping ----------
            gi = s.get_gate_idx()
            if phase == "B":
                g_i = wi // 2
                if g_i < ng:
                    gp = main._course_pts[g_i]
                    dh = wps[2 * g_i + 1][:2] - wps[2 * g_i][:2]; dh /= max(np.linalg.norm(dh), 1e-6)
                    s_now = float(np.sign((pos[:2] - gp[:2]) @ dh)) or 1.0
                    key = f"_s{g_i}"
                    s_old = getattr(main, key, None)
                    if s_old is not None and s_now != s_old:
                        perp = float(np.hypot((pos[:2] - gp[:2]) @ np.array([-dh[1], dh[0]]),
                                              pos[2] - gp[2]))
                        inside = perp < half_ap
                        crossings += int(inside)
                        print(f"PLANE-CROSS gate {g_i} t={t:.1f} offset={perp:.2f} "
                              f"({'INSIDE' if inside else 'OUTSIDE'} half={half_ap:.2f}) gi={gi}",
                              flush=True)
                    setattr(main, key, s_now)
                ztol = 0.6 if wi % 2 == 0 else 1.5         # PRE point: SETTLE at gate altitude
                                                            # (run 4 crossed 1.1 m high + sinking
                                                            # under the loose tol -> frame strike)
                if dist < wp_r[wi] and abs(pos[2] - wp[2]) < ztol:
                    wi += 1; wp_t0 = now
                    main._leg_z = float(pos[2])
                    if wi < len(wps):
                        main._leg_d = float(np.linalg.norm(wps[wi][:2] - pos[:2]))
                    if wi % 2 == 0:
                        u_track = None; sz_mark = 0.0      # fresh acquisition for the next gate
                    print(f"wp {wi}/{len(wps)} t={t:.1f}", flush=True)
                    if wi >= len(wps):
                        break
                elif now - wp_t0 > WP_TIMEOUT + dist / 1.0:
                    print(f"WP {wi + 1} TIMEOUT t={t:.1f} dist={dist:.1f} -> skip", flush=True)
                    wi += 1; wp_t0 = now
                    main._leg_z = float(pos[2])
                    if wi < len(wps):
                        main._leg_d = float(np.linalg.norm(wps[wi][:2] - pos[:2]))
                    if wi >= len(wps):
                        break
            # RACE-FIELD WATCHER: print the moment ANY scoring field moves
            rc = s.get_race()
            if rc:
                sig = (rc["active_gate_index"], rc["last_gate_time"], rc["race_finish_ns"])
                if getattr(main, "_rsig", None) is not None and sig != main._rsig:
                    print(f"*** RACE-FIELD CHANGE t={t:.1f}: {main._rsig} -> {sig} ***", flush=True)
                main._rsig = sig
            if gi > prev_gi:
                print(f"*** GATE_IDX TICK t={t:.1f} ({prev_gi}->{gi}) ***", flush=True)
                prev_gi = gi

            max_tilt = max(max_tilt, tilt_true)
            if t < 3.5:
                _, coll0 = s.get_collision()
            else:
                _, coll = s.get_collision()
                if coll != coll0:
                    print(f"COLLISION t={t:.1f} gi={gi} wp={wi} -> stop", flush=True); break
            if tilt_true > ABORT_TILT:
                print(f"TUMBLE tilt={tilt_true:.0f} t={t:.1f} -> stop", flush=True); break
            if DUMP:
                fr2, _ = s.get_frame()
                if fr2 and now - getattr(main, "_dump_t", 0) > 0.4:
                    main._dump_t = now
                    os.makedirs("frames_dump", exist_ok=True)
                    cv2.imwrite(f"frames_dump/c{t:06.1f}.jpg", fr2[0])
            k = int(t / 2.0)
            if k != lastlog:
                lastlog = k
                uu = f"{det.u:.0f}" if det else "--"
                print(f"t={t:5.1f} ph={phase} u={uu} wp={wi + 1 if phase == 'B' else '-'} "
                      f"dist={dist:4.1f} spd={np.linalg.norm(vel_w):4.1f} z={pos[2] - spawn[2]:+5.1f} "
                      f"tilt={tilt_true:3.0f} gi={gi} x={crossings}", flush=True)
            time.sleep(LOOP_DT)
    finally:
        idle(); rec.close()
        if flog is not None:
            flog.close()
    time.sleep(3.0)                        # let any post-course scoring fields land
    print(f"final race: {s.get_race()}", flush=True)
    print(f"\nRESULT: {crossings}/{ng} apertures crossed (self-judged), gate_idx delta "
          f"{s.get_gate_idx() - gi0}, in {time.time() - t0:.0f}s, max_tilt={max_tilt:.0f} "
          f"(run {rec.run_id})", flush=True)


if __name__ == "__main__":
    main()
