"""COR-139 frame probe: is the drone's ODOMETRY frame (pos_ned, which origin-jumps) offset from the
absolute TRACK_INFO gate frame, and can we ANCHOR by localizing gate0 with the camera at spawn?
Resets once, reads ODOM pos vs LPN pos vs gate0/gate5, then runs the vq_gate_wp vision pipeline ONE
shot at spawn (drone stationary, camera looking forward at gate0) to localize gate0 in the drone/odom
frame. Prints gate0_vision vs gate0_track + the anchor translation T. No flight, no mixer (cheap).
If gate0_vision is sane (e.g. ~8.5 m up = canonical) -> wire the anchor into trpy_gate_fly.
Usage: python scripts/probe_frame.py
"""
import os, sys, time
import numpy as np
import cv2
from pymavlink import mavutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aigp.commander import Commander
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store
from aigp.geometry import quat_to_R
from aigp.gate_detect import load_params, red_mask, GateDetection
from fit_model import qfix

IDLE = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE

# --- vq_gate_wp vision constants + transforms (copied VERBATIM, frame-critical) ---
GATE_AP = 1.5                                  # spec aperture (m)
CX = 320.0; FX = 320.0; FY = 320.0; CY = 180.0
T20 = np.radians(20.0)
H_DEFAULT = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])  # H_CANDS[1][1] = refl0 (y-flip)


def r_opt_body_true(s_cam):
    S, C = np.sin(T20), np.cos(T20)
    return np.array([[0.0, s_cam * S, s_cam * C],
                     [s_cam, 0.0, 0.0],
                     [0.0, C, -S]])


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
    return max(cands, key=lambda d: d[0].area)   # spawn: biggest red square = gate0


def localize_gate0(s, secs=3.0):
    """One-shot gate0 localization at spawn (drone stationary, camera forward). Returns the MEDIAN
    over `secs` of vision gate0 estimates in the drone/odom frame, plus the sample count + ranges."""
    p = load_params()
    ds0 = s.get_drone()
    yaw0 = float(np.arctan2(quat_to_R(ds0.quat_wxyz)[1, 0], quat_to_R(ds0.quat_wxyz)[0, 0]))
    cam_live = -np.array([np.cos(yaw0), np.sin(yaw0)])
    R_t0 = quat_to_R(qfix(ds0.quat_wxyz))
    s_cam = 1.0 if float(R_t0[:2, 0] @ cam_live) > 0 else -1.0
    R_OPT_T = r_opt_body_true(s_cam)
    print(f"localize_gate0: s_cam={s_cam:+.0f} yaw0={np.degrees(yaw0):+.0f}", flush=True)
    pts = []; rngs = []
    t0 = time.time()
    while time.time() - t0 < secs:
        fr, _ = s.get_frame(); bgr = fr[0] if fr else None
        ds = s.get_drone()
        if bgr is not None and ds is not None:
            det, hole_wh = detect_tracked(bgr, p, None, require_hole=True)
            if det is not None and hole_wh is not None:
                ap_px = max(hole_wh)
                if ap_px >= 6.0:
                    d_opt = np.array([(det.u - CX) / FX, (det.v - CY) / FY, 1.0]); d_opt /= np.linalg.norm(d_opt)
                    R_t = quat_to_R(qfix(ds.quat_wxyz))
                    ray_t = R_t @ (R_OPT_T @ d_opt)
                    Z = FX * GATE_AP / ap_px
                    if Z < 60.0:
                        pts.append(ds.pos_ned + Z * (H_DEFAULT @ ray_t)); rngs.append(Z)
        time.sleep(0.02)
    if not pts:
        return None, 0, None
    return np.median(np.array(pts), 0), len(pts), (float(np.min(rngs)), float(np.median(rngs)), float(np.max(rngs)))


def main():
    s = Store(); m = MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); VisionIO(s).start()
    boot = int(time.time() * 1000); c = Commander(m.conn, boot)

    def idle():
        m.conn.mav.set_attitude_target_send(int(time.time() * 1000) - boot, m.conn.target_system,
                                             m.conn.target_component, IDLE, [1., 0, 0, 0], 0, 0, 0, 0)
    prev = s.get_race(); pb = prev["boot_ms"] if prev else None
    c.sim_reset(); t = time.time()
    while time.time() - t < 30:
        idle(); r2 = s.get_race()
        if r2 and (not r2["race_live"] or (pb and r2["boot_ms"] < pb)):
            break
        time.sleep(0.02)
    t = time.time()
    while time.time() - t < 30:
        idle(); d = s.get_drone()
        if s.get_race_live() and d is not None:
            break
        time.sleep(0.02)
    time.sleep(2.0)   # let LPN + TRACK_INFO + camera arrive
    ds = s.get_drone(); lpn = s.get_lpn(); gates = s.get_gates()
    print("ODOM pos_ned :", None if ds is None else np.asarray(ds.pos_ned).round(2), flush=True)
    print("LPN pos      :", None if lpn is None else np.asarray(lpn[0]).round(2), flush=True)
    g0 = None if not gates else np.asarray(gates[0].pos_ned, float)
    g5 = None if not gates or len(gates) < 6 else np.asarray(gates[5].pos_ned, float)
    print("gate0 TRACK  :", None if g0 is None else g0.round(2), flush=True)
    print("gate5 TRACK  :", None if g5 is None else g5.round(2), flush=True)

    # vision anchor probe (keep idling so the FC stays happy during the 3 s capture)
    gv, n, rng = localize_gate0(s, secs=3.0)
    if gv is None:
        print("VISION gate0 : NO DETECTION (gate0 not seen at spawn -- range too far / not in FOV)", flush=True)
    else:
        print(f"VISION gate0 : {gv.round(2)}  (n={n} dets, range min/med/max={tuple(round(x,1) for x in rng)})", flush=True)
        if g0 is not None:
            T = gv - g0
            print(f"ANCHOR T     : {T.round(2)}  (add to every TRACK gate)", flush=True)
            if g5 is not None:
                print(f"gate5 anchored: {(g5 + T).round(2)}", flush=True)
        print(f"gate0 height vs spawn: {gv[2] - (0.0 if ds is None else ds.pos_ned[2]):+.2f} m (NED z; negative = UP)", flush=True)
    idle()


if __name__ == "__main__":
    main()
