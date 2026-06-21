"""Spline-engine demo on the SAME 3D figure-eight (vs the legs-engine run in live_fig8.py).

Drives navigator._follow_spline directly (the public goto/follow force engine='legs' for every yaw
mode). GateTrajectory builds a smooth, curvature-aware trajectory through the eight; tracked with
pre-stabilize + V-ramp + local-window projection. Scored against the SAME 72-pt commanded polyline
(min point-to-segment) so the tracking error is directly comparable to the legs run.

Run on the laptop (sim up):  python scripts/live_fig8_spline.py
"""
import os, sys, time, threading
import numpy as np
from pymavlink import mavutil

sys.path.insert(0, os.getcwd())

import aigp.flight_telemetry as ftm
from aigp.commander import Commander
from aigp.drone import Drone, FlightConfig
from aigp.geometry import quat_to_R
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.navigator import load_plant
from aigp.state import Store

IDLE = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE


def idle(m, boot):
    m.conn.mav.set_attitude_target_send(int(time.time() * 1000) - boot, m.conn.target_system,
                                        m.conn.target_component, IDLE, [1.0, 0, 0, 0], 0, 0, 0, 0)


def fresh_start(s, c, m, boot):
    t = time.time()
    while time.time() - t < 1.0:
        idle(m, boot); time.sleep(0.02)
    prev = s.get_race(); pb = prev["boot_ms"] if prev else None
    c.sim_reset(); t = time.time()
    while time.time() - t < 30:
        idle(m, boot); r2 = s.get_race()
        if r2 and (not r2["race_live"] or (pb and r2["boot_ms"] < pb)):
            break
        time.sleep(0.02)
    while time.time() - t < 30:
        idle(m, boot); d = s.get_drone()
        if s.get_race_live() and d is not None and np.linalg.norm(d.vel_ned) < 3.0:
            return True
        time.sleep(0.02)
    return False


def figure_eight(cx, A, W, H, base_alt, n):
    pts = []
    for k in range(n):
        t = 2.0 * np.pi * k / n
        pts.append(np.array([cx + A * np.sin(t),
                             0.5 * W * np.sin(2.0 * t),
                             -base_alt - H * np.sin(t)]))
    return pts


def _pt_seg(p, a, b):
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom < 1e-12 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return p - (a + t * ab)


def poly_err(p, P):
    bn = 1e18; best = None
    for i in range(len(P)):
        r = _pt_seg(p, P[i], P[(i + 1) % len(P)])
        n = float(np.linalg.norm(r))
        if n < bn:
            bn = n; best = r
    return best


def tilt_of(quat):
    R = quat_to_R(np.asarray(quat, float))
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))


def main():
    s = Store()
    m = MavlinkIO(s); assert m.wait_heartbeat(10), "no heartbeat"; m.start(); VisionIO(s).start()
    boot = int(time.time() * 1000); c = Commander(m.conn, boot)
    plant = load_plant("sysid/sim_response.json")
    assert fresh_start(s, c, m, boot), "not live"
    ds0 = s.get_drone(); spawn = ds0.pos_ned.copy()
    yaw0 = float(np.arctan2(*(quat_to_R(ds0.quat_wxyz)[[1, 0], 0])))
    flog = ftm.from_args(sys.argv, plant[2], "live_fig8_spline", store=s)

    cfg = FlightConfig(vmax=4.0, amax=3.5, tilt_deg=18.0, default_speed=3.0)
    drone = Drone(s, c, plant, config=cfg, flog=flog)
    drone.set_origin(pos_ned=spawn, yaw=yaw0)
    drone.nav._t0 = time.time()                                 # _follow_spline/flog.push need it
    wmax = float(sys.argv[sys.argv.index("--wmax") + 1]) if "--wmax" in sys.argv else drone.nav.gains.WMAX
    drone.nav.gains.WMAX = wmax
    if "--kdr" in sys.argv:                                     # spline rate-D (passed as kd_rate=g.KD_ATT)
        drone.nav.gains.KD_ATT = float(sys.argv[sys.argv.index("--kdr") + 1])
    print(f"WMAX={wmax} KD_ATT(rateD)={drone.nav.gains.KD_ATT}", flush=True)
    c.arm()
    print(f"spawn={np.round(spawn,2)} yaw0={yaw0:.2f}", flush=True)

    body_pts = figure_eight(cx=7.0, A=5.0, W=6.0, H=1.2, base_alt=3.0, n=72)
    world_pts = [drone.nav._resolve(p, "body") for p in body_pts]
    P = world_pts + [world_pts[0]]

    if flog:
        flog.clear_trail()

    stop = threading.Event()
    errs, xy, ze, tilts = [], [], [], []

    def watch():
        while not stop.is_set():
            d = s.get_drone()
            if d is not None and np.all(np.isfinite(d.pos_ned)):
                r = poly_err(d.pos_ned.astype(float), P)
                errs.append(float(np.linalg.norm(r)))
                xy.append(float(np.linalg.norm(r[:2]))); ze.append(abs(float(r[2])))
                tilts.append(tilt_of(d.quat_wxyz))
            time.sleep(0.05)

    th = threading.Thread(target=watch, daemon=True); th.start()
    t0 = time.time()
    res = drone.nav._follow_spline(world_pts, yaw="course", v_cruise=3.0)   # SPLINE ENGINE
    dur = time.time() - t0
    stop.set(); th.join(timeout=1)

    if flog:
        flog.clear_trail()
    if res != "abort":
        drone.land(3.0).wait(timeout=10)
    else:
        print("aborted -> skip land (avoid post-abort flail/wedge)", flush=True)

    # error-vs-polyline only counts the tracking phase (skip the spawn->eight lead-in's first ~3 s
    # pre-stab + climb), to compare like-for-like with the legs run that started from altitude.
    def stat(a): return (round(np.max(a), 2), round(float(np.mean(a)), 2)) if a else (None, None)
    e_mx, e_mn = stat(errs); xy_mx, xy_mn = stat(xy); z_mx, z_mn = stat(ze)
    print("\n=== FIGURE-EIGHT TRACKING (SPLINE) ===", flush=True)
    print(f"  result={res}  dur={dur:.1f}s  pts={len(world_pts)}  samples={len(errs)}", flush=True)
    print(f"  3D err   max={e_mx} mean={e_mn} m", flush=True)
    print(f"  XY  err  max={xy_mx} mean={xy_mn} m", flush=True)
    print(f"  Z   err  max={z_mx} mean={z_mn} m", flush=True)
    print(f"  tilt     max={round(max(tilts),1) if tilts else None} deg", flush=True)
    print("LIVE_FIG8_SPLINE_DONE", flush=True)
    time.sleep(1.0)


if __name__ == "__main__":
    main()
