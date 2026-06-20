"""GOTO — high-level waypoint navigation CLI over the WaypointNavigator API.

Thin wrapper: does lifecycle (fresh_start + arm), then hands a live store + commander to
aigp.navigator.WaypointNavigator and calls follow(). The proven control loop now lives in
aigp/navigator.py (unit-tested); this file is just the CLI + sim lifecycle.

Usage (live dashboard ON by default -> --no-viz to disable, --rrd <path> to record):
  python goto.py                       # safe default: a small box out front (body frame)
  python goto.py body 6 0 0  6 4 0     # body-relative triples: (fwd, right, down) from spawn heading
  python goto.py world 8 0 0  0 8 0    # world-NED triples: (N, E, D) offsets from spawn
  python goto.py body 12 0 0  12 12 0  0 12 0  0 0 0 --yaw hold   # CAMERA-DECOUPLED square (fixed heading)

CAMERA / HEADING MODES (--yaw): course (camera along travel, nose-first), face (nose at target),
hold (FIXED spawn heading — camera-decoupled strafe), lookat (camera about a point, WIP). Nose-first
modes fly the smooth spline; strafe modes (hold) auto-use the gentle legs engine + true-frame control
(s_lat auto-probed, world-y mirror) — the proven vq_waypoint2 --square chain, no rate-loop ring.

STATUS (live-tested on the VQ sim): nose-first single/multi-waypoint flies cleanly. Camera-decoupled
fixed-heading strafe (--yaw hold) ported from the proven vq_waypoint2 --square (4/4, ~18 deg tilt).
"""
import sys
import time

import numpy as np
from pymavlink import mavutil

import aigp.flight_telemetry as ftm
from aigp.commander import Commander
from aigp.geometry import quat_to_R
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.navigator import WaypointNavigator, load_plant, NavGains
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


def parse_opts(argv):
    """Pull non-positional flags out of argv, returning (remaining_positionals, opts).
      --yaw {hold|face|lookat|course}   camera/heading mode (default course)
      --lookat X Y Z                    point the camera at (in the same frame as the waypoints)
      --osgn ROLL PITCH YAW             per-axis body-rate sign override (default from NavGains)
      --maxspeed V                      along-track cruise speed cap (m/s)
      --vcruise V                       spline cruise speed (m/s, default 2.5)
      --legs                            use point-to-point legs engine instead of spline
      --zvd                             enable ZVD input shaper on rate commands (ring killer)
      --zvddelay N                      uniform ZVD delay in loop frames (default 14)
      --slat S                          force strafe lateral sign (+1/-1), skip the live probe
      --ymirror {0|1}                   force the true-frame world-y mirror (default 1)
    """
    opts = {"yaw": "course", "lookat": None, "osgn": None, "maxspeed": None,
            "vcruise": 2.5, "legs": False, "zvd": False, "zvddelay": (14, 14, 14),
            "slat": None, "ymirror": None}
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a == "--yaw" and i + 1 < len(argv):
            opts["yaw"] = argv[i + 1]; i += 2
        elif a == "--lookat" and i + 3 < len(argv):
            opts["lookat"] = (float(argv[i + 1]), float(argv[i + 2]), float(argv[i + 3])); i += 4
        elif a == "--osgn" and i + 3 < len(argv):
            opts["osgn"] = [float(argv[i + 1]), float(argv[i + 2]), float(argv[i + 3])]; i += 4
        elif a == "--maxspeed" and i + 1 < len(argv):
            opts["maxspeed"] = float(argv[i + 1]); i += 2
        elif a == "--vcruise" and i + 1 < len(argv):
            opts["vcruise"] = float(argv[i + 1]); i += 2
        elif a == "--legs":
            opts["legs"] = True; i += 1
        elif a == "--zvd":
            opts["zvd"] = True; i += 1
        elif a == "--zvddelay" and i + 1 < len(argv):
            n = int(argv[i + 1]); opts["zvddelay"] = (n, n, n); i += 2
        elif a == "--slat" and i + 1 < len(argv):
            opts["slat"] = float(argv[i + 1]); i += 2
        elif a == "--ymirror" and i + 1 < len(argv):
            opts["ymirror"] = bool(int(argv[i + 1])); i += 2
        else:
            out.append(a); i += 1
    return out, opts


def parse_args(argv):
    body = True
    if argv and argv[0] in ("body", "world"):
        body = (argv[0] == "body"); argv = argv[1:]
    if len(argv) >= 3:
        f = [float(x) for x in argv]
        return body, [tuple(f[i:i + 3]) for i in range(0, len(f) - len(f) % 3, 3)]
    return True, [(6.0, 0.0, 0.0), (6.0, 4.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 0.0)]


def main():
    pos_argv, opts = parse_opts(ftm.strip_viz_args(sys.argv[1:]))
    body, wps = parse_args(pos_argv)
    s = Store()
    m = MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); VisionIO(s).start()
    boot = int(time.time() * 1000)
    c = Commander(m.conn, boot)
    plant = load_plant("sysid/sim_response.json")
    assert fresh_start(s, c, m, boot), "not live"

    ds0 = s.get_drone(); spawn = ds0.pos_ned.copy()
    R0 = quat_to_R(ds0.quat_wxyz)
    yaw0 = float(np.arctan2(R0[1, 0], R0[0, 0]))

    # 'world' triples preserve goto.py's historical semantics (offsets from spawn) -> convert to
    # absolute NED here and call frame='world'; 'body' triples pass straight through.
    if body:
        targets = wps
        look = opts["lookat"]                                   # body coords; follow resolves
        frame = "body"
    else:
        targets = [tuple((spawn + np.array(o, float)).tolist()) for o in wps]
        look = (tuple((spawn + np.array(opts["lookat"], float)).tolist())
                if opts["lookat"] is not None else None)         # offset-from-spawn -> absolute
        frame = "world"

    print(f"GOTO {len(wps)} waypoints ({'body fwd/right/down' if body else 'world NED'}) "
          f"yaw={opts['yaw']} lookat={opts['lookat']} engine={'legs' if opts['legs'] else 'spline'}"
          f" vcruise={opts['vcruise']}: {wps}", flush=True)

    gains = NavGains()
    if opts["osgn"] is not None:
        gains.RATE_SIGN = np.array(opts["osgn"], float)
    if opts["maxspeed"] is not None:
        gains.MAX_SPEED = opts["maxspeed"]
    if opts["zvd"]:
        gains.ZVD = True
        gains.ZVD_DELAY = opts["zvddelay"]

    # real rate_gain (plant[2]) so FlightLog reconstructs rad/s for display (parity with old goto.py);
    # store=s streams the COLLISION flag too (dashboard hard rule).
    flog = ftm.from_args(sys.argv, plant[2], "goto", store=s)
    nav = WaypointNavigator(s, c, plant, gains=gains, flog=flog)
    nav.set_origin(pos_ned=spawn, yaw=yaw0)
    if opts["ymirror"] is not None:
        nav._tf_ymirror = opts["ymirror"]
    if opts["slat"] is not None:
        nav._s_lat = float(opts["slat"])     # forced sign -> _probe_s_lat is skipped
    print(f"calib: s_cam={nav._s_cam:+.0f} yaw0_t={np.degrees(nav._yaw0_t):+.0f} "
          f"cam_live=({nav._cam_live[0]:+.2f},{nav._cam_live[1]:+.2f}) "
          f"ymirror={nav._tf_ymirror} s_lat={nav._s_lat:+.0f}", flush=True)
    c.arm()
    res = nav.follow(targets, yaw=opts["yaw"], look_point=look, v_cruise=opts["vcruise"],
                     engine=("legs" if opts["legs"] else "spline"), frame=frame)
    if flog is not None:
        flog.close()
    print(f"mission {res}", flush=True)


if __name__ == "__main__":
    main()
