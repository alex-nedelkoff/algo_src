"""GOTO — high-level waypoint navigation CLI over the WaypointNavigator API.

Thin wrapper: does lifecycle (fresh_start + arm), then hands a live store + commander to
aigp.navigator.WaypointNavigator and calls follow(). The proven control loop now lives in
aigp/navigator.py (unit-tested); this file is just the CLI + sim lifecycle.

Usage (live dashboard ON by default -> --no-viz to disable, --rrd <path> to record):
  python goto.py                       # safe default: a small box out front (body frame)
  python goto.py body 6 0 0  6 4 0     # body-relative triples: (fwd, right, down) from spawn heading
  python goto.py world 8 0 0  0 8 0    # world-NED triples: (N, E, D) offsets from spawn

STATUS (live-tested on the VQ sim): single-waypoint go-to flies cleanly. Multi-waypoint paths with
sharp 90-deg turns / precise arrivals are MARGINAL — the weathervane instability fights stop-and-turn.
This platform wants to flow forward; precise waypoint following is better served by the RL policy.
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
      --yaw {hold|face|lookat|course}   camera/heading mode (default hold)
      --lookat X Y Z                    point the camera at (in the same frame as the waypoints)
      --osgn ROLL PITCH YAW             per-axis body-rate sign override (default from NavGains)
      --maxspeed V                      along-track cruise speed cap (m/s)
    """
    opts = {"yaw": "hold", "lookat": None, "osgn": None, "maxspeed": None}
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
          f"yaw={opts['yaw']} lookat={opts['lookat']}: {wps}", flush=True)

    gains = NavGains()
    if opts["osgn"] is not None:
        gains.RATE_SIGN = np.array(opts["osgn"], float)
    if opts["maxspeed"] is not None:
        gains.MAX_SPEED = opts["maxspeed"]

    # real rate_gain (plant[2]) so FlightLog reconstructs rad/s for display (parity with old goto.py);
    # store=s streams the COLLISION flag too (dashboard hard rule).
    flog = ftm.from_args(sys.argv, plant[2], "goto", store=s)
    nav = WaypointNavigator(s, c, plant, gains=gains, flog=flog)
    nav.set_origin(pos_ned=spawn, yaw=yaw0)
    c.arm()
    res = nav.follow(targets, frame=frame, yaw=opts["yaw"], settle=True, look_point=look)
    if flog is not None:
        flog.close()
    print(f"mission {res}", flush=True)


if __name__ == "__main__":
    main()
