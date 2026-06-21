"""Demo: lock the camera on the FIRST RACE GATE (gate 0 from the sim's TRACK_INFO broadcast)
while the drone strafes side to side, so the camera pans to keep the gate in view.

Run on the laptop worktree (sim up):  python examples/look_at_gate.py
Watch Mac Rerun: the cyan `world/expected_los` ray + `world/los_target` marker sit ON gate 0;
the camera frustum (and FPV feed) should hold the gate.
"""
import os
import sys
import time

import numpy as np
from pymavlink import mavutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path

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


def main():
    s = Store()
    m = MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); VisionIO(s).start()
    boot = int(time.time() * 1000); c = Commander(m.conn, boot)
    plant = load_plant("sysid/sim_response.json")
    assert fresh_start(s, c, m, boot), "not live"

    ds0 = s.get_drone(); spawn = ds0.pos_ned.copy()

    # gate 0 world pose (TRACK_INFO). The first post-reset broadcast can be GARBAGE (z = -4569 m
    # seen) -> poll until a sane reading: within ~150 m of spawn and |z| < 30 m. (gates[].pos_ned is
    # the same world-NED frame as the drone's pos_ned -- vq_track_wp flies to it directly.)
    gate0 = None
    t = time.time()
    while time.time() - t < 12.0:
        gates = s.get_gates()
        if gates:
            g0 = np.asarray(gates[0].pos_ned, float)
            if np.linalg.norm((g0 - spawn)[:2]) < 150.0 and abs(g0[2] - spawn[2]) < 30.0:
                gate0 = g0; break
            print(f"  ...rejecting garbage gate0 {np.round(g0, 1).tolist()}", flush=True)
        time.sleep(0.2)
    assert gate0 is not None, "no SANE gate0 from TRACK_INFO (only garbage reads)"
    print(f"gate0 world NED: {np.round(gate0, 1).tolist()} (width {gates[0].width:.2f}); "
          f"spawn {np.round(spawn, 1).tolist()}", flush=True)

    yaw0 = float(np.arctan2(*(quat_to_R(ds0.quat_wxyz)[[1, 0], 0])))
    flog = ftm.from_args(sys.argv, plant[2], "look_at_gate", store=s)

    # BEST CASE for a FIXED 20deg-up camera + yaw-only aim: view the gate from a point R_VIEW in
    # front of it and DZ = R_VIEW*tan(20) BELOW it, so the gate sits at the camera's 20deg elevation
    # (gel ~= cel ~= +20 -> centered). Gentle config -> minimal body tilt -> steady camera. A slow
    # lateral sweep keeps the gate ~centered while the heading tracks it.
    R_VIEW = 12.0
    DZ = R_VIEW * np.tan(np.radians(20.0))               # ~4.4 m below the gate
    front = (spawn - gate0)[:2]; front = front / max(np.linalg.norm(front), 1e-6)  # gate -> spawn dir
    perp = np.array([-front[1], front[0]])
    view0 = np.array([gate0[0] + front[0] * R_VIEW, gate0[1] + front[1] * R_VIEW, gate0[2] + DZ])
    viewL = view0 + np.array([perp[0] * 4.0, perp[1] * 4.0, 0.0])
    viewR = view0 - np.array([perp[0] * 4.0, perp[1] * 4.0, 0.0])

    drone = Drone(s, c, plant, config=FlightConfig(vmax=2.0, amax=1.5, tilt_deg=12.0,
                                                   default_speed=1.5), flog=flog)
    drone.nav.set_origin(pos_ned=spawn, yaw=yaw0)
    c.arm()

    drone.look_at(tuple(gate0), frame="world")           # persistent world look point = gate 0
    print(f"move to view point {np.round(view0, 1).tolist()} (in front of + below gate)...", flush=True)
    drone.goto(tuple(view0), yaw="lookat", frame="world", speed=1.5).wait()
    print("slow sweep left, camera holding gate 0...", flush=True)
    drone.goto(tuple(viewL), yaw="lookat", frame="world", speed=1.2).wait()
    print("slow sweep right...", flush=True)
    drone.goto(tuple(viewR), yaw="lookat", frame="world", speed=1.2).wait()
    print("center...", flush=True)
    drone.goto(tuple(view0), yaw="lookat", frame="world", speed=1.2).wait()
    print("done", flush=True)


if __name__ == "__main__":
    main()
