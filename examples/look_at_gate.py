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

    # gate 0 world pose, broadcast via TRACK_INFO after the reset
    gates = None
    t = time.time()
    while gates is None and time.time() - t < 5.0:
        gates = s.get_gates(); time.sleep(0.05)
    assert gates, "no gates received (TRACK_INFO)"
    gate0 = np.asarray(gates[0].pos_ned, float)
    print(f"gate0 world NED: {np.round(gate0, 1).tolist()} (width {gates[0].width:.2f})", flush=True)

    ds0 = s.get_drone(); spawn = ds0.pos_ned.copy()
    yaw0 = float(np.arctan2(*(quat_to_R(ds0.quat_wxyz)[[1, 0], 0])))
    flog = ftm.from_args(sys.argv, plant[2], "look_at_gate", store=s)

    drone = Drone(s, c, plant, config=FlightConfig(), flog=flog)
    drone.nav.set_origin(pos_ned=spawn, yaw=yaw0)
    c.arm()

    drone.look_at(tuple(gate0), frame="world")           # persistent world look point = gate 0
    print("strafe right, camera on gate 0...", flush=True)
    drone.goto((0, 6, 0), yaw="lookat", frame="body").wait()    # strafe right; camera pans to gate
    print("strafe left...", flush=True)
    drone.goto((0, -6, 0), yaw="lookat", frame="body").wait()   # strafe left
    print("back to center...", flush=True)
    drone.goto((0, 0, 0), yaw="lookat", frame="body").wait()
    print("done", flush=True)


if __name__ == "__main__":
    main()
