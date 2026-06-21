"""Demo: connect + arm the VQ sim, then drive the Drone control API.

Run on the laptop worktree (sim up):  python examples/drone_demo.py
"""
import sys
import threading
import time

import numpy as np
from pymavlink import mavutil

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
    yaw0 = float(np.arctan2(*(quat_to_R(ds0.quat_wxyz)[[1, 0], 0])))
    flog = ftm.from_args(sys.argv, plant[2], "drone_demo", store=s)

    drone = Drone(s, c, plant, config=FlightConfig(), flog=flog)
    drone.set_origin(pos_ned=spawn, yaw=yaw0)
    c.arm()

    drone.takeoff(2.0).wait()                                  # climb 2 m
    drone.orbit((10, 0, 0), radius=5.0, seconds=8.0).wait()    # circle a point, camera locked
    m_far = drone.goto((12, 8, 0), yaw="lookat", look_at=(0, 0, 0))   # fly while watching spawn
    time.sleep(2.0); print("status:", m_far.status().phase, m_far.status().tilt_deg)
    drone.goto((0, 0, 0), yaw="hold")                          # preempts m_far (auto-abort)
    drone.land(2.0).wait()
    print("done")


if __name__ == "__main__":
    main()
