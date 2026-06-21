"""Clean orbit-only live run: clear the trail between missions so the dashboard shows
actual-vs-commanded for ONE maneuver (the orbit), not the whole session tangle.

takeoff -> [clear trail] -> orbit (2 laps, camera locked on center) -> [clear trail] -> land.

Run on the laptop (sim up):  python scripts/live_orbit.py
"""
import os, sys, time
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


def main():
    s = Store()
    m = MavlinkIO(s); assert m.wait_heartbeat(10), "no heartbeat"; m.start(); VisionIO(s).start()
    boot = int(time.time() * 1000); c = Commander(m.conn, boot)
    plant = load_plant("sysid/sim_response.json")
    assert fresh_start(s, c, m, boot), "not live"
    ds0 = s.get_drone(); spawn = ds0.pos_ned.copy()
    yaw0 = float(np.arctan2(*(quat_to_R(ds0.quat_wxyz)[[1, 0], 0])))
    flog = ftm.from_args(sys.argv, plant[2], "live_orbit", store=s)

    cfg = FlightConfig(vmax=4.0, amax=3.5, tilt_deg=18.0, default_speed=3.0)
    drone = Drone(s, c, plant, config=cfg, flog=flog)
    drone.set_origin(pos_ned=spawn, yaw=yaw0)
    c.arm()
    print(f"spawn={np.round(spawn,2)} yaw0={yaw0:.2f}", flush=True)

    if flog:
        flog.clear_trail()
    drone.takeoff(3.0).wait(timeout=10)                         # climb to orbit altitude

    if flog:
        flog.clear_trail()                                     # <-- only the orbit trail from here
    # center 6 m forward, 3 m up (body frame -z = up), radius 4, ~2 laps; camera locked on center
    st = drone.orbit((6, 0, -3), radius=4.0, seconds=16.0).wait(timeout=45)
    print(f"orbit result={st.result.value} end={np.round(s.get_drone().pos_ned,2)}", flush=True)

    if flog:
        flog.clear_trail()
    drone.land(3.0).wait(timeout=10)
    print("LIVE_ORBIT_DONE", flush=True)
    time.sleep(1.0)                                            # let the daemon flush the last frames


if __name__ == "__main__":
    main()
