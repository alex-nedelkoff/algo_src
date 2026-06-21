"""Restart the race via the SIM_RESET MAVLink command (no GUI). Sends a few resets, waits for the
race to go live + the drone to settle at spawn. Prints the outcome."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.getcwd())
from aigp.commander import Commander
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store
from pymavlink import mavutil

IDLE = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE

s = Store()
m = MavlinkIO(s)
assert m.wait_heartbeat(10), "no heartbeat"
m.start(); VisionIO(s).start()
boot = int(time.time() * 1000); c = Commander(m.conn, boot)


def idle():
    m.conn.mav.set_attitude_target_send(int(time.time() * 1000) - boot, m.conn.target_system,
                                        m.conn.target_component, IDLE, [1.0, 0, 0, 0], 0, 0, 0, 0)


print("before:", s.get_race())
for attempt in range(3):
    c.sim_reset()
    print(f"sent SIM_RESET #{attempt+1}", flush=True)
    t = time.time()
    while time.time() - t < 12:
        idle()
        d = s.get_drone(); r = s.get_race()
        if s.get_race_live() and d is not None and np.linalg.norm(d.vel_ned) < 3.0:
            print(f"LIVE: pos={np.round(d.pos_ned,2)} |v|={np.linalg.norm(d.vel_ned):.2f} race={r}", flush=True)
            sys.exit(0)
        time.sleep(0.05)
    print(f"  not live yet (race={s.get_race()})", flush=True)
print("SIM_RESET did NOT bring race live", flush=True)
sys.exit(1)
