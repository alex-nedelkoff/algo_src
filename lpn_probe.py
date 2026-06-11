"""lpn_probe.py -- side-by-side LOCAL_POSITION_NED vs ODOMETRY: same chart? native world vel?"""
import time, os
import numpy as np
os.environ.setdefault("MAVLINK20", "1")
from pymavlink import mavutil

conn = mavutil.mavlink_connection("udpin:0.0.0.0:14550", source_system=255)
conn.wait_heartbeat(timeout=10)
print("listening 12 s...", flush=True)
lpn = None; odo = None; n = 0
t0 = time.time()
while time.time() - t0 < 12 and n < 6:
    m = conn.recv_match(type=["LOCAL_POSITION_NED", "ODOMETRY"], blocking=True, timeout=1)
    if m is None:
        continue
    if m.get_type() == "LOCAL_POSITION_NED":
        lpn = m
    else:
        odo = m
    if lpn is not None and odo is not None:
        dp = np.array([lpn.x - odo.x, lpn.y - odo.y, lpn.z - odo.z])
        print(f"pos diff LPN-ODO: [{dp[0]:+.3f},{dp[1]:+.3f},{dp[2]:+.3f}]  "
              f"LPN v=[{lpn.vx:+.2f},{lpn.vy:+.2f},{lpn.vz:+.2f}]  "
              f"ODO v=[{odo.vx:+.2f},{odo.vy:+.2f},{odo.vz:+.2f}]", flush=True)
        lpn = None; odo = None; n += 1
