"""Read-only laptop connectivity check: import the new API in env aigp, connect MAVLink,
confirm sim telemetry is flowing. NO arm, NO attitude commands. Run from the deploy dir root."""
import sys, time, os
import numpy as np
sys.path.insert(0, os.getcwd())
from aigp.drone import Drone, FlightConfig         # exercises new-code imports (navigator, gate_traj)
from aigp.state import Store
from aigp.io_layer import MavlinkIO, VisionIO

print("imports OK (drone+navigator+gate_traj)")
s = Store()
m = MavlinkIO(s)
print("waiting heartbeat...")
ok = m.wait_heartbeat(10)
print("heartbeat:", ok)
if not ok:
    sys.exit("NO HEARTBEAT — sim not feeding MAVLink to this host on 14550")
m.start(); VisionIO(s).start()
t = time.time(); seen = 0
while time.time() - t < 5:
    d = s.get_drone(); r = s.get_race()
    if d is not None:
        seen += 1
        print(f"  live={s.get_race_live()} pos={np.round(d.pos_ned,2)} "
              f"|v|={np.linalg.norm(d.vel_ned):.2f} race={r}")
    time.sleep(0.5)
print(f"telemetry frames seen: {seen}")
print("CONN_CHECK_DONE")
