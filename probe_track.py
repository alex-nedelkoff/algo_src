"""probe_track.py -- dump the sim's TRACK_INFO gate poses + race status + drone pos (no flight)."""
import time
import numpy as np
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store

from aigp.commander import Commander
s = Store(); m = MavlinkIO(s); assert m.wait_heartbeat(10); m.start()
boot = int(time.time() * 1000); c = Commander(m.conn, boot)
c.sim_reset()
print("reset sent, listening for TRACK_INFO...", flush=True)
t0 = time.time()
gates = None
while time.time() - t0 < 30:
    gates = s.get_gates()
    if gates is not None:
        break
    time.sleep(0.2)
print("race:", s.get_race(), flush=True)
ds = s.get_drone()
if ds is not None:
    print("drone pos_ned:", np.round(ds.pos_ned, 2).tolist(), flush=True)
print("gate_idx:", s.get_gate_idx(), flush=True)
if gates is None:
    print("NO TRACK_INFO received in 15 s", flush=True)
else:
    print(f"gates type={type(gates)}", flush=True)
    try:
        for i, g in enumerate(gates):
            print(f"gate {i}: {g}", flush=True)
    except TypeError:
        print(gates, flush=True)
