import time
import numpy as np
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store
from aigp.commander import Commander
s = Store(); m = MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); VisionIO(s).start()
boot = int(time.time()*1000); c = Commander(m.conn, boot)
print("pos before:", np.round(s.get_drone().pos_ned,1) if s.get_drone() else None, flush=True)
c.arm(); time.sleep(0.5); c.sim_reset()
t0=time.time(); last=-1
while time.time()-t0 < 8:
    d=s.get_drone(); r=s.get_race()
    n=int((time.time()-t0)/1)
    if n!=last:
        last=n
        print(f"t={time.time()-t0:.0f} pos={np.round(d.pos_ned,1) if d else None} start_ms={r['race_start_ms'] if r else '?'} live={s.get_race_live()}", flush=True)
    time.sleep(0.05)
