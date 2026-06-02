import time
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store
s = Store(); m = MavlinkIO(s); assert m.wait_heartbeat(10); m.start()
t0 = time.time(); last = None
while time.time() - t0 < 8:
    r = s.get_race()
    if r is not None and r != last:
        last = dict(r)
        print(f"t={time.time()-t0:4.1f} race={r} live={s.get_race_live()}", flush=True)
    time.sleep(0.1)
d = s.get_drone()
print("drone:", None if d is None else f"pos={d.pos_ned}", flush=True)
print("final race:", s.get_race(), "live:", s.get_race_live(), flush=True)
