"""Clear the latched-high throttle and get the race live, ACRO style.
Send zero-rate, zero-thrust idle frames (the min-throttle state the countdown
gate wants), reset, and hold idle through the countdown. Read-only otherwise.
Reports race status + drone respawn. NO climb/flight commands."""
import time
import numpy as np
from pymavlink import mavutil
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store
from aigp.commander import Commander

IDLE_MASK = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE


def idle(conn, boot):
    conn.mav.set_attitude_target_send(
        int(time.time() * 1000) - boot, conn.target_system, conn.target_component,
        IDLE_MASK, [1.0, 0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0)   # rates 0, thrust 0


s = Store(); m = MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); VisionIO(s).start()
boot = int(time.time() * 1000); c = Commander(m.conn, boot)

print("phase 1: idle thrust=0 to clear latched throttle (2s)", flush=True)
t0 = time.time()
while time.time() - t0 < 2.0:
    idle(m.conn, boot); time.sleep(0.02)

print("phase 2: sim_reset, then hold idle and wait for countdown", flush=True)
c.sim_reset()
t0 = time.time(); last = -1
while time.time() - t0 < 30.0:
    idle(m.conn, boot)
    r = s.get_race(); d = s.get_drone()
    n = int((time.time() - t0) / 1.0)
    if n != last:
        last = n
        pos = None if d is None else np.round(d.pos_ned, 1)
        print(f"t={time.time()-t0:4.1f} start_ms={r['race_start_ms'] if r else '?'} "
              f"boot_ms={r['boot_ms'] if r else '?'} live={s.get_race_live()} pos={pos}", flush=True)
    if s.get_race_live():
        print(f">>> RACE LIVE at t={time.time()-t0:.1f}", flush=True)
        break
    time.sleep(0.02)
else:
    print(">>> timed out, never went live", flush=True)
