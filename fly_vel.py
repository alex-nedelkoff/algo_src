"""Fly straight 45 s using the SIM'S OWN velocity controller (the official
template's interface): SET_POSITION_TARGET_LOCAL_NED, MAV_FRAME_LOCAL_NED,
velocity = (2,0,0) m/s = 2 m/s north. No attitude quaternion, no hand-rolled
controller. Matches PyAIPilotExample/controller.py update_position_flight_control
exactly (yaw ignored). Success = flies north ~2 m/s, stays controlled 45 s."""
import time
import numpy as np
from pymavlink import mavutil
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store
from aigp.commander import Commander
from aigp.geometry import quat_to_R
from aigp.race import wait_for_fresh_race_live

SPEED = 2.0; DURATION = 45.0
MASK = (mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE)

s = Store(); m = MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); VisionIO(s).start()
boot = int(time.time() * 1000); c = Commander(m.conn, boot)
assert wait_for_fresh_race_live(s, c), "not live"
spawn = s.get_drone().pos_ned.copy()
c.arm()
t0 = time.time(); last = -1; tumbled = False
while time.time() - t0 < DURATION:
    now_ms = int(time.time() * 1000) - boot
    m.conn.mav.set_position_target_local_ned_send(
        now_ms, m.conn.target_system, m.conn.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED, MASK,
        0.0, 0.0, 0.0, SPEED, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ds = s.get_drone()
    if ds is not None:
        zb = quat_to_R(ds.quat_wxyz)[:, 2]
        tilt = np.degrees(np.arccos(max(-1, min(1, zb[2]))))
        if tilt > 75:
            tumbled = True
        n = int((time.time() - t0) / 3.0)
        if n != last:
            last = n
            d = ds.pos_ned - spawn
            print(f"t={time.time()-t0:4.1f} N={d[0]:7.1f} E={d[1]:6.1f} D={d[2]:+6.1f} "
                  f"vN={ds.vel_ned[0]:+4.1f} vE={ds.vel_ned[1]:+4.1f} tilt={tilt:3.0f}", flush=True)
    time.sleep(0.004)
d = s.get_drone().pos_ned - spawn
print(f"\nRESULT: N={d[0]:.0f}m E={d[1]:.0f}m D={d[2]:+.0f}m  "
      f"{'TUMBLED' if tumbled else 'CONTROLLED'} over {DURATION:.0f}s", flush=True)
