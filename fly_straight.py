"""Fly straight from spawn for 45 s continuous (velocity tracking, no position
hold -> no windup). Command constant forward velocity + hold altitude; log forward
distance, lateral deviation, altitude error, tilt. Success = stays controlled
(no tumble, bounded tilt, altitude held) flying roughly straight the whole time."""
import json, time
import numpy as np
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store
from aigp.commander import Commander
from aigp.attitude_control import AttitudeSetpointController
from aigp.geometry import quat_to_R
from aigp.race import wait_for_fresh_race_live

SPEED = 2.0          # m/s forward
DURATION = 45.0

r = json.load(open("sysid/sim_response.json"))
s = Store(); m = MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); VisionIO(s).start()
boot = int(time.time() * 1000); c = Commander(m.conn, boot)
assert wait_for_fresh_race_live(s, c), "not live"
ds0 = s.get_drone(); spawn = ds0.pos_ned.copy()
R0 = quat_to_R(ds0.quat_wxyz); yaw0 = float(np.arctan2(R0[1, 0], R0[0, 0]))
fwd = np.array([np.cos(yaw0), np.sin(yaw0), 0.0])     # body-forward heading in world
ctl = AttitudeSetpointController(hover_thrust=r["hover_thrust"], k_a=r["k_a"],
                                 kp_pos=[1.0, 1.0, 1.8], kd_pos=[3.2, 3.2, 3.0], tilt_max_deg=12.0)
print(f"spawn yaw={yaw0:.2f} fwd=({fwd[0]:+.2f},{fwd[1]:+.2f}) speed={SPEED}", flush=True)
c.arm()
dt = 0.02; t0 = time.time(); last = -1; tumbled = False
while time.time() - t0 < DURATION:
    ds = s.get_drone()
    if ds is not None:
        vel_sp = SPEED * fwd
        pos_sp = np.array([ds.pos_ned[0] + vel_sp[0] * 0.3,
                           ds.pos_ned[1] + vel_sp[1] * 0.3, spawn[2]])  # advance horiz, hold alt
        q, th = ctl.update(ds.pos_ned, ds.vel_ned, ds.quat_wxyz, pos_sp, vel_sp, yaw0, dt=dt)
        c.send_attitude_setpoint(q, th)
        zb = quat_to_R(ds.quat_wxyz)[:, 2]
        tilt = np.degrees(np.arccos(max(-1, min(1, zb[2]))))
        if tilt > 70:
            tumbled = True
        n = int((time.time() - t0) / 3.0)
        if n != last:
            last = n
            d = ds.pos_ned - spawn
            fdist = float(d @ fwd)
            lat = float(np.linalg.norm(d[:2] - fdist * fwd[:2]))
            spd = float(np.linalg.norm(ds.vel_ned[:2]))
            print(f"t={time.time()-t0:4.1f} fwd={fdist:6.1f}m lat={lat:5.1f}m "
                  f"alt_err={d[2]:+5.1f}m spd={spd:.1f} tilt={tilt:3.0f}", flush=True)
    time.sleep(dt)
d = s.get_drone().pos_ned - spawn
fdist = float(d @ fwd); lat = float(np.linalg.norm(d[:2] - fdist * fwd[:2]))
print(f"\nRESULT: flew {fdist:.0f}m forward, lateral dev {lat:.1f}m, alt_err {d[2]:+.1f}m, "
      f"{'TUMBLED' if tumbled else 'no tumble'} over {DURATION:.0f}s", flush=True)
