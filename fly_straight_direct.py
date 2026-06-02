"""Fly straight 45 s, building the SEND quaternion DIRECTLY from the empirical
send->accel map (probe ground truth: send +y->north, send -x->east), bypassing our
quaternion convention + the remap. Yaw command held at 0 (no heading coupling).
Tilt angles clamped. Success = flies north, east stays ~0, survives 45 s."""
import json, time
import numpy as np
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store
from aigp.commander import Commander
from aigp.geometry import quat_to_R
from aigp.control_math import desired_accel, quat_mul, collective_accel, accel_to_thrust_norm, G
from aigp.race import wait_for_fresh_race_live

SPEED = 2.0; DURATION = 45.0; TILT_MAX = np.radians(12)
KP = np.array([1.0, 1.0, 1.8]); KD = np.array([3.2, 3.2, 3.0])

r = json.load(open("sysid/sim_response.json")); HOVER = r["hover_thrust"]; KA = r["k_a"]
s = Store(); m = MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); VisionIO(s).start()
boot = int(time.time() * 1000); c = Commander(m.conn, boot)
assert wait_for_fresh_race_live(s, c), "not live"
ds0 = s.get_drone(); spawn = ds0.pos_ned.copy()
fwd = np.array([1.0, 0.0, 0.0])      # world north
c.arm()
dt = 0.02; t0 = time.time(); last = -1; tumbled = False
while time.time() - t0 < DURATION:
    ds = s.get_drone()
    if ds is not None:
        vel_sp = SPEED * fwd
        pos_sp = np.array([ds.pos_ned[0] + vel_sp[0] * 0.3, ds.pos_ned[1], spawn[2]])
        a = desired_accel(ds.pos_ned, ds.vel_ned, pos_sp, vel_sp, KP, KD)
        thN = np.clip(np.arctan2(a[0], G), -TILT_MAX, TILT_MAX)   # +y rot -> north
        thE = np.clip(np.arctan2(a[1], G), -TILT_MAX, TILT_MAX)   # -x rot -> east
        qy = np.array([np.cos(thN / 2), 0, np.sin(thN / 2), 0])
        qx = np.array([np.cos(thE / 2), -np.sin(thE / 2), 0, 0])
        q_send = quat_mul(qy, qx)
        thr = accel_to_thrust_norm(collective_accel(a, ds.quat_wxyz), HOVER, KA)
        c.send_attitude_setpoint(q_send, thr)
        zb = quat_to_R(ds.quat_wxyz)[:, 2]
        tilt = np.degrees(np.arccos(max(-1, min(1, zb[2]))))
        if tilt > 70:
            tumbled = True
        n = int((time.time() - t0) / 3.0)
        if n != last:
            last = n
            d = ds.pos_ned - spawn
            print(f"t={time.time()-t0:4.1f} fwd={d[0]:7.1f}m lat={d[1]:7.1f}m alt_err={d[2]:+6.1f}m "
                  f"spd={np.linalg.norm(ds.vel_ned[:2]):4.1f} tilt={tilt:3.0f}", flush=True)
    time.sleep(dt)
d = s.get_drone().pos_ned - spawn
print(f"\nRESULT: fwd(N)={d[0]:.0f}m lateral(E)={d[1]:.0f}m alt_err={d[2]:+.0f}m "
      f"{'TUMBLED' if tumbled else 'NO TUMBLE'} over {DURATION:.0f}s", flush=True)
