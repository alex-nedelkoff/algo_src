"""OPEN-LOOP attitude->accel ground truth. Send a FIXED raw quaternion (no
controller, no position loop) for 2.5 s and log where the drone goes (dpos NED)
and the tilt it achieved (zb = body-down in world, from odometry). Reveals the
sim's true send-quaternion convention so we can replace the guessed remap.
Physics check: the drone should accelerate along -zb_horizontal (thrust=-zb)."""
import json, time
import numpy as np
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store
from aigp.commander import Commander
from aigp.geometry import quat_to_R
from aigp.race import wait_for_fresh_race_live

r = json.load(open("sysid/sim_response.json")); HOVER = r["hover_thrust"]
s = Store(); m = MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); VisionIO(s).start()
boot = int(time.time() * 1000); c = Commander(m.conn, boot)


def probe(q, label, T=2.5):
    assert wait_for_fresh_race_live(s, c), "not live"
    ds0 = s.get_drone(); p0 = ds0.pos_ned.copy()
    R0 = quat_to_R(ds0.quat_wxyz); yaw0 = float(np.degrees(np.arctan2(R0[1, 0], R0[0, 0])))
    c.arm(); t0 = time.time()
    while time.time() - t0 < T:
        c.send_attitude_setpoint(np.asarray(q, float), HOVER)
        time.sleep(0.02)
    ds = s.get_drone(); dp = ds.pos_ned - p0; v = ds.vel_ned
    zb = quat_to_R(ds.quat_wxyz)[:, 2]
    tilt = np.degrees(np.arccos(max(-1, min(1, zb[2]))))
    print(f"{label:12s} q={np.round(np.asarray(q,float),3)} | dpos N={dp[0]:+5.1f} E={dp[1]:+5.1f} "
          f"D={dp[2]:+5.1f} | vN={v[0]:+5.1f} vE={v[1]:+5.1f} | yaw0={yaw0:+4.0f} "
          f"tilt={tilt:3.0f} -zb_h=(N{-zb[0]:+.2f},E{-zb[1]:+.2f})", flush=True)


h = np.radians(10) / 2; cw, sw = np.cos(h), np.sin(h)   # 10 deg rotation quaternion
probe([1, 0, 0, 0], "level")
probe([cw, sw, 0, 0], "rot+x")
probe([cw, -sw, 0, 0], "rot-x")
probe([cw, 0, sw, 0], "rot+y")
probe([cw, 0, -sw, 0], "rot-y")
probe([cw, 0, 0, sw], "rot+z(yaw)")
print("done", flush=True)
