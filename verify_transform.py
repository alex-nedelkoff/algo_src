"""Verify candidate send-transform q_sim=[w,x,-y,z] (negate pitch/y). Open-loop:
for each desired world accel (N/E/S/W) build the controller's desired attitude,
apply the candidate transform, hold 2 s, log which way the drone actually went.
Success = NORTH->north, EAST->east, SOUTH->south, WEST->west."""
import json, time
import numpy as np
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store
from aigp.commander import Commander
from aigp.geometry import quat_to_R
from aigp.control_math import desired_attitude, mat_to_quat, collective_accel, accel_to_thrust_norm
from aigp.race import wait_for_fresh_race_live

r = json.load(open("sysid/sim_response.json")); HOVER = r["hover_thrust"]; KA = r["k_a"]
s = Store(); m = MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); VisionIO(s).start()
boot = int(time.time() * 1000); c = Commander(m.conn, boot)


def cand(q):                       # candidate transform: negate y (pitch) component
    return np.array([q[0], q[1], -q[2], q[3]])


def go(a_des, label, T=2.0):
    a_des = np.asarray(a_des, float)
    assert wait_for_fresh_race_live(s, c), "not live"
    ds0 = s.get_drone(); p0 = ds0.pos_ned.copy()
    yaw0 = float(np.arctan2(quat_to_R(ds0.quat_wxyz)[1, 0], quat_to_R(ds0.quat_wxyz)[0, 0]))
    c.arm(); t0 = time.time()
    while time.time() - t0 < T:
        ds = s.get_drone()
        if ds is not None:
            q_des = mat_to_quat(desired_attitude(a_des, yaw0))
            thr = accel_to_thrust_norm(collective_accel(a_des, ds.quat_wxyz), HOVER, KA)
            c.send_attitude_setpoint(cand(q_des), thr)
        time.sleep(0.02)
    dp = s.get_drone().pos_ned - p0
    want = {"NORTH": "N+", "EAST": "E+", "SOUTH": "N-", "WEST": "E-"}[label]
    ok = {"NORTH": dp[0] > 1, "EAST": dp[1] > 1, "SOUTH": dp[0] < -1, "WEST": dp[1] < -1}[label]
    print(f"{label:6s} a_des={a_des} -> dpos N={dp[0]:+5.1f} E={dp[1]:+5.1f} D={dp[2]:+5.1f}  "
          f"want {want}  {'OK' if ok else 'WRONG'}", flush=True)


go([3, 0, 0], "NORTH")
go([0, 3, 0], "EAST")
go([-3, 0, 0], "SOUTH")
go([0, -3, 0], "WEST")
print("done", flush=True)
