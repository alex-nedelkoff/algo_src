"""GOTO — high-level waypoint navigation over the ACRO interface (a set_position replacement).

Flies through a list of NED waypoints (offsets from spawn) using GoToWaypoint guidance +
race_cruise's proven pos/vel -> ACRO controller: velocity-toward-target (slowed on approach),
yaw faces the target en route (camera-forward = weathervane-stable), modest speed cap + tilt cap
for the weathervane-unstable platform. Arrival-detected sequencing; per-waypoint timeout.

Usage: python goto.py            # flies the default square
       python goto.py 8 0 0  0 8 0   # flat list of N,E,D triples (NED, rel. to spawn)
"""
import sys, json, time
import numpy as np
from pymavlink import mavutil
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store
from aigp.commander import Commander
from aigp.geometry import quat_to_R
from aigp.control_math import (desired_attitude, mat_to_quat, attitude_error_quat,
                               collective_accel, accel_to_thrust_norm)
from aigp.guidance import GoToWaypoint

# --- race_cruise winning gains ---
KP_ATT = np.array([0.5, 1.6, 1.0]); KP_YAW = 3.0; KD_YAW = 0.3
KD_AL = 1.2; KP_Z = 1.8; KD_Z = 3.0
WMAX = 4.0; LOOP_DT = 0.004; TILT_MAX_ACC = np.tan(np.radians(15)) * 9.81; ABORT_TILT = 80.0
WP_TIMEOUT = 30.0; MAX_SPEED = 2.0; ARRIVE = 0.6     # modest speed: weathervane-safe

r = json.load(open("sysid/sim_response.json")); HOVER = r["hover_thrust"]; KA = r["k_a"]
RG = np.array([r["rate_gain_axes"]["roll"], r["rate_gain_axes"]["pitch"], r["rate_gain_axes"]["yaw"]])
IDLE = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE

s = Store(); m = MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); VisionIO(s).start()
boot = int(time.time() * 1000); c = Commander(m.conn, boot)


def idle():
    m.conn.mav.set_attitude_target_send(int(time.time() * 1000) - boot, m.conn.target_system,
                                        m.conn.target_component, IDLE, [1.0, 0, 0, 0], 0, 0, 0, 0)


def fresh_start():
    t = time.time()
    while time.time() - t < 1.0:
        idle(); time.sleep(0.02)
    prev = s.get_race(); pb = prev["boot_ms"] if prev else None
    c.sim_reset(); t = time.time()
    while time.time() - t < 30:
        idle(); r2 = s.get_race()
        if r2 and (not r2["race_live"] or (pb and r2["boot_ms"] < pb)):
            break
        time.sleep(0.02)
    while time.time() - t < 30:
        idle(); d = s.get_drone()
        if s.get_race_live() and d is not None and np.linalg.norm(d.vel_ned) < 3.0:
            return True
        time.sleep(0.02)
    return False


def parse_waypoints(argv):
    if len(argv) >= 3:
        f = [float(x) for x in argv]
        return [tuple(f[i:i + 3]) for i in range(0, len(f) - len(f) % 3, 3)]
    return [(8.0, 0.0, 0.0), (8.0, 8.0, 0.0), (0.0, 8.0, 0.0), (0.0, 0.0, 0.0)]  # square


def control(ds, vsp, target_z, yaw_sp):
    """velocity setpoint (world NED) + target altitude + yaw setpoint -> (wcmd, thr, tilt)."""
    a = np.zeros(3)
    a[:2] = KD_AL * (vsp[:2] - ds.vel_ned[:2])
    ah = a[:2]; n = float(np.linalg.norm(ah))
    if n > TILT_MAX_ACC:
        a[:2] = ah / n * TILT_MAX_ACC
    a[2] = KP_Z * (target_z - ds.pos_ned[2]) + KD_Z * (vsp[2] - ds.vel_ned[2])
    Rc = quat_to_R(ds.quat_wxyz); yaw_cur = float(np.arctan2(Rc[1, 0], Rc[0, 0]))
    q_des = mat_to_quat(desired_attitude(a, yaw_cur))
    w_des = KP_ATT * attitude_error_quat(ds.quat_wxyz, q_des)
    w_des[2] = KP_YAW * ((yaw_sp - yaw_cur + np.pi) % (2 * np.pi) - np.pi) - KD_YAW * float(ds.omega[2])
    thr = accel_to_thrust_norm(collective_accel(a, ds.quat_wxyz), HOVER, KA)
    zb = Rc[:, 2]; tilt = np.degrees(np.arccos(max(-1, min(1, zb[2]))))
    return np.clip(w_des / RG, -WMAX, WMAX), thr, tilt


def main():
    wps = parse_waypoints(sys.argv[1:])
    assert fresh_start(), "not live"
    ds0 = s.get_drone(); spawn = ds0.pos_ned.copy()
    yaw0 = float(np.arctan2(quat_to_R(ds0.quat_wxyz)[1, 0], quat_to_R(ds0.quat_wxyz)[0, 0]))
    g = GoToWaypoint(kp_pos=0.8, max_speed=MAX_SPEED, arrival_radius=ARRIVE)
    print(f"GOTO {len(wps)} waypoints (NED rel. spawn): {wps}", flush=True)
    c.arm()
    for wi, off in enumerate(wps):
        target = spawn + np.array(off, float); t_wp = time.time(); last = -1
        reached = False
        while time.time() - t_wp < WP_TIMEOUT:
            ds = s.get_drone()
            if ds is not None:
                if g.arrived(ds.pos_ned, target):
                    reached = True; break
                sp = g.update(ds.pos_ned, ds.vel_ned, target, hold_yaw=yaw0)
                wcmd, thr, tilt = control(ds, np.array([sp.vx, sp.vy, sp.vz]), target[2], sp.yaw)
                c.send_attitude_target(wcmd, thr)
                if tilt > ABORT_TILT:
                    print(f"ABORT tilt={tilt:.0f}", flush=True); return
                k = int((time.time() - t_wp) / 2.0)
                if k != last:
                    last = k
                    d = float(np.linalg.norm(target - ds.pos_ned))
                    print(f"  wp{wi} dist={d:4.1f} spd={np.linalg.norm(ds.vel_ned[:2]):4.1f} "
                          f"tilt={tilt:3.0f}", flush=True)
            time.sleep(LOOP_DT)
        print(f"{'REACHED' if reached else 'TIMEOUT'} wp{wi} {off}", flush=True)
    print("mission done", flush=True)


if __name__ == "__main__":
    main()
