"""GOTO — high-level waypoint navigation over the ACRO interface (a set_position replacement).

Flies a list of waypoints by running race_cruise's PROVEN control toward each (along-track speed +
cross-track line-following, heading HELD = strafe not yaw, forward-accel cap + soft roll for the
weathervane-unstable camera-forward direction), then SETTLES (kills momentum) at each waypoint
before turning — arrival/stopping is the hard part on this platform, and momentum carried into a
new leg excites the instability. Arrival-detected sequencing; per-waypoint timeout; tilt abort.

Usage:
  python goto.py                       # safe default: a small box out front (body frame)
  python goto.py body 6 0 0  6 4 0     # body-relative triples: (fwd, right, down) from spawn heading
  python goto.py world 8 0 0  0 8 0    # world-NED triples: (N, E, D) offsets from spawn

STATUS (live-tested on the VQ sim): single-waypoint go-to flies cleanly (e.g. `body 6 0 0` ->
reached, tilt ~2 deg). Multi-waypoint paths with sharp 90-deg turns / precise arrivals are
MARGINAL — the weathervane instability fights stop-and-turn (overshoot + tilt spikes on transitions).
This platform wants to flow forward (like racing), not stop-and-go; precise/fast waypoint following
is better served by the learned RL policy than this hand-tuned PID. Tuning TODO: settle harder
between legs, slower cruise, or an integral term on velocity.
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

KP_ATT = np.array([0.5, 1.6, 1.0]); KP_YAW = 3.0; KD_YAW = 0.3
KP_CT = 0.5; KD_CT = 1.2; AL_MAX = 0.5; KD_AL = 1.2; KP_Z = 1.8; KD_Z = 3.0
WMAX = 4.0; LOOP_DT = 0.004; TILT_MAX_ACC = np.tan(np.radians(15)) * 9.81; ABORT_TILT = 80.0
WP_TIMEOUT = 40.0; MAX_SPEED = 1.2; ARRIVE = 1.5; DECEL_MAX = 2.0   # single-wp validated at these
SETTLE_T = 2.5; SETTLE_V = 0.4    # station-keep at each wp until speed < SETTLE_V (or SETTLE_T s)

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


def parse_args(argv):
    body = False
    if argv and argv[0] in ("body", "world"):
        body = (argv[0] == "body"); argv = argv[1:]
    if len(argv) >= 3:
        f = [float(x) for x in argv]
        return body, [tuple(f[i:i + 3]) for i in range(0, len(f) - len(f) % 3, 3)]
    return True, [(6.0, 0.0, 0.0), (6.0, 4.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 0.0)]  # safe fwd box


def cmd(ds, a2, z_sp, yaw0):
    """Send one ACRO command for a desired horizontal accel a2 (2,), altitude z_sp, heading yaw0.
    Returns achieved tilt (deg)."""
    a = np.zeros(3); a[:2] = a2
    n = float(np.linalg.norm(a[:2]))
    if n > TILT_MAX_ACC:
        a[:2] = a[:2] / n * TILT_MAX_ACC
    a[2] = KP_Z * (z_sp - ds.pos_ned[2]) + KD_Z * (0.0 - ds.vel_ned[2])
    Rc = quat_to_R(ds.quat_wxyz); yaw_cur = float(np.arctan2(Rc[1, 0], Rc[0, 0]))
    q_des = mat_to_quat(desired_attitude(a, yaw_cur))
    w_des = KP_ATT * attitude_error_quat(ds.quat_wxyz, q_des)
    w_des[2] = KP_YAW * ((yaw0 - yaw_cur + np.pi) % (2 * np.pi) - np.pi) - KD_YAW * float(ds.omega[2])
    thr = accel_to_thrust_norm(collective_accel(a, ds.quat_wxyz), HOVER, KA)
    c.send_attitude_target(np.clip(w_des / RG, -WMAX, WMAX), thr)
    zb = Rc[:, 2]; return float(np.degrees(np.arccos(max(-1, min(1, zb[2])))))


def fly_leg(target, leg_start, yaw0):
    seg = (target - leg_start)[:2]; seglen = float(np.linalg.norm(seg))
    travel = seg / seglen if seglen > 1e-3 else None
    z_sp = float(target[2]); t_wp = time.time(); last = -1
    while time.time() - t_wp < WP_TIMEOUT:
        ds = s.get_drone()
        if ds is not None:
            rel = target - ds.pos_ned
            if float(np.linalg.norm(rel)) < ARRIVE:
                return "reached"
            tv = travel if travel is not None else (rel[:2] / (np.linalg.norm(rel[:2]) + 1e-9))
            lat = np.array([-tv[1], tv[0]])
            spd = float(np.clip(0.6 * float(rel[:2] @ tv), 0.0, MAX_SPEED))   # ramp to a stop
            a_al = float(np.clip(KD_AL * (spd - float(ds.vel_ned[:2] @ tv)), -DECEL_MAX, AL_MAX))
            a_ct = -KP_CT * float((ds.pos_ned - leg_start)[:2] @ lat) - KD_CT * float(ds.vel_ned[:2] @ lat)
            tilt = cmd(ds, a_al * tv + a_ct * lat, z_sp, yaw0)
            if tilt > ABORT_TILT:
                print(f"  ABORT tilt={tilt:.0f}", flush=True); return "abort"
            k = int((time.time() - t_wp) / 2.0)
            if k != last:
                last = k
                print(f"  dist={float(np.linalg.norm(rel)):5.1f} spd={np.linalg.norm(ds.vel_ned[:2]):4.1f} "
                      f"tilt={tilt:3.0f}", flush=True)
        time.sleep(LOOP_DT)
    return "timeout"


def settle(target, yaw0):
    """Station-keep at target (damp velocity + hold position) to kill momentum before the next leg."""
    z_sp = float(target[2]); t0 = time.time()
    while time.time() - t0 < SETTLE_T:
        ds = s.get_drone()
        if ds is not None:
            if float(np.linalg.norm(ds.vel_ned[:2])) < SETTLE_V:
                return
            err = (target - ds.pos_ned)[:2]
            a2 = np.clip(KP_CT * err, -1.0, 1.0) - KD_CT * ds.vel_ned[:2]   # pull to point + damp vel
            tilt = cmd(ds, a2, z_sp, yaw0)
            if tilt > ABORT_TILT:
                return
        time.sleep(LOOP_DT)


def main():
    body, wps = parse_args(sys.argv[1:])
    assert fresh_start(), "not live"
    ds0 = s.get_drone(); spawn = ds0.pos_ned.copy()
    yaw0 = float(np.arctan2(quat_to_R(ds0.quat_wxyz)[1, 0], quat_to_R(ds0.quat_wxyz)[0, 0]))
    if body:
        cy, sy = np.cos(yaw0), np.sin(yaw0)
        fwd = np.array([-cy, -sy]); right = np.array([-sy, cy])
        offs = [np.array([o[0] * fwd[0] + o[1] * right[0], o[0] * fwd[1] + o[1] * right[1], o[2]]) for o in wps]
    else:
        offs = [np.array(o, float) for o in wps]
    targets = [spawn + o for o in offs]
    print(f"GOTO {len(wps)} waypoints ({'body fwd/right/down' if body else 'world NED'}): {wps}", flush=True)
    c.arm()
    leg_start = spawn.copy()
    for wi, target in enumerate(targets):
        print(f"-> wp{wi} {wps[wi]}", flush=True)
        res = fly_leg(target, leg_start, yaw0)
        print(f"   {res.upper()} wp{wi}", flush=True)
        if res == "abort":
            return
        settle(target, yaw0)
        ds = s.get_drone()
        leg_start = ds.pos_ned.copy() if ds is not None else target
    print("mission done", flush=True)


if __name__ == "__main__":
    main()
