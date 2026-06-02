"""RACE CRUISE — controllable straight flight via the ACRO rate interface, including
the camera-forward (racing) direction that previously diverged on roll.

Resolves the 2026-06-02 open blocker. Three fixes vs the old rate_fly.py:
  1. CROSS-TRACK line-following (pos+vel feedback to the track line). The old loop only
     damped lateral velocity, never pulled back to the line -> drift integrated (140 m).
  2. FORWARD-ACCEL CAP (AL_MAX). The velocity loop overshoots (no integral + telemetry
     lag, settles well above setpoint); capping forward accel keeps speed under the
     weathervane-instability threshold.
  3. SOFTENED per-axis attitude gains, esp. ROLL. Measured rate gain is ~-2.5/axis
     (not the sysID -1.9), so the old gains over-rotated ~1.3x and the slow high-inertia
     roll axis (Ixx ~3.7x Iyy) limit-cycled. KP_R=0.5 breaks it.

Diagnosis notes (data-backed, see vault 'AI-GP flight-control handoff'):
  - The cam-forward instability is AERODYNAMIC (weathervane / tail-first), not a control
    bug: the attitude controller is direction-symmetric yet only tail-first diverges, and
    the divergence is speed-dependent. Keeping speed modest + line-following + soft roll
    makes it controllable.
  - NO roll<->yaw rate-command swap: wcmd->omega telemetry maps each axis to itself.

Result (DIR=fwd, racing direction): 75 m straight, lateral <1 m, tilt <=6 deg, ~2.8 m/s.

Usage: python race_cruise.py [DIR=fwd|back] [DURATION] [SPEED]
  fwd  = camera-forward / racing (-body-x, tail-first) -- the hard, now-solved case.
  back = camera-backward (+body-x, nose-first) -- the already-easy case.
"""
import sys, json, time
import numpy as np
from pymavlink import mavutil
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store
from aigp.commander import Commander
from aigp.geometry import quat_to_R
from aigp.control_math import (desired_accel, desired_attitude, mat_to_quat,
                               attitude_error_quat, collective_accel, accel_to_thrust_norm)

DIR      = sys.argv[1] if len(sys.argv) > 1 else "fwd"
DURATION = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
SPEED    = float(sys.argv[3]) if len(sys.argv) > 3 else 1.8

# --- Tuned gains (winning config 2026-06-02) ---
KP_ATT = np.array([0.5, 1.6, 1.0])   # per-axis attitude P: [roll(soft, hi-inertia), pitch, yaw-unused]
KP_YAW = 3.0                          # heading hold
KD_YAW = 0.3                          # light yaw-rate damping (omega_z); >~0.7 injects noise -> unstable
KP_CT  = 0.5; KD_CT = 1.2            # cross-track position / velocity (line-following)
AL_MAX = 0.5                          # forward-accel cap (m/s^2) -- holds speed under weathervane threshold
KD_AL  = 1.2                          # along-track velocity gain
KP_Z   = 1.8; KD_Z = 3.0            # altitude hold
WMAX   = 4.0; LOOP_DT = 0.004; TILT_MAX_ACC = np.tan(np.radians(15)) * 9.81

r = json.load(open("sysid/sim_response.json"))
HOVER = r["hover_thrust"]; KA = r["k_a"]
RG = np.array([r["rate_gain_axes"]["roll"], r["rate_gain_axes"]["pitch"], r["rate_gain_axes"]["yaw"]])
IDLE = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE

s = Store(); m = MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); VisionIO(s).start()
boot = int(time.time() * 1000); c = Commander(m.conn, boot)


def idle():
    m.conn.mav.set_attitude_target_send(int(time.time() * 1000) - boot, m.conn.target_system,
                                        m.conn.target_component, IDLE, [1.0, 0, 0, 0], 0, 0, 0, 0)


def fresh_start():
    """Clear latched throttle (idle thrust=0) -> sim_reset -> wait for the countdown to
    recycle (race_live False->True) with the drone at rest. REQUIRED before each run."""
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


def roll_deg(R):
    return float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))


assert fresh_start(), "not live"
ds0 = s.get_drone(); spawn = ds0.pos_ned.copy(); z_sp = spawn[2]
yaw0 = float(np.arctan2(quat_to_R(ds0.quat_wxyz)[1, 0], quat_to_R(ds0.quat_wxyz)[0, 0]))
sign = -1.0 if DIR == "fwd" else 1.0                       # fwd = camera-forward = -body-x
travel = sign * np.array([np.cos(yaw0), np.sin(yaw0)])
lat_hat = np.array([-travel[1], travel[0]])
print(f"race_cruise DIR={DIR} SPEED={SPEED} DUR={DURATION}", flush=True)
c.arm()
t0 = time.time(); last = -1; max_tilt = 0.0; peak_lat = 0.0; bad = False
while time.time() - t0 < DURATION:
    ds = s.get_drone()
    if ds is not None:
        d = ds.pos_ned - spawn
        v_al = float(ds.vel_ned[:2] @ travel); v_ct = float(ds.vel_ned[:2] @ lat_hat)
        p_ct = float(d[:2] @ lat_hat)
        a_al = float(np.clip(KD_AL * (SPEED - v_al), -4.0, AL_MAX))
        a_ct = -KP_CT * p_ct - KD_CT * v_ct
        a = np.zeros(3); a[:2] = a_al * travel + a_ct * lat_hat
        a[2] = KP_Z * (z_sp - ds.pos_ned[2]) + KD_Z * (0.0 - ds.vel_ned[2])
        ah = a[:2]; n = float(np.linalg.norm(ah))
        if n > TILT_MAX_ACC:
            a[:2] = ah / n * TILT_MAX_ACC
        Rc = quat_to_R(ds.quat_wxyz); yaw_cur = float(np.arctan2(Rc[1, 0], Rc[0, 0]))
        q_des = mat_to_quat(desired_attitude(a, yaw_cur))
        w_des = KP_ATT * attitude_error_quat(ds.quat_wxyz, q_des)
        w_des[2] = KP_YAW * ((yaw0 - yaw_cur + np.pi) % (2 * np.pi) - np.pi) - KD_YAW * float(ds.omega[2])
        thr = accel_to_thrust_norm(collective_accel(a, ds.quat_wxyz), HOVER, KA)
        c.send_attitude_target(np.clip(w_des / RG, -WMAX, WMAX), thr)
        zb = Rc[:, 2]; tilt = np.degrees(np.arccos(max(-1, min(1, zb[2]))))
        max_tilt = max(max_tilt, tilt); peak_lat = max(peak_lat, abs(p_ct))
        if tilt > 75:
            bad = True
        k = int((time.time() - t0) / 3.0)
        if k != last:
            last = k
            print(f"t={time.time()-t0:4.1f} fwd={float(d[:2]@travel):6.1f} lat={p_ct:+5.1f} "
                  f"roll={roll_deg(Rc):+4.0f} yaw={np.degrees(yaw_cur):+4.0f} "
                  f"spd={np.linalg.norm(ds.vel_ned[:2]):4.1f} tilt={tilt:3.0f}", flush=True)
    time.sleep(LOOP_DT)
d = s.get_drone().pos_ned - spawn
print(f"\nRESULT[{DIR}]: fwd={float(d[:2]@travel):.0f}m lat={float(d[:2]@lat_hat):+.0f}m "
      f"peak_lat={peak_lat:.0f}m alt_err={d[2]:+.0f}m max_tilt={max_tilt:.0f} "
      f"{'TUMBLED' if bad else 'OK'} over {DURATION:.0f}s", flush=True)
