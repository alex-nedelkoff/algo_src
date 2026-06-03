"""AERO CAPTURE — fly an excitation maneuver and log per-sample rows for aero sysID.

Reuses race_cruise's fresh_start() + ACRO rate-control inner loop. Each loop iteration logs:
  t, vx,vy,vz (world NED), qw,qx,qy,qz (body->world), wx,wy,wz (gyro, body),
  fx,fy,fz (HIGHRES_IMU specific force, body FRD), thrust_accel (commanded collective accel
  along body -z, m/s^2; 0 on coast), coast (bool)
to aero_data/<maneuver>_<stamp>.parquet + a one-line append to aero_data/manifest.csv.

Heading is HELD throughout (no yaw-chasing) so the drone stays nose-/tail-fixed and we don't trip
the weathervane tumble; excitation comes from translation speed, coast bursts, and rate doublets.

Usage: python aero_capture.py <maneuver> [args...]
  sweep <vmax> [dir=back|fwd] [dur]   powered straight ramp 0->vmax then hold (drag)
  coast <entry_speed> [dir=back|fwd]  build to entry_speed, cut thrust, log ~0.6s decel/twist
                                      (clean aero force: thrust=0 -> specific force == aero/m)
  doublet <axis r|p|y> <speed> [amp]  reach speed, inject +/- body-rate doublet (damping + control)
  tumble  <axis r|p|y> <speed> [dir]  build speed, pulse a body rate, then CUT MOTORS and log the
                                      free rotational+translational decay (clean damping+weathervane:
                                      motor torque = 0, so I*dw/dt = tau_aero - w x (I w))
"""
import sys, csv, json, time, os
import numpy as np
from pymavlink import mavutil
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store
from aigp.commander import Commander
from aigp.geometry import quat_to_R
from aigp.control_math import (desired_attitude, mat_to_quat, attitude_error_quat,
                               collective_accel, accel_to_thrust_norm)

# --- tuned gains (race_cruise winning config 2026-06-02) ---
KP_ATT = np.array([0.5, 1.6, 1.0]); KP_YAW = 3.0; KD_YAW = 0.3
KP_CT = 0.5; KD_CT = 1.2; KD_AL = 1.2; AL_MAX = 0.8
KP_Z = 1.8; KD_Z = 3.0
WMAX = 4.0; LOOP_DT = 0.004; TILT_MAX_ACC = np.tan(np.radians(20)) * 9.81
ABORT_TILT = 80.0

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


class Logger:
    COLS = ["t", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "wx", "wy", "wz",
            "fx", "fy", "fz", "thrust_accel", "coast",
            "u0", "u1", "u2", "u3", "wcx", "wcy", "wcz", "thr_cmd"]  # motors + control effort

    def __init__(self):
        self.rows = []

    def log(self, t, ds, imu, thrust_accel, coast, wcmd=None, thr_cmd=0.0):
        acc, gyro, _ = imu
        q = ds.quat_wxyz; v = ds.vel_ned
        a = s.get_actuators(); u = a[0] if a is not None else np.zeros(4)   # ACTUATOR_OUTPUT_STATUS
        w = np.zeros(3) if wcmd is None else np.asarray(wcmd, float)
        self.rows.append([t, v[0], v[1], v[2], q[0], q[1], q[2], q[3],
                          gyro[0], gyro[1], gyro[2], acc[0], acc[1], acc[2],
                          float(max(0.0, thrust_accel)), bool(coast),    # actual collective >= 0
                          float(u[0]), float(u[1]), float(u[2]), float(u[3]),
                          float(w[0]), float(w[1]), float(w[2]), float(thr_cmd)])

    def save(self, maneuver):
        import pandas as pd
        os.makedirs("aero_data", exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = f"aero_data/{maneuver}_{stamp}.parquet"
        df = pd.DataFrame(self.rows, columns=self.COLS)
        df.to_parquet(path)
        man = "aero_data/manifest.csv"; new = not os.path.exists(man)
        with open(man, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["file", "maneuver", "rows", "dur_s", "vmax_mps", "n_coast"])
            spd = np.linalg.norm(df[["vx", "vy", "vz"]].to_numpy(), axis=1)
            w.writerow([os.path.basename(path), maneuver, len(df),
                        round(df["t"].iloc[-1] - df["t"].iloc[0], 2) if len(df) else 0,
                        round(float(spd.max()), 2) if len(df) else 0, int(df["coast"].sum())])
        return path, df


def control(ds, vel_sp_world, z_sp, yaw0, extra_w=None, tilt_acc=TILT_MAX_ACC,
            al_dir=None, al_max=None):
    """Compute (wcmd, thr, thrust_accel, tilt) to track a world-velocity setpoint at fixed heading.
    extra_w: optional additive body-rate command (rad/s) for doublet injection.
    tilt_acc: cap on commanded horizontal accel (governor passes a tighter value).
    al_dir/al_max: cap the ACCELERATION along al_dir to al_max (allow strong decel) — the race_cruise
    forward-accel cap that keeps tail-first flight under the weathervane runaway threshold."""
    a = np.zeros(3)
    a[:2] = KD_AL * (vel_sp_world[:2] - ds.vel_ned[:2])
    a[2] = KP_Z * (z_sp - ds.pos_ned[2]) + KD_Z * (vel_sp_world[2] - ds.vel_ned[2])
    if al_max is not None and al_dir is not None:
        hat = np.asarray(al_dir, float)[:2]; hat = hat / (np.linalg.norm(hat) + 1e-9)
        a_al = float(a[:2] @ hat)
        a[:2] = a[:2] + (np.clip(a_al, -4.0, al_max) - a_al) * hat   # cap accel, allow decel
    ah = a[:2]; n = float(np.linalg.norm(ah))
    if n > tilt_acc:
        a[:2] = ah / n * tilt_acc
    Rc = quat_to_R(ds.quat_wxyz); yaw_cur = float(np.arctan2(Rc[1, 0], Rc[0, 0]))
    q_des = mat_to_quat(desired_attitude(a, yaw_cur))
    w_des = KP_ATT * attitude_error_quat(ds.quat_wxyz, q_des)
    w_des[2] = KP_YAW * ((yaw0 - yaw_cur + np.pi) % (2 * np.pi) - np.pi) - KD_YAW * float(ds.omega[2])
    if extra_w is not None:
        w_des = w_des + extra_w
    ta = collective_accel(a, ds.quat_wxyz)
    thr = accel_to_thrust_norm(ta, HOVER, KA)
    zb = Rc[:, 2]; tilt = np.degrees(np.arccos(max(-1, min(1, zb[2]))))
    return np.clip(w_des / RG, -WMAX, WMAX), thr, ta, tilt


def setup():
    assert fresh_start(), "not live"
    ds0 = s.get_drone()
    yaw0 = float(np.arctan2(quat_to_R(ds0.quat_wxyz)[1, 0], quat_to_R(ds0.quat_wxyz)[0, 0]))
    z_sp = ds0.pos_ned[2]
    c.arm()
    return ds0, yaw0, z_sp


def heading_vec(yaw0, direction):
    """World-frame unit velocity direction for a held heading yaw0.
    fwd = camera-forward (-body-x, toward course); back = nose-first (+body-x, toward spawn obstacle);
    right/left = pure lateral (±body-y, strafe); diag/diagl = 45 deg fwd+lateral (excites D_x AND D_y
    while net-moving toward the open course)."""
    h = np.array([np.cos(yaw0), np.sin(yaw0), 0.0])       # +body-x (nose) in world
    lat = np.array([-np.sin(yaw0), np.cos(yaw0), 0.0])    # +body-y (right) in world
    fwd = -h
    table = {"fwd": fwd, "back": h, "right": lat, "left": -lat,
             "diag": (fwd + lat) / np.sqrt(2.0), "diagl": (fwd - lat) / np.sqrt(2.0)}
    return table[direction]


def run_sweep(vmax, direction="back", dur=14.0):
    ds0, yaw0, z_sp = setup()
    hat = heading_vec(yaw0, direction); lg = Logger()
    t0 = time.time(); ramp = dur * 0.6; last = -1
    while time.time() - t0 < dur:
        tau = time.time() - t0
        sp = vmax * min(1.0, tau / ramp)
        ds = s.get_drone(); imu = s.get_imu()
        if ds is not None and imu is not None:
            wcmd, thr, ta, tilt = control(ds, sp * hat, z_sp, yaw0)
            c.send_attitude_target(wcmd, thr)
            lg.log(tau, ds, imu, ta, coast=False)
            if tilt > ABORT_TILT:
                print(f"ABORT tilt={tilt:.0f}", flush=True); break
            k = int(tau / 2.0)
            if k != last:
                last = k
                print(f"t={tau:4.1f} spd={np.linalg.norm(ds.vel_ned[:2]):4.1f} sp={sp:4.1f} "
                      f"tilt={tilt:3.0f}", flush=True)
        time.sleep(LOOP_DT)
    path, df = lg.save("sweep")
    print(f"\nSAVED {path}  rows={len(df)} vmax={np.linalg.norm(df[['vx','vy','vz']].to_numpy(),axis=1).max():.1f}", flush=True)


def run_coast(entry_speed, direction="back", coast_s=0.6):
    ds0, yaw0, z_sp = setup()
    hat = heading_vec(yaw0, direction); lg = Logger()
    t0 = time.time(); last = -1
    # phase 1: build speed (powered, logged coast=False)
    while time.time() - t0 < 12.0:
        tau = time.time() - t0
        ds = s.get_drone(); imu = s.get_imu()
        if ds is not None and imu is not None:
            wcmd, thr, ta, tilt = control(ds, entry_speed * hat, z_sp, yaw0)
            c.send_attitude_target(wcmd, thr)
            lg.log(tau, ds, imu, ta, coast=False)
            v = float(np.linalg.norm(ds.vel_ned[:2]))
            if tilt > ABORT_TILT:
                print(f"ABORT tilt={tilt:.0f}", flush=True); path, df = lg.save("coast"); return
            if v >= 0.92 * entry_speed and tau > 4.0:
                break
        time.sleep(LOOP_DT)
    # phase 2: cut thrust, log the decel/twist (coast=True, thrust_accel=0)
    print(f"COAST entry v={np.linalg.norm(s.get_drone().vel_ned[:2]):.1f}", flush=True)
    tc = time.time()
    while time.time() - tc < coast_s:
        tau = time.time() - t0
        ds = s.get_drone(); imu = s.get_imu()
        if ds is not None and imu is not None:
            idle()                                  # zero thrust, attitude-ignore
            lg.log(tau, ds, imu, 0.0, coast=True)
        time.sleep(LOOP_DT)
    path, df = lg.save("coast")
    print(f"\nSAVED {path}  rows={len(df)} n_coast={int(df['coast'].sum())}", flush=True)


def run_doublet(axis, speed, amp=2.0, direction="back"):
    ds0, yaw0, z_sp = setup()
    hat = heading_vec(yaw0, direction); lg = Logger()
    ax = {"r": 0, "p": 1, "y": 2}[axis]
    t0 = time.time(); reached = None; phase = 0
    while time.time() - t0 < 16.0:
        tau = time.time() - t0
        ds = s.get_drone(); imu = s.get_imu()
        if ds is not None and imu is not None:
            v = float(np.linalg.norm(ds.vel_ned[:2]))
            extra = None
            if reached is None and v >= 0.9 * speed and tau > 4.0:
                reached = tau
            if reached is not None:
                dt = tau - reached                  # +amp for 0.25s, -amp for 0.25s, then settle
                if dt < 0.25:
                    extra = np.zeros(3); extra[ax] = amp
                elif dt < 0.5:
                    extra = np.zeros(3); extra[ax] = -amp
                elif dt > 1.5:
                    break
            wcmd, thr, ta, tilt = control(ds, speed * hat, z_sp, yaw0, extra_w=extra)
            c.send_attitude_target(wcmd, thr)
            lg.log(tau, ds, imu, ta, coast=False)
            if tilt > ABORT_TILT:
                print(f"ABORT tilt={tilt:.0f}", flush=True); break
        time.sleep(LOOP_DT)
    path, df = lg.save(f"doublet_{axis}")
    print(f"\nSAVED {path}  rows={len(df)}", flush=True)


def run_circle(speed, radius, dur=22.0):
    """STEADY constant-speed circle, FIXED heading (body velocity sweeps all sideslip angles ->
    excites D_x AND D_y). Centered AHEAD of spawn (toward the open course) so it stays clear of the
    obstacle behind spawn. Position feedback keeps it on the circle. Steady-state lateral velocity =>
    phase/latency-robust (unlike the transient weave). Logs raw rows; latency + lever-arm handled offline."""
    ds0, yaw0, z_sp = setup()
    fwd2 = heading_vec(yaw0, "fwd")[:2]                       # toward course (horizontal)
    alpha0 = float(np.arctan2(fwd2[1], fwd2[0]))            # start velocity pointing at the course
    wc = speed / radius                                       # velocity-direction rate (rad/s); curves into open space
    spawn = ds0.pos_ned[:2].copy()
    lg = Logger(); t0 = time.time(); last = -1
    while time.time() - t0 < dur:
        tau = time.time() - t0
        alpha = alpha0 + wc * tau                             # PURE rotating constant-speed velocity (no position target)
        ds = s.get_drone(); imu = s.get_imu()
        if ds is not None and imu is not None:
            vsp = np.zeros(3)
            vsp[:2] = speed * np.array([np.cos(alpha), np.sin(alpha)])
            wcmd, thr, ta, tilt = control(ds, vsp, z_sp, yaw0)
            c.send_attitude_target(wcmd, thr)
            lg.log(tau, ds, imu, ta, coast=False)
            if tilt > ABORT_TILT:
                print(f"ABORT tilt={tilt:.0f}", flush=True); break
            k = int(tau / 3.0)
            if k != last:
                last = k
                R = quat_to_R(ds.quat_wxyz); vb = R.T @ ds.vel_ned
                drift = float(np.linalg.norm(ds.pos_ned[:2] - spawn))
                print(f"t={tau:4.1f} spd={np.linalg.norm(ds.vel_ned[:2]):4.1f} "
                      f"v_body=({vb[0]:+4.1f},{vb[1]:+4.1f}) drift={drift:4.1f} tilt={tilt:3.0f}", flush=True)
        time.sleep(LOOP_DT)
    path, df = lg.save("circle")
    print(f"\nSAVED {path}  rows={len(df)}", flush=True)


def run_weave(v_fwd, v_lat, period=4.0, dur=16.0):
    """Powered lemniscate-style weave, FIXED heading: net-forward toward the course (tail-first) with a
    lateral velocity sinusoid -> sweeps body-y velocity (sideslip) while netting toward open space.
    Horizontal aero force a_aero_x/y = f_body_x/y is thrust-INDEPENDENT, so this gives clean D_x AND D_y."""
    ds0, yaw0, z_sp = setup()
    fwd = heading_vec(yaw0, "fwd"); lat = heading_vec(yaw0, "right")
    lg = Logger(); t0 = time.time(); last = -1
    while time.time() - t0 < dur:
        tau = time.time() - t0
        vsp = v_fwd * fwd + v_lat * np.sin(2 * np.pi * tau / period) * lat
        ds = s.get_drone(); imu = s.get_imu()
        if ds is not None and imu is not None:
            wcmd, thr, ta, tilt = control(ds, vsp, z_sp, yaw0)
            c.send_attitude_target(wcmd, thr)
            lg.log(tau, ds, imu, ta, coast=False)
            if tilt > ABORT_TILT:
                print(f"ABORT tilt={tilt:.0f}", flush=True); break
            k = int(tau / 2.0)
            if k != last:
                last = k
                R = quat_to_R(ds.quat_wxyz); vby = float((R.T @ ds.vel_ned)[1])
                print(f"t={tau:4.1f} spd={np.linalg.norm(ds.vel_ned[:2]):4.1f} v_body_y={vby:+4.1f} "
                      f"tilt={tilt:3.0f}", flush=True)
        time.sleep(LOOP_DT)
    path, df = lg.save("weave")
    print(f"\nSAVED {path}  rows={len(df)}", flush=True)


def run_tumble(axis, speed, direction="back", spin_amp=4.0, free_s=1.2):
    ds0, yaw0, z_sp = setup()
    hat = heading_vec(yaw0, direction); lg = Logger()
    ax = {"r": 0, "p": 1, "y": 2}[axis]
    t0 = time.time()
    while time.time() - t0 < 10.0:                       # phase 1: build speed (powered)
        tau = time.time() - t0; ds = s.get_drone(); imu = s.get_imu()
        if ds is not None and imu is not None:
            wcmd, thr, ta, tilt = control(ds, speed * hat, z_sp, yaw0)
            c.send_attitude_target(wcmd, thr); lg.log(tau, ds, imu, ta, coast=False)
            if np.linalg.norm(ds.vel_ned[:2]) >= 0.9 * speed and tau > 4.0:
                break
        time.sleep(LOOP_DT)
    ts = time.time()                                     # phase 2: spin-up pulse (powered, brief)
    while time.time() - ts < 0.3:
        tau = time.time() - t0; ds = s.get_drone(); imu = s.get_imu()
        if ds is not None and imu is not None:
            extra = np.zeros(3); extra[ax] = spin_amp
            wcmd, thr, ta, tilt = control(ds, speed * hat, z_sp, yaw0, extra_w=extra)
            c.send_attitude_target(wcmd, thr); lg.log(tau, ds, imu, ta, coast=False)
        time.sleep(LOOP_DT)
    tc = time.time()                                     # phase 3: MOTORS OFF, log free decay (coast)
    while time.time() - tc < free_s:
        tau = time.time() - t0; ds = s.get_drone(); imu = s.get_imu()
        if ds is not None and imu is not None:
            idle(); lg.log(tau, ds, imu, 0.0, coast=True)
        time.sleep(LOOP_DT)
    path, df = lg.save(f"tumble_{axis}")
    pw = np.abs(df[["wx", "wy", "wz"]].to_numpy()).max() if len(df) else 0.0
    print(f"\nSAVED {path}  rows={len(df)} n_coast={int(df['coast'].sum())} peak_|w|={pw:.1f} rad/s", flush=True)


class ReferenceGovernor:
    """Envelope protection for closed-loop ID: cap the velocity-setpoint magnitude and slew rate so
    excitation can't trigger the speed overshoot / tilt-spike / divergence seen with open-loop circles.
    (The tilt cap is applied separately via control(tilt_acc=...).)"""
    def __init__(self, max_speed=4.0, max_slew=2.0):
        self.max_speed = max_speed; self.max_slew = max_slew; self.prev = np.zeros(3)

    def govern(self, vsp_raw, dt):
        v = np.asarray(vsp_raw, float).copy()
        sp = float(np.linalg.norm(v[:2]))
        if sp > self.max_speed:
            v[:2] = v[:2] / sp * self.max_speed
        dv = v - self.prev; n = float(np.linalg.norm(dv))
        if n > self.max_slew * dt:
            dv = dv / n * self.max_slew * dt
        v = self.prev + dv; self.prev = v
        return v


def run_doublet211(axis, trim_speed, amp, dwell=0.5):
    """Closed-loop ID excitation: hold a modest fwd (toward-course) trim under the reference governor,
    then inject a 2-1-1 multistep (signs +,-,+ with durations 2d,d,d) on the lateral-velocity ('lat')
    or yaw-rate ('yaw') channel. Stays near trim -> safe on the weathervane-unstable platform. Logs the
    measured motor outputs + commanded effort so tau_motor can be reconstructed offline."""
    ds0, yaw0, z_sp = setup()
    fwd = heading_vec(yaw0, "fwd"); lat = heading_vec(yaw0, "right")
    gov = ReferenceGovernor(max_speed=max(trim_speed + 1.5, 3.0))
    tilt_acc = np.tan(np.radians(18.0)) * 9.81
    settle = 4.0
    segs = [(+1, 2 * dwell), (-1, 1 * dwell), (+1, 1 * dwell)]
    total = settle + sum(d for _, d in segs) + 2.5

    def seg_sign(te):
        acc = settle
        for sgn, dur in segs:
            if te < acc + dur:
                return sgn
            acc += dur
        return 0

    lg = Logger(); t0 = time.time(); last = -1
    while time.time() - t0 < total:
        tau = time.time() - t0; sgn = seg_sign(tau)
        ds = s.get_drone(); imu = s.get_imu()
        if ds is not None and imu is not None:
            vsp = gov.govern(trim_speed * fwd, LOOP_DT)   # gently-ramped trim (no overshoot)
            extra = None
            if axis == "lat":
                vsp = vsp + sgn * amp * lat                # sharp lateral doublet ON TOP (not slewed)
            elif axis == "yaw":
                extra = np.zeros(3); extra[2] = sgn * amp
            wcmd, thr, ta, tilt = control(ds, vsp, z_sp, yaw0, extra_w=extra, tilt_acc=tilt_acc,
                                          al_dir=fwd, al_max=0.6)   # cap forward accel (anti-runaway)
            c.send_attitude_target(wcmd, thr)
            lg.log(tau, ds, imu, ta, coast=False, wcmd=wcmd, thr_cmd=thr)
            if tilt > ABORT_TILT:
                print(f"ABORT tilt={tilt:.0f}", flush=True); break
            k = int(tau / 2.0)
            if k != last:
                last = k
                R = quat_to_R(ds.quat_wxyz); vb = R.T @ ds.vel_ned
                a = s.get_actuators(); u = a[0] if a else np.zeros(4)
                print(f"t={tau:4.1f} seg={sgn:+d} v_body=({vb[0]:+4.1f},{vb[1]:+4.1f}) "
                      f"spd={np.linalg.norm(ds.vel_ned[:2]):4.1f} tilt={tilt:3.0f} "
                      f"u=[{u[0]:.2f},{u[1]:.2f},{u[2]:.2f},{u[3]:.2f}]", flush=True)
        time.sleep(LOOP_DT)
    path, df = lg.save(f"d211_{axis}")
    print(f"\nSAVED {path}  rows={len(df)} cols={len(df.columns)}", flush=True)


def run_pulse(axis, amp=1.5, reps=8, trim=1.5):
    """KAPPA-ISOLATION (sysID): cruise slowly toward the OPEN course (governed + forward-accel-capped =
    the proven doublet211 safe envelope, heading AWAY from the spawn obstacle) and inject a train of
    SHARP, brief roll('roll')/pitch('pitch') body-rate pulses. Each pulse's leading edge has large
    omega_dot while omega~0 and v is small (aero negligible), so I*omega_dot ~= tau_motor isolates the
    inertia scale kappa from damping — breaking the omega_dot<->omega collinearity that makes kappa
    unidentifiable in oscillatory doublets. Motor torque is known via the hover-calibrated k_f, so a
    clean kappa propagates the gravity anchor to yaw (pins k_q and wv_z). Moving into open space avoids
    gate contact (no collision flag exists in telemetry). Logs motors + effort."""
    ds0, yaw0, z_sp = setup()
    ax = {"roll": 0, "pitch": 1, "yaw": 2}[axis]
    fwd = heading_vec(yaw0, "fwd")
    gov = ReferenceGovernor(max_speed=max(trim + 1.0, 2.5))
    tilt_acc = np.tan(np.radians(18.0)) * 9.81
    pulse_dur, rest_dur, settle = 0.18, 0.6, 3.0
    period = pulse_dur + rest_dur
    lg = Logger(); t0 = time.time(); last = -1
    total = settle + reps * period + 0.6
    while time.time() - t0 < total:
        tau = time.time() - t0
        ds = s.get_drone(); imu = s.get_imu()
        if ds is not None and imu is not None:
            vsp = gov.govern(trim * fwd, LOOP_DT)              # gentle fwd cruise into open space
            extra = None; te = tau - settle
            if te >= 0:
                k = int(te // period); ph = te - k * period
                if k < reps and ph < pulse_dur:
                    extra = np.zeros(3); extra[ax] = (1.0 if k % 2 == 0 else -1.0) * amp
            wcmd, thr, ta, tilt = control(ds, vsp, z_sp, yaw0, extra_w=extra, tilt_acc=tilt_acc,
                                          al_dir=fwd, al_max=0.6)
            c.send_attitude_target(wcmd, thr)
            lg.log(tau, ds, imu, ta, coast=False, wcmd=wcmd, thr_cmd=thr)
            if tilt > ABORT_TILT:
                print(f"ABORT tilt={tilt:.0f}", flush=True); break
            k2 = int(tau / 1.0)
            if k2 != last:
                last = k2
                R = quat_to_R(ds.quat_wxyz); vb = R.T @ ds.vel_ned
                print(f"t={tau:4.1f} spd={np.linalg.norm(ds.vel_ned[:2]):4.1f} "
                      f"w=({ds.omega[0]:+4.1f},{ds.omega[1]:+4.1f},{ds.omega[2]:+4.1f}) tilt={tilt:3.0f}", flush=True)
        time.sleep(LOOP_DT)
    path, df = lg.save(f"pulse_{axis}")
    print(f"\nSAVED {path}  rows={len(df)} cols={len(df.columns)}", flush=True)


if __name__ == "__main__":
    man = sys.argv[1]
    if man == "sweep":
        run_sweep(float(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "back",
                  float(sys.argv[4]) if len(sys.argv) > 4 else 14.0)
    elif man == "coast":
        run_coast(float(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "back")
    elif man == "doublet":
        run_doublet(sys.argv[2], float(sys.argv[3]),
                    float(sys.argv[4]) if len(sys.argv) > 4 else 2.0)
    elif man == "tumble":
        run_tumble(sys.argv[2], float(sys.argv[3]),
                   sys.argv[4] if len(sys.argv) > 4 else "back",
                   spin_amp=float(sys.argv[5]) if len(sys.argv) > 5 else 4.0)
    elif man == "weave":
        run_weave(float(sys.argv[2]), float(sys.argv[3]),
                  float(sys.argv[4]) if len(sys.argv) > 4 else 4.0)
    elif man == "circle":
        run_circle(float(sys.argv[2]), float(sys.argv[3]))
    elif man == "doublet211":
        run_doublet211(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]),
                       float(sys.argv[5]) if len(sys.argv) > 5 else 0.5)
    elif man == "pulse":
        run_pulse(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 1.5,
                  int(sys.argv[4]) if len(sys.argv) > 4 else 8,
                  float(sys.argv[5]) if len(sys.argv) > 5 else 1.5)
    else:
        print(f"unknown maneuver {man}"); sys.exit(1)
