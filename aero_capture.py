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
            "fx", "fy", "fz", "thrust_accel", "coast"]

    def __init__(self):
        self.rows = []

    def log(self, t, ds, imu, thrust_accel, coast):
        acc, gyro, _ = imu
        q = ds.quat_wxyz; v = ds.vel_ned
        self.rows.append([t, v[0], v[1], v[2], q[0], q[1], q[2], q[3],
                          gyro[0], gyro[1], gyro[2], acc[0], acc[1], acc[2],
                          float(thrust_accel), bool(coast)])

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


def control(ds, vel_sp_world, z_sp, yaw0, extra_w=None):
    """Compute (wcmd, thr, thrust_accel, tilt) to track a world-velocity setpoint at fixed heading.
    extra_w: optional additive body-rate command (rad/s) for doublet injection."""
    a = np.zeros(3)
    a[:2] = KD_AL * (vel_sp_world[:2] - ds.vel_ned[:2])
    a[2] = KP_Z * (z_sp - ds.pos_ned[2]) + KD_Z * (vel_sp_world[2] - ds.vel_ned[2])
    ah = a[:2]; n = float(np.linalg.norm(ah))
    if n > TILT_MAX_ACC:
        a[:2] = ah / n * TILT_MAX_ACC
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
    sign = -1.0 if direction == "fwd" else 1.0   # fwd = camera-forward (-body-x, tail-first)
    return sign * np.array([np.cos(yaw0), np.sin(yaw0), 0.0])


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
    else:
        print(f"unknown maneuver {man}"); sys.exit(1)
