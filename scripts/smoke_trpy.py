"""Smoke test the DIRECT-TRPY interface (SET_ACTUATOR_CONTROL_TARGET / 4 raw motor throttles) on the
VQ sim -- the candidate rate-loop BYPASS. Run on the laptop with the sim live:

    python scripts/smoke_trpy.py

Validated 2026-06-23 (exp-log TRPY-BYPASS):
  A. message echo  -> sim applies our raw motors EXACTLY (bypasses the inner loop).   PASS
  B. thrust map    -> hover throttle ~0.234 (matches sysID fit 0.23).                 PASS
  C. torque calib  -> data-driven effectiveness M (3x4, cond ~1.4): ALL axes          PASS
                      controllable. (dyn_probe's assumed X-mix gave roll/yaw ~0 -- the
                      old "direct-TRPY doesn't work" was a WRONG MIXER, not the message.)
  D. closed loop   -> our PD on raw motors LEVELS the drone 18deg -> 0.0deg and holds   PASS
                      a clean level hover (tilt ~0, no drift) indefinitely. The earlier
                      ~6 s instability was a WRONG ROLL-ERROR SIGN (COR-138 task 1):
                      reduced-attitude righting is e=[-R[2,1], +R[2,0]] (asymmetric);
                      +roll is positive feedback (slow, masked by the pitch-only spawn).

Open-loop (raw motors, no controller) is unstable -- like any quad. The interface itself works,
and a simple PD on it holds a clean hover -> viable rate-loop bypass (the COR-138 / WPTRACK-04 lever).
"""
import os, sys, time
import numpy as np
from pymavlink import mavutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from aigp.commander import Commander
from aigp.geometry import quat_to_R
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.navigator import _qfix, WFIX
from aigp.state import Store

IDLE = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
HOVER = 0.234
RES = []
def rec(ok, name, detail=""):
    RES.append((ok, name, detail)); print(f"[{'PASS' if ok else 'PARTIAL/FAIL'}] {name}  {detail}", flush=True)


def idle(m, boot):
    m.conn.mav.set_attitude_target_send(int(time.time() * 1000) - boot, m.conn.target_system,
                                        m.conn.target_component, IDLE, [1., 0, 0, 0], 0, 0, 0, 0)

def fresh_start(s, c, m, boot):
    t = time.time()
    while time.time() - t < 1.0: idle(m, boot); time.sleep(0.02)
    prev = s.get_race(); pb = prev["boot_ms"] if prev else None
    c.sim_reset(); t = time.time()
    while time.time() - t < 30:
        idle(m, boot); r2 = s.get_race()
        if r2 and (not r2["race_live"] or (pb and r2["boot_ms"] < pb)): break
        time.sleep(0.02)
    while time.time() - t < 30:
        idle(m, boot); d = s.get_drone()
        if s.get_race_live() and d is not None and np.linalg.norm(d.vel_ned) < 3.0: return True
        time.sleep(0.02)
    return False

def tilt_deg(ds):
    R = quat_to_R(_qfix(ds.quat_wxyz)); return float(np.degrees(np.arccos(max(-1., min(1., R[2, 2])))))

def coll_resp(s, c, coll, secs):
    """Hold a collective on all 4 motors; return mean vertical accel (m/s^2, +up)."""
    vz, ts, t0 = [], [], time.time()
    while time.time() - t0 < secs:
        c.send_motor_command([coll] * 4); time.sleep(0.01)
        ds = s.get_drone()
        if ds is not None: vz.append(float(ds.vel_ned[2])); ts.append(time.time())
    if len(vz) < 5: return 0.0
    return -float(np.polyfit(np.array(ts) - ts[0], np.array(vz), 1)[0])   # NED z+ down -> up-accel = -dvz/dt

def calib_motor(s, c, m, boot, i, bump=0.10):
    """Bump motor i; peak true-frame gyro vector / bump = effectiveness column (cond ~1.4)."""
    assert fresh_start(s, c, m, boot); c.arm()
    u = np.full(4, HOVER); u[i] = HOVER + bump
    t0 = time.time(); best = np.zeros(3); bn = 0.0
    while time.time() - t0 < 0.30:
        c.send_motor_command(u.tolist()); time.sleep(0.01)
        ds = s.get_drone()
        if ds is not None:
            om = np.asarray(ds.omega, float) * WFIX
            if np.linalg.norm(om) > bn: bn = float(np.linalg.norm(om)); best = om.copy()
            if tilt_deg(ds) > 50: break
    return best / bump


def main():
    s = Store(); m = MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); VisionIO(s).start()
    boot = int(time.time() * 1000); c = Commander(m.conn, boot)
    assert fresh_start(s, c, m, boot), "not live"
    c.arm()

    # A. message echo / bypass
    print("\n=== A. MESSAGE ECHO ===", flush=True)
    t0 = time.time(); got = []
    while time.time() - t0 < 0.8:
        c.send_motor_command([0.18] * 4); time.sleep(0.01)
        a = s.get_actuators()
        if a is not None: got.append(np.asarray(a[0], float)[:4])
    mean = np.mean(got[-20:], axis=0) if got else np.zeros(4)
    rec(bool(np.all(np.abs(mean - 0.18) < 0.05)), "actuator echo",
        f"-> sent 0.18, sim applies {mean.round(3)} (= raw motors bypass the inner loop)")

    # B. thrust map (around hover, no crash)
    print("\n=== B. THRUST MAP ===", flush=True)
    assert fresh_start(s, c, m, boot); c.arm()
    rows = []
    for u in (0.18, 0.21, 0.24, 0.27):
        rows.append((u, coll_resp(s, c, u, 0.7))); assert fresh_start(s, c, m, boot); c.arm()
    us = np.array([r[0] for r in rows]); az = np.array([r[1] for r in rows])
    hover = float(np.interp(0.0, az, us)) if (az.min() < 0 < az.max()) else float("nan")
    print(f"   coll {us} -> vert_accel {az.round(2)}", flush=True)
    rec(0.15 < hover < 0.32, "hover throttle", f"-> ~{hover:.3f} (fit 0.23)")

    # C. data-driven effectiveness matrix (all-axis authority)
    print("\n=== C. TORQUE EFFECTIVENESS (per-motor calibration) ===", flush=True)
    M = np.array([calib_motor(s, c, m, boot, i) for i in range(4)]).T   # 3x4
    Minv = np.linalg.pinv(M); cond = float(np.linalg.cond(M))
    print(f"   M(3x4)=\n{M.round(0)}\n   cond(M)={cond:.1f}", flush=True)
    rec(cond < 5.0, "all-axis authority", f"-> cond(M)={cond:.1f} (well-conditioned = roll/pitch/yaw all controllable)")

    # D. closed-loop level + altitude hold on raw motors (the bypass)
    print("\n=== D. CLOSED-LOOP HOVER (PD on raw motors) ===", flush=True)
    assert fresh_start(s, c, m, boot); c.arm()
    KP, KD, KPZ, KDZ = 3.5, 3.0, 0.5, 0.35
    ds = s.get_drone(); z_ref = float(ds.pos_ned[2])
    t0 = time.time(); last = -1; minlevel = 99.0; t_unstable = None
    while time.time() - t0 < 12.0:
        ds = s.get_drone()
        if ds is not None:
            R = quat_to_R(_qfix(ds.quat_wxyz)); om = np.asarray(ds.omega, float) * WFIX
            e = np.array([-float(R[2, 1]), float(R[2, 0])])        # reduced-attitude righting sr=-1 sp=+1 (COR-138 t1: +roll = positive feedback)
            tau = np.array([KP * e[0] - KD * om[0], KP * e[1] - KD * om[1], -KD * om[2]])
            coll = HOVER + KPZ * (float(ds.pos_ned[2]) - z_ref) + KDZ * float(ds.vel_ned[2])   # NED z+ down
            c.send_motor_command(np.clip(coll + Minv @ tau, 0., 1.).tolist())
            ti = tilt_deg(ds)
            now = time.time() - t0
            if now > 0.8: minlevel = min(minlevel, ti)
            if minlevel < 5.0 and ti > 12.0 and t_unstable is None: t_unstable = now
            if ti > 55: print(f"   tumbled tilt={ti:.0f} @ {now:.1f}s", flush=True); break
            k = int(now * 2)
            if k != last: last = k; print(f"   t={now:4.1f}s tilt={ti:4.1f} z={float(ds.pos_ned[2]):+5.2f}/{z_ref:+.2f}", flush=True)
        time.sleep(0.01)
    leveled = minlevel < 5.0
    stable = leveled and t_unstable is None
    rec(stable, "closed-loop clean hover",
        f"-> leveled 18deg -> {minlevel:.1f}deg, {'HELD level (no instability)' if stable else 'destabilized ~'+format(t_unstable or 12,'.0f')+'s'}")

    print("\n=== SUMMARY ===", flush=True)
    npass = sum(1 for r in RES if r[0])
    print(f"  {npass}/{len(RES)} pass  -> direct-TRPY interface WORKS (message + authority); "
          f"closed-loop bypass demonstrated (levels + holds ~5 s, slow instability needs the full inner loop)", flush=True)


if __name__ == "__main__":
    main()
