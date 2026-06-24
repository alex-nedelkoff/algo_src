"""COR-138 ladder rungs 2-4: attitude SETPOINT tracking on direct-TRPY (the inner loop that replaces
the sim's explosive rate loop). Run live on the laptop:  python scripts/trpy_nav.py

trpy_motors() = the validated level-hover controller generalized to track a commanded thrust direction:
desired body-down from the accel command, the reduced-attitude error e=[-R[2,1], +R[2,0]] (sr=-1 sp=+1,
COR-138 t1), Minv mixer (cond ~1.4), HOVER*|f|/g collective.

STATUS (2026-06-23):
  rung 2 HELD TILT  -> PASS: holds a commanded 10deg tilt rock-steady (fast attitude KP8/KD2; the
                       earlier over-damping was masking the roll-sign bug, now fixed).
  rung 4 GOTO       -> BLOCKED on the HORIZONTAL FRAME MAP. Direction probe: cmd world +x -> drone
                       accelerates -x (mirrored); cmd +y -> +y with a +x cross-leak. The accel->motion
                       map is a rotation+mirror (the project's recurring lateral-sign/frame issue, here
                       in the qfix/WFIX frame). A hand sign-flip runs away (it's not a clean flip).
  NEXT: do NOT re-solve the frame here. Reuse aigp.navigator.attitude_command_tf's PROVEN a2->q_des
        mapping (s_cam/WFIX/ymirror) to get the desired attitude, then track q_des with this torque
        loop instead of the sim rate command. Replaces only the final rate->motor step.
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
HOVER = 0.234; G = 9.81
KP, KD, KPY = 8.0, 2.0, 3.0; KPZ, KDZ = 3.0, 2.0; TILT_MAX = np.radians(20)


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
def yaw_of(ds):
    R = quat_to_R(_qfix(ds.quat_wxyz)); return float(np.arctan2(R[1, 0], R[0, 0]))

def calib(s, c, m, boot, i, bump=0.10):
    """Per-motor torque effectiveness (peak true-frame gyro / bump) -> column of M (3x4), cond ~1.4."""
    assert fresh_start(s, c, m, boot); c.arm(); u = np.full(4, HOVER); u[i] = HOVER + bump
    t0 = time.time(); best = np.zeros(3); bn = 0.0
    while time.time() - t0 < 0.30:
        c.send_motor_command(u.tolist()); time.sleep(0.01); ds = s.get_drone()
        if ds is not None:
            om = np.asarray(ds.omega, float) * WFIX
            if np.linalg.norm(om) > bn: bn = float(np.linalg.norm(om)); best = om.copy()
            if tilt_deg(ds) > 50: break
    return best / bump

def trpy_motors(ds, a2, z_ref, yaw_ref, Minv):
    """Direct-TRPY inner loop: world horizontal accel a2 (+ altitude PD) -> 4 motor throttles.
    NOTE: a2 must be in the frame the drone actually accelerates in -- the raw world a2 is mirrored
    (see module docstring). Use the navigator's a2->q_des mapping upstream until that's wired in."""
    R = quat_to_R(_qfix(ds.quat_wxyz)); om = np.asarray(ds.omega, float) * WFIX
    a2 = np.asarray(a2, float); n = np.linalg.norm(a2); amax = np.tan(TILT_MAX) * G
    if n > amax: a2 = a2 / n * amax
    az = KPZ * (z_ref - float(ds.pos_ned[2])) - KDZ * float(ds.vel_ned[2])
    f = np.array([a2[0], a2[1], az - G]); fmag = float(np.linalg.norm(f))   # thrust accel (NED); up = -z
    d_body = R.T @ (-f / max(fmag, 1e-6))                                   # desired body-down dir, body frame
    e = np.array([-d_body[1], d_body[0]])                                   # reduced-attitude err; sr=-1 sp=+1
    yaw = np.arctan2(R[1, 0], R[0, 0]); ye = ((yaw_ref - yaw + np.pi) % (2 * np.pi)) - np.pi
    tau = np.array([KP * e[0] - KD * om[0], KP * e[1] - KD * om[1], KPY * ye - KD * om[2]])
    return np.clip(HOVER * fmag / G + Minv @ tau, 0., 1.)


def main():
    s = Store(); m = MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); VisionIO(s).start()
    boot = int(time.time() * 1000); c = Commander(m.conn, boot)
    M = np.array([calib(s, c, m, boot, i) for i in range(4)]).T; Minv = np.linalg.pinv(M)
    print(f"cond(M)={np.linalg.cond(M):.1f}", flush=True)

    # RUNG 2: hold a commanded 10deg tilt (validates setpoint tracking, not just righting)
    print("\n=== RUNG 2: hold 10deg tilt ===", flush=True)
    assert fresh_start(s, c, m, boot); c.arm()
    ds = s.get_drone(); z_ref = float(ds.pos_ned[2]); yaw0 = yaw_of(ds)
    a2 = np.array([G * np.tan(np.radians(10)), 0.0]); t0 = time.time(); last = -1; tl = []
    while time.time() - t0 < 8.0:
        ds = s.get_drone()
        if ds is not None:
            c.send_motor_command(trpy_motors(ds, a2, z_ref, yaw0, Minv).tolist())
            ti = tilt_deg(ds); now = time.time() - t0
            if now > 3: tl.append(ti)
            if ti > 55: print(f"   tumbled @ {now:.1f}s", flush=True); break
            k = int(now * 2)
            if k != last: last = k; print(f"   t={now:4.1f}s tilt={ti:4.1f} (target 10)", flush=True)
        time.sleep(0.01)
    print(f"   -> steady tilt {np.mean(tl) if tl else 99:.1f} deg (target 10)  [PASS if ~10, stable]", flush=True)
    # rung 4 (goto) intentionally omitted -- blocked on the horizontal frame map (see docstring).


if __name__ == "__main__":
    main()
