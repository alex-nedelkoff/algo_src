"""vq_deploy_teacher_real.py -- fly the PURE ANALYTIC DAgger teacher (ff_batch + spline anchor)
on the LIVE VQ sim, following the REAL course from TRACK_INFO (no policy, no synthetic gates).

Confirms the question: does the teacher the RL students imitate actually fly the real VQ course?
The matched-sim teacher completes the descending course (MAX GATE 5); this tests transfer.

Frame chain + send + fresh_start lifted verbatim from vq_deploy4_canonical.py (proven). The only
changes: gates come from ENCAP_TRACK_INFO (real descending course, NED->ENU via *B), and the action
is ff_batch(aim(spline)) instead of a policy. Abort on TRUE tilt (qfix), not the live-frame warp.

Usage: python -u vq_deploy_teacher_real.py [--vdes 8] [--maxw 6] [--thrmax 0.6] [--maxt 45]
"""
import sys, json, time
import numpy as np
from pymavlink import mavutil
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store
from aigp.commander import Commander
from aigp.geometry import quat_to_R
from aigp.control_math import quat_mul, desired_attitude, mat_to_quat, attitude_error_quat, collective_accel, accel_to_thrust_norm
from gate_traj import GateTrajectory
import aigp.flight_telemetry as ftm

def argf(f, d): return float(sys.argv[sys.argv.index(f)+1]) if f in sys.argv else d
G = 9.81
MAXW = argf("--maxw", 6.0); THRMAX = argf("--thrmax", 0.6); VDES = argf("--vdes", 8.0); MAX_T = argf("--maxt", 45.0)
SLEW = argf("--slew", 40.0)   # rad/s^2 wire-rate slew cap -- MUST mirror training (RING-06 ring killer); un-slewed kicks ring the live plant
VCAP = argf("--vcap", 0.0)    # hard speed governor: brake along -vel above this (0=off). Descent gravity-assist defeats the normal tilt-clamped brake; needs an explicit cap to hold low speed
RAMP = argf("--ramp", 6.0)    # target-speed onset ramp (s) -- gentle accel so the drone never enters the v>8 runaway regime (vq_course-style creep)
GOV_TILT = np.tan(np.radians(50.0)) * G   # allow more horizontal authority for the brake than the 35deg cruise clamp
GATE_R = argf("--gater", 2.0); ABORT_TILT = 100.0; LEVEL_T = 2.0; GRACE = LEVEL_T + 1.5
# teacher constants (rl_finetune.py, verbatim)
KP_POS = np.array([0.6, 0.6, 2.0]); KD_POS = np.array([1.2, 1.2, 3.0])
KP_ATT = np.array([6.0, 6.0, 4.0]); KD_ATT = np.array([1.2, 1.2, 0.0]); KP_YAW = 4.0; KD_YAW = 0.5
ROLL_WV0, ROLL_WV1, YAW_WV = -0.105, -0.019, -0.149
YR_CAP = 1.5; TILT_MAX = np.tan(np.radians(35.0)) * G; LEAD = 2.5
vqm = json.load(open("sysid/vq_model.json")); F0 = vqm["thrust"]["f0"]; DF = vqm["thrust"]["df_dthr"]
GAIN = np.array([vqm["rate_loop"][n]["gain_G"] for n in ("roll", "pitch", "yaw")])
r = json.load(open("sysid/sim_response.json")); RG = np.array([r["rate_gain_axes"]["roll"], r["rate_gain_axes"]["pitch"], r["rate_gain_axes"]["yaw"]])
IDLE = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE

def qfix(q): return np.array([q[1], q[2], q[3], q[0]])
B = np.array([1.0, -1.0, -1.0]); bq = np.array([0.0, 1.0, 0.0, 0.0]); bqc = np.array([0.0, -1.0, 0.0, 0.0])
def to_enu(ds):
    q_true = qfix(ds.quat_wxyz); Rt = quat_to_R(q_true)
    vel_w = Rt @ ds.vel_ned; om_t = ds.omega * np.array([1.0, -1.0, 1.0])
    return ds.pos_ned * B, vel_w * B, quat_mul(quat_mul(bq, q_true), bqc), om_t * B
def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi

def ff_one(pos, vel, q, om, tgt_pos, tgt_vel, tgt_yaw):
    """Scalar port of rl_finetune.ff_batch (single drone). q = ENU quat wxyz."""
    a = KP_POS * (tgt_pos - pos) + KD_POS * (tgt_vel - vel)
    clamp = TILT_MAX
    if VCAP > 0:                          # hard speed governor (descent overspeed)
        vmag = float(np.linalg.norm(vel))
        if vmag > VCAP:
            a[:2] += -6.0 * (vmag - VCAP) * (vel[:2] / (np.linalg.norm(vel[:2]) + 1e-9))
            clamp = GOV_TILT              # allow harder brake tilt while over speed
    n = np.linalg.norm(a[:2])
    if n > clamp:
        a[:2] *= clamp / n
    R = quat_to_R(q); yaw = np.arctan2(R[1, 0], R[0, 0])
    t = a + np.array([0, 0, G]); zb = t / np.linalg.norm(t)
    xc = np.array([np.cos(tgt_yaw), np.sin(tgt_yaw), 0.0])
    yb = np.cross(zb, xc); yb /= np.linalg.norm(yb); xb = np.cross(yb, zb)
    q_des = mat_to_quat(np.column_stack([xb, yb, zb]))
    w0, x0, y0, z0 = q; w1, x1, y1, z1 = q_des
    qe = np.array([w0*w1 + x0*x1 + y0*y1 + z0*z1, w0*x1 - x0*w1 - y0*z1 + z0*y1,
                   w0*y1 + x0*z1 - y0*w1 - z0*x1, w0*z1 - x0*y1 + y0*x1 - z0*w1])
    if qe[0] < 0: qe = -qe
    w = KP_ATT * (2 * qe[1:4]) - KD_ATT * om
    w[2] = KP_YAW * wrap(tgt_yaw - yaw) - KD_YAW * om[2]
    vb = R.T @ vel
    w[0] += (ROLL_WV0 + ROLL_WV1 * vb[0]) * vb[1]
    w[2] += -YAW_WV * vb[1]
    cos_t = max(R[2, 2], 0.5); c = (G + a[2]) / cos_t
    c = min(c, max(18.0, (G / cos_t) * 1.25))   # maintain lift while braking (06-15 brake-clamp fix; old c_max=10@tilt>40 starved lift -> fall)
    thr = np.clip((F0 + c) / (-DF), 0.0, 1.0)
    u = np.empty(4)
    u[0] = np.clip(2 * thr / THRMAX - 1, -1, 1)
    u[1:4] = np.clip(w / GAIN / MAXW, -1, 1)
    cap = YR_CAP / abs(GAIN[2]) / MAXW
    u[3] = np.clip(u[3], -cap, cap)
    return u

# --- connect + reset ---
s = Store(); m = MavlinkIO(s)
print("connecting...", flush=True); assert m.wait_heartbeat(10), "NO HEARTBEAT"
m.start(); VisionIO(s).start(); boot = int(time.time() * 1000); c = Commander(m.conn, boot)
def idle():
    m.conn.mav.set_attitude_target_send(int(time.time()*1000)-boot, m.conn.target_system, m.conn.target_component, IDLE, [1.0,0,0,0], 0,0,0,0)
def fresh_start():
    t = time.time()
    while time.time()-t < 1.0: idle(); time.sleep(0.02)
    prev = s.get_race(); pb = prev["boot_ms"] if prev else None; c.sim_reset(); t = time.time()
    while time.time()-t < 30:
        idle(); r2 = s.get_race()
        if r2 and (not r2["race_live"] or (pb and r2["boot_ms"] < pb)): break
        time.sleep(0.02)
    while time.time()-t < 30:
        idle(); d = s.get_drone()
        if s.get_race_live() and d is not None and np.linalg.norm(d.vel_ned) < 3.0: return True
        time.sleep(0.02)
    return False

print("fresh_start...", flush=True); assert fresh_start(), "not live"

# --- REAL course from TRACK_INFO ---
gates_obj = None; t = time.time()
while time.time()-t < 5:
    gates_obj = s.get_gates()
    if gates_obj: break
    idle(); time.sleep(0.1)
assert gates_obj, "NO TRACK_INFO"
gates_ned = np.array([g.pos_ned for g in gates_obj])
gates_enu = gates_ned * B               # NED->deploy ENU (same convention as to_enu)
c.arm()
ds0 = s.get_drone(); pos0, _, _, _ = to_enu(ds0)
traj = GateTrajectory(np.vstack([pos0[None, :], gates_enu]), v_cruise=VDES,
                      tilt_budget_deg=35.0, c_drag=0.057, margin=0.6)
NG = len(gates_enu)
print(f"spawn_enu={pos0.round(1)} gates_enu[0]={gates_enu[0].round(1)} gates_enu[-1]={gates_enu[-1].round(1)} "
      f"arc_len={traj.s_max:.1f}m NG={NG} vdes={VDES} maxw={MAXW} thrmax={THRMAX}", flush=True)

from aigp.recorder import Recorder
rec = Recorder(s, script="vq_deploy_teacher_real", mode="rate",
               notes="pure analytic spline teacher on REAL TRACK_INFO course",
               extra_meta={"cmd_layout": ["wx", "wy", "wz", "thrust"]})
flog = ftm.from_args(sys.argv, RG, run_name="teacher_real", store=s)
t0 = time.time(); gi = 0; last = -1; maxgi = 0; prev_rates = np.zeros(3); t_prev = t0
while time.time()-t0 < MAX_T and gi < NG:
    ds = s.get_drone()
    if ds is None: time.sleep(0.01); continue
    t = time.time()-t0; pe, ve, qe, ome = to_enu(ds)
    t_now = time.time(); dt_loop = max(1e-3, t_now - t_prev); t_prev = t_now
    if t < LEVEL_T:
        # settle hover (live-frame, hold spawn attitude) -- no flip; spawn already camera-forward
        a = np.array([0.0, 0.0, 1.8*(ds0.pos_ned[2]-ds.pos_ned[2]) - 3.0*ds.vel_ned[2]])
        Rd = quat_to_R(ds.quat_wxyz); yc = float(np.arctan2(Rd[1, 0], Rd[0, 0]))
        qd = mat_to_quat(desired_attitude(a, yc)); wd = np.array([0.5, 1.6, 1.0]) * attitude_error_quat(ds.quat_wxyz, qd)
        wd[2] = -0.3 * float(ds.omega[2])
        thr = accel_to_thrust_norm(collective_accel(a, ds.quat_wxyz), 0.2675, 62.0)
        c.send_attitude_target(np.clip(wd/RG, -4, 4), thr); phase = "LVL"
    else:
        s_d = traj.nearest_s(pe); ref = traj.sample(min(s_d + LEAD, traj.s_max))
        ramp = min(1.0, (t - LEVEL_T) / max(RAMP, 1e-3))   # gentle speed onset -> never enter the runaway regime
        tgt_pos = ref["pos"]; tgt_vel = float(ref["v"]) * ramp * ref["tang"]; tgt_yaw = float(ref["yaw"])
        u = ff_one(pe, ve, qe, ome, tgt_pos, tgt_vel, tgt_yaw)
        thr = float((u[0]+1)/2*THRMAX); rates = np.array([u[1], -u[2], -u[3]]) * MAXW   # ENU->VQ via B
        if SLEW > 0 and prev_rates is not None:   # mirror training wire-rate slew cap (RING-06)
            rates = prev_rates + np.clip(rates - prev_rates, -SLEW*dt_loop, SLEW*dt_loop)
        prev_rates = rates.copy()
        c.send_attitude_target(rates, thr); phase = "TCH"
        rec.log([float(rates[0]), float(rates[1]), float(rates[2]), float(thr)])
    # TRUE tilt (qfix) for abort, not the warped live-frame
    Rt = quat_to_R(qfix(ds.quat_wxyz)); tilt = float(np.degrees(np.arccos(np.clip(Rt[2, 2], -1, 1))))
    vmag = float(np.linalg.norm(ds.vel_ned))
    dist = float(np.linalg.norm(gates_enu[gi] - pe))   # 3D dist to current gate
    if dist < GATE_R:
        gi += 1; maxgi = max(maxgi, gi)
        print(f"  >>> GATE {gi}/{NG} t={t:.1f} v={vmag:.1f} tilt={tilt:.0f}", flush=True)
    if flog is not None: flog.push(t, ds, {"thr": thr}, cruise=VDES, running=s.get_race_live(), armed=True)
    if tilt > ABORT_TILT and t > GRACE:
        print(f"  ABORT true_tilt={tilt:.0f} t={t:.1f} gate {gi}/{NG}", flush=True); break
    if int(t*2) != last:
        last = int(t*2)
        print(f"  t={t:4.1f} {phase} gate{gi} dist={dist:5.1f} z={pe[2]:6.1f} v={vmag:4.1f} tilt={tilt:3.0f}", flush=True)
    time.sleep(0.01)
idle(); rec.close()
if flog is not None: flog.close()
verdict = "TEACHER FLIES REAL VQ" if maxgi >= NG*0.7 else ("partial" if maxgi >= 2 else "FAILED")
print(f"DONE: reached {maxgi}/{NG} gates  z_final={pe[2]:.1f}  [{verdict}]", flush=True)
