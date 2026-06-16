"""vq_deploy_policy_real.py -- fly a trained RL policy on the LIVE VQ, REAL TRACK_INFO course.

Same validated frame/IO/send/slew as vq_deploy_teacher_real.py, but runs a policy zip instead
of the analytic teacher. Obs = the training _compute_obs layout (59-dim: 27 base + 8x4 action
history), gate-yaw-relative (frame-robust -> built in the deploy ENU frame), arena 17.92
(=arena_bounds 179/10 for the real course), lookahead width/height 2.0 (training default).

Usage: python -u vq_deploy_policy_real.py --policy ft_vq1_v3_dag_best.zip --maxw 6 --thrmax 0.6 --slew 40 --hist 8
"""
import sys, json, time, zipfile, io
import numpy as np, torch
from pymavlink import mavutil
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store
from aigp.commander import Commander
from aigp.geometry import quat_to_R
from aigp.control_math import quat_mul, desired_attitude, mat_to_quat, attitude_error_quat, collective_accel, accel_to_thrust_norm
import aigp.flight_telemetry as ftm

def argf(f, d): return float(sys.argv[sys.argv.index(f)+1]) if f in sys.argv else d
def args(f, d): return sys.argv[sys.argv.index(f)+1] if f in sys.argv else d
POLICY = args("--policy", "ft_vq1_v3_dag_best.zip")
G = 9.81
MAXW = argf("--maxw", 6.0); THRMAX = argf("--thrmax", 0.6); MAX_T = argf("--maxt", 70.0)
SLEW = argf("--slew", 40.0); HIST = int(argf("--hist", 8)); GATE_R = argf("--gater", 2.0)
ZVD = "--zvd" in sys.argv   # EXP-20b ring killer on the policy rate cmds; MUST match training --zvd
ZVD_AMP = np.array([0.371, 0.476, 0.153]); ZVD_DELAY = (7, 7, 4); _zvd_buf = np.zeros((15, 3))
ABORT_TILT = 100.0; LEVEL_T = 2.0; GRACE = LEVEL_T + 1.5; MOTOR_NORM = 0.5114015
vqm = json.load(open("sysid/vq_model.json")); F0 = vqm["thrust"]["f0"]; DF = vqm["thrust"]["df_dthr"]
r = json.load(open("sysid/sim_response.json")); RG = np.array([r["rate_gain_axes"]["roll"], r["rate_gain_axes"]["pitch"], r["rate_gain_axes"]["yaw"]])
IDLE = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE

z = zipfile.ZipFile(POLICY); sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu")
Wt = {k: v.numpy() for k, v in sd.items()}
def policy(obs):
    h = obs.astype(np.float64)
    for i in (0, 2, 4):
        h = np.tanh(Wt[f"mlp_extractor.policy_net.{i}.weight"] @ h + Wt[f"mlp_extractor.policy_net.{i}.bias"])
    return Wt["action_net.weight"] @ h + Wt["action_net.bias"]

def qfix(q): return np.array([q[1], q[2], q[3], q[0]])
B = np.array([1.0, -1.0, -1.0]); bq = np.array([0.0, 1.0, 0.0, 0.0]); bqc = np.array([0.0, -1.0, 0.0, 0.0])
MIRROR = "--nomirror" not in sys.argv   # ADAPTER FIX (default ON): to_enu inverted lateral-y vs the training env (the "world-y mirror")
# -> the policy's lateral sign was inverted live -> lateral runaway/crab. Flip drone pos/vel y + gate y so deploy frame matches train.
def to_enu(ds):
    q_true = qfix(ds.quat_wxyz); Rt = quat_to_R(q_true)
    pos, vel, q, om = ds.pos_ned*B, (Rt @ ds.vel_ned)*B, quat_mul(quat_mul(bq, q_true), bqc), (ds.omega*np.array([1.0, -1.0, 1.0]))*B
    if MIRROR: pos = pos*np.array([1.,-1.,1.]); vel = vel*np.array([1.,-1.,1.])
    return pos, vel, q, om
def wrap(a): return (a + np.pi) % (2*np.pi) - np.pi
def rotxy(v, yaw):
    c, sn = np.cos(-yaw), np.sin(-yaw)
    return np.array([c*v[0]-sn*v[1], sn*v[0]+c*v[1]])
def quat_to_euler(q):
    w, x, y, zz = q
    return (np.arctan2(2*(w*x+y*zz), 1-2*(x*x+y*y)), np.arcsin(np.clip(2*(w*y-zz*x), -1, 1)), np.arctan2(2*(w*zz+x*y), 1-2*(y*y+zz*zz)))

# --- connect + reset ---
s = Store(); m = MavlinkIO(s)
print("connecting...", flush=True); assert m.wait_heartbeat(10), "NO HEARTBEAT"
m.start(); VisionIO(s).start(); boot = int(time.time()*1000); c = Commander(m.conn, boot)
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
gates_obj = None; t = time.time()
while time.time()-t < 5:
    gates_obj = s.get_gates()
    if gates_obj: break
    idle(); time.sleep(0.1)
assert gates_obj, "NO TRACK_INFO"
gates = np.array([g.pos_ned for g in gates_obj]) * B          # deploy ENU
if MIRROR: gates[:, 1] *= -1.0                                # match the to_enu lateral-y fix (consistent gate-relative obs)
NG = len(gates)
# gate yaw = heading of the leg into each gate (matches make_env REALCOURSE _yaw_quat(arctan2(dy,dx)))
gyaws = np.zeros(NG); prev = np.array([0.0, 0.0, gates[0, 2]])
ds0 = s.get_drone(); spawn, _, _, _ = to_enu(ds0); prev = spawn.copy()
for i in range(NG):
    d = gates[i] - prev; gyaws[i] = np.arctan2(d[1], d[0]); prev = gates[i]
ARENA = (float(np.abs(gates[:, :2]).max()) + 20.0) / 10.0     # 17.92, mirror training arena_bounds/10
c.arm()
print(f"spawn={spawn.round(1)} gate0={gates[0].round(1)} gate5={gates[-1].round(1)} NG={NG} arena_obs={ARENA:.2f} policy={POLICY}", flush=True)

def build_obs(pe, ve, qe, ome, gi, prev_act, hist):
    gate = gates[gi]; gy = gyaws[gi]; gi1 = (gi+1) % NG
    o = np.zeros(20 + 6 + 1 + 4*HIST, dtype=np.float32)
    o[0:2] = rotxy((pe-gate)[:2], gy); o[2] = pe[2]-gate[2]
    o[3:5] = rotxy(ve[:2], gy); o[5] = ve[2]
    roll, pitch, dyaw = quat_to_euler(qe); o[6] = roll; o[7] = pitch; o[8] = wrap(dyaw - gy)
    o[9:12] = ome; o[12:16] = MOTOR_NORM; o[16:20] = prev_act
    dg = gates[gi1] - gate
    o[20:22] = rotxy(dg[:2], gy); o[22] = dg[2]; o[23] = wrap(gyaws[gi1]-gy); o[24] = 2.0; o[25] = 2.0
    o[26] = ARENA
    if HIST > 0: o[27:27+4*HIST] = hist.flatten()
    return o

from aigp.recorder import Recorder
rec = Recorder(s, script="vq_deploy_policy_real", mode="rate", notes="RL policy on REAL TRACK_INFO course", extra_meta={"cmd_layout": ["wx","wy","wz","thrust"], "policy": POLICY})
flog = ftm.from_args(sys.argv, RG, run_name="policy_real", store=s)
t0 = time.time(); gi = 0; last = -1; maxgi = 0; prev_act = np.zeros(4); hist = np.zeros((HIST, 4)); prev_rates = np.zeros(3); t_prev = t0
while time.time()-t0 < MAX_T and gi < NG:
    ds = s.get_drone()
    if ds is None: time.sleep(0.01); continue
    t = time.time()-t0; pe, ve, qe, ome = to_enu(ds)
    t_now = time.time(); dt_loop = max(1e-3, t_now - t_prev); t_prev = t_now
    if t < LEVEL_T:
        a = np.array([0.0, 0.0, 1.8*(ds0.pos_ned[2]-ds.pos_ned[2]) - 3.0*ds.vel_ned[2]])
        Rd = quat_to_R(ds.quat_wxyz); yc = float(np.arctan2(Rd[1, 0], Rd[0, 0]))
        qd = mat_to_quat(desired_attitude(a, yc)); wd = np.array([0.5, 1.6, 1.0]) * attitude_error_quat(ds.quat_wxyz, qd)
        wd[2] = -0.3*float(ds.omega[2]); thr = accel_to_thrust_norm(collective_accel(a, ds.quat_wxyz), 0.2675, 62.0)
        c.send_attitude_target(np.clip(wd/RG, -4, 4), thr); phase = "LVL"
    else:
        obs = build_obs(pe, ve, qe, ome, gi, prev_act, hist)
        u = np.clip(policy(obs), -1, 1)
        thr = float((u[0]+1)/2*THRMAX); rates = np.array([u[1], -u[2], -u[3]]) * MAXW
        if ZVD:   # ring killer BEFORE slew (matches env vq_zvd->vq_slew order)
            _zvd_buf = np.roll(_zvd_buf, 1, axis=0); _zvd_buf[0] = rates
            rates = np.array([ZVD_AMP[0]*_zvd_buf[0, ax] + ZVD_AMP[1]*_zvd_buf[ZVD_DELAY[ax], ax]
                              + ZVD_AMP[2]*_zvd_buf[2*ZVD_DELAY[ax], ax] for ax in range(3)])
        if SLEW > 0:
            rates = prev_rates + np.clip(rates - prev_rates, -SLEW*dt_loop, SLEW*dt_loop)
        prev_rates = rates.copy()
        c.send_attitude_target(rates, thr); phase = "POL"
        rec.log([float(rates[0]), float(rates[1]), float(rates[2]), float(thr)])
        hist[1:] = hist[:-1]; hist[0] = u; prev_act = u
    Rt = quat_to_R(qfix(ds.quat_wxyz)); tilt = float(np.degrees(np.arccos(np.clip(Rt[2, 2], -1, 1))))
    vmag = float(np.linalg.norm(ds.vel_ned)); dist = float(np.linalg.norm(gates[gi] - pe))
    if dist < GATE_R:
        gi += 1; maxgi = max(maxgi, gi); print(f"  >>> GATE {gi}/{NG} t={t:.1f} v={vmag:.1f} tilt={tilt:.0f}", flush=True)
    if flog is not None: flog.push(t, ds, {"thr": thr}, cruise=3.0, running=s.get_race_live(), armed=True)
    if tilt > ABORT_TILT and t > GRACE:
        print(f"  ABORT true_tilt={tilt:.0f} t={t:.1f} gate {gi}/{NG}", flush=True); break
    if int(t*2) != last:
        last = int(t*2); print(f"  t={t:4.1f} {phase} gate{gi} dist={dist:5.1f} z={pe[2]:6.1f} v={vmag:4.1f} tilt={tilt:3.0f}", flush=True)
    time.sleep(0.01)
idle(); rec.close()
if flog is not None: flog.close()
verdict = "POLICY FLIES VQ1" if maxgi >= NG*0.7 else ("partial" if maxgi >= 2 else "FAILED")
print(f"DONE: reached {maxgi}/{NG} gates  z_final={pe[2]:.1f}  [{verdict}]", flush=True)
