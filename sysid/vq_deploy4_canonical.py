"""vq_deploy4.py -- fly the matched-sim policy on the LIVE VQ (transfer test, obs-corrected).

Fixes vs vq_deploy: (1) obs in the GATE-YAW-RELATIVE frame (training _compute_obs rotates XY by
-gate_yaw and uses drone_yaw-gate_yaw); (2) arena-extent obs = 12.0 (arena_bounds 120 in training,
not 60); (3) synthetic course CURVES like the training distribution (+-0.12-0.28 rad/gate -- the
policy never saw a straight course). --turn sets rad/gate (sign = direction), default +0.2.

Frame chain verified by vq_deploy_check (policy u0~hover_u0 at hover, obs sane). Now SEND the policy
action: thr=(u0+1)/2 ; body-rate cmd to VQ = B@(u[1:4]*MAXW) = [u1,-u2,-u3]*MAXW (ENU->VQ rate cmd).
fresh_start -> fly toward synthetic ENU gate corridor -> abort on tilt. Live -u log. Answers: does the
matched-sim-trained policy hold nose / pass gates on the real VQ, or hit the weathervane wall?
Usage: python -u vq_deploy3.py --policy X.zip [--gates 6] [--space 10] [--turn 0.2]
v4: NO FLIP AT ALL -- course runs along the CAMERA direction, so the spawn attitude (nose
anti-course) IS the racing attitude; policy hands off straight into its trained racing
distribution (obs[8]~pi) with zero yaw rotation. 180-deg hover yaw-flips tumble the live
plant whoever commands them (run2 policy-flip, run3 analytic-flip) -- that regime was never
fit; avoid, do not fight. (a) was PRE-FLIP -- the LVL phase yaws the nose 180 deg (capped 1.2 rad/s) to the
anti-course attitude BEFORE handoff, so the policy starts inside its post-flip racing
distribution (the live plant diverges from the matched plant during the aggressive flip
transient -- offline chain check races with lag+latency, live tumbles only in the flip);
(b) Recorder ON -- every run is sysID data.
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
POLICY_ZIP = args("--policy", "ft_dagger.zip")
G = 9.81; MAXW = argf("--maxw", 17.45); THRMAX = argf("--thrmax", 1.0); NG = int(argf("--gates", 6)); SPACE = argf("--space", 10.0)
ZVD = "--zvd" in sys.argv   # ZVD shaper on wire rate cmds; MUST match training (EXP-20b ring killer on the policy path)
ZVD_AMP = np.array([0.371, 0.476, 0.153]); ZVD_DELAY = (7, 7, 4); _zvd_buf = np.zeros((15, 3))
GATE_R = 1.5; ABORT_TILT = 110.0; LEVEL_T = 3.0; GRACE = LEVEL_T + 1.0; MAX_T = 33.0
vqm = json.load(open("sysid/vq_model.json")); F0 = vqm["thrust"]["f0"]; DF = vqm["thrust"]["df_dthr"]
HOVER_U0 = 2*((F0+G)/(-DF))-1
r = json.load(open("sysid/sim_response.json")); RG = np.array([r["rate_gain_axes"]["roll"], r["rate_gain_axes"]["pitch"], r["rate_gain_axes"]["yaw"]])
IDLE = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE

z = zipfile.ZipFile(POLICY_ZIP); sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu")
Wt = {k: v.numpy() for k, v in sd.items()}
def policy(obs):
    h = obs.astype(np.float64)
    for i in (0, 2, 4):
        h = np.tanh(Wt[f"mlp_extractor.policy_net.{i}.weight"] @ h + Wt[f"mlp_extractor.policy_net.{i}.bias"])
    return Wt["action_net.weight"] @ h + Wt["action_net.bias"]

def qfix(q): return np.array([q[1], q[2], q[3], q[0]])
B = np.array([1.0, -1.0, -1.0]); bq = np.array([0.0, 1.0, 0.0, 0.0]); bqc = np.array([0.0, -1.0, 0.0, 0.0])
def quat_to_euler(q):
    w, x, y, zz = q
    return (np.arctan2(2*(w*x+y*zz), 1-2*(x*x+y*y)), np.arcsin(np.clip(2*(w*y-zz*x), -1, 1)),
            np.arctan2(2*(w*zz+x*y), 1-2*(y*y+zz*zz)))
MIRROR = "--nomirror" not in sys.argv   # ADAPTER FIX (default ON, 06-16): to_enu inverted lateral-y vs the RL env (B=[1,-1,-1]
# y-flip = "world-y mirror") -> policy lateral sign inverted live -> lateral runaway/crab. Flip drone pos/vel y. (Synthetic gates
# here are built from camfwd=to_enu(ds0) so they inherit the fix.) `--nomirror` = old buggy frame.
MIRRORATT = "--mirroratt" in sys.argv   # COMPLETE the mirror: also reflect the ENU attitude quat [w,x,y,z]->[w,-x,y,-z] (flips world
# roll + yaw to match the mirrored pos/vel; pitch invariant; body omega untouched = body frame). Plain MIRROR flips pos/vel only ->
# roll/yaw stayed un-mirrored, consistent only at ~0 tilt (strafe). Banked turns need this. A/B this vs plain MIRROR.
def to_enu(ds):
    q_true = qfix(ds.quat_wxyz); Rt = quat_to_R(q_true)
    vel_w = Rt @ ds.vel_ned; om_t = ds.omega*np.array([1.0, -1.0, 1.0])
    pos, vel, q, om = ds.pos_ned*B, vel_w*B, quat_mul(quat_mul(bq, q_true), bqc), om_t*B
    if MIRROR: pos = pos*np.array([1.,-1.,1.]); vel = vel*np.array([1.,-1.,1.])
    if MIRROR and MIRRORATT: q = q*np.array([1.,-1.,1.,-1.])
    return pos, vel, q, om

s = Store(); m = MavlinkIO(s)
print("connecting...", flush=True); assert m.wait_heartbeat(10), "NO HEARTBEAT"
m.start(); VisionIO(s).start(); boot = int(time.time()*1000); c = Commander(m.conn, boot)
def idle():
    m.conn.mav.set_attitude_target_send(int(time.time()*1000)-boot, m.conn.target_system, m.conn.target_component, IDLE, [1.0,0,0,0], 0,0,0,0)
def fresh_start():
    t=time.time()
    while time.time()-t<1.0: idle(); time.sleep(0.02)
    prev=s.get_race(); pb=prev["boot_ms"] if prev else None; c.sim_reset(); t=time.time()
    while time.time()-t<30:
        idle(); r2=s.get_race()
        if r2 and (not r2["race_live"] or (pb and r2["boot_ms"]<pb)): break
        time.sleep(0.02)
    while time.time()-t<30:
        idle(); d=s.get_drone()
        if s.get_race_live() and d is not None and np.linalg.norm(d.vel_ned)<3.0: return True
        time.sleep(0.02)
    return False

print("fresh_start...", flush=True); assert fresh_start(), "not live"; c.arm()
ds0 = s.get_drone(); pos0, _, _, _ = to_enu(ds0)
camfwd = -(quat_to_R(qfix(ds0.quat_wxyz))[:, 0]) * B; camfwd[2] = 0; camfwd /= (np.linalg.norm(camfwd)+1e-9)
TURN = argf("--turn", 0.2)                      # rad/gate, matches training curve distribution
# course extends along the NOSE (+body_x), like training (spawn obs[8]~0); the policy yaw-flips
# during entry and races camera-first toward the gates (anti-velocity nose = camera leads travel)
hd0 = float(np.arctan2(camfwd[1], camfwd[0]))   # course along CAMERA: no flip needed
gates = []; gyaws = []; hd = hd0; p = pos0.copy()
for i in range(NG):
    hd += TURN; p = p + SPACE*np.array([np.cos(hd), np.sin(hd), 0.0])
    gates.append(p.copy()); gyaws.append(hd)
gates = np.array(gates); gyaws = np.array(gyaws)
def rotxy(v, yaw):
    c, sn = np.cos(-yaw), np.sin(-yaw)
    return np.array([c*v[0]-sn*v[1], sn*v[0]+c*v[1]])
def wrap(a): return (a+np.pi) % (2*np.pi) - np.pi
Rd0 = quat_to_R(ds0.quat_wxyz)
yaw_live0 = float(np.arctan2(Rd0[1,0], Rd0[0,0]))
yaw_flip_tgt = yaw_live0                          # v4: NO flip -- hold spawn yaw
print(f"spawn_enu={pos0.round(1)} camfwd={camfwd.round(2)} turn={TURN} gates0={gates[0].round(1)} "
      f"hover_u0={HOVER_U0:.2f} preflip {np.degrees(yaw_live0):.0f}->{np.degrees(yaw_flip_tgt):.0f}deg", flush=True)
from aigp.recorder import Recorder
rec_d = Recorder(s_store := s, script="vq_deploy4", mode="rate",
                 notes="policy deploy + preflip; every run = sysID data",
                 extra_meta={"cmd_layout": ["wx", "wy", "wz", "thrust"], "policy": POLICY_ZIP})
flog = ftm.from_args(sys.argv, RG, run_name="vq_deploy4", store=s)
prev_act = np.zeros(4); t0 = time.time(); gi = 0; betas = []; last = -1
OBSLOG = [] if "--obslog" in sys.argv else None
while time.time()-t0 < MAX_T and gi < NG:
    ds = s.get_drone()
    if ds is None: time.sleep(0.01); continue
    t = time.time()-t0; pe, ve, qe, ome = to_enu(ds)
    gate = gates[gi]; gi1 = (gi+1) % NG; gyaw = gyaws[gi]   # training env wraps lookahead
    roll, pitch, dyaw = quat_to_euler(qe)
    obs = np.zeros(27, dtype=np.float32)
    obs[0:2] = rotxy((pe-gate)[:2], gyaw); obs[2] = pe[2]-gate[2]      # gate-yaw frame
    obs[3:5] = rotxy(ve[:2], gyaw); obs[5] = ve[2]
    obs[6] = roll; obs[7] = pitch; obs[8] = wrap(dyaw - gyaw); obs[9:12] = ome
    obs[12:16] = 0.5114015; obs[16:20] = prev_act   # matched-sim motor slots sit at the reset hover value
    dg = gates[gi1]-gate
    obs[20:22] = rotxy(dg[:2], gyaw); obs[22] = dg[2]; obs[23] = wrap(gyaws[gi1]-gyaw)
    obs[24] = 2.0; obs[25] = 2.0; obs[26] = 12.0                        # arena 120/10 (training)
    if "--noobsclip" not in sys.argv:
        obs[0:2] = np.clip(obs[0:2], -44.0, 44.0); obs[2] = float(np.clip(obs[2], -8.0, 8.0))  # OOD guard (run-7 flee)
    u = policy(obs); uc = np.clip(u, -1, 1)
    if OBSLOG is not None:
        OBSLOG.append(np.concatenate([[t], obs.astype(np.float64), u]))
    if t >= LEVEL_T:
        prev_act = uc   # training episodes start with prev_actions=0; do NOT shadow during LVL (OOD obs[16:20] at handoff)
    if t < LEVEL_T:
        # Phase 1: leveling hover + PRE-FLIP: yaw the nose to anti-course (the policy's racing
        # attitude) analytically, capped 1.2 rad/s, so the handoff skips the flip transient.
        a = np.array([0.0, 0.0, 1.8*(ds0.pos_ned[2]-ds.pos_ned[2]) - 3.0*ds.vel_ned[2]])
        Rd = quat_to_R(ds.quat_wxyz); yc = float(np.arctan2(Rd[1,0], Rd[0,0]))
        y_tgt = yaw_flip_tgt   # live-frame yaw pointing nose anti-course
        yerr = (y_tgt - yc + np.pi) % (2*np.pi) - np.pi
        qd = mat_to_quat(desired_attitude(a, yc)); wd = np.array([0.5,1.6,1.0])*attitude_error_quat(ds.quat_wxyz, qd)
        wd[2] = float(np.clip(3.0*yerr, -1.2, 1.2)) - 0.3*float(ds.omega[2])
        thr = accel_to_thrust_norm(collective_accel(a, ds.quat_wxyz), 0.2675, 62.0)
        c.send_attitude_target(np.clip(wd/RG, -4, 4), thr); phase = "LVL"
    else:
        # Phase 2: POLICY flies
        thr = float((uc[0]+1)/2*THRMAX); rates = np.array([uc[1], uc[2], -uc[3]])*MAXW   # pitch un-flipped (06-16 PITCH-FIX: model pitch sign corrected to live)
        if ZVD:
            _zvd_buf = np.roll(_zvd_buf, 1, axis=0); _zvd_buf[0] = rates
            rates = np.array([ZVD_AMP[0]*_zvd_buf[0, ax] + ZVD_AMP[1]*_zvd_buf[ZVD_DELAY[ax], ax]
                              + ZVD_AMP[2]*_zvd_buf[2*ZVD_DELAY[ax], ax] for ax in range(3)])
        c.send_attitude_target(rates, thr); phase = "POL"
        rec_d.log([float(rates[0]), float(rates[1]), float(rates[2]), float(thr)])
    Rc = quat_to_R(ds.quat_wxyz); tilt = float(np.degrees(np.arccos(max(-1, min(1, Rc[2, 2])))))
    vmag = float(np.linalg.norm(ds.vel_ned))
    # sideslip in ENU body frame
    if np.linalg.norm(ve[:2]) > 0.5:
        nose = quat_to_R(qe)[:2, 0]; vv = ve[:2]
        beta = abs(np.degrees(np.arctan2(nose[0]*vv[1]-nose[1]*vv[0], nose@vv))); betas.append(beta)
    else:
        beta = 0.0
    dist = float(np.linalg.norm((gate-pe)[:2]))
    if dist < GATE_R:
        gi += 1; print(f"  GATE {gi}/{NG} t={t:.1f} v={vmag:.1f} tilt={tilt:.0f} |beta|={beta:.0f}", flush=True)
    if flog is not None: flog.push(t, ds, {"thr": thr}, cruise=4.0, running=s.get_race_live(), armed=True)
    if tilt > ABORT_TILT and t > GRACE:
        print(f"  ABORT tilt={tilt:.0f} t={t:.1f} gate {gi}/{NG} |beta|={beta:.0f}", flush=True); break
    if int(t*2) != last:
        last = int(t*2)
        print(f"  t={t:4.1f} {phase} gate{gi} dist={dist:4.1f} v={vmag:4.1f} tilt={tilt:3.0f} |beta|={beta:3.0f} u=[{u[0]:+.2f},{u[1]:+.2f},{u[2]:+.2f},{u[3]:+.2f}]", flush=True)
    time.sleep(0.01)
idle()
if OBSLOG is not None:
    np.save("obslog_last.npy", np.array(OBSLOG))
    print(f"obslog saved: {len(OBSLOG)} rows -> obslog_last.npy", flush=True)
rec_d.close()
if flog is not None: flog.close()
print(f"DEPLOY DONE: reached {gi}/{NG} gates | mean|beta|={np.mean(betas) if betas else 0:.0f} | "
      f"{'POLICY FLIES VQ' if gi>=NG*0.7 else ('weathervane/tail-first' if (betas and np.mean(betas)>90) else 'failed-other')}", flush=True)
