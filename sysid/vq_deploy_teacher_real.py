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
def args(f, d): return sys.argv[sys.argv.index(f)+1] if f in sys.argv else d
G = 9.81
MAXW = argf("--maxw", 6.0); THRMAX = argf("--thrmax", 0.6); VDES = argf("--vdes", 8.0); MAX_T = argf("--maxt", 45.0)
SLEW = argf("--slew", 40.0)   # rad/s^2 wire-rate slew cap -- MUST mirror training (RING-06 ring killer); un-slewed kicks ring the live plant
VCAP = argf("--vcap", 0.0)    # hard speed governor: brake along -vel above this (0=off). Descent gravity-assist defeats the normal tilt-clamped brake; needs an explicit cap to hold low speed
RAMP = argf("--ramp", 6.0)    # target-speed onset ramp (s) -- gentle accel so the drone never enters the v>8 runaway regime (vq_course-style creep)
VZMAX = argf("--vzmax", 1e9)  # controlled-sink: cap descent rate (m/s); forces slow forward speed on the steep descent. 1e9=off
TILT_SOFT = argf("--tiltsoft", 30.0); TILT_HARD = argf("--tilthard", 90.0)  # actual-tilt cap: fade tilt-drive from SOFT->HARD measured tilt (stay below the ~45deg ring detonation). HARD>=90 = off
STRAIGHT = int(argf("--straight", 0))  # N>0: override the TRACK_INFO course with N straight LEVEL gates 30m apart along camera-forward (drag-saturation speed test, no descent/turns)
ZVD = "--zvd" in sys.argv   # EXP-20b ring killer on the wire-rate cmds (3-impulse ZVD prefilter)
CAMFLIP = "--camflip" in sys.argv   # flip nose 180deg so the CAMERA faces the direction of travel
STRAFE = "--strafe" in sys.argv     # hold spawn yaw (camera fixed forward), translate FWD/BACK/LEFT/RIGHT in sequence
STRAFEV = argf("--strafev", 3.0); STRAFEDUR = argf("--strafedur", 4.0)  # strafe speed (m/s) + seconds per direction
STRAFEDIR = int(argf("--strafedir", -1))   # -1=cycle FWD/BACK/LEFT/RGHT; 0/1/2/3 = hold ONE direction from hover (isolate it)
STRAFESWEEP = argf("--strafesweep", 0.0)    # >0: ramp commanded lateral speed 0->STRAFEV over this many s (find the ceiling)
CIRCLE = "--circle" in sys.argv     # coordinated-turn probe: fly a constant-radius camera-forward circle (isolates the roll<->yaw
CIRCR = argf("--circr", 12.0); CIRCV = argf("--circv", 5.0)   # turn coupling -- the gate-turn regime without the full course)
ZVD_AMP = np.array([0.371, 0.476, 0.153]); _zd = int(argf("--zvddelay", 3)); ZVD_DELAY = (_zd, _zd, _zd); _zvd_buf = np.zeros((4*_zd+2, 3))  # delay = ring half-period; 11Hz@75Hz = 3 (default; old 7 was mistuned to ~5Hz). live-confirmed: pitch-gyro HF 0.60->0.11
GOV_TILT = np.tan(np.radians(50.0)) * G   # allow more horizontal authority for the brake than the 35deg cruise clamp
GATE_R = argf("--gater", 2.0); ABORT_TILT = 100.0; LEVEL_T = 2.0; GRACE = LEVEL_T + 1.5
# teacher constants (rl_finetune.py, verbatim)
KP_POS = np.array([0.6, 0.6, 2.0]); KD_POS = np.array([1.2, 1.2, 3.0])
KP_ATT = np.array([6.0, 6.0, 4.0]) * argf("--kpatt", 1.0); KD_ATT = np.array([1.2, 1.2, 0.0]) * argf("--kdatt", 1.0); KP_YAW = 4.0; KD_YAW = 0.5  # lower kpatt / raise kdatt to stop the 11Hz ring exciting
ROLL_WV0, ROLL_WV1, YAW_WV = -0.105, -0.019, -0.149
YR_CAP = 1.5; TILT_MAX = np.tan(np.radians(argf("--tiltbudget", 35.0))) * G; LEAD = 2.5   # cap commanded tilt -> bounds the live rate-loop overshoot (1.8x) that rings into runaway
vqm = json.load(open("sysid/vq_model.json")); F0 = vqm["thrust"]["f0"]; DF = vqm["thrust"]["df_dthr"]
GAIN = np.array([vqm["rate_loop"][n]["gain_G"] for n in ("roll", "pitch", "yaw")])
r = json.load(open("sysid/sim_response.json")); RG = np.array([r["rate_gain_axes"]["roll"], r["rate_gain_axes"]["pitch"], r["rate_gain_axes"]["yaw"]])
IDLE = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE

def qfix(q): return np.array([q[1], q[2], q[3], q[0]])
B = np.array([1.0, -1.0, -1.0]); bq = np.array([0.0, 1.0, 0.0, 0.0]); bqc = np.array([0.0, -1.0, 0.0, 0.0])
MIRROR = "--mirror" in sys.argv         # legacy opt-in, default OFF (gates_enu[:,1] flip line below). The lateral latflip bug is
# NOT in this adapter -- verified: attitude conjugation == clean M-conjugation (dot=1.0); omega map [1,1,-1] is correct (the unified
# all-M rebuild was REFUTED live: omega->[1,-1,-1] inverts pitch-rate damping -> tumble). Adapter is correct; latflip = plant aero.
def to_enu(ds):
    q_true = qfix(ds.quat_wxyz); Rt = quat_to_R(q_true)
    vel_w = Rt @ ds.vel_ned; om_t = ds.omega * np.array([1.0, -1.0, 1.0])
    pos, vel, q, om = ds.pos_ned * B, vel_w * B, quat_mul(quat_mul(bq, q_true), bqc), om_t * B
    if MIRROR:
        pos = pos * np.array([1.0, -1.0, 1.0]); vel = vel * np.array([1.0, -1.0, 1.0])
    return pos, vel, q, om
def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi

LATFLIP = "--nolatflip" not in sys.argv   # DEFAULT ON (06-16 live A/B): negate the lateral (world-y) accel command -- THE lateral-sign
# fix. --nomirror --latflip captured gates 1-3 live; plain input-mirror tumbled 0/6. `--nolatflip` = old buggy (positive-feedback) sign.
def ff_one(pos, vel, q, om, tgt_pos, tgt_vel, tgt_yaw):
    """Scalar port of rl_finetune.ff_batch (single drone). q = ENU quat wxyz."""
    a = KP_POS * (tgt_pos - pos) + KD_POS * (tgt_vel - vel)
    if LATFLIP:
        a[1] *= -1.0
    clamp = TILT_MAX
    if VCAP > 0:                          # hard speed governor (descent overspeed)
        vmag = float(np.linalg.norm(vel))
        if vmag > VCAP:
            a[:2] += -6.0 * (vmag - VCAP) * (vel[:2] / (np.linalg.norm(vel[:2]) + 1e-9))
            clamp = GOV_TILT              # allow harder brake tilt while over speed
    n = np.linalg.norm(a[:2])
    if n > clamp:
        a[:2] *= clamp / n
    # ACTUAL-tilt cap: the 12Hz ring detonates above ~45deg ACTUAL tilt (the rate loop overshoots
    # commands, so capping commanded tilt isn't enough). Fade the tilt-driving accel out as MEASURED
    # tilt -> TILT_HARD, so the attitude controller levels the drone before it pokes the ring.
    tilt_now = np.degrees(np.arccos(np.clip(1 - 2 * (q[1]**2 + q[2]**2), -1.0, 1.0)))
    if TILT_HARD < 90.0 and tilt_now > TILT_SOFT:
        a[:2] *= max(0.0, 1.0 - (tilt_now - TILT_SOFT) / max(TILT_HARD - TILT_SOFT, 1e-3))
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

# --- open-loop roll-rate sign probe (isolate plant/adapter lateral sign; convention-free pos log) ---
OSGN = [float(x) for x in args("--osign", "1,1,-1").split(",")]   # output [roll,pitch,yaw]. pitch un-flipped (PITCH-FIX: vq_model
# pitch gain_G/B sign-corrected to live -> teacher u2 flipped -> keep wire identical by NOT flipping pitch output). yaw stays flipped.
ROLLPROBE = "--rollprobe" in sys.argv
PROBEROLL = argf("--proberoll", 0.3)    # constant WIRE rate sent open-loop on PROBEAX (matches vq_matched wcmd[ax])
PROBEAX   = int(argf("--probeax", 0))   # 0=roll 1=pitch 2=yaw -- which body-rate axis to probe open-loop
PROBEDUR  = argf("--probedur", 1.2)     # seconds of probe after LVL settle
PROBETHR  = argf("--probethr", 0.27)    # hover thrust (wire)

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
if MIRROR: gates_enu[:, 1] *= -1.0       # match the to_enu lateral-y fix (consistent gate-relative obs); STRAIGHT rebuilds from camfwd below so it's already consistent
c.arm()
ds0 = s.get_drone(); pos0, _, _, _ = to_enu(ds0)
ANCHOR = "--noanchor" not in sys.argv   # DEFAULT ON: the TRACK_INFO ABSOLUTE origin jumps per reset (gate0 seen at
# -23 one run, +102/+886 the next) but the gate-to-gate RELATIVE layout is constant (shared world orientation, only the
# translation jumps). Trusting absolute coords -> drone arrives off-center (clips/pins gate0) or chases a gate 886m up.
# Fix: re-anchor the reliable relative layout to the SPAWN via camfwd (drone faces down-course at spawn), so every run is
# identical regardless of the absolute jump. D0 = spawn->gate0 distance (canonical real course). --noanchor = old absolute.
if ANCHOR and STRAIGHT == 0:
    camfwd_a = -(quat_to_R(qfix(ds0.quat_wxyz))[:, 0]) * B; camfwd_a[2] = 0.0
    camfwd_a /= (np.linalg.norm(camfwd_a) + 1e-9)
    D0 = argf("--d0", 23.3)
    rel = gates_enu - gates_enu[0]              # constant gate-to-gate layout (incl. z descents)
    gate0_anchor = pos0 + camfwd_a * D0; gate0_anchor[2] = pos0[2]   # gate0 straight ahead, course-centered, ~spawn alt
    course_dir = rel[1] / (np.linalg.norm(rel[1]) + 1e-9)            # track gate0->gate1 dir, for the sanity check
    gates_enu = gate0_anchor + rel
    print(f"ANCHOR: camfwd={camfwd_a.round(2)} track_dir={course_dir.round(2)} D0={D0} gate0 {gates_enu[0].round(1)} "
          f"gate5 {gates_enu[-1].round(1)} (re-anchored to spawn; abs origin jump ignored)", flush=True)
if STRAIGHT > 0:   # flat straight line along the SPAWN camera-forward dir (where the drone is pointing),
    # NOT toward TRACK_INFO gate0 (its coords jump per session -> mis-aligned course -> crab/wrong-way)
    fwd = -(quat_to_R(qfix(ds0.quat_wxyz))[:, 0]) * B; fwd[2] = 0.0; fwd /= (np.linalg.norm(fwd) + 1e-9)
    gates_enu = np.array([pos0 + fwd * 30.0 * (i + 1) for i in range(STRAIGHT)])
    gates_enu[:, 2] = pos0[2]
    print(f"STRAIGHT course along camfwd={fwd.round(2)}", flush=True)
STRAFE_DIRS = STRAFE_NAMES = None; yaw_fixed = 0.0
if STRAFE:
    cam = -(quat_to_R(qfix(ds0.quat_wxyz))[:, 0]) * B; cam[2] = 0.0; cam /= (np.linalg.norm(cam) + 1e-9)
    perp = np.array([-cam[1], cam[0], 0.0])                       # left of the camera
    nose0 = (quat_to_R(qfix(ds0.quat_wxyz))[:, 0]) * B
    yaw_fixed = float(np.arctan2(nose0[1], nose0[0]))            # hold spawn nose -> camera stays forward
    up = np.array([0.0, 0.0, 1.0])   # ENU up (+z); 6-DOF sanity: FWD/BACK/LEFT/RGHT/UP/DOWN, heading locked
    STRAFE_DIRS = [cam, -cam, perp, -perp, up, -up]; STRAFE_NAMES = ['FWD ', 'BACK', 'LEFT', 'RGHT', 'UP  ', 'DOWN']
    print(f"STRAFE: cam={cam.round(2)} perp={perp.round(2)} yaw_fixed={np.degrees(yaw_fixed):.0f} v={STRAFEV} dur={STRAFEDUR}s/dir", flush=True)
traj = GateTrajectory(np.vstack([pos0[None, :], gates_enu]), v_cruise=VDES,
                      tilt_budget_deg=(argf("--tiltbudget", 45.0) if STRAIGHT > 0 else 35.0), c_drag=0.057, margin=0.6, vz_max=VZMAX)
NG = len(gates_enu)
print(f"spawn_enu={pos0.round(1)} gates_enu[0]={gates_enu[0].round(1)} gates_enu[-1]={gates_enu[-1].round(1)} "
      f"arc_len={traj.s_max:.1f}m NG={NG} vdes={VDES} maxw={MAXW} thrmax={THRMAX}", flush=True)

from aigp.recorder import Recorder
rec = Recorder(s, script="vq_deploy_teacher_real", mode="rate",
               notes="pure analytic spline teacher on REAL TRACK_INFO course",
               extra_meta={"cmd_layout": ["wx", "wy", "wz", "thrust"]})
flog = ftm.from_args(sys.argv, RG, run_name="teacher_real", store=s)
t0 = time.time(); gi = 0; last = -1; maxgi = 0; prev_rates = np.zeros(3); t_prev = t0; sname = "LVL"
if STRAFE:
    MAX_T = STRAFESWEEP + LEVEL_T + 3 if STRAFESWEEP > 0 else (STRAFEDUR + LEVEL_T + 1 if STRAFEDIR >= 0 else STRAFEDUR * 6 + LEVEL_T + 1)
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
    elif ROLLPROBE:   # open-loop constant wire rate on PROBEAX, NO feedback -> measure plant response sign vs the cmd
        rates = np.zeros(3); rates[PROBEAX] = PROBEROLL; thr = PROBETHR; sname = "PRB "
        c.send_attitude_target(rates, thr); phase = "PRB"
        rec.log([float(rates[0]), float(rates[1]), float(rates[2]), float(thr)])
        if (t - LEVEL_T) > PROBEDUR:
            R = quat_to_R(qfix(ds.quat_wxyz)); hd = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
            print(f"  PROBE DONE ax={PROBEAX} cmd={PROBEROLL:+.2f} pe=[{pe[0]:+.2f},{pe[1]:+.2f},{pe[2]:+.2f}] "
                  f"omega={ds.omega.round(2)} hdg={hd:+.0f} pos_ned={ds.pos_ned.round(2)}", flush=True); break
    else:
        if CIRCLE:   # coordinated camera-forward circle: tgt follows a constant-radius arc, nose anti-tangent
            tau = t - LEVEL_T; w = CIRCV / CIRCR; th = w * tau; ramp = min(1.0, tau / 2.0)
            tgt_pos = pos0 + CIRCR * np.array([np.sin(th), 1.0 - np.cos(th), 0.0])
            tgt_vel = CIRCV * ramp * np.array([np.cos(th), np.sin(th), 0.0])
            tgt_yaw = float(np.arctan2(tgt_vel[1], tgt_vel[0])) + np.pi   # camera-forward (anti-tangent)
            sname = "CIRC"
        elif STRAFE:   # hold yaw (camera fixed), translate in the current direction (velocity-only, no spline)
            if STRAFESWEEP > 0:   # ramp commanded lateral speed 0->STRAFEV slowly to find the runaway ceiling
                sidx = STRAFEDIR if STRAFEDIR >= 0 else 2; ramp = min(1.0, (t - LEVEL_T) / STRAFESWEEP)
            elif STRAFEDIR >= 0:
                sidx = STRAFEDIR; ramp = min(1.0, (t - LEVEL_T) / 0.8)   # single direction from hover
            else:
                sidx = int((t - LEVEL_T) / STRAFEDUR) % len(STRAFE_DIRS); ramp = min(1.0, ((t - LEVEL_T) % STRAFEDUR) / 0.8)
            tgt_vel = STRAFE_DIRS[sidx] * STRAFEV * ramp; tgt_pos = pe.copy(); tgt_yaw = yaw_fixed
            sname = STRAFE_NAMES[sidx] + (f"{STRAFEV*ramp:.1f}" if STRAFESWEEP > 0 else "")
        else:
            s_d = traj.nearest_s(pe); ref = traj.sample(min(s_d + LEAD, traj.s_max))
            ramp = min(1.0, (t - LEVEL_T) / max(RAMP, 1e-3))   # gentle speed onset -> never enter the runaway regime
            tgt_pos = ref["pos"]; tgt_vel = float(ref["v"]) * ramp * ref["tang"]
            tgt_yaw = float(ref["yaw"]) + (np.pi if CAMFLIP else 0.0); sname = "TCH"
        u = ff_one(pe, ve, qe, ome, tgt_pos, tgt_vel, tgt_yaw)
        thr = float((u[0]+1)/2*THRMAX)
        rates = np.array([OSGN[0]*u[1], OSGN[1]*u[2], OSGN[2]*u[3]]) * MAXW   # ENU->VQ wire output sign map (sweepable, default [1,-1,-1])
        if ZVD:   # EXP-20b ring killer BEFORE slew
            _zvd_buf = np.roll(_zvd_buf, 1, axis=0); _zvd_buf[0] = rates
            rates = np.array([ZVD_AMP[0]*_zvd_buf[0, ax] + ZVD_AMP[1]*_zvd_buf[ZVD_DELAY[ax], ax]
                              + ZVD_AMP[2]*_zvd_buf[2*ZVD_DELAY[ax], ax] for ax in range(3)])
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
        Rq = quat_to_R(qe); nose = Rq[:, 0]
        vh = np.degrees(np.arctan2(ve[1], ve[0])); beta = ((np.degrees(np.arctan2(nose[1], nose[0])) - vh + 180) % 360) - 180
        print(f"  t={t:4.1f} {sname} x={pe[0]:6.1f} y={pe[1]:6.1f} z={pe[2]:5.1f} v={vmag:4.1f} velhd={vh:4.0f} beta={beta:4.0f} tilt={tilt:3.0f}", flush=True)
    time.sleep(0.01)
idle(); rec.close()
if flog is not None: flog.close()
verdict = "TEACHER FLIES REAL VQ" if maxgi >= NG*0.7 else ("partial" if maxgi >= 2 else "FAILED")
print(f"DONE: reached {maxgi}/{NG} gates  z_final={pe[2]:.1f}  [{verdict}]", flush=True)
