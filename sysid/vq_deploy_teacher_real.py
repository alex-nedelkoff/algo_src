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
CAMFLIP = "--nocamflip" not in sys.argv   # DEFAULT ON: nose holds spawn heading so the CAMERA faces travel (no 180deg turn-around)
STRAFE = "--strafe" in sys.argv     # hold spawn yaw (camera fixed forward), translate FWD/BACK/LEFT/RIGHT in sequence
STRAFEV = argf("--strafev", 3.0); STRAFEDUR = argf("--strafedur", 4.0)  # strafe speed (m/s) + seconds per direction
STRAFEDIR = int(argf("--strafedir", -1))   # -1=cycle FWD/BACK/LEFT/RGHT; 0/1/2/3 = hold ONE direction from hover (isolate it)
STRAFESWEEP = argf("--strafesweep", 0.0)    # >0: ramp commanded lateral speed 0->STRAFEV over this many s (find the ceiling)
CIRCLE = "--circle" in sys.argv     # coordinated-turn probe: fly a constant-radius camera-forward circle (isolates the roll<->yaw
CIRCR = argf("--circr", 12.0); CIRCV = argf("--circv", 5.0)   # turn coupling -- the gate-turn regime without the full course)
ZVD_AMP = np.array([0.371, 0.476, 0.153]); _zd = int(argf("--zvddelay", 3)); ZVD_DELAY = (_zd, _zd, _zd); _zvd_buf = np.zeros((4*_zd+2, 3))  # delay = ring half-period; 11Hz@75Hz = 3 (default; old 7 was mistuned to ~5Hz). live-confirmed: pitch-gyro HF 0.60->0.11
GOV_TILT = np.tan(np.radians(50.0)) * G   # allow more horizontal authority for the brake than the 35deg cruise clamp
GATE_R = argf("--gater", 2.0); ABORT_TILT = 100.0; LEVEL_T = 2.0; GRACE = LEVEL_T + 1.5
HOME = "--home" in sys.argv         # gate-homing override (default OFF -- superseded by the perpendicular-spline waypoints)
DHOME = argf("--dhome", 8.0)        # start homing this many m before the gate plane (along the normal)
HOME_LEAD = argf("--homelead", 2.5) # aim point past the centroid along the normal -> drive THROUGH centered (no stall)
GATED = argf("--gated", 3.0)        # spline perpendicular-crossing: insert approach/exit waypoints gate +/- gnorm*GATED so the
# spline is locally STRAIGHT and perpendicular through each (vertical) gate plane -> centered crossing, no corner-cut. 0=off.
RECOVER = "--norecover" not in sys.argv  # PIN-RECOVERY (default ON): if the drone wedges on a gate frame (low v + ongoing gate
V_STALL = argf("--vstall", 0.9)          # collisions), back off along -gnorm and re-approach -> a graze is no longer terminal (kills the grind).
RECOVER_BACK = argf("--recoverback", 5.0); RECOVER_DUR = argf("--recoverdur", 1.3); MAX_RETRY = int(argf("--maxretry", 3))
VGATE = argf("--vgate", 5.0)             # cap target speed within SLOWD of a gate plane -> gentler contact (no crash-reset) + better centering
SLOWD = argf("--slowd", 7.0)             # gate-slowdown zone (m before the plane). VGATE=0 disables.
# teacher constants (rl_finetune.py, verbatim) -- lateral (xy) PD now tunable to cut the ~0.7m cross-track LAG that clips gates
KPXY = argf("--kpxy", 1.4); KDXY = argf("--kdxy", 3.5)   # KPXY cuts cross-track lag; KDXY damps the jog overshoot -> 0-collision threading (06-17 clean 6/6)
KP_POS = np.array([KPXY, KPXY, 2.0]); KD_POS = np.array([KDXY, KDXY, 3.0])
KP_ATT = np.array([6.0, 6.0, 4.0]) * argf("--kpatt", 1.0); KD_ATT = np.array([1.2, 1.2, 0.0]) * argf("--kdatt", 1.0); KP_YAW = 4.0; KD_YAW = 0.5  # lower kpatt / raise kdatt to stop the 11Hz ring exciting
ROLL_WV0, ROLL_WV1, YAW_WV = -0.105, -0.019, -0.149
YR_CAP = 1.5; TILT_MAX = np.tan(np.radians(argf("--tiltbudget", 35.0))) * G; LEAD = argf("--lead", 2.6)   # spline lookahead (m); lower = aim closer to the gate = less corner-cut lateral miss on curves (gate threading)
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
camfwd_g = -(quat_to_R(qfix(ds0.quat_wxyz))[:, 0]) * B; camfwd_g[2] = 0.0; camfwd_g /= (np.linalg.norm(camfwd_g) + 1e-9)
gnorm = -camfwd_g   # travel / gate-normal direction (course goes -camfwd; gates are vertical, normal = this horizontal axis)
ANCHOR = "--noanchor" not in sys.argv   # ALWAYS ON: the TRACK_INFO ABSOLUTE origin changes EVERY run (gate0 observed at
# -23/+88/+886, z 0/+25/+886) -> raw coords are unusable. The gate-to-gate RELATIVE layout is constant. So rebuild the
# course from the reliable relative layout, anchored to the SPAWN along -camfwd (course dir). --noanchor = old raw absolute.
if ANCHOR and STRAIGHT == 0:
    D0 = argf("--d0", 23.3)
    rel = gates_enu - gates_enu[0]              # constant gate-to-gate layout (incl. z descents + true lateral jogs)
    # COURSE IS ALONG -camfwd (verified live 06-16 via gate viz: camfwd=+x but real TRACK_INFO gates at -x).
    gate0_anchor = pos0 + gnorm * D0; gate0_anchor[2] = pos0[2]   # gate0 along travel (-camfwd), ~spawn alt
    gates_enu = gate0_anchor + rel
    print(f"ANCHOR: gnorm={gnorm.round(2)} D0={D0} gate0 {gates_enu[0].round(1)} gate5 {gates_enu[-1].round(1)}", flush=True)
ZLIFT = argf("--zlift", 1.5)   # raise all gate-z targets (m, ENU up). Drone was passing UNDER gates -> TRACK_INFO z may be the
if ZLIFT != 0.0 and STRAIGHT == 0:   # gate base (not aperture center) and/or the analytic z-loop sags ~1m below z_ref (z-hang).
    gates_enu[:, 2] += ZLIFT
    print(f"ZLIFT: raised gate z by {ZLIFT}m -> gate0_z={gates_enu[0,2]:.1f} gate5_z={gates_enu[-1,2]:.1f}", flush=True)
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
NG = len(gates_enu)
# Build the spline waypoints. With GATED>0, surround each gate centroid with approach/exit points along the gate normal
# (gnorm) so the spline crosses each (vertical) gate plane STRAIGHT + perpendicular + centered (no corner-cut on the y-jogs).
if GATED > 0 and STRAIGHT == 0:
    wps = [pos0]
    for gc in gates_enu:
        wps += [gc - gnorm * GATED, gc, gc + gnorm * GATED]   # gnorm = travel dir: approach, center, exit
    spline_wps = np.array(wps)
    print(f"SPLINE: perpendicular gate waypoints (GATED={GATED}m), {len(spline_wps)} pts", flush=True)
else:
    spline_wps = np.vstack([pos0[None, :], gates_enu])
traj = GateTrajectory(spline_wps, v_cruise=VDES,
                      tilt_budget_deg=(argf("--tiltbudget", 45.0) if STRAIGHT > 0 else 35.0), c_drag=0.057, margin=0.6, vz_max=VZMAX)
print(f"spawn_enu={pos0.round(1)} gates_enu[0]={gates_enu[0].round(1)} gates_enu[-1]={gates_enu[-1].round(1)} "
      f"arc_len={traj.s_max:.1f}m NG={NG} vdes={VDES} maxw={MAXW} thrmax={THRMAX}", flush=True)

from aigp.recorder import Recorder
rec = Recorder(s, script="vq_deploy_teacher_real", mode="rate",
               notes="pure analytic spline teacher on REAL TRACK_INFO course",
               extra_meta={"cmd_layout": ["wx", "wy", "wz", "thrust"]})
flog = ftm.from_args(sys.argv, RG, run_name="teacher_real", store=s)
if flog is not None:   # overlay gate positions on the 3D NED trajectory (telemetry world frame = NED = ds.pos_ned)
    flog.log_gates(gates_enu * B, name="Gaim", color=(0, 255, 0), radii=GATE_R)  # ANCHORED centers the drone aims at (green, 2m = the geometric-pass radius)
    flog.log_gates(gates_ned, name="Gtrk", color=(255, 0, 255), radii=1.0)       # RAW TRACK_INFO absolute gates (magenta, may be origin-jumped)
    print(f"VIZ gates: anchored(green) gate0_ned={(gates_enu[0]*B).round(1)} | trackinfo(magenta) gate0_ned={gates_ned[0].round(1)}", flush=True)
    for _gi in range(NG): print(f"   GATE{_gi} enu={gates_enu[_gi].round(1)}", flush=True)
t0 = time.time(); gi = 0; last = -1; maxgi = 0; prev_rates = np.zeros(3); t_prev = t0; sname = "LVL"
# GROUND-TRUTH watchers (sim judge + collisions) -- the geometric gi counter (dist<GATE_R) is an ESTIMATE; these are real.
rs0 = s.get_race(); prev_agi = int(rs0["active_gate_index"]) if rs0 else -1; prev_lgt = (rs0 or {}).get("last_gate_time", 0)
_c0 = s.get_collision(); prev_cseq = _c0[1] if _c0 else 0; judge_passes = 0; ncoll = 0; last_coll_t = -1.0
hgi = 0   # homing target gate index (advances when the drone crosses each gate plane along gnorm)
stall_t0 = -1.0; recover_until = -1.0; recover_anchor = pos0.copy(); retry_n = 0   # pin-recovery state
was_live = False; finished = False   # finish detection: at 6/6 the course ENDS (odometry cuts, record page) -> race finishes
if STRAFE:
    MAX_T = STRAFESWEEP + LEVEL_T + 3 if STRAFESWEEP > 0 else (STRAFEDUR + LEVEL_T + 1 if STRAFEDIR >= 0 else STRAFEDUR * 6 + LEVEL_T + 1)
while time.time()-t0 < MAX_T:   # run on the REAL judge (below), NOT the geometric gi (which trips ~2m early and cut gate5 short)
    ds = s.get_drone()
    if ds is None: time.sleep(0.01); continue
    t = time.time()-t0; pe, ve, qe, ome = to_enu(ds)
    vmag = float(np.linalg.norm(ds.vel_ned))
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
            ramp = min(1.0, (t - LEVEL_T) / max(RAMP, 1e-3))   # gentle speed onset -> never enter the runaway regime
            # advance the homing gate once the drone crosses its plane (along the gate normal gnorm)
            while hgi < NG - 1 and float(np.dot(gates_enu[hgi] - pe, gnorm)) < 0.0:
                hgi += 1; retry_n = 0; stall_t0 = -1.0   # cleanly crossed a gate -> reset recovery state
            gate_c = gates_enu[hgi]; to_plane = float(np.dot(gate_c - pe, gnorm))
            cyaw = float(np.arctan2(gnorm[1], gnorm[0])) + (np.pi if CAMFLIP else 0.0)
            # PIN-RECOVERY: detect a wedge (slow + ongoing gate collisions near a gate) -> back off along -gnorm, re-approach
            if RECOVER and t > recover_until and t > GRACE and vmag < V_STALL and (t - last_coll_t) < 0.3 and 0 < to_plane < DHOME:
                if stall_t0 < 0: stall_t0 = t
                if t - stall_t0 > 0.4:   # confirmed wedge
                    retry_n += 1
                    if retry_n > MAX_RETRY:   # give up on this gate -> skip past it (avoid an infinite back-off loop)
                        print(f"  ~~~ PIN-RECOVER: GATE{hgi} unthreadable after {MAX_RETRY} tries -> skip ~~~", flush=True)
                        hgi = min(hgi + 1, NG - 1); retry_n = 0; stall_t0 = -1.0
                    else:
                        recover_until = t + RECOVER_DUR; recover_anchor = pe - gnorm * RECOVER_BACK; recover_anchor[2] = gate_c[2]
                        print(f"  ~~~ PIN-RECOVER #{retry_n} GATE{hgi} t={t:.1f} v={vmag:.1f}: back off {RECOVER_BACK}m + re-approach ~~~", flush=True)
                        stall_t0 = -1.0
            elif vmag > V_STALL:
                stall_t0 = -1.0   # moving fine -> clear the stall timer
            if t < recover_until:   # RECOVERING: drive back off the gate frame, then normal guidance resumes
                tgt_pos = recover_anchor; tgt_vel = -gnorm * 1.5; tgt_yaw = cyaw; sname = "RCVR"
            elif HOME and to_plane < DHOME:   # GATE-HOMING: aim THROUGH the centroid perpendicular to the (vertical) gate plane
                tgt_pos = gate_c + gnorm * HOME_LEAD          # centerline point just past the gate -> centered + keeps drive (no stall)
                tgt_vel = gnorm * (VDES * ramp)               # cross perpendicular -> kills the curve's lateral velocity at the plane
                tgt_yaw = cyaw; sname = "HOME"
            else:   # between gates: follow the spline path
                s_d = traj.nearest_s(pe); ref = traj.sample(min(s_d + LEAD, traj.s_max))
                tgt_pos = ref["pos"]; tgt_vel = float(ref["v"]) * ramp * ref["tang"]
                tgt_yaw = float(ref["yaw"]) + (np.pi if CAMFLIP else 0.0); sname = "TCH"
            if VGATE > 0 and sname != "RCVR" and 0 < to_plane < SLOWD:   # gate-slowdown: gentle contact + centering time
                sp = float(np.linalg.norm(tgt_vel))
                if sp > VGATE: tgt_vel = tgt_vel * (VGATE / sp)
                sname = sname + "s"
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
    dist = float(np.linalg.norm(gates_enu[min(gi, NG-1)] - pe))   # 3D dist to current gate (clamped; loop runs on the judge now)
    if dist < GATE_R and gi < NG:
        gi += 1; maxgi = max(maxgi, gi)
        print(f"  >>> GATE {gi}/{NG} (geom dist<{GATE_R}) t={t:.1f} v={vmag:.1f} tilt={tilt:.0f}", flush=True)
    # --- GROUND TRUTH: sim judge (active_gate_index / last_gate_time) = the REAL "gate threaded" broadcast ---
    rs = s.get_race()
    if rs is not None:
        agi = int(rs["active_gate_index"]); lgt = rs.get("last_gate_time", prev_lgt)
        if agi > prev_agi and lgt > 0:   # REAL forward gate pass (ignore agi->0 / lgt=-1 race resets)
            judge_passes += 1
            print(f"  >>>JUDGE GATE PASS<<<: active_gate_index {prev_agi}->{agi} last_gate_time={lgt} t={t:.1f} v={vmag:.1f} (geom gi={gi})", flush=True)
        elif agi < prev_agi:
            print(f"  (race reset: active_gate_index {prev_agi}->{agi} t={t:.1f} v={vmag:.1f})", flush=True)
        prev_agi = agi; prev_lgt = lgt
        if judge_passes >= NG:   # all gates ticked per the sim judge -> done
            print(f"  *** ALL {NG} GATES THREADED (judge) t={t:.1f} ***", flush=True); finished = True; break
        # FINISH: threading the LAST gate ENDS the course (odometry cuts, record page) -> the race finishes (no 6th tick fires).
        live = bool(rs.get("race_live", False)); fin = rs.get("race_finish_ns", 0) or 0
        if live: was_live = True
        if (fin > 0) or (was_live and not live and judge_passes >= NG - 1):
            print(f"  *** COURSE COMPLETE / FINISH t={t:.1f}: judge_passes={judge_passes}, race ended (finish_ns={fin}, live={live}) -> 6/6 ***", flush=True)
            finished = True; break
    # --- collisions (the COLLISION broadcast): id 1001=gate, 1002=env; impulse = hit magnitude ---
    cev, cseq = s.get_collision()
    if cseq != prev_cseq and cev is not None:
        ncoll += 1
        if (cev.get("impulse", 0.0) > 0.4 or cev.get("id") == 1001) and (t - last_coll_t) > 0.25:   # throttle grind spam
            ng_i = int(np.argmin(np.linalg.norm(gates_enu - pe, axis=1))); off = pe - gates_enu[ng_i]   # miss vector vs nearest gate
            print(f"  !!! COLLISION id={cev.get('id')} imp={cev.get('impulse'):.2f} t={t:.1f} nearest=GATE{ng_i} off(along,lat,vert? enu xyz)=[{off[0]:+.1f},{off[1]:+.1f},{off[2]:+.1f}] !!!", flush=True)
            last_coll_t = t
        prev_cseq = cseq
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
verdict = "COURSE COMPLETE 6/6 (FINISH)" if finished else ("TEACHER FLIES REAL VQ" if judge_passes >= NG*0.7 else ("partial" if judge_passes >= 2 else "FAILED"))
print(f"DONE: JUDGE gate passes={judge_passes}/{NG} finished={finished} (geom maxgi={maxgi})  collisions={ncoll}  z_final={pe[2]:.1f}  [{verdict}]", flush=True)
