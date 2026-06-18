"""rl_finetune.py -- PPO fine-tune on the matched sim, warm-started from FF/DAgger (COR-127 step 5).

Flags: --steps --envs --dagger --bc_epochs --vdes --warmstart 0/1 --curve --recurrent --tag NAME
Pipeline: matched-corridor VecEnv (vq_rate) -> (Recurrent)PPO (log_std_init -2.5 when warm so
exploration ~ the tiny rate-action scale; hover thrust action-bias) -> BC+DAgger warm-start on FF
demos (MLP only; expert reads ground truth via VecEnvAdapter.env) -> learn() with gate-chaining
reward -> periodic gate-eval. Saves ft_<tag>.zip.
"""
import sys, json
import numpy as np
import torch, torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from sim.envs.gate_race_env import GateRaceEnv
from sim.envs.vec_env_adapter import VecEnvAdapter
from sim.tracks import Track, GateState
from sim.dynamics.numpy_quad import POS, VEL, QUAT, OMEGA, quat_to_rotmat_batch
from sim.gate_traj import GateTrajectory
_TRAJS = []                                       # per-env spline planners, populated by make_env

G = 9.81
def argf(f, d): return float(sys.argv[sys.argv.index(f)+1]) if f in sys.argv else d
def args(f, d): return sys.argv[sys.argv.index(f)+1] if f in sys.argv else d
STEPS = int(argf("--steps", 5_000_000)); NENV = int(argf("--envs", 64))
DAGGER = int(argf("--dagger", 4)); BC_EPOCHS = int(argf("--bc_epochs", 20)); NG = 6
VDES = argf("--vdes", 4.0); WARM = int(argf("--warmstart", 1)); CURVE = "--curve" in sys.argv
REALCOURSE = "--realcourse" in sys.argv     # train on the ACTUAL VQ1 track (live TRACK_INFO) not the synthetic arc
RCPATH = args("--rcpath", "sysid/vq1_track_real_course.json")   # optional json override of the embedded coords
VDES_WARM = argf("--vdes_warm", VDES)   # curriculum: BC/DAgger demos at an easier speed (teacher 48/48 @v4-curve)
LAT = argf("--latency", 0.0); TLAG = argf("--thrust_lag", 0.0)
REC = "--recurrent" in sys.argv; TAG = args("--tag", "v1"); DR = "--dr" in sys.argv
if REC:
    WARM = 0  # BC-into-LSTM not wired; recurrent variant leans on PPO
M = json.load(open("sysid/vq_model.json"))
GAIN = np.array([M["rate_loop"][n]["gain_G"] for n in ("roll", "pitch", "yaw")])
F0 = M["thrust"]["f0"]; DF = M["thrust"]["df_dthr"]
MAXW = argf("--maxw", 17.45)   # action-authority cap: max wire rate cmd (rad/s); deploy chain must use the same value
THRMAX = argf("--thrmax", 1.0)  # thrust-authority cap: u0=+1 -> thr THRMAX (teacher p99 ~0.33; live bang-bang thr 1.0 was the 06-12 killer)
ZVD = "--zvd" in sys.argv       # ZVD-shape the wire rate cmds (EXP-20b ring killer on the policy path); deploy must match
SLEW = argf("--slew", 0.0)      # rad/s^2 wire-rate slew cap (RING-06: prevents the hard kick that triggers the nonlinear 11 Hz ring); 0=off
SCATTER = "--scatter" in sys.argv  # miss-recovery starts: random gate, behind 1-34 m, racing yaw (training envs only)
DT = 1.0 / 72.0
# EP_STEPS: synthetic=1080 (~15s). REALCOURSE is 164m -> scale with vdes so the episode COVERS the full course + margin.
# FIDELITY (06-17): the old fixed 1080 TRUNCATED the real course mid-flight -> the in-sim teacher "capped" at 2-3 gates by
# TIMEOUT (not dynamics; lower vdes -> fewer gates in 15s) -> RL learned a failing teacher. Now ~36s at v5, ~60s at v3.
EP_STEPS = int(argf("--ep_steps", int(185.0 / max(VDES, 2.0) / DT * 1.25) if REALCOURSE else 1080))
# weathervane-FF (corner_speed teacher term, WV-DYNAMIC coeffs), ENU env frame signs
ROLL_WV0, ROLL_WV1, YAW_WV = -0.105, -0.019, -0.149
YR_CAP = 1.5   # rad/s yaw-rate cap (corner_speed)
KPXY = argf("--kpxy", 1.4); KDXY = argf("--kdxy", 3.5)   # clean-teacher lateral PD (ported from live 06-17): KPXY cuts cross-track
KP_POS = np.array([KPXY, KPXY, 2.0]); KD_POS = np.array([KDXY, KDXY, 3.0])   # lag, KDXY damps the jog overshoot -> 0-collision threading
KP_ATT = np.array([6.0, 6.0, 4.0]); KD_ATT = np.array([1.2, 1.2, 0.0]); KP_YAW = 4.0; KD_YAW = 0.5
TILT_MAX = np.tan(np.radians(35)) * G
HOVER_U0 = 2*((F0+G)/(-DF))/THRMAX - 1
# Racing reward: gate_passage (per-gate, the objective) + delta gate_progress (loiter-safe guidance)
# + small body-rate penalty + crash. NO dense per-step survival terms (heading_alignment/speed_bonus
# are gameable on a finite course -> reward-hacking: survive+point-at-gate without threading).
WSIDE = argf("--ws", 0.0)   # TRANSFER-06: sideslip penalty weight; 0 = off (campaign default)
WAPER = argf("--wa", 0.0)   # TRANSFER-08: aperture-proximity bonus weight (Gaussian, sigma 5 m)
WCNTR = argf("--wc", 0.0)   # gate-centering weight (existing, near-plane lateral penalty)
RAD_START = argf("--rad_start", 1.36 if REALCOURSE else 1.0)  # FIDELITY (06-17): REALCOURSE gate aperture = live W/H 2.72m -> radius
RAD_END   = argf("--rad_end",   1.36 if REALCOURSE else 1.0)  # 1.36 (the old 1.0 was TIGHTER than live -> teacher clipped gate0 @ lat 1.34)
RAD_STEPS = int(argf("--rad_steps", 0))   # anneal duration in PPO steps; 0 = no curriculum (rad=RAD_START)
HIST   = int(argf("--hist", 0))           # NeuroBEM history stack: prev-action steps in obs (handoff fix)
LEANOBS = "--leanobs" in sys.argv
CAMFWD = "--camfwd" in sys.argv   # nose along +tangent (camera-forward); flip teacher tgt_yaw by pi (anti-tangent->tangent)         # 32-dim lean obs: drop motor/w-h/arena/prev-action; keep [0:12]+lookahead[20:24]+history
WSMOOTH = argf("--wsmooth", 0.0)          # NLM rec: action smoothness weight (anti 11Hz ring); try 1.4
BRAKE = argf("--brake", 0.5)              # aim() brake-taper slope; bigger = brake harder per meter
VZMAX = argf("--vzmax", 1e9)              # controlled-sink: cap spline descent rate (m/s); slow forward speed on descents (1e9=off)
GATED = argf("--gated", 3.0)              # clean-teacher: perpendicular gate-crossing waypoints (gate +/- leg_normal*GATED) -> straight,
VGATE = argf("--vgate", 5.0); SLOWD = argf("--slowd", 7.0)   # centered plane crossing; + gate-approach speed cap within SLOWD m (gentle, centered)
GATEFIXED = "--gatefixed" in sys.argv  # experiment: gates share ONE fixed course-axis orientation. Default OFF -- per-leg threads better in-sim
ZLIFT = argf("--zlift", 1.2)             # raise gate-z targets to cancel the analytic z-loop SAG (~1.2m z-hang, KP/KD-only no integral);
# live masked this via the TRACK_INFO zlift -> without it the sim teacher arrives ~1.2m low at the descent gates -> clips the tight radius
V_GATE = argf("--vgate", 1.5)             # min commit speed near gate; lower = arrive slower
# Residual learning (HANDOFF 06-14 #1): action = clip(FF_spline_anchor + tanh(net)*delta_scale).
# The anchor guarantees stability so PPO can only add bounded deltas -> CANNOT erode it. Deploy mirrors.
RESIDUAL = "--residual_ff" in sys.argv
DELTA_SCALE = argf("--delta_scale", 0.15)  # residual delta magnitude cap
# Asymmetric privileged critic (HANDOFF #2): critic sees true per-env plant params, actor only obs.
ASYM = "--asymmetric_critic" in sys.argv
DR_WIDTH = argf("--dr_width", 0.4)         # per-env plant DR width feeding the privileged channel
WSPEED = argf("--wspeed", 0.0)   # 06-15: penalize |v| above VCAP_TRAIN (keep policy below the ~v5 live runaway threshold)
VCAP_TRAIN = argf("--vcap", 4.0) # m/s speed cap for the penalty
REWARD = {"gate_progress": 2.0, "gate_passage": 15.0, "gate_offset": 0.5,
          "body_rate": 0.005, "crash_penalty": 10.0, "sideslip": WSIDE,
          "aperture": WAPER, "gate_centering": WCNTR, "action_smoothness": WSMOOTH,
          "speed": WSPEED, "speed_cap": VCAP_TRAIN}


def mat_to_quat_batch(m):
    N = m.shape[0]; t = m[:, 0, 0]+m[:, 1, 1]+m[:, 2, 2]; q = np.zeros((N, 4))
    cA = t > 0; cB = (~cA) & (m[:, 0, 0] >= m[:, 1, 1]) & (m[:, 0, 0] >= m[:, 2, 2])
    cC = (~cA) & (~cB) & (m[:, 1, 1] >= m[:, 2, 2]); cD = (~cA) & (~cB) & (~cC)
    sA = np.sqrt(np.maximum(t+1, 1e-12))*2
    q[cA, 0] = .25*sA[cA]; q[cA, 1] = (m[cA, 2, 1]-m[cA, 1, 2])/sA[cA]; q[cA, 2] = (m[cA, 0, 2]-m[cA, 2, 0])/sA[cA]; q[cA, 3] = (m[cA, 1, 0]-m[cA, 0, 1])/sA[cA]
    sB = np.sqrt(np.maximum(1+m[:, 0, 0]-m[:, 1, 1]-m[:, 2, 2], 1e-12))*2
    q[cB, 0] = (m[cB, 2, 1]-m[cB, 1, 2])/sB[cB]; q[cB, 1] = .25*sB[cB]; q[cB, 2] = (m[cB, 0, 1]+m[cB, 1, 0])/sB[cB]; q[cB, 3] = (m[cB, 0, 2]+m[cB, 2, 0])/sB[cB]
    sC = np.sqrt(np.maximum(1+m[:, 1, 1]-m[:, 0, 0]-m[:, 2, 2], 1e-12))*2
    q[cC, 0] = (m[cC, 0, 2]-m[cC, 2, 0])/sC[cC]; q[cC, 1] = (m[cC, 0, 1]+m[cC, 1, 0])/sC[cC]; q[cC, 2] = .25*sC[cC]; q[cC, 3] = (m[cC, 1, 2]+m[cC, 2, 1])/sC[cC]
    sD = np.sqrt(np.maximum(1+m[:, 2, 2]-m[:, 0, 0]-m[:, 1, 1], 1e-12))*2
    q[cD, 0] = (m[cD, 1, 0]-m[cD, 0, 1])/sD[cD]; q[cD, 1] = (m[cD, 0, 2]+m[cD, 2, 0])/sD[cD]; q[cD, 2] = (m[cD, 1, 2]+m[cD, 2, 1])/sD[cD]; q[cD, 3] = .25*sD[cD]
    return q/np.linalg.norm(q, axis=1, keepdims=True)


def ff_batch(S, tgt_pos, tgt_vel, tgt_yaw):
    pos = S[:, POS]; vel = S[:, VEL]; q = S[:, QUAT]; om = S[:, OMEGA]
    a = KP_POS*(tgt_pos-pos) + KD_POS*(tgt_vel-vel)
    n = np.linalg.norm(a[:, :2], axis=1); sc = np.where(n > TILT_MAX, TILT_MAX/np.maximum(n, 1e-9), 1.0)
    a[:, 0] *= sc; a[:, 1] *= sc
    R = quat_to_rotmat_batch(q); yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    t = a + np.array([0, 0, G]); zb = t/np.linalg.norm(t, axis=1, keepdims=True)
    xc = np.stack([np.cos(tgt_yaw), np.sin(tgt_yaw), np.zeros_like(tgt_yaw)], axis=1)
    yb = np.cross(zb, xc); yb /= np.linalg.norm(yb, axis=1, keepdims=True); xb = np.cross(yb, zb)
    q_des = mat_to_quat_batch(np.stack([xb, yb, zb], axis=2))
    w0, x0, y0, z0 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]; w1, x1, y1, z1 = q_des[:, 0], q_des[:, 1], q_des[:, 2], q_des[:, 3]
    qe = np.stack([w0*w1+x0*x1+y0*y1+z0*z1, w0*x1-x0*w1-y0*z1+z0*y1, w0*y1+x0*z1-y0*w1-z0*x1, w0*z1-x0*y1+y0*x1-z0*w1], axis=1)
    qe[qe[:, 0] < 0] *= -1
    w = KP_ATT*(2*qe[:, 1:4]) - KD_ATT*om
    w[:, 2] = KP_YAW*(((tgt_yaw-yaw+np.pi) % (2*np.pi))-np.pi) - KD_YAW*om[:, 2]
    # weathervane-FF: cancel the sideslip moment (validated live in corner_speed; ENU signs)
    vb = np.einsum("nji,nj->ni", R, vel)            # body velocity = R^T v
    w[:, 0] += (ROLL_WV0 + ROLL_WV1*vb[:, 0]) * vb[:, 1]
    w[:, 2] += -YAW_WV * vb[:, 1]
    cos_t = np.maximum(R[:, 2, 2], 0.5); c = (G + a[:, 2])/cos_t
    # Collective cap that MAINTAINS LIFT while braking. The old flat c_max=10 at tilt>40 starved
    # vertical thrust (c*cos_t < g) -> the drone fell during the brake -> gravity ran the speed away
    # on the descent (06-15 brake diagnosis: thrust opposed velocity correctly but lift collapsed).
    # New cap = max(18, hover-lift/cos_t * 1.25): always >= the collective needed to hover at this
    # tilt, with 25% brake headroom; thrmax bounds the final thrust downstream.
    c_max = np.maximum(18.0, (G / cos_t) * 1.25)
    c = np.minimum(c, c_max)
    thr = np.clip((F0+c)/(-DF), 0.0, 1.0)
    u = np.empty((S.shape[0], 4)); u[:, 0] = np.clip(2*thr/THRMAX-1, -1, 1); u[:, 1:4] = np.clip(w/GAIN/MAXW, -1, 1)
    cap = YR_CAP/abs(GAIN[2])/MAXW   # corner_speed yaw-rate cap (resonance guard)
    u[:, 3] = np.clip(u[:, 3], -cap, cap)
    return u.astype(np.float32)


def aim(S, gates, gp, trajs=None):
    # SPLINE-FOLLOWING teacher (TRACK-01 port). Each env has an open arc-length spline through
    # spawn + gates (built in make_env). Drone projects to spline (nearest_s), reference is
    # LEAD m ahead. tgt_pos = ref["pos"], tgt_vel = ref["v"] * ref["tang"] (tilt-budget speed),
    # tgt_yaw = ref["yaw"] (nose anti-tangent, camera-forward). Smooth tangent + analytic
    # curvature + tilt-budget speed schedule -> drone follows a smoother trajectory than
    # point-to-point PD; live tilt budget pre-scheduled before each gate (predictive braking).
    n_envs = len(gp)
    LEAD = 2.5
    T = trajs if trajs is not None else _TRAJS   # explicit per-env splines (anti global-clobber)
    out_pos = np.zeros((n_envs, 3))
    out_vel = np.zeros((n_envs, 3))
    out_yaw = np.zeros(n_envs)
    if len(T) < n_envs:                               # fallback if trajs not built (smoke/test path)
        idx = np.clip(gp, 0, NG-1); gate = gates[np.arange(n_envs), idx]
        dxy = (gate - S[:, POS])[:, :2]; dist = np.linalg.norm(dxy, axis=1, keepdims=True) + 1e-6
        nose_dir = (dxy if CAMFWD else -dxy) / dist
        return gate, np.zeros((n_envs, 3)), np.arctan2(nose_dir[:, 1], nose_dir[:, 0])
    for i in range(n_envs):
        traj = T[i]
        s_d = traj.nearest_s(S[i, POS])
        s_ref = float(min(s_d + LEAD, traj.s_max))
        ref = traj.sample(s_ref)
        out_pos[i] = ref["pos"]
        out_vel[i] = float(ref["v"]) * ref["tang"]
        out_yaw[i] = float(ref["yaw"]) + (np.pi if CAMFWD else 0.0)
        # clean-teacher gate-slowdown: cap target speed near the current gate -> gentle, centered crossing
        if VGATE > 0:
            gi = int(np.clip(gp[i], 0, gates.shape[1] - 1)); d_gate = float(np.linalg.norm(gates[i, gi] - S[i, POS]))
            if d_gate < SLOWD:
                sp = float(np.linalg.norm(out_vel[i]))
                if sp > VGATE: out_vel[i] *= VGATE / sp
    return out_pos, out_vel, out_yaw


def _yaw_quat(th):
    return np.array([np.cos(th/2), 0.0, 0.0, np.sin(th/2)])


# Real VQ1 course (captured live via probe_track.py -> ENCAP_TRACK_INFO, 2026-06-15).
# Course frame = env ENU (x forward, z up), spawn-relative: x=-x_ned, y=y_ned, z=-z_ned+1.
# Row 0 = spawn; rows 1-6 = the 6 gates. Real track: ~164 m, DESCENDS 26 m, spacing 23-37 m
# (vs the synthetic CURVE arc: flat, 9-18 m spacing). Embedded so a pod git-pull is self-contained;
# --rcpath json (sysid/vq1_track_real_course.json) overrides if present.
VQ1_REAL_COURSE = np.array([
    [0.0,          0.0,         1.0],          # spawn
    [23.29761696, -0.39988279,  1.0524707],    # gate 0
    [46.89339828, -2.49997067, -4.04752922],   # gate 1
    [74.59339905,  1.20002925, -12.64752865],  # gate 2
    [111.49339294, -5.09997082, -23.54752922], # gate 3
    [135.49339294, -0.79997069, -24.33514023], # gate 4
    [159.19338989, -4.39997053, -24.94752884], # gate 5
])

def _real_course():
    """(spawn(3,), gates(NG,3)) in env ENU course frame. JSON at RCPATH overrides embedded."""
    import os
    arr = VQ1_REAL_COURSE
    if os.path.exists(RCPATH):
        arr = np.array(json.load(open(RCPATH))["gates"], float)
    assert arr.shape[0] == NG + 1, f"real course needs {NG+1} rows (spawn+{NG} gates), got {arr.shape}"
    return arr[0], arr[1:]


def make_env(n, seed, vq_path="sysid/vq_model.json", train=True, gate_radius=None):
    rng = np.random.default_rng(seed); tracks = []; G3 = []
    rc_spawn, rc_gates = (_real_course() if REALCOURSE else (np.array([0.0, 0.0, 1.0]), None))
    rc_ceiling = 10.0
    if REALCOURSE:
        # The real sim descends 26m into a pit; the env floors at z<=0 and ceils at 10. Lift the whole
        # course so its LOWEST point clears the floor by FLOOR_MARGIN, and raise the ceiling to fit.
        # Absolute z is invisible to the gate-relative obs -- only the 26m descent geometry matters.
        FLOOR_MARGIN = 3.0
        allz = np.concatenate([[rc_spawn[2]], rc_gates[:, 2]])
        lift = FLOOR_MARGIN - float(allz.min())
        rc_spawn = rc_spawn + np.array([0.0, 0.0, lift])
        rc_gates = rc_gates + np.array([0.0, 0.0, lift])
        rc_ceiling = float(allz.max()) + lift + 6.0
    for _ in range(n):
        sp = rng.uniform(9, 18); amp = rng.uniform(1.0, 3.0); ph = rng.uniform(0, 6.28)   # deploy courses use space 14-16; 9-13 left them OOD-long (grid weak cells)
        gs = []
        if REALCOURSE:
            # fixed VQ1 track (live TRACK_INFO): descends 26 m, spacing 23-37 m. FIDELITY FIX (06-17): the LIVE
            # gates ALL share ONE fixed orientation along the course axis (TRACK_INFO quat identical for every gate,
            # normal=course dir) -- NOT the per-leg heading. _gate_normal = local-x, so a fixed course-axis yaw makes
            # every gate plane perpendicular to travel (matches live; the per-leg version clipped descent gate3 in sim).
            cyaw = float(np.arctan2(rc_gates[-1][1] - rc_spawn[1], rc_gates[-1][0] - rc_spawn[0]))   # course primary axis
            for i in range(NG):
                gyaw = cyaw if GATEFIXED else float(np.arctan2((rc_gates[i] - (rc_gates[i-1] if i > 0 else rc_spawn))[1],
                                                               (rc_gates[i] - (rc_gates[i-1] if i > 0 else rc_spawn))[0]))
                gs.append(GateState(position=rc_gates[i].copy(), orientation=_yaw_quat(gyaw)))
        elif CURVE:
            hd = 0.0; pos = np.array([8.0, 0.0, 1.0]); turn = rng.uniform(0.12, 0.28)*rng.choice([-1, 1])
            for i in range(NG):
                hd += turn; pos = pos + sp*np.array([np.cos(hd), np.sin(hd), 0.0])
                z = 1.0 + 0.6*np.sin(i*0.7+ph)
                gs.append(GateState(position=np.array([pos[0], pos[1], z]), orientation=_yaw_quat(hd)))
        else:
            for i in range(NG):
                gs.append(GateState(position=np.array([8.0+sp*i, amp*np.sin(i*0.9+ph), 1.0]),
                                    orientation=np.array([1.0, 0, 0, 0])))
        tracks.append(Track(gates=gs, name=("vq1" if REALCOURSE else "c"), start_position=rc_spawn.copy())); G3.append([g.position for g in gs])
    scat = SCATTER and train
    env = GateRaceEnv(n_envs=n, dt=DT, max_steps=EP_STEPS, action_mode="vq_rate",
                      max_body_rate=MAXW, vq_max_thrust=THRMAX, vq_zvd=ZVD,
                      vq_slew=(SLEW if SLEW > 0 else None),
                      start_behind_max=34.0 if scat else None,
                      start_pos=(rc_spawn if REALCOURSE else None),   # spawn at the descending course's top, not the z=1 default
                      vq_model_path=vq_path, tracks=tracks, random_gate_start=scat,
                      start_behind_dist=1.0, start_vel_std=0.4, start_att_std=0.08, start_omega_std=0.3,
                      gate_collision=True, gate_passage_radius=(gate_radius if gate_radius is not None else RAD_START),
                      arena_bounds=(float(np.abs(rc_gates[:, :2]).max()) + 20.0 if REALCOURSE else 120.0),  # 164m course needs >120; DEPLOY obs arena_extent must mirror /10
                      ceiling=rc_ceiling,   # realcourse: ~35m to fit the lifted descent (default 10)
                      reward_weights=REWARD, vq_latency_s=LAT, vq_thrust_lag_s=TLAG,
                      n_action_history=HIST, lean_obs=LEANOBS,
                      privileged_obs=ASYM, dr_width=(DR_WIDTH if ASYM else 0.0))
    # Per-env open spline trajectories (TRACK-01 06-09: tilt-budget speed-scheduled controller; STRAIGHT
    # 8.6 m/s clean live). Prepend spawn so spline covers the launch leg too.
    global _TRAJS
    spawn3 = rc_spawn.copy()
    def _gate_wps(seq):   # clean-teacher: surround each gate with approach/exit points along the leg normal so the
        pts = [spawn3]; prev = spawn3   # spline crosses each gate plane STRAIGHT + perpendicular + centered (no corner-cut)
        for g in seq:
            g = np.asarray(g, float); nrm = (g - prev).copy(); nrm[2] = 0.0; nl = np.linalg.norm(nrm)
            nrm = nrm / nl if nl > 1e-6 else np.array([1.0, 0.0, 0.0])
            gz = g + np.array([0.0, 0.0, ZLIFT])   # raise gate z to cancel the z-hang sag (matched-sim z is true; drone sags ~ZLIFT below)
            pts += ([gz - nrm * GATED, gz, gz + nrm * GATED] if GATED > 0 else [gz]); prev = g
        return np.array(pts)
    _TRAJS = [GateTrajectory(_gate_wps(g3_env),
                              v_cruise=VDES, tilt_budget_deg=35.0, c_drag=0.057, margin=0.6, vz_max=VZMAX)
              for g3_env in G3]
    env._trajs = _TRAJS   # bind this env's splines so the residual anchor survives global clobber
    return env, np.array(G3)


class ResidualVecEnv(VecEnvAdapter):
    """Policy outputs a bounded delta; the env steps with anchor(FF spline)+delta.
    The analytic anchor guarantees stability, so PPO can only add tanh-bounded
    corrections and CANNOT erode it (HANDOFF 06-14 #1). The deploy script must
    mirror this exact anchor+delta math."""
    def __init__(self, env, gates):
        super().__init__(env)
        self._gates = gates

    def step_async(self, deltas):
        S = self.env._states
        anchor = ff_batch(S, *aim(S, self._gates, self.env._gate_indices, trajs=self.env._trajs))
        applied = np.clip(anchor + np.tanh(deltas) * DELTA_SCALE, -1.0, 1.0)
        super().step_async(applied.astype(np.float32))


def policy_mean(model, obs_np):
    obs_t, _ = model.policy.obs_to_tensor(obs_np)
    feat = model.policy.extract_features(obs_t)
    latent_pi = model.policy.mlp_extractor.forward_actor(feat)
    return model.policy.action_net(latent_pi)


def eval_gates(model, seed=7, NE=16, gate_radius=None):
    # TRANSFER-10 fix: eval at the SAME radius as current training (passed by callback during
    # curriculum); else falls back to RAD_END (the final/target radius, NOT RAD_START -- prior
    # bug was eval always at RAD_START so curriculum eval lied)
    eval_rad = gate_radius if gate_radius is not None else RAD_END
    env, gates = make_env(NE, seed, train=False, gate_radius=eval_rad)
    venv = ResidualVecEnv(env, gates) if RESIDUAL else VecEnvAdapter(env); obs = venv.reset()
    peak = np.zeros(NE, int); fr = np.zeros(NE, bool); lstm = None; starts = np.ones(NE, bool)
    for _ in range(EP_STEPS + 200):
        if REC:
            act, lstm = model.predict(obs, state=lstm, episode_start=starts, deterministic=True)
        else:
            act, _ = model.predict(obs, deterministic=True)
        peak = np.maximum(peak, np.where(fr, peak, env._gates_passed))
        obs, r, dones, infos = venv.step(act); starts = dones; fr |= dones
        if fr.all():
            break
    return np.maximum(peak, env._gates_passed)


def teacher_eval(seed=7, NE=16, gate_radius=None):
    # Eval the analytic ff_batch+aim teacher itself (the DAgger demonstrator) in the matched sim.
    # A gate collision TERMINATES the episode -> peak gates == NG iff the teacher threaded CLEAN (0 collisions).
    eval_rad = gate_radius if gate_radius is not None else RAD_END
    env, gates = make_env(NE, seed, train=False, gate_radius=eval_rad)
    venv = VecEnvAdapter(env); venv.reset()
    peak = np.zeros(NE, int); fr = np.zeros(NE, bool)
    for _ in range(EP_STEPS + 200):
        S = env._states
        u = ff_batch(S, *aim(S, gates, env._gate_indices, trajs=env._trajs))
        peak = np.maximum(peak, np.where(fr, peak, env._gates_passed))
        _, _, dones, _ = venv.step(u); fr |= dones
        if fr.all():
            break
    return np.maximum(peak, env._gates_passed)


def teacher_trace(seed=7, gate_radius=1.36):
    # NE=1 raw-env trace. The env auto-resets done sub-envs INSIDE step() (clearing reason), so capture the reason
    # from BEFORE the step and keep a ring buffer of the last states -> SEE the exact crash mode on the descent leg.
    from sim.envs.gate_race_env import _gate_normal, GATE_RACE_TERM_NAMES
    from collections import deque
    env, gates = make_env(1, seed, train=False, gate_radius=gate_radius)
    env.reset(); prev_p = 0; ring = deque(maxlen=10)
    for step in range(EP_STEPS):
        S = env._states; gi = int(env._gate_indices[0]); trk = env._tracks[0]
        gate = trk.gates[gi % trk.num_gates]; nrm = _gate_normal(gate); rel = S[0, POS] - gate.position
        along = float(np.dot(rel, nrm)); latv = rel - along*nrm; lat = float(np.linalg.norm(latv))
        q = S[0, QUAT]; tilt = np.degrees(np.arccos(np.clip(1 - 2*(q[2]**2 + q[3]**2), -1, 1)))
        wmax = float(np.max(np.abs(S[0, OMEGA]))); v = float(np.linalg.norm(S[0, VEL]))
        ring.append(f"  s{step:4d} gi={gi} pos={S[0,POS].round(1)} gate={gate.position.round(1)} lat={lat:.2f} along={along:+5.1f} z={S[0,POS][2]:.1f} v={v:.1f} tilt={tilt:.0f} wmax={wmax:.1f}")
        u = ff_batch(S, *aim(S, gates, env._gate_indices, trajs=env._trajs))
        before = int(env._termination_reasons[0])
        obs, r, term, trunc, info = env.step(u)
        if env._gates_passed[0] != prev_p and env._gates_passed[0] > prev_p:
            print(f"  PASS gate{prev_p}->{int(env._gates_passed[0])} step{step} v={v:.1f} lat={lat:.2f}", flush=True); prev_p = int(env._gates_passed[0])
        if step % 40 == 0:
            print(ring[-1], flush=True)
        if bool(term[0]) or bool(trunc[0]):
            rsn = int(env._termination_reasons[0]) or before   # reason may be cleared by the in-step reset
            print(f"  *** END step{step} reason={rsn} ({GATE_RACE_TERM_NAMES.get(rsn,'?')}) -- last 10 steps: ***", flush=True)
            for ln in ring: print(ln, flush=True)
            break
    return


def main():
    global VDES, LAT, TLAG
    torch.manual_seed(0)
    print(f"FT[{TAG}]: vdes={VDES} warm={WARM} curve={CURVE} realcourse={REALCOURSE} rec={REC} lat={LAT} tlag={TLAG} maxw={MAXW} thrmax={THRMAX} zvd={ZVD} slew={SLEW} ws={WSIDE} wa={WAPER} wc={WCNTR} wspeed={WSPEED}@vcap{VCAP_TRAIN} hist={HIST} leanobs={LEANOBS} camfwd={CAMFWD} wsm={WSMOOTH} rad_curr={RAD_START}->{RAD_END}@{RAD_STEPS} residual={RESIDUAL} asym={ASYM} dscale={DELTA_SCALE} drw={DR_WIDTH} steps={STEPS}", flush=True)
    if "--tracegate" in sys.argv:   # NE=1 trace of the teacher (debug the descent-gate clip), then exit
        teacher_trace(gate_radius=1.36); return
    if "--evalteacher" in sys.argv:   # verify the clean teacher in-sim, then exit (no training)
        for rad in (RAD_END, 1.36):   # sim default radius + the live VQ aperture (1.36m)
            pk = teacher_eval(NE=16, gate_radius=rad)
            print(f"TEACHER EVAL (rad={rad}): gates {pk.mean():.2f}/{NG}  CLEAN(full-course) {int((pk>=NG).sum())}/16  dist={np.bincount(pk, minlength=NG+1).tolist()}", flush=True)
        return
    env, gates = make_env(NENV, 1)
    venv = ResidualVecEnv(env, gates) if RESIDUAL else VecEnvAdapter(env)
    # Fine-tuning from a BC/DAgger warm-start: tiny exploration (rate actions are ~0.01-0.03; std must
    # not swamp them), no entropy bonus, gentle LR + tight trust region, few epochs -> don't destroy the
    # warm-start. From-scratch (warm=0) needs real exploration, so log_std 0 + entropy + larger LR.
    log_std = argf("--log_std", -4.0 if WARM else 0.0)
    LR = argf("--lr", 1e-4 if WARM else 3e-4); ENT = argf("--ent", 0.0 if WARM else 0.01)
    EPO = int(argf("--epo", 4 if WARM else 8)); CLIP = argf("--clip", 0.1 if WARM else 0.2)
    TKL = argf("--target_kl", 0.0)  # 0 = disabled; e.g. 0.02 enables KL early-stop
    print(f"FT[{TAG}] PPO: log_std={log_std} lr={LR} ent={ENT} epo={EPO} clip={CLIP} target_kl={TKL}", flush=True)
    if REC:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO("MlpLstmPolicy", venv, n_steps=512, batch_size=8192, n_epochs=EPO, gamma=0.999,
                             gae_lambda=0.95, clip_range=CLIP, ent_coef=ENT, learning_rate=LR,
                             policy_kwargs=dict(net_arch=[128, 128], log_std_init=log_std, lstm_hidden_size=128),
                             device="cuda", verbose=1)
    else:
        ppo_kwargs = dict(n_steps=512, batch_size=8192, n_epochs=EPO, gamma=0.999,
                          gae_lambda=0.95, clip_range=CLIP, ent_coef=ENT, learning_rate=LR,
                          policy_kwargs=dict(net_arch=[128, 128, 128], log_std_init=log_std),
                          device="cuda" if torch.cuda.is_available() else "cpu", verbose=1)
        if TKL > 0:
            ppo_kwargs["target_kl"] = TKL   # KL early-stop guard against destroying warm-start
        if ASYM:
            from control.policies.asymmetric import AsymmetricActorCriticPolicy
            ppo_policy = AsymmetricActorCriticPolicy   # Dict obs -> actor=obs, critic=obs+privileged
        else:
            ppo_policy = "MlpPolicy"
        model = PPO(ppo_policy, venv, **ppo_kwargs)
    with torch.no_grad():
        if ASYM:
            pass   # asymmetric actor_mlp is near-zero-init -> policy starts at the FF anchor
        elif RESIDUAL:
            model.policy.action_net.bias[:] = 0.0   # delta starts ~0 -> policy = anchor (no hover bias)
        else:
            model.policy.action_net.bias[:] = torch.tensor([HOVER_U0, 0, 0, 0], dtype=model.policy.action_net.bias.dtype)

    best_dag = -1.0   # defined even when BC/DAgger is skipped (residual/asym anchor IS the warm start)
    if WARM and not RESIDUAL and not ASYM:
        vdes_run = VDES; VDES = VDES_WARM   # teacher demos/relabels at the curriculum speed
        print(f"FT: FF demos + BC... (vdes_warm={VDES_WARM})", flush=True)
        obs = venv.reset(); X = []; Y = []
        for _ in range(EP_STEPS):
            # teacher targets the env's CURRENT gate (_gate_indices), not gates_passed --
            # with scattered starts the episode begins at a random gate, so gates_passed (0)
            # would aim the teacher backward across the course (poisoned demos)
            S = env._states; u = ff_batch(S, *aim(S, gates, env._gate_indices))
            X.append(obs.copy()); Y.append(u.copy()); obs, r, d, info = venv.step(u)
        X = np.concatenate(X); Y = np.concatenate(Y)
        opt = torch.optim.Adam(model.policy.parameters(), 1e-3)
        def bc(X, Y, epochs):
            # per-dim standardized MSE (dagger_v2's fix): rate channels are ~0.02 vs thrust ~0.5;
            # raw MSE under-weights them and the ~-2.5 rate-loop gain amplifies the error.
            ystd = torch.tensor(Y.std(0) + 1e-6, dtype=torch.float32, device=model.device)
            Yt = torch.tensor(Y, dtype=torch.float32, device=model.device); n = len(X); bs = 8192
            for ep in range(epochs):
                perm = np.random.permutation(n)
                for j in range(0, n, bs):
                    idx = perm[j:j+bs]; opt.zero_grad()
                    l = (((policy_mean(model, X[idx]) - Yt[idx]) / ystd) ** 2).mean()
                    l.backward(); opt.step()
            return l.item()
        l = bc(X, Y, BC_EPOCHS); pk = eval_gates(model)
        print(f"  BC: mse {l:.4f} | gates {pk.mean():.1f}/{NG} fin {int((pk>=NG).sum())}/16", flush=True)
        best_dag = pk.mean()   # a late DAgger round can collapse (dr_field r6: 6.0 -> 0.0); keep best-by-eval
        if best_dag > 0: model.save(f"ft_{TAG}_best")
        def dr_model_path(rd):
            # domain randomization: perturbed plant per DAgger round (policy must not exploit
            # the nominal model's precision -- DEPLOY-01 transfer gap)
            rng = np.random.default_rng(1000 + rd)
            mv = json.loads(json.dumps(M))
            mm = mv.get("rate_loop_mimo")
            if mm is not None:
                A = np.array(mm["A"]); Bm = np.array(mm["B"])
                for _ in range(50):
                    # MonoRace-width scatter (POD_SCALE_PLAN: bumped diag ±20% -> ±40% for VQ2 push).
                    # Need the policy to hold its basin across real-plant-sized variation.
                    off = np.ones((3, 3)) + (rng.uniform(-0.6, 0.6, (3, 3)) * (1 - np.eye(3)))
                    dia = 1 + rng.uniform(-0.40, 0.40, 3)
                    A2 = A * off * dia[:, None]; B2 = Bm * off * (1 + rng.uniform(-0.30, 0.30, 3))[:, None]
                    if np.abs(np.linalg.eigvals(A2)).max() < 0.995:
                        break
                mm["A"] = A2.tolist(); mm["B"] = B2.tolist()
            r2 = mv.get("rate_loop_2nd")
            if r2 is not None:                       # scatter the underdamped loop (RING-01): damping + freq + gain
                for _ in range(50):
                    A1 = np.array(r2["A1"]); A2 = np.array(r2["A2"])
                    B0 = np.array(r2["B0"]); B1 = np.array(r2["B1"])
                    dia = 1 + rng.uniform(-0.20, 0.20, 3)   # POD_SCALE_PLAN: pole-mag ±10% -> ±20%
                    A1s = A1 * dia[:, None]; A2s = A2 * (1 + rng.uniform(-0.20, 0.20, 3))[:, None]
                    g = (1 + rng.uniform(-0.30, 0.30, 3))[:, None]
                    B0s = B0 * g; B1s = B1 * g
                    C = np.zeros((6, 6)); C[0:3, 0:3] = A1s; C[0:3, 3:6] = A2s; C[3:6, 0:3] = np.eye(3)
                    if np.abs(np.linalg.eigvals(C)).max() < 0.985:   # base |p|=0.92 (RING-04); allow scatter up to live-ring stiffness
                        r2["A1"] = A1s.tolist(); r2["A2"] = A2s.tolist()
                        r2["B0"] = B0s.tolist(); r2["B1"] = B1s.tolist()
                        break
            mv["weathervane"]["wv_coeff"] *= 1 + rng.uniform(-0.5, 0.5)
            if "weathervane_v2" in mv:               # scatter the ACTIVE wv surface (speed-dependent, 06-10)
                for ax in ("roll", "yaw"):
                    for k in ("a", "b"):
                        mv["weathervane_v2"][ax][k] *= 1 + rng.uniform(-0.5, 0.5)
            if "dr_rate_disturbance" in mv:          # inject the MEASURED unmodeled-moment field
                # (tilt-scaled sustained rate biases, BRAKE-WV-01) -- the DEPLOY-02 snap mechanism;
                # scale randomized so the policy tolerates the field's magnitude range, not a value
                mv["dr_rate_disturbance"]["scale"] = float(rng.uniform(0.5, 2.0))
            mv["thrust"]["df_dthr"] *= 1 + rng.uniform(-0.15, 0.15)
            mv["thrust"]["f0"] *= 1 + rng.uniform(-0.10, 0.10)
            for kk in ("Dx", "Dy"):
                mv["drag_linear_body"][kk] *= 1 + rng.uniform(-0.4, 0.4)
            for kk in ("qx", "qy", "qz", "Dz"):
                mv["drag_quadratic_body"][kk] *= 1 + rng.uniform(-0.4, 0.4)
            path = f"/tmp/vq_dr_{rd}.json"
            json.dump(mv, open(path, "w"))
            return path

        lat0, tlag0 = LAT, TLAG
        for rd in range(1, DAGGER+1):
            vqp = dr_model_path(rd) if DR else "sysid/vq_model.json"
            if DR:   # per-round timing DR: live chain has ~12-22 ms latency + ~85 ms thrust lag
                # POD_SCALE_PLAN: extended latency to 0-30 ms range, finer steps
                rngt = np.random.default_rng(2000 + rd)
                LAT = float(rngt.uniform(0.0, 0.030))
                TLAG = float(rngt.uniform(0.0, 0.12))
            ed, gd = make_env(NENV, 100+rd, vq_path=vqp); vd = VecEnvAdapter(ed); o = vd.reset(); Xn = []; Yn = []
            for _ in range(EP_STEPS):
                S = ed._states; uff = ff_batch(S, *aim(S, gd, ed._gate_indices))
                act, _ = model.predict(o, deterministic=True); Xn.append(o.copy()); Yn.append(uff.copy()); o, r, dn, inf = vd.step(act)
            X = np.concatenate([X, np.concatenate(Xn)]); Y = np.concatenate([Y, np.concatenate(Yn)])
            LAT, TLAG = lat0, tlag0   # eval on the nominal-timing env
            l = bc(X, Y, BC_EPOCHS); pk = eval_gates(model)
            print(f"  DAgger r{rd}: D={len(X)} mse {l:.4f} | gates {pk.mean():.1f}/{NG} fin {int((pk>=NG).sum())}/16", flush=True)
            if pk.mean() > best_dag:
                best_dag = pk.mean(); model.save(f"ft_{TAG}_best")
                print(f"  DAgger r{rd}: new best {best_dag:.2f} -> ft_{TAG}_best.zip", flush=True)
        VDES = vdes_run   # PPO + eval at the target speed

    class GateEval(BaseCallback):
        def __init__(s, every): super().__init__(); s.every = every; s.last = 0; s.best = best_dag
        def _on_step(s):
            # TRANSFER-10 radius curriculum: linearly anneal gate_passage_radius RAD_START -> RAD_END
            # over RAD_STEPS PPO steps. Wider gates first = looser basin of attraction = more
            # robust to sim-to-sim drift (user concern: in-env perfect but live miss grows; this
            # widens the threading window the policy MUST get into so it can't overfit precision).
            if RAD_STEPS > 0 and RAD_START != RAD_END:
                frac = min(1.0, s.num_timesteps / RAD_STEPS)
                new_rad = RAD_START + (RAD_END - RAD_START) * frac
                env.gate_passage_radius = new_rad
            if s.num_timesteps - s.last >= s.every:
                s.last = s.num_timesteps; pk = eval_gates(model, gate_radius=env.gate_passage_radius)
                rad_str = f" rad={env.gate_passage_radius:.2f}" if RAD_STEPS > 0 else ""
                print(f"  [t={s.num_timesteps}]{rad_str} gates {pk.mean():.1f}/{NG} fin {int((pk>=NG).sum())}/16", flush=True)
                if pk.mean() > s.best:   # PPO can degrade the warm-start; keep the best-by-eval policy
                    s.best = pk.mean(); model.save(f"ft_{TAG}_best")
                    print(f"  [t={s.num_timesteps}] new best {s.best:.2f} -> ft_{TAG}_best.zip", flush=True)
            return True

    if STEPS > 0:
        print(f"FT[{TAG}]: learn {STEPS}...", flush=True)
        model.learn(total_timesteps=STEPS, progress_bar=False, callback=GateEval(int(argf("--eval_every", 250_000))))
    else:
        print(f"FT[{TAG}]: STEPS=0 -> save DAgger warm-start only (no PPO)", flush=True)
    pk = eval_gates(model)
    print(f"FT[{TAG}] DONE: gates {pk.mean():.1f}/{NG} max {pk.max()} fin {int((pk>=NG).sum())}/16 dist {np.bincount(np.clip(pk,0,NG),minlength=NG+1)}", flush=True)
    model.save(f"ft_{TAG}")
    print(f"saved ft_{TAG}.zip", flush=True)


if __name__ == "__main__":
    main()
