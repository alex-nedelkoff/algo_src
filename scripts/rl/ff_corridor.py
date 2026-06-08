"""ff_corridor.py -- STEP 1: does the analytic coord-turn FF race the corridor on the MATCHED sim?

Drives the ff_teacher FF controller (state->desired collective-accel + body-rates) through
GateRaceEnv(action_mode="vq_rate") = VQMatchedDynamics(ENU) with the DIRECTIONAL roll weathervane.
Two interface adapters vs ff_teacher's old numpy_quad path:
  (1) THRUST: desired collective accel c -> VQ normalized thrust  thr = (f0 + c)/(-df)
  (2) RATE:   the VQ rate loop achieves omega = gain_G*wcmd, so command wcmd = omega_des/gain_G
              (ff_teacher fed a feedforward mixer with unit gain; here the rate loop has gain ~-2.5)
FF aims each env at its next un-passed gate at v_des, nose-to-velocity. Reports gates passed,
sideslip, completion. Pass = the analytic demonstrator survives the correct/harder aero -> DAgger.
Usage: python ff_corridor.py [--v 4] [--envs 16] [--gates 6] [--collide]
"""
import sys, json
import numpy as np
from sim.envs.gate_race_env import GateRaceEnv
from sim.tracks import Track, GateState
from sim.dynamics.numpy_quad import POS, VEL, QUAT, OMEGA

G = 9.81
def argf(f, d): return float(sys.argv[sys.argv.index(f)+1]) if f in sys.argv else d

VDES = argf("--v", 4.0); NENV = int(argf("--envs", 16)); NG = int(argf("--gates", 6))
COLLIDE = "--collide" in sys.argv
MODEL = json.load(open("sysid/vq_model.json"))
GAIN = np.array([MODEL["rate_loop"][n]["gain_G"] for n in ("roll", "pitch", "yaw")])
F0 = MODEL["thrust"]["f0"]; DF = MODEL["thrust"]["df_dthr"]
MAXW = 17.45
KP_POS = np.array([0.6, 0.6, 2.0]); KD_POS = np.array([1.2, 1.2, 3.0])
KP_ATT = np.array([6.0, 6.0, 4.0]); KD_ATT = np.array([1.2, 1.2, 0.0]); KP_YAW = 4.0; KD_YAW = 0.5
TILT_MAX = np.tan(np.radians(35)) * G


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def mat_to_quat(R):
    t = R[0, 0]+R[1, 1]+R[2, 2]
    if t > 0:
        s = 0.5/np.sqrt(t+1); w = 0.25/s; x = (R[2, 1]-R[1, 2])*s; y = (R[0, 2]-R[2, 0])*s; z = (R[1, 0]-R[0, 1])*s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2*np.sqrt(1+R[0, 0]-R[1, 1]-R[2, 2]); w = (R[2, 1]-R[1, 2])/s; x = 0.25*s; y = (R[0, 1]+R[1, 0])/s; z = (R[0, 2]+R[2, 0])/s
    elif R[1, 1] > R[2, 2]:
        s = 2*np.sqrt(1+R[1, 1]-R[0, 0]-R[2, 2]); w = (R[0, 2]-R[2, 0])/s; x = (R[0, 1]+R[1, 0])/s; y = 0.25*s; z = (R[1, 2]+R[2, 1])/s
    else:
        s = 2*np.sqrt(1+R[2, 2]-R[0, 0]-R[1, 1]); w = (R[1, 0]-R[0, 1])/s; x = (R[0, 2]+R[2, 0])/s; y = (R[1, 2]+R[2, 1])/s; z = 0.25*s
    q = np.array([w, x, y, z]); return q/np.linalg.norm(q)


def desired_attitude(a, yaw):
    t = a + np.array([0, 0, G]); zb = t/np.linalg.norm(t)
    xc = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    yb = np.cross(zb, xc); yb /= np.linalg.norm(yb); xb = np.cross(yb, zb)
    return np.column_stack([xb, yb, zb])


def att_err_quat(qc, qd):
    w0, x0, y0, z0 = qc; w1, x1, y1, z1 = qd
    qe = np.array([w0*w1+x0*x1+y0*y1+z0*z1, w0*x1-x0*w1-y0*z1+z0*y1,
                   w0*y1+x0*z1-y0*w1-z0*x1, w0*z1-x0*y1+y0*x1-z0*w1])
    if qe[0] < 0:
        qe = -qe
    return 2*qe[1:4]


def ff_action(state, tgt_pos, tgt_vel, tgt_yaw):
    """Return VQ_RATE normalized action u[-1,1]^4 for one env state."""
    pos = state[POS]; vel = state[VEL]; q = state[QUAT]; om = state[OMEGA]
    a = KP_POS*(tgt_pos-pos) + KD_POS*(tgt_vel-vel)
    ah = a[:2]; n = np.linalg.norm(ah)
    if n > TILT_MAX:
        a[:2] = ah/n*TILT_MAX
    Rc = quat_to_R(q); yaw = np.arctan2(Rc[1, 0], Rc[0, 0])
    q_des = mat_to_quat(desired_attitude(a, tgt_yaw))
    w_des = KP_ATT*att_err_quat(q, q_des) - KD_ATT*om                 # desired body rate (rad/s)
    w_des[2] = KP_YAW*((tgt_yaw-yaw+np.pi) % (2*np.pi)-np.pi) - KD_YAW*om[2]
    cos_t = max(quat_to_R(q)[2, 2], 0.5); c = (G + a[2])/cos_t        # desired collective accel (body-up)
    thr = np.clip((F0 + c)/(-DF), 0.0, 1.0)                           # VQ thrust inversion
    wcmd = w_des / GAIN                                               # rate-loop gain inversion
    u = np.empty(4)
    u[0] = np.clip(2*thr - 1.0, -1, 1)
    u[1:4] = np.clip(wcmd/MAXW, -1, 1)
    return u


def build_corridor(n_gates, rng):
    gates = []
    for i in range(n_gates):
        x = 8.0 + 10.0*i
        y = 2.0*np.sin(i*1.1)                       # lateral jog -> excites the weathervane
        gates.append(GateState(position=np.array([x, y, 1.0]),   # spawn height (env resets z=1.0)
                               orientation=np.array([1.0, 0.0, 0.0, 0.0])))   # all same orientation (+x)
    start = np.array([0.0, 0.0, 1.0])
    return Track(gates=gates, name="corridor", start_position=start)


def main():
    track = build_corridor(NG, np.random.default_rng(0))
    env = GateRaceEnv(n_envs=NENV, dt=0.01, max_steps=2500, action_mode="vq_rate",
                      vq_model_path="sysid/vq_model.json", track=track,
                      random_gate_start=False, start_behind_dist=1.0, start_vel_std=0.0,
                      start_att_std=0.0, start_omega_std=0.0, gate_collision=COLLIDE,
                      gate_passage_radius=1.0, v_max=30.0, arena_bounds=120.0)
    obs, info = env.reset(seed=0)
    gates_xyz = np.array([g.position for g in track.gates])
    peak = np.zeros(NENV, dtype=int)      # per-env peak gates passed (pre auto-reset)
    frozen = np.zeros(NENV, dtype=bool)   # stop counting an env after its first episode ends
    betas = [[] for _ in range(NENV)]
    term_reason = np.zeros(NENV, dtype=int)
    for step in range(2500):
        S = env._states.copy()
        gp = env._gates_passed.copy()
        u = np.zeros((NENV, 4))
        for i in range(NENV):
            idx = min(int(gp[i]), NG-1)
            gate = gates_xyz[idx]
            d = gate - S[i, POS]; dxy = d[:2]; dist = np.linalg.norm(dxy) + 1e-6
            tgt_vel = np.array([VDES*dxy[0]/dist, VDES*dxy[1]/dist, 0.0])
            tgt_yaw = np.arctan2(dxy[1], dxy[0])
            u[i] = ff_action(S[i], gate, tgt_vel, tgt_yaw)
            if not frozen[i]:
                peak[i] = max(peak[i], int(gp[i]))
                v = S[i, VEL][:2]
                if np.linalg.norm(v) > 0.5:
                    nose = quat_to_R(S[i, QUAT])[:2, 0]
                    b = np.degrees(np.arctan2(nose[0]*v[1]-nose[1]*v[0], nose@v)); betas[i].append(abs(b))
        if step % 200 == 0:
            i0 = 0
            print(f"  [t={step*0.01:4.1f}] env0 pos=({S[i0,0]:5.1f},{S[i0,1]:5.1f},{S[i0,2]:4.1f}) "
                  f"|v|={np.linalg.norm(S[i0,VEL]):4.1f} tgtgate={min(int(gp[i0]),NG-1)} passed={gp[i0]}", flush=True)
        obs, r, term, trunc, info = env.step(u)
        peak = np.maximum(peak, np.where(frozen, peak, env._gates_passed))
        newly = (term | trunc) & (~frozen)
        if "term_reason" in info:
            term_reason = np.where(newly, info.get("term_reason", term_reason), term_reason)
        frozen |= (term | trunc)
        if frozen.all():
            break
    finished = peak >= NG
    mb = np.array([np.mean(b) if b else 0 for b in betas])
    print(f"\nFF on MATCHED sim (vq_rate, directional wv) | v_des={VDES} envs={NENV} gates={NG} collide={COLLIDE}", flush=True)
    print(f"  gates passed (peak/episode): mean {peak.mean():.1f}/{NG}  max {peak.max()}  min {peak.min()}  | finished {finished.sum()}/{NENV}", flush=True)
    print(f"  distribution: {np.bincount(np.clip(peak,0,NG).astype(int), minlength=NG+1)}", flush=True)
    print(f"  mean|beta|: {mb.mean():.0f} deg (median {np.median(mb):.0f}; low=nose-first/no sideslip)", flush=True)
    print(f"  steps run: {step+1}", flush=True)
    print("VERDICT:", "FF RACES corridor -> good DAgger demonstrator" if peak.mean() >= NG*0.6
          else "FF struggles -> see env0 trace (setup vs aero)", flush=True)


if __name__ == "__main__":
    main()
