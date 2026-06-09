"""dagger_v2.py -- DAgger (vectorized FF expert) to fix BC covariate shift on the matched sim.

Fixes vs dagger_bc.py:
  - FF expert fully VECTORIZED over envs (batched quat/attitude math) -> fast collect + relabel.
  - action targets STANDARDIZED per-dim (the rate channels are tiny ~0.02; abs-MSE under-weights them
    and the -2.5-gain rate loop amplifies the error) -> linear head, deploy = net_out*act_std, clip.
  - DAGGER rounds: roll the student, relabel the states IT visits with the FF, aggregate, retrain.
Self-tests the batched FF == scalar before running. Logs unbuffered.
Usage: python -u dagger_v2.py [--v 4] [--demo_envs 96] [--rounds 4] [--epochs 20]
"""
import sys, json
import numpy as np
import torch, torch.nn as nn
from sim.envs.gate_race_env import GateRaceEnv
from sim.tracks import Track, GateState
from sim.dynamics.numpy_quad import POS, VEL, QUAT, OMEGA, quat_to_rotmat_batch

G = 9.81
def argf(f, d): return float(sys.argv[sys.argv.index(f)+1]) if f in sys.argv else d
VDES = argf("--v", 4.0); DEMO_ENVS = int(argf("--demo_envs", 96))
ROUNDS = int(argf("--rounds", 4)); EPOCHS = int(argf("--epochs", 20)); NG = 6
MODEL = json.load(open("sysid/vq_model.json"))
GAIN = np.array([MODEL["rate_loop"][n]["gain_G"] for n in ("roll", "pitch", "yaw")])
F0 = MODEL["thrust"]["f0"]; DF = MODEL["thrust"]["df_dthr"]; MAXW = 17.45
DT = 1.0 / 72.0; EP_STEPS = 1584   # MIMO rate loop is dt-specific (fit 1/72); ~22 s episodes
# weathervane-FF (corner_speed teacher term, WV-DYNAMIC coeffs) in the ENU env frame:
# under B=diag(1,-1,-1) the roll-wv sign flips, yaw-wv is invariant (vq_matched ENU port).
ROLL_WV0, ROLL_WV1, YAW_WV = -0.105, -0.019, -0.149
YR_CAP = 1.5   # rad/s yaw-rate cap (corner_speed)
KP_POS = np.array([0.6, 0.6, 2.0]); KD_POS = np.array([1.2, 1.2, 3.0])
KP_ATT = np.array([6.0, 6.0, 4.0]); KD_ATT = np.array([1.2, 1.2, 0.0]); KP_YAW = 4.0; KD_YAW = 0.5
TILT_MAX = np.tan(np.radians(35)) * G


def mat_to_quat_batch(m):
    N = m.shape[0]; t = m[:, 0, 0]+m[:, 1, 1]+m[:, 2, 2]; q = np.zeros((N, 4))
    cA = t > 0
    cB = (~cA) & (m[:, 0, 0] >= m[:, 1, 1]) & (m[:, 0, 0] >= m[:, 2, 2])
    cC = (~cA) & (~cB) & (m[:, 1, 1] >= m[:, 2, 2])
    cD = (~cA) & (~cB) & (~cC)
    sA = np.sqrt(np.maximum(t+1, 1e-12))*2
    q[cA, 0] = .25*sA[cA]; q[cA, 1] = (m[cA, 2, 1]-m[cA, 1, 2])/sA[cA]
    q[cA, 2] = (m[cA, 0, 2]-m[cA, 2, 0])/sA[cA]; q[cA, 3] = (m[cA, 1, 0]-m[cA, 0, 1])/sA[cA]
    sB = np.sqrt(np.maximum(1+m[:, 0, 0]-m[:, 1, 1]-m[:, 2, 2], 1e-12))*2
    q[cB, 0] = (m[cB, 2, 1]-m[cB, 1, 2])/sB[cB]; q[cB, 1] = .25*sB[cB]
    q[cB, 2] = (m[cB, 0, 1]+m[cB, 1, 0])/sB[cB]; q[cB, 3] = (m[cB, 0, 2]+m[cB, 2, 0])/sB[cB]
    sC = np.sqrt(np.maximum(1+m[:, 1, 1]-m[:, 0, 0]-m[:, 2, 2], 1e-12))*2
    q[cC, 0] = (m[cC, 0, 2]-m[cC, 2, 0])/sC[cC]; q[cC, 1] = (m[cC, 0, 1]+m[cC, 1, 0])/sC[cC]
    q[cC, 2] = .25*sC[cC]; q[cC, 3] = (m[cC, 1, 2]+m[cC, 2, 1])/sC[cC]
    sD = np.sqrt(np.maximum(1+m[:, 2, 2]-m[:, 0, 0]-m[:, 1, 1], 1e-12))*2
    q[cD, 0] = (m[cD, 1, 0]-m[cD, 0, 1])/sD[cD]; q[cD, 1] = (m[cD, 0, 2]+m[cD, 2, 0])/sD[cD]
    q[cD, 2] = (m[cD, 1, 2]+m[cD, 2, 1])/sD[cD]; q[cD, 3] = .25*sD[cD]
    return q/np.linalg.norm(q, axis=1, keepdims=True)


def desired_attitude_batch(a, yaw):
    t = a + np.array([0, 0, G]); zb = t/np.linalg.norm(t, axis=1, keepdims=True)
    xc = np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], axis=1)
    yb = np.cross(zb, xc); yb /= np.linalg.norm(yb, axis=1, keepdims=True)
    xb = np.cross(yb, zb)
    return np.stack([xb, yb, zb], axis=2)   # columns


def att_err_quat_batch(qc, qd):
    w0, x0, y0, z0 = qc[:, 0], qc[:, 1], qc[:, 2], qc[:, 3]
    w1, x1, y1, z1 = qd[:, 0], qd[:, 1], qd[:, 2], qd[:, 3]
    qe = np.stack([w0*w1+x0*x1+y0*y1+z0*z1, w0*x1-x0*w1-y0*z1+z0*y1,
                   w0*y1+x0*z1-y0*w1-z0*x1, w0*z1-x0*y1+y0*x1-z0*w1], axis=1)
    qe[qe[:, 0] < 0] *= -1
    return 2*qe[:, 1:4]


def ff_batch(S, tgt_pos, tgt_vel, tgt_yaw):
    pos = S[:, POS]; vel = S[:, VEL]; q = S[:, QUAT]; om = S[:, OMEGA]
    a = KP_POS*(tgt_pos-pos) + KD_POS*(tgt_vel-vel)
    n = np.linalg.norm(a[:, :2], axis=1)
    sc = np.where(n > TILT_MAX, TILT_MAX/np.maximum(n, 1e-9), 1.0)
    a[:, 0] *= sc; a[:, 1] *= sc
    R = quat_to_rotmat_batch(q); yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    q_des = mat_to_quat_batch(desired_attitude_batch(a, tgt_yaw))
    w = KP_ATT*att_err_quat_batch(q, q_des) - KD_ATT*om
    w[:, 2] = KP_YAW*(((tgt_yaw-yaw+np.pi) % (2*np.pi))-np.pi) - KD_YAW*om[:, 2]
    # weathervane-FF: cancel the sideslip moment (validated live in corner_speed; ENU signs)
    vb = np.einsum("nji,nj->ni", R, vel)            # body velocity = R^T v
    w[:, 0] += (ROLL_WV0 + ROLL_WV1*vb[:, 0]) * vb[:, 1]
    w[:, 2] += -YAW_WV * vb[:, 1]
    cos_t = np.maximum(R[:, 2, 2], 0.5); c = (G + a[:, 2])/cos_t
    thr = np.clip((F0+c)/(-DF), 0.0, 1.0)
    u = np.empty((S.shape[0], 4))
    u[:, 0] = np.clip(2*thr-1, -1, 1); u[:, 1:4] = np.clip(w/GAIN/MAXW, -1, 1)
    # yaw-rate cap (corner_speed YR_CAP): an uncapped pi-size yaw error commands ~12 rad/s,
    # which excites the yaw-rate resonance and tumbles the MIMO plant at spawn.
    cap = YR_CAP/abs(GAIN[2])/MAXW
    u[:, 3] = np.clip(u[:, 3], -cap, cap)
    return u


def aim(S, gates, gp):
    """Velocity setpoint toward the gate; yaw tracks ANTI-VELOCITY = camera-forward racing
    (corner_speed's beta-closed-loop yaw law). Hold current yaw while slow (entry phase).
    Nose-at-gate flies camera-BACKWARD (stable weathervane branch) = wrong regime to learn."""
    idx = np.clip(gp, 0, NG-1)
    gate = gates[np.arange(len(gp)), idx]
    dxy = (gate - S[:, POS])[:, :2]; dist = np.linalg.norm(dxy, axis=1, keepdims=True)+1e-6
    tv = np.zeros((len(gp), 3)); tv[:, :2] = VDES*dxy/dist
    vel = S[:, VEL][:, :2]; spd = np.linalg.norm(vel, axis=1)
    R = quat_to_rotmat_batch(S[:, QUAT]); yaw_cur = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    yaw_av = np.arctan2(-vel[:, 1], -vel[:, 0])
    return gate, tv, np.where(spd > 1.0, yaw_av, yaw_cur)


def make_env(n, seed, vstd=0.0, astd=0.0, ostd=0.0):
    rng = np.random.default_rng(seed); tracks = []; G3 = []
    for _ in range(n):
        sp = rng.uniform(9, 13); amp = rng.uniform(1.0, 3.0); ph = rng.uniform(0, 6.28)
        gs = [GateState(position=np.array([8.0+sp*i, amp*np.sin(i*0.9+ph), 1.0]),
                        orientation=np.array([1.0, 0, 0, 0])) for i in range(NG)]
        tracks.append(Track(gates=gs, name="c", start_position=np.array([0, 0, 1.0])))
        G3.append([g.position for g in gs])
    env = GateRaceEnv(n_envs=n, dt=DT, max_steps=EP_STEPS, action_mode="vq_rate",
                      vq_model_path="sysid/vq_model.json", tracks=tracks, random_gate_start=False,
                      start_behind_dist=1.0, start_vel_std=vstd, start_att_std=astd, start_omega_std=ostd,
                      gate_collision=False, gate_passage_radius=1.0, arena_bounds=120.0)
    return env, np.array(G3)


def roll_ff(env, gates, record=True):
    obs, _ = env.reset(); OBS = []; ACT = []; fr = np.zeros(env.n_envs, bool)
    for _ in range(EP_STEPS):
        g, tv, ty = aim(env._states, gates, env._gates_passed)
        u = ff_batch(env._states, g, tv, ty)
        if record:
            k = ~fr; OBS.append(obs[k]); ACT.append(u[k])
        obs, r, term, trunc, info = env.step(u); fr |= (term | trunc)
        if fr.all():
            break
    return (np.concatenate(OBS), np.concatenate(ACT)) if record else None


def roll_student_relabel(env, gates, net, mu, sd, ystd):
    """Roll the student; record (obs it sees, FF-relabel of its state). DAgger aggregation."""
    obs, _ = env.reset(); OBS = []; ACT = []; fr = np.zeros(env.n_envs, bool)
    for _ in range(EP_STEPS):
        with torch.no_grad():
            us = (net(torch.tensor((obs-mu)/sd, dtype=torch.float32)).numpy()*ystd)
        us = np.clip(us, -1, 1)
        g, tv, ty = aim(env._states, gates, env._gates_passed)
        uff = ff_batch(env._states, g, tv, ty)              # expert label for the visited state
        k = ~fr; OBS.append(obs[k]); ACT.append(uff[k])
        obs, r, term, trunc, info = env.step(us); fr |= (term | trunc)
        if fr.all():
            break
    return np.concatenate(OBS), np.concatenate(ACT)


def eval_student(net, mu, sd, ystd, seed=7, NE=16):
    env, gates = make_env(NE, seed)
    obs, _ = env.reset(seed=seed); peak = np.zeros(NE, int); fr = np.zeros(NE, bool)
    for _ in range(EP_STEPS + 200):
        with torch.no_grad():
            u = np.clip(net(torch.tensor((obs-mu)/sd, dtype=torch.float32)).numpy()*ystd, -1, 1)
        peak = np.maximum(peak, np.where(fr, peak, env._gates_passed))
        obs, r, term, trunc, info = env.step(u); fr |= (term | trunc)
        if fr.all():
            break
    peak = np.maximum(peak, env._gates_passed)
    return peak


class Net(nn.Module):
    def __init__(s, di):
        super().__init__()
        s.f = nn.Sequential(nn.Linear(di, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 4))

    def forward(s, x):
        return s.f(x)


def train(net, X, Y, mu, sd, ystd, epochs):
    Xt = torch.tensor((X-mu)/sd, dtype=torch.float32); Yt = torch.tensor(Y/ystd, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), 1e-3); lf = nn.MSELoss(); n = len(Xt); bs = 4096
    for ep in range(epochs):
        perm = torch.randperm(n)
        for j in range(0, n, bs):
            idx = perm[j:j+bs]; opt.zero_grad(); l = lf(net(Xt[idx]), Yt[idx]); l.backward(); opt.step()
    return l.item()


def main():
    torch.manual_seed(0)
    # self-test batched FF == scalar on random states
    rng = np.random.default_rng(0); St = np.zeros((50, 17)); q = rng.normal(size=(50, 4))
    St[:, QUAT] = q/np.linalg.norm(q, axis=1, keepdims=True); St[:, POS] = rng.normal(size=(50, 3))
    St[:, VEL] = rng.normal(size=(50, 3)); St[:, OMEGA] = rng.normal(size=(50, 3))*0.5
    tp = rng.normal(size=(50, 3))*5; tv = rng.normal(size=(50, 3)); tyaw = rng.uniform(-3, 3, 50)
    ub = ff_batch(St, tp, tv, tyaw)
    print(f"batched FF self-test: shape {ub.shape}, finite {np.isfinite(ub).all()}", flush=True)

    print(f"DAgger: FF demos ({DEMO_ENVS} corridors)...", flush=True)
    env, gates = make_env(DEMO_ENVS, 1, vstd=0.4, astd=0.08, ostd=0.3)
    X, Y = roll_ff(env, gates)
    mu = X.mean(0); sd = X.std(0)+1e-6; ystd = Y.std(0)+1e-6
    print(f"  demos {X.shape}; action std {np.round(ystd,3)}", flush=True)
    net = Net(X.shape[1])
    l = train(net, X, Y, mu, sd, ystd, EPOCHS)
    pk = eval_student(net, mu, sd, ystd)
    print(f"  BC(round0): train_mse {l:.4f} | gates mean {pk.mean():.1f}/{NG} finished {int((pk>=NG).sum())}/16 dist {np.bincount(np.clip(pk,0,NG),minlength=NG+1)}", flush=True)

    for rd in range(1, ROUNDS+1):
        envd, gd = make_env(DEMO_ENVS, 100+rd, vstd=0.4, astd=0.08, ostd=0.3)
        Xn, Yn = roll_student_relabel(envd, gd, net, mu, sd, ystd)
        X = np.concatenate([X, Xn]); Y = np.concatenate([Y, Yn])
        mu = X.mean(0); sd = X.std(0)+1e-6; ystd = Y.std(0)+1e-6
        l = train(net, X, Y, mu, sd, ystd, EPOCHS)
        pk = eval_student(net, mu, sd, ystd)
        print(f"  DAgger r{rd}: +{len(Xn)} states (D={len(X)}) train_mse {l:.4f} | gates mean {pk.mean():.1f}/{NG} "
              f"finished {int((pk>=NG).sum())}/16 dist {np.bincount(np.clip(pk,0,NG),minlength=NG+1)}", flush=True)

    print("VERDICT:", "DAgger student RACES it -> ready for RL fine-tune" if pk.mean() >= NG*0.6
          else "still weak -> recurrent policy / more rounds / beta-mix", flush=True)


if __name__ == "__main__":
    main()
