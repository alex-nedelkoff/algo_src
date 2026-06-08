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

G = 9.81
def argf(f, d): return float(sys.argv[sys.argv.index(f)+1]) if f in sys.argv else d
def args(f, d): return sys.argv[sys.argv.index(f)+1] if f in sys.argv else d
STEPS = int(argf("--steps", 5_000_000)); NENV = int(argf("--envs", 64))
DAGGER = int(argf("--dagger", 4)); BC_EPOCHS = int(argf("--bc_epochs", 20)); NG = 6
VDES = argf("--vdes", 4.0); WARM = int(argf("--warmstart", 1)); CURVE = "--curve" in sys.argv
REC = "--recurrent" in sys.argv; TAG = args("--tag", "v1")
if REC:
    WARM = 0  # BC-into-LSTM not wired; recurrent variant leans on PPO
M = json.load(open("sysid/vq_model.json"))
GAIN = np.array([M["rate_loop"][n]["gain_G"] for n in ("roll", "pitch", "yaw")])
F0 = M["thrust"]["f0"]; DF = M["thrust"]["df_dthr"]; MAXW = 17.45
KP_POS = np.array([0.6, 0.6, 2.0]); KD_POS = np.array([1.2, 1.2, 3.0])
KP_ATT = np.array([6.0, 6.0, 4.0]); KD_ATT = np.array([1.2, 1.2, 0.0]); KP_YAW = 4.0; KD_YAW = 0.5
TILT_MAX = np.tan(np.radians(35)) * G
HOVER_U0 = 2*((F0+G)/(-DF)) - 1
# Racing reward: gate_passage (per-gate, the objective) + delta gate_progress (loiter-safe guidance)
# + small body-rate penalty + crash. NO dense per-step survival terms (heading_alignment/speed_bonus
# are gameable on a finite course -> reward-hacking: survive+point-at-gate without threading).
REWARD = {"gate_progress": 2.0, "gate_passage": 15.0, "gate_offset": 0.5,
          "body_rate": 0.005, "crash_penalty": 10.0}


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
    cos_t = np.maximum(R[:, 2, 2], 0.5); c = (G + a[:, 2])/cos_t
    thr = np.clip((F0+c)/(-DF), 0.0, 1.0)
    u = np.empty((S.shape[0], 4)); u[:, 0] = np.clip(2*thr-1, -1, 1); u[:, 1:4] = np.clip(w/GAIN/MAXW, -1, 1)
    return u.astype(np.float32)


def aim(S, gates, gp):
    idx = np.clip(gp, 0, NG-1); gate = gates[np.arange(len(gp)), idx]
    dxy = (gate - S[:, POS])[:, :2]; dist = np.linalg.norm(dxy, axis=1, keepdims=True)+1e-6
    tv = np.zeros((len(gp), 3)); tv[:, :2] = VDES*dxy/dist
    return gate, tv, np.arctan2(dxy[:, 1], dxy[:, 0])


def _yaw_quat(th):
    return np.array([np.cos(th/2), 0.0, 0.0, np.sin(th/2)])


def make_env(n, seed):
    rng = np.random.default_rng(seed); tracks = []; G3 = []
    for _ in range(n):
        sp = rng.uniform(9, 13); amp = rng.uniform(1.0, 3.0); ph = rng.uniform(0, 6.28)
        gs = []
        if CURVE:
            hd = 0.0; pos = np.array([8.0, 0.0, 1.0]); turn = rng.uniform(0.12, 0.28)*rng.choice([-1, 1])
            for i in range(NG):
                hd += turn; pos = pos + sp*np.array([np.cos(hd), np.sin(hd), 0.0])
                z = 1.0 + 0.6*np.sin(i*0.7+ph)
                gs.append(GateState(position=np.array([pos[0], pos[1], z]), orientation=_yaw_quat(hd)))
        else:
            for i in range(NG):
                gs.append(GateState(position=np.array([8.0+sp*i, amp*np.sin(i*0.9+ph), 1.0]),
                                    orientation=np.array([1.0, 0, 0, 0])))
        tracks.append(Track(gates=gs, name="c", start_position=np.array([0, 0, 1.0]))); G3.append([g.position for g in gs])
    env = GateRaceEnv(n_envs=n, dt=0.01, max_steps=1500, action_mode="vq_rate",
                      vq_model_path="sysid/vq_model.json", tracks=tracks, random_gate_start=False,
                      start_behind_dist=1.0, start_vel_std=0.4, start_att_std=0.08, start_omega_std=0.3,
                      gate_collision=True, gate_passage_radius=1.0, arena_bounds=120.0, reward_weights=REWARD)
    return env, np.array(G3)


def policy_mean(model, obs_np):
    obs_t, _ = model.policy.obs_to_tensor(obs_np)
    feat = model.policy.extract_features(obs_t)
    latent_pi = model.policy.mlp_extractor.forward_actor(feat)
    return model.policy.action_net(latent_pi)


def eval_gates(model, seed=7, NE=16):
    env, gates = make_env(NE, seed); venv = VecEnvAdapter(env); obs = venv.reset()
    peak = np.zeros(NE, int); fr = np.zeros(NE, bool); lstm = None; starts = np.ones(NE, bool)
    for _ in range(2200):
        if REC:
            act, lstm = model.predict(obs, state=lstm, episode_start=starts, deterministic=True)
        else:
            act, _ = model.predict(obs, deterministic=True)
        peak = np.maximum(peak, np.where(fr, peak, env._gates_passed))
        obs, r, dones, infos = venv.step(act); starts = dones; fr |= dones
        if fr.all():
            break
    return np.maximum(peak, env._gates_passed)


def main():
    torch.manual_seed(0)
    print(f"FT[{TAG}]: vdes={VDES} warm={WARM} curve={CURVE} rec={REC} steps={STEPS}", flush=True)
    env, gates = make_env(NENV, 1); venv = VecEnvAdapter(env)
    log_std = -2.5 if WARM else 0.0
    if REC:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO("MlpLstmPolicy", venv, n_steps=512, batch_size=8192, n_epochs=8, gamma=0.999,
                             gae_lambda=0.95, clip_range=0.2, ent_coef=0.004, learning_rate=3e-4,
                             policy_kwargs=dict(net_arch=[128, 128], log_std_init=log_std, lstm_hidden_size=128),
                             device="cuda", verbose=1)
    else:
        model = PPO("MlpPolicy", venv, n_steps=512, batch_size=8192, n_epochs=8, gamma=0.999,
                    gae_lambda=0.95, clip_range=0.2, ent_coef=0.004, learning_rate=3e-4,
                    policy_kwargs=dict(net_arch=[128, 128, 128], log_std_init=log_std), device="cuda", verbose=1)
    with torch.no_grad():
        model.policy.action_net.bias[:] = torch.tensor([HOVER_U0, 0, 0, 0], dtype=model.policy.action_net.bias.dtype)

    if WARM:
        print("FT: FF demos + BC...", flush=True)
        obs = venv.reset(); X = []; Y = []
        for _ in range(1500):
            S = env._states; u = ff_batch(S, *aim(S, gates, env._gates_passed))
            X.append(obs.copy()); Y.append(u.copy()); obs, r, d, info = venv.step(u)
        X = np.concatenate(X); Y = np.concatenate(Y)
        opt = torch.optim.Adam(model.policy.parameters(), 1e-3); lf = nn.MSELoss()
        def bc(X, Y, epochs):
            Yt = torch.tensor(Y, dtype=torch.float32, device=model.device); n = len(X); bs = 8192
            for ep in range(epochs):
                perm = np.random.permutation(n)
                for j in range(0, n, bs):
                    idx = perm[j:j+bs]; opt.zero_grad(); l = lf(policy_mean(model, X[idx]), Yt[idx]); l.backward(); opt.step()
            return l.item()
        l = bc(X, Y, BC_EPOCHS); pk = eval_gates(model)
        print(f"  BC: mse {l:.4f} | gates {pk.mean():.1f}/{NG} fin {int((pk>=NG).sum())}/16", flush=True)
        for rd in range(1, DAGGER+1):
            ed, gd = make_env(NENV, 100+rd); vd = VecEnvAdapter(ed); o = vd.reset(); Xn = []; Yn = []
            for _ in range(1500):
                S = ed._states; uff = ff_batch(S, *aim(S, gd, ed._gates_passed))
                act, _ = model.predict(o, deterministic=True); Xn.append(o.copy()); Yn.append(uff.copy()); o, r, dn, inf = vd.step(act)
            X = np.concatenate([X, np.concatenate(Xn)]); Y = np.concatenate([Y, np.concatenate(Yn)])
            l = bc(X, Y, BC_EPOCHS); pk = eval_gates(model)
            print(f"  DAgger r{rd}: D={len(X)} mse {l:.4f} | gates {pk.mean():.1f}/{NG} fin {int((pk>=NG).sum())}/16", flush=True)

    class GateEval(BaseCallback):
        def __init__(s, every): super().__init__(); s.every = every; s.last = 0
        def _on_step(s):
            if s.num_timesteps - s.last >= s.every:
                s.last = s.num_timesteps; pk = eval_gates(model)
                print(f"  [t={s.num_timesteps}] gates {pk.mean():.1f}/{NG} fin {int((pk>=NG).sum())}/16", flush=True)
            return True

    print(f"FT[{TAG}]: learn {STEPS}...", flush=True)
    model.learn(total_timesteps=STEPS, progress_bar=False, callback=GateEval(500_000))
    pk = eval_gates(model)
    print(f"FT[{TAG}] DONE: gates {pk.mean():.1f}/{NG} max {pk.max()} fin {int((pk>=NG).sum())}/16 dist {np.bincount(np.clip(pk,0,NG),minlength=NG+1)}", flush=True)
    model.save(f"ft_{TAG}")
    print(f"saved ft_{TAG}.zip", flush=True)


if __name__ == "__main__":
    main()
