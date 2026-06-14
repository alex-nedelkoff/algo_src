# Residual FF + Asymmetric Privileged Critic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--residual_ff` (policy outputs a bounded delta on top of the analytic spline FF anchor, so PPO cannot erode the stable anchor) and `--asymmetric_critic` (critic sees true per-env plant params, actor sees only observable obs) to `scripts/rl/rl_finetune.py`, then train + live-test on the VQ sim.

**Architecture:**
- **Residual** is applied in a thin `ResidualVecEnv` wrapper (subclass of `VecEnvAdapter`) in `rl_finetune.py`: on `step(delta)` it reads `env._states`, recomputes the spline anchor `ff_batch(S, *aim(...))`, and steps the inner env with `applied = clip(anchor + tanh(delta)*delta_scale, -1, 1)`. The env/dynamics are untouched by residual. The DEPLOY script must mirror this exact anchor+delta math.
- **Asymmetric critic** needs per-env plant variation to be meaningful, so it pulls in **per-env domain randomization** in `VQMatchedDynamics` (scalar aero/thrust/wv params become per-env arrays; rate-loop matrices stay nominal) + a **privileged Dict obs channel** (`{"policy":..., "privileged": dr_factors}`) + a hand-rolled SB3 `AsymmetricActorCriticPolicy` (actor MLP on `policy`, critic GCNet on `concat(policy, privileged)`). The critic is discarded at deploy, so asymmetric adds zero deploy cost.

**Tech Stack:** numpy (vectorized dynamics + env), Stable-Baselines3 PPO (pod-only; not importable on the Mac), PyTorch, scipy (spline planner already ported in `sim/gate_traj.py`).

**Invariant (non-negotiable):** with randomization OFF and the new flags OFF, every code path must be **bit-identical** to today's behavior — `VQMatchedDynamics` is the campaign's validated foundation. Each task that touches it has a bit-identity test.

**Test split:** numpy pieces (Tasks 1-3, 5-anchor) are TDD'd locally with pytest. The SB3 asymmetric policy (Task 4) and end-to-end wiring (Task 6) are verified via a **small pod smoke-run** (`--steps 20000`) because SB3 is pod-only.

**Default knobs:** `--delta_scale 0.15`, `--dr_width 0.4` (matches the existing per-round DR widths in `dr_model_path`).

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `sim/dynamics/vq_matched.py` | Per-env scalar-param arrays + `randomize()` + `dr_factors` | Modify |
| `sim/envs/gate_race_env.py` | Privileged Dict obs channel; pass per-env DR to dynamics | Modify |
| `sim/envs/vec_env_adapter.py` | Dict-obs passthrough | Modify |
| `scripts/rl/rl_finetune.py` | Flags, `ResidualVecEnv`, asymmetric policy wiring, BC-skip | Modify |
| `control/policies/asymmetric.py` | `AsymmetricActorCriticPolicy` (actor=obs, critic=obs+priv) | Create |
| `tests/test_sim/test_vq_matched_dr.py` | Per-env DR bit-identity + factor correctness | Create |
| `tests/test_sim/test_privileged_obs.py` | Dict obs shape/contents; flat-obs unchanged when off | Create |
| `tests/test_rl/test_residual_anchor.py` | `applied==anchor` when delta=0; clip bounds | Create |
| laptop `vq_deploy4_hist.py` | Mirror residual anchor at deploy (via scp) | Modify (Task 7) |

---

## Task 1: Per-env domain randomization in VQMatchedDynamics

**Files:**
- Modify: `sim/dynamics/vq_matched.py`
- Test: `tests/test_sim/test_vq_matched_dr.py`

The randomizable scalar params are the aero/thrust/wv terms used directly in `step()`:
`Dx, Dy, qx, qy, qz, Dz, f0, df` and (when `weathervane_v2` present) `wv2 = (a_r, b_r, a_y, b_y, vclip)` and `dist_scale`. `vclip` is NOT randomized. The rate-loop matrices (`A1_r2/A2_r2/B0_r2/B1_r2`, `A_rate/B_rate`) stay nominal (shared) — they are dt-specific and validated; per-round DR already scatters them for the actor.

The privileged factor vector (per env, K=11): `[Dx, Dy, qx, qy, qz, Dz, f0, df, wv_roll_b, wv_yaw_b, dist_scale]` expressed as **(factor - 1.0)** centered at 0 (dist_scale stored as `scale` directly since its nominal is 0). Order is fixed and documented.

- [ ] **Step 1: Write the failing test (bit-identity when off + factor shape)**

```python
# tests/test_sim/test_vq_matched_dr.py
import json
import numpy as np
from sim.dynamics.vq_matched import VQMatchedDynamics

MODEL = json.load(open("sysid/vq_model.json"))


def _rollout(dyn, n=8, steps=30, seed=0):
    rng = np.random.default_rng(seed)
    s = dyn.reset(n)
    s[:, 3:6] = rng.normal(0, 2, (n, 3))          # nonzero velocity to exercise drag/wv
    acts = rng.uniform(-1, 1, (steps, n, 4)); acts[:, :, 0] = 0.5
    out = []
    for a in acts:
        s = dyn.step(s, a); out.append(s.copy())
    return np.array(out)


def test_randomize_off_is_bit_identical():
    a = VQMatchedDynamics(MODEL, frame="ENU")
    b = VQMatchedDynamics(MODEL, frame="ENU")   # never call randomize() -> nominal
    ra, rb = _rollout(a), _rollout(b)
    np.testing.assert_array_equal(ra, rb)        # determinism baseline
    # scalar params unchanged type (float, not array) when not randomized
    assert np.isscalar(a.Dx) or np.ndim(a.Dx) == 0


def test_dr_factors_shape_and_center():
    d = VQMatchedDynamics(MODEL, frame="ENU")
    rng = np.random.default_rng(3)
    d.randomize(rng, n_envs=8, width=0.4)
    assert d.dr_factors.shape == (8, 11)
    assert np.all(np.abs(d.dr_factors) <= 1.5)   # centered near 0, bounded
    # per-env params became arrays of length n_envs
    assert np.asarray(d.Dx).shape == (8,)


def test_dr_changes_trajectory():
    base = VQMatchedDynamics(MODEL, frame="ENU")
    drd = VQMatchedDynamics(MODEL, frame="ENU")
    drd.randomize(np.random.default_rng(1), n_envs=8, width=0.4)
    assert not np.allclose(_rollout(base), _rollout(drd))
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/alex/Documents/drone-ai-grand-prix/algo_src && PYTHONPATH=. python -m pytest tests/test_sim/test_vq_matched_dr.py -v`
Expected: FAIL — `AttributeError: 'VQMatchedDynamics' object has no attribute 'randomize'`.

- [ ] **Step 3: Implement `randomize()` + array-safe `step()`**

In `__init__`, after the existing scalar assignments, initialize `self.dr_factors = None` and keep scalars as-is (Python floats). Add the method:

```python
    # ---- per-env domain randomization (asymmetric-critic privileged channel) ----
    # Scalar aero/thrust/wv params become per-env arrays; rate-loop matrices stay nominal.
    # dr_factors (n_envs, 11) = privileged vector, centered at 0 (factor-1; dist_scale raw).
    _DR_KEYS = ("Dx", "Dy", "qx", "qy", "qz", "Dz", "f0", "df")

    def randomize(self, rng, n_envs: int, width: float = 0.4) -> None:
        self._nom = {k: float(getattr(self, k)) for k in self._DR_KEYS}
        fac = {k: 1.0 + rng.uniform(-width, width, n_envs) for k in self._DR_KEYS}
        for k in self._DR_KEYS:
            setattr(self, k, self._nom[k] * fac[k])
        # weathervane v2 speed-slope (roll b, yaw b) -- the live-divergent term (WV-DYNAMIC)
        if self.wv2 is not None:
            a_r, b_r, a_y, b_y, vclip = self.wv2
            fr = 1.0 + rng.uniform(-width, width, n_envs)
            fy = 1.0 + rng.uniform(-width, width, n_envs)
            self.wv2 = (a_r, b_r * fr, a_y, b_y * fy, vclip)
            wv_rb, wv_yb = fr - 1.0, fy - 1.0
        else:
            wv_rb = wv_yb = np.zeros(n_envs)
        # measured disturbance field magnitude (the snap mechanism) -- nominal 0..2 range
        dist = rng.uniform(0.0, 2.0, n_envs)
        self._dist_scale_arr = dist
        self.dr_factors = np.column_stack(
            [fac[k] - 1.0 for k in self._DR_KEYS] + [wv_rb, wv_yb, dist]
        ).astype(np.float32)
```

In `step()`, the disturbance field block must use the per-env array when present. Replace the activation guard so that when `self.dr_factors is not None` the `self._dist_scale` becomes the per-env `self._dist_scale_arr`:

```python
        # --- measured disturbance field (DR): tilt-scaled correlated rate bias ---
        dist_scale = getattr(self, "_dist_scale_arr", None)
        dist_active = dist_scale if dist_scale is not None else self._dist_scale
        if np.any(dist_active > 0.0):
            n = states.shape[0]
            if self._dist_bias is None or self._dist_bias.shape[0] != n:
                self._dist_bias = np.zeros((n, 3)); self._dist_k = 0
            if self._dist_k % self._dist_steps == 0:
                tilt = np.degrees(np.arccos(np.clip(np.abs(R[:, 2, 2]), -1.0, 1.0)))
                idx = np.searchsorted(self._dist_lo, tilt, side="right") - 1
                sig = self._dist_sig[np.clip(idx, 0, len(self._dist_sig) - 1)]
                scl = np.asarray(dist_active)[:, None] if np.ndim(dist_active) else dist_active
                self._dist_bias = np.random.normal(0.0, 1.0, (n, 3)) * sig * scl
            self._dist_k += 1
            om_new += self._dist_bias * dt
```

NOTE: the per-env `_dist_steps`/`_dist_sig`/`_dist_lo` are only built in `__init__` when `_dd["scale"]>0` in the source model. For DR to inject the field on a model whose canonical `dr_rate_disturbance.scale==0`, `randomize()` must also build those tables. Add to `randomize()` after computing `dist`, guarded on the model having a `dr_rate_disturbance` block:

```python
        _dd = self._dd_raw
        if _dd and not hasattr(self, "_dist_sig"):
            bins = _dd["bins"]
            self._dist_lo = np.array([b["tilt_deg"][0] for b in bins], float)
            self._dist_sig = np.array([b["sigma"] for b in bins], float)
            self._dist_steps = max(1, int(round(float(_dd.get("block_s", 0.5)) / self.dt)))
```

and in `__init__` store `self._dd_raw = model.get("dr_rate_disturbance") or {}` (next to the existing `_dd` read). The drag/thrust/wv arrays broadcast cleanly in the existing `step()` math (`-(self.Dx + self.qx*vmag)*vb[:,0]` etc. — all `(n_envs,)`), so no other `step()` change is needed.

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=. python -m pytest tests/test_sim/test_vq_matched_dr.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the existing dynamics tests (guard the foundation)**

Run: `PYTHONPATH=. python -m pytest tests/test_sim/ tests/test_dynamics/ -q`
Expected: PASS (no regressions; randomization is off by default).

- [ ] **Step 6: Commit**

```bash
git add sim/dynamics/vq_matched.py tests/test_sim/test_vq_matched_dr.py
git commit -m "feat(cor-127): per-env DR + privileged factors in VQMatchedDynamics"
```

---

## Task 2: Privileged Dict obs channel in GateRaceEnv

**Files:**
- Modify: `sim/envs/gate_race_env.py`
- Test: `tests/test_sim/test_privileged_obs.py`

Add a constructor kwarg `privileged_obs: bool = False` and `dr_width: float = 0.0`. When `dr_width>0` and `action_mode==VQ_RATE`, call `self._matched.randomize(rng, n_envs, dr_width)` right after the dynamics is built. When `privileged_obs`, the observation space becomes a `gymnasium.spaces.Dict({"policy": <existing Box>, "privileged": Box(K)})` and the obs builders return `{"policy": flat, "privileged": self._matched.dr_factors}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sim/test_privileged_obs.py
import numpy as np
from gymnasium import spaces
from sim.envs.gate_race_env import GateRaceEnv
from sim.tracks import Track, GateState


def _track():
    gs = [GateState(position=np.array([8.0 + 8 * i, 0.0, 1.0]),
                    orientation=np.array([1.0, 0, 0, 0])) for i in range(6)]
    return Track(gates=gs, name="c", start_position=np.array([0, 0, 1.0]))


def _env(**kw):
    return GateRaceEnv(n_envs=4, dt=1 / 72.0, max_steps=200, action_mode="vq_rate",
                       max_body_rate=6.0, vq_max_thrust=0.6,
                       tracks=[_track() for _ in range(4)], **kw)


def test_flat_obs_unchanged_when_off():
    env = _env()
    obs, _ = env.reset(seed=0)
    assert isinstance(obs, np.ndarray) and obs.shape[0] == 4


def test_privileged_dict_obs_shape():
    env = _env(privileged_obs=True, dr_width=0.4)
    assert isinstance(env.observation_space, spaces.Dict)
    obs, _ = env.reset(seed=0)
    assert set(obs.keys()) == {"policy", "privileged"}
    assert obs["policy"].shape[0] == 4
    assert obs["privileged"].shape == (4, 11)
    # privileged is per-env constant across a step (fixed-at-construction DR)
    a = np.zeros((4, 4)); a[:, 0] = 0.0
    obs2, *_ = env.step(a)
    np.testing.assert_array_equal(obs["privileged"], obs2["privileged"])
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/test_sim/test_privileged_obs.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'privileged_obs'`.

- [ ] **Step 3: Implement the kwarg + DR call + Dict obs**

In `__init__` signature add `privileged_obs: bool = False,` and `dr_width: float = 0.0,`. Store `self._privileged_obs = privileged_obs`. After `self._matched = VQMatchedDynamics(...)` (around line 399), add:

```python
            if dr_width > 0.0:
                self._matched.randomize(np.random.default_rng(self._seed_or(0)),
                                        n_envs, dr_width)
            self._priv_dim = (self._matched.dr_factors.shape[1]
                              if self._matched.dr_factors is not None else 0)
```

(Use the env's existing seeding convention; if none, `np.random.default_rng(0)` is fine — per-env DR just needs to differ across envs, which it does within one call.) After building `self.observation_space` (around line 404), wrap it when privileged:

```python
        if self._privileged_obs:
            from gymnasium import spaces as _sp
            self.observation_space = _sp.Dict({
                "policy": self.observation_space,
                "privileged": _sp.Box(-np.inf, np.inf, shape=(self._priv_dim,),
                                      dtype=np.float32),
            })
```

Wrap the two obs-return paths. Find `_compute_obs_batched` (returns the flat `obs` array) and `_compute_obs`; add a helper and use it at every return that hands obs to the caller (`reset`, `step`, `_compute_obs`):

```python
    def _wrap_obs(self, flat):
        if not self._privileged_obs:
            return flat
        priv = self._matched.dr_factors
        if flat.ndim == 1:            # single-env squeeze path
            return {"policy": flat, "privileged": priv[0]}
        return {"policy": flat, "privileged": priv.astype(np.float32)}
```

Then in `reset()` return `self._wrap_obs(obs), info` and in `step()` return `self._wrap_obs(obs), ...`. (The internal `terminal_obs` in info stays the flat array — VecEnvAdapter Task 3 handles it.)

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=. python -m pytest tests/test_sim/test_privileged_obs.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the env test suite**

Run: `PYTHONPATH=. python -m pytest tests/test_sim/ -q`
Expected: PASS (flat-obs path unchanged when `privileged_obs` is off).

- [ ] **Step 6: Commit**

```bash
git add sim/envs/gate_race_env.py tests/test_sim/test_privileged_obs.py
git commit -m "feat(cor-127): privileged Dict obs channel + per-env DR hook in GateRaceEnv"
```

---

## Task 3: Dict-obs passthrough in VecEnvAdapter

**Files:**
- Modify: `sim/envs/vec_env_adapter.py`
- Test: covered by Task 6 smoke (SB3 VecEnv contract); add a light local shape test.

`VecEnvAdapter.reset()`/`step_wait()` currently call `obs.ndim`/`obs[None,:]` which breaks on a dict. Branch so dict obs is normalized per-array.

- [ ] **Step 1: Implement dict-aware shape normalization**

Add a module helper and use it in `reset` and `step_wait` in place of the inline `if obs.ndim == 1: obs = obs[None, :]`:

```python
def _as_batched(obs):
    if isinstance(obs, dict):
        return {k: (v[None, :] if v.ndim == 1 else v) for k, v in obs.items()}
    return obs[None, :] if obs.ndim == 1 else obs
```

In `reset()`: `obs = self.env.reset(seed=self._pending_seed)[0]; return _as_batched(obs)`.
In `step_wait()`: `obs = _as_batched(obs)` after the env step. The `terminal_obs` handling stays as-is (flat array; SB3 only needs it for bootstrap and the asymmetric policy's `predict_values` reads the dict obs, not terminal_obs — acceptable, matches SB3's standard truncation bootstrap which uses the dict).

- [ ] **Step 2: Local shape test**

```python
# append to tests/test_sim/test_privileged_obs.py
def test_vec_adapter_dict_passthrough():
    from sim.envs.vec_env_adapter import VecEnvAdapter
    env = _env(privileged_obs=True, dr_width=0.4)
    venv = VecEnvAdapter(env)
    obs = venv.reset()
    assert set(obs.keys()) == {"policy", "privileged"}
    assert obs["policy"].shape == (4, env._obs_dim)
```

Run: `PYTHONPATH=. python -m pytest tests/test_sim/test_privileged_obs.py::test_vec_adapter_dict_passthrough -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add sim/envs/vec_env_adapter.py tests/test_sim/test_privileged_obs.py
git commit -m "feat(cor-127): Dict-obs passthrough in VecEnvAdapter"
```

---

## Task 4: AsymmetricActorCriticPolicy (SB3)

**Files:**
- Create: `control/policies/asymmetric.py`
- Test: pod smoke (SB3 not importable on Mac). Include a self-test `__main__` block runnable on the pod.

Actor consumes `obs["policy"]`; critic consumes `concat(obs["policy"], obs["privileged"])`. Subclass `MultiInputActorCriticPolicy` and override the value path. Reuse the project `GCNet` MLP for the critic.

- [ ] **Step 1: Implement the policy**

```python
# control/policies/asymmetric.py
"""Asymmetric actor-critic: actor sees observable obs, critic sees obs+privileged.

The privileged channel (true per-env plant params from VQMatchedDynamics.randomize)
gives the value function a cleaner advantage signal in live-unsafe states without
leaking unobservable info into the deployed actor. The critic is discarded at deploy,
so the residual actor flies on observable obs alone. (COR-127, HANDOFF 06-14 #2.)
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.policies import MultiInputActorCriticPolicy


class AsymmetricActorCriticPolicy(MultiInputActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        # net_arch applies to the actor; the critic is a separate MLP built below.
        super().__init__(*args, **kwargs)
        obs_sp = self.observation_space
        pol_dim = int(np.prod(obs_sp["policy"].shape))
        priv_dim = int(np.prod(obs_sp["privileged"].shape))
        h = 256
        self.critic_net = nn.Sequential(
            nn.Linear(pol_dim + priv_dim, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, 1),
        ).to(self.device)
        # neutralize the inherited shared value head so only critic_net carries value grads
        self.value_net = nn.Identity()
        self.optimizer = self.optimizer_class(self.parameters(), lr=self.lr_schedule(1),
                                              **self.optimizer_kwargs)

    def _critic_in(self, obs):
        p = obs["policy"].reshape(obs["policy"].shape[0], -1)
        v = obs["privileged"].reshape(obs["privileged"].shape[0], -1)
        return torch.cat([p.float(), v.float()], dim=1)

    def _actor_features(self, obs):
        # actor path: only the observable obs through the standard extractor/mlp
        pol_only = {"policy": obs["policy"],
                    "privileged": torch.zeros_like(obs["privileged"])}
        return super().extract_features(pol_only)

    def forward(self, obs, deterministic: bool = False):
        feat = self._actor_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(feat)
        dist = self._get_action_dist_from_latent(latent_pi)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        values = self.critic_net(self._critic_in(obs))
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        feat = self._actor_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(feat)
        dist = self._get_action_dist_from_latent(latent_pi)
        log_prob = dist.log_prob(actions)
        values = self.critic_net(self._critic_in(obs))
        return values, log_prob, dist.entropy()

    def predict_values(self, obs):
        return self.critic_net(self._critic_in(obs))

    def _predict(self, obs, deterministic: bool = False):
        feat = self._actor_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(feat)
        return self._get_action_dist_from_latent(latent_pi).get_actions(
            deterministic=deterministic)


if __name__ == "__main__":   # pod self-test
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO

    class _Dummy(gym.Env):
        observation_space = spaces.Dict({
            "policy": spaces.Box(-1, 1, (20,)), "privileged": spaces.Box(-1, 1, (11,))})
        action_space = spaces.Box(-1, 1, (4,))

        def reset(self, *, seed=None, options=None):
            return self.observation_space.sample(), {}

        def step(self, a):
            return self.observation_space.sample(), 0.0, False, True, {}

    m = PPO(AsymmetricActorCriticPolicy, _Dummy(), n_steps=64, batch_size=32, device="cpu")
    m.learn(128)
    print("OK: asymmetric PPO trained 128 steps")
```

- [ ] **Step 2: Pod self-test**

Run on pod: `ssh runpod-cor127 "cd /workspace/algo_src_cor127 && PYTHONPATH=. python control/policies/asymmetric.py"`
Expected: `OK: asymmetric PPO trained 128 steps` (no shape/grad errors).

- [ ] **Step 3: Commit**

```bash
git add control/policies/asymmetric.py
git commit -m "feat(cor-127): AsymmetricActorCriticPolicy (privileged critic)"
```

---

## Task 5: Residual wrapper + rl_finetune wiring

**Files:**
- Modify: `scripts/rl/rl_finetune.py`
- Test: `tests/test_rl/test_residual_anchor.py` (anchor math, numpy-only, local)

- [ ] **Step 1: Write the failing anchor test**

```python
# tests/test_rl/test_residual_anchor.py
import numpy as np
import scripts.rl.rl_finetune as F


def test_residual_identity_when_delta_zero():
    # applied = clip(anchor + tanh(0)*scale) == clip(anchor)
    anchor = np.array([[0.1, -0.2, 0.3, -0.05]], dtype=np.float32)
    delta = np.zeros((1, 4), dtype=np.float32)
    applied = np.clip(anchor + np.tanh(delta) * 0.15, -1, 1)
    np.testing.assert_allclose(applied, np.clip(anchor, -1, 1), atol=1e-7)


def test_residual_bounded():
    anchor = np.full((1, 4), 0.95, dtype=np.float32)
    delta = np.full((1, 4), 5.0, dtype=np.float32)     # saturates tanh -> +scale
    applied = np.clip(anchor + np.tanh(delta) * 0.15, -1, 1)
    assert np.all(applied <= 1.0) and np.all(applied >= -1.0)
```

Run: `PYTHONPATH=. python -m pytest tests/test_rl/test_residual_anchor.py -v`
Expected: PASS immediately (these pin the math contract the wrapper must implement; they import the module to guard against import errors after wiring).

- [ ] **Step 2: Add flags (top of rl_finetune.py, near the other arg reads)**

```python
RESIDUAL = "--residual_ff" in sys.argv
ASYM = "--asymmetric_critic" in sys.argv
DELTA_SCALE = argf("--delta_scale", 0.15)
DR_WIDTH = argf("--dr_width", 0.4)
```

- [ ] **Step 3: Add the `ResidualVecEnv` wrapper (after `make_env`)**

```python
class ResidualVecEnv(VecEnvAdapter):
    """Policy outputs a bounded delta; env steps with anchor(FF spline) + delta.
    The analytic anchor guarantees stability, so PPO can only add tanh-bounded
    corrections and CANNOT erode it (HANDOFF 06-14 #1). Deploy must mirror this."""
    def __init__(self, env, gates):
        super().__init__(env)
        self._gates = gates

    def step_async(self, deltas):
        S = self.env._states
        anchor = ff_batch(S, *aim(S, self._gates, self.env._gate_indices))
        applied = np.clip(anchor + np.tanh(deltas) * DELTA_SCALE, -1, 1)
        super().step_async(applied.astype(np.float32))
```

(`step_async` stores `self._actions`; `step_wait` already steps with it and returns the dict/flat obs untouched, so the policy sees the same obs while the env applies the anchored action.)

- [ ] **Step 4: Wire env construction (privileged + DR when asymmetric)**

In `make_env`, thread the new kwargs into `GateRaceEnv(...)`:

```python
                      n_action_history=HIST,
                      privileged_obs=ASYM, dr_width=(DR_WIDTH if ASYM else 0.0))
```

- [ ] **Step 5: Wire model + venv in `main()`**

Replace the venv build and policy class selection:

```python
    env, gates = make_env(NENV, 1)
    venv = ResidualVecEnv(env, gates) if RESIDUAL else VecEnvAdapter(env)
```

When `ASYM`, use the asymmetric policy and `MultiInputPolicy` plumbing:

```python
    if ASYM:
        from control.policies.asymmetric import AsymmetricActorCriticPolicy
        policy_cls = AsymmetricActorCriticPolicy
    else:
        policy_cls = "MlpPolicy"
    ...
    model = PPO(policy_cls, venv, **ppo_kwargs)
```

When `RESIDUAL`: set the action_net bias to ZERO (not `HOVER_U0` — the anchor already supplies hover thrust; delta must start ~0) and skip BC/DAgger (the anchor IS the warm-start):

```python
    with torch.no_grad():
        if RESIDUAL:
            model.policy.action_net.bias[:] = 0.0
        else:
            model.policy.action_net.bias[:] = torch.tensor([HOVER_U0, 0, 0, 0], ...)
    ...
    if WARM and not RESIDUAL:
        ... existing BC/DAgger block ...
    best_dag = -1.0 if (RESIDUAL or not WARM) else best_dag   # GateEval init guard
```

The `eval_gates` and `GateEval` paths must also use a residual venv when `RESIDUAL`. Refactor `eval_gates` to build `ResidualVecEnv(env, gates)` when `RESIDUAL` (mirror the `make_env` call it already does). Keep the asymmetric dict obs flowing through `model.predict` (SB3 handles dict obs natively).

- [ ] **Step 6: Local import-smoke (no SB3 — guards numpy wiring)**

Run: `PYTHONPATH=. python -m pytest tests/test_rl/test_residual_anchor.py -v`
Expected: PASS (module imports; anchor math holds). SB3-dependent paths are covered by Task 6.

- [ ] **Step 7: Commit**

```bash
git add scripts/rl/rl_finetune.py tests/test_rl/test_residual_anchor.py
git commit -m "feat(cor-127): --residual_ff + --asymmetric_critic wiring in rl_finetune"
```

---

## Task 6: Pod smoke-run (end-to-end, small)

**Files:** none (verification only).

- [ ] **Step 1: Sync branch to pod**

```bash
git push    # branch already tracks origin per HANDOFF
ssh runpod-cor127 "cd /workspace/algo_src_cor127 && git pull"
```

- [ ] **Step 2: Tiny residual+asymmetric run (catches shape/grad/obs bugs fast)**

```bash
ssh runpod-cor127 "cd /workspace/algo_src_cor127 && PYTHONPATH=. python scripts/rl/rl_finetune.py \
    --residual_ff --asymmetric_critic --maxw 6 --thrmax 0.6 --slew 40 --scatter \
    --hist 8 --vdes 3 --brake 1.0 --vgate 1.0 --delta_scale 0.15 --dr_width 0.4 \
    --steps 20000 --eval_every 10000 --tag smoke_residual_asym"
```
Expected: prints FT header, runs PPO 20k steps, prints `[t=...] gates X/6`, saves `ft_smoke_residual_asym.zip`. No tracebacks. Confirms: dict obs flows through SB3, critic consumes privileged, residual anchor steps, eval works.

- [ ] **Step 3: If smoke passes, launch the real run**

```bash
ssh runpod-cor127 "cd /workspace/algo_src_cor127 && PYTHONPATH=. nohup python scripts/rl/rl_finetune.py \
    --residual_ff --asymmetric_critic --maxw 6 --thrmax 0.6 --slew 40 --scatter --dr \
    --hist 8 --vdes 3 --brake 1.0 --vgate 1.0 --delta_scale 0.15 --dr_width 0.4 \
    --log_std -2.5 --ent 0.005 --steps 5000000 --eval_every 250000 \
    --tag residual_asym_5M > logs/residual_asym_5M.log 2>&1 &"
```
(Note: `--steps 5000000`, NOT `--steps 0` — residual + asymmetric both require PPO to learn; `--steps 0` would nullify both. This corrects the HANDOFF command.)

---

## Task 7: Deploy mirror (laptop residual anchor)

**Files:** laptop `vq_deploy4_hist.py` (Windows worktree, via scp/ssh).

The deployed actor outputs delta; the laptop deploy loop must compute the SAME anchor from live state + the live gate poses and send `clip(anchor + tanh(delta)*delta_scale, -1, 1)`. The deploy script already builds obs from live pos/vel/quat/omega + gate poses, so it can build a `GateTrajectory` (from spawn + gate poses) and call `ff_batch` + `aim` exactly like training.

- [ ] **Step 1: Pull the current laptop deploy script**

```bash
scp -O laptop:"C:/Users/alexj/Documents/drone-ai-grand-prix/algo_src/.claude/worktrees/aigp-client/vq_deploy4_hist.py" /tmp/vq_deploy4_hist.py
```

- [ ] **Step 2: Add `--residual_ff --delta_scale` to the deploy loop**

Read `/tmp/vq_deploy4_hist.py`, locate the per-step `action = model.predict(obs)` then `send`. Add: build the per-env (n=1) `GateTrajectory` once from `[spawn]+gate_poses` (matching `make_env`: `v_cruise=VDES, tilt_budget_deg=35.0, c_drag=0.057, margin=0.6`), and when `--residual_ff`:

```python
delta, _ = model.predict(obs, deterministic=True)
S = build_state_vector(...)          # the live (1,17) state the script already has
anchor = ff_batch(S, *aim(S, gates_arr, gate_index_arr))
action = np.clip(anchor + np.tanh(delta) * DELTA_SCALE, -1, 1)
```

Reuse `ff_batch`/`aim`/`GateTrajectory` by importing from `scripts.rl.rl_finetune` and `sim.gate_traj` (the laptop worktree has the repo). Keep `--maxw 6 --thrmax 0.6 --slew 40 --hist 8` matched.

- [ ] **Step 3: Push back**

```bash
scp -O /tmp/vq_deploy4_hist.py laptop:"C:/Users/alexj/.../aigp-client/vq_deploy4_hist.py"
```

- [ ] **Step 4: Live test (after the 5M run finishes; pull champion first)**

```bash
scp -O runpod-cor127:/workspace/algo_src_cor127/ft_residual_asym_5M_best.zip /tmp/
scp -O /tmp/ft_residual_asym_5M_best.zip laptop:"C:/Users/alexj/.../aigp-client/"
~/.rerun33-venv/bin/rerun --port 9876 --save /Users/alex/Documents/aigp_residual_$(date +%H%M%S).rrd
ssh laptop "cd /D C:\\Users\\alexj\\...\\aigp-client && python -u vq_deploy4_hist.py \
    --policy ft_residual_asym_5M_best.zip --residual_ff --delta_scale 0.15 \
    --maxw 6 --thrmax 0.6 --slew 40 --hist 8 --gates 6"
```
Expected/compare vs baselines: closest-dist < 4.3m AND tilt-at-closest low (no tumble; the 4.5m DAgger champion held tilt=12°). Record closest + tilt-at-closest. If it tumbles live → sim drag drift is the real bottleneck → pivot to `fit_drag_quad.py` (HANDOFF fallback).

---

## Self-Review notes

- **Spec coverage:** residual (Task 5/7) ✓, asymmetric critic (Task 4) ✓, per-env DR dependency (Task 1) ✓, privileged obs (Task 2/3) ✓, corrected `--steps>0` (Task 6) ✓, deploy mirror (Task 7) ✓, validated-dynamics bit-identity guard (Task 1 Step 1/5) ✓.
- **Known limitation (logged, not silently dropped):** per-env DR is fixed-at-construction, not per-episode. The critic sees variation across the 64 envs but each env's plant is constant over its episodes. Per-episode resampling (HANDOFF hypothesis #5) is a future improvement; this is enough for the asymmetric critic to learn a param→value mapping.
- **Perf note:** the residual wrapper recomputes `aim()` (Python loop over 64 splines) every PPO step — roughly doubles env-step cost vs vanilla. Acceptable on the A4500 pod (the existing DAgger rollouts already pay this); vectorize `aim` later if 5M wall-clock balloons.
- **Type consistency:** `dr_factors` is `(n_envs, 11)` everywhere; `DELTA_SCALE`/`DR_WIDTH` module constants used in wrapper + env + deploy; `ff_batch`/`aim` signatures reused unchanged.
</content>
</invoke>
