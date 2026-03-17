# Multi-Gate Lookahead Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the observation space include N configurable lookahead gates (default 1 = current behavior) so the policy can anticipate upcoming turns.

**Architecture:** Add `n_lookahead_gates` parameter to `GateRaceEnv`. Reorder obs so prev_action sits at [16:20] and the variable-length gate lookahead block fills [20:20+4*N]. Loop over `range(1, N+1)` with per-env modulo wrapping. Factory and config pass the parameter through.

**Tech Stack:** NumPy, Gymnasium, Hydra/OmegaConf, pytest

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `sim/envs/gate_race_env.py` | Modify | Add `n_lookahead_gates` param, compute dynamic `OBS_DIM`, refactor `_compute_obs_batched()` |
| `sim/envs/numpy_quad_factory.py` | Modify | Pass `n_lookahead_gates` through to `GateRaceEnv` |
| `configs/sim/numpy_quad.yaml` | Modify | Add `n_lookahead_gates: 1` default (note: spec referenced `configs/train.yaml` but the correct location is the sim config group where `NumpyQuadEnvFactory` `_target_` is defined) |
| `tests/test_sim/test_obs_vector.py` | Modify | Update hardcoded obs indices for reordered layout, add lookahead tests |

**Other test files importing `OBS_DIM`:** `test_envs.py`, `test_vec_env_adapter.py`, `test_gate_passage.py`, `test_random_start.py`, `tests/test_training/test_train_entrypoint.py` — these all use the default `n_lookahead_gates=1` which preserves `OBS_DIM=24`, so they do NOT need changes. The module-level `OBS_DIM = 24` constant is kept for backwards compatibility with these tests.

---

## Chunk 1: Core Implementation

### Task 1: Update obs layout and add n_lookahead_gates to GateRaceEnv

**Files:**
- Modify: `sim/envs/gate_race_env.py:46-50` (OBS_DIM constant)
- Modify: `sim/envs/gate_race_env.py:256-330` (constructor)
- Modify: `sim/envs/gate_race_env.py:811-888` (`_compute_obs_batched`)
- Modify: `tests/test_sim/test_obs_vector.py`

- [ ] **Step 1: Write failing tests for new obs layout and lookahead**

Add a new test class to `tests/test_sim/test_obs_vector.py`. These tests verify (a) obs dim changes with `n_lookahead_gates`, (b) the reordered layout puts prev_action at [16:20] and gate block at [20:], and (c) multiple lookahead gates produce correct values.

```python
class TestObsLookaheadGates:
    """Multi-gate lookahead observation tests."""

    def test_obs_dim_default(self) -> None:
        """Default n_lookahead_gates=1 gives 24 dims (20 base + 4*1)."""
        env = _make_env()
        obs, _ = env.reset(seed=0)
        obs_flat = obs.flatten()
        assert obs_flat.shape == (24,), f"Expected (24,), got {obs_flat.shape}"

    def test_obs_dim_two_lookahead(self) -> None:
        """n_lookahead_gates=2 gives 28 dims (20 base + 4*2)."""
        env = _make_env(n_lookahead_gates=2)
        obs, _ = env.reset(seed=0)
        obs_flat = obs.flatten()
        assert obs_flat.shape == (28,), f"Expected (28,), got {obs_flat.shape}"

    def test_obs_dim_three_lookahead(self) -> None:
        """n_lookahead_gates=3 gives 32 dims (20 base + 4*3)."""
        env = _make_env(n_lookahead_gates=3)
        obs, _ = env.reset(seed=0)
        obs_flat = obs.flatten()
        assert obs_flat.shape == (32,), f"Expected (32,), got {obs_flat.shape}"

    def test_prev_action_at_16_20(self) -> None:
        """After reorder, prev_action is at [16:20], zeros after reset."""
        env = _make_env()
        obs, _ = env.reset(seed=0)
        obs_flat = obs.flatten()
        np.testing.assert_allclose(
            obs_flat[16:20], [0.0, 0.0, 0.0, 0.0], atol=1e-5,
            err_msg="prev_action at [16:20] should be zero after reset",
        )

    def test_lookahead_gate_values_no_rotation(self) -> None:
        """3 gates in a line along X, yaw=0. Verify lookahead[0] = gate-to-gate vec."""
        # Gate 0 at [0,0,2], Gate 1 at [5,0,2], Gate 2 at [10,3,2]
        env = _make_env(
            gate_positions=[[0.0, 0.0, 2.0], [5.0, 0.0, 2.0], [10.0, 3.0, 2.0]],
            gate_yaws=[0.0, 0.0, 0.0],
            n_lookahead_gates=2,
        )
        env.reset(seed=0)
        obs = env._compute_obs()
        obs_flat = obs.flatten()
        # obs[20:24] = lookahead gate 1: pos(gate1 - gate0) in gate0 yaw frame + yaw delta
        # gate0 yaw=0 so no rotation. delta_pos = [5, 0, 0], yaw_delta = 0
        np.testing.assert_allclose(obs_flat[20:23], [5.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(obs_flat[23], 0.0, atol=1e-5)
        # obs[24:28] = lookahead gate 2: pos(gate2 - gate0) in gate0 yaw frame + yaw delta
        # delta_pos = [10, 3, 0], yaw_delta = 0
        np.testing.assert_allclose(obs_flat[24:27], [10.0, 3.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(obs_flat[27], 0.0, atol=1e-5)

    def test_lookahead_gate_with_rotation(self) -> None:
        """Current gate yaw=pi/2. Lookahead position rotated into gate frame."""
        # Gate 0 at [0,0,2] yaw=pi/2, Gate 1 at [10,5,3] yaw=0
        env = _make_env(
            gate_positions=[[0.0, 0.0, 2.0], [10.0, 5.0, 3.0]],
            gate_yaws=[np.pi / 2, 0.0],
        )
        env.reset(seed=0)
        obs = env._compute_obs()
        obs_flat = obs.flatten()
        # delta_pos = [10, 5, 1]. With gate yaw=pi/2 (cos=0, sin=1):
        # rx = 0*10 + 1*5 = 5, ry = -1*10 + 0*5 = -10
        # obs[20:23] = [5, -10, 1]
        np.testing.assert_allclose(obs_flat[20:23], [5.0, -10.0, 1.0], atol=1e-5)
        # yaw delta = wrap(0 - pi/2) = -pi/2
        np.testing.assert_allclose(obs_flat[23], -np.pi / 2, atol=1e-4)

    def test_lookahead_wraps_around(self) -> None:
        """With n_lookahead_gates=3 on a 3-gate track, gate indices wrap via modulo."""
        env = _make_env(
            gate_positions=[[0.0, 0.0, 2.0], [5.0, 0.0, 2.0], [10.0, 0.0, 2.0]],
            gate_yaws=[0.0, 0.0, 0.0],
            n_lookahead_gates=3,
        )
        env.reset(seed=0)
        obs = env._compute_obs()
        obs_flat = obs.flatten()
        # Lookahead gate 3 = (0+3) % 3 = gate 0 = same as current gate
        # delta_pos = gate0.pos - gate0.pos = [0, 0, 0]
        np.testing.assert_allclose(obs_flat[28:31], [0.0, 0.0, 0.0], atol=1e-5)

    def test_lookahead_exceeds_track_length(self) -> None:
        """n_lookahead_gates=5 on a 4-gate track wraps correctly via modulo."""
        # 4 gates along X-axis at x=0,3,6,9
        env = _make_env(
            gate_positions=[
                [0.0, 0.0, 2.0], [3.0, 0.0, 2.0],
                [6.0, 0.0, 2.0], [9.0, 0.0, 2.0],
            ],
            gate_yaws=[0.0, 0.0, 0.0, 0.0],
            n_lookahead_gates=5,
        )
        env.reset(seed=0)
        obs = env._compute_obs()
        obs_flat = obs.flatten()
        # gate_idx=0. Lookahead k=1 -> gate1, k=2 -> gate2, k=3 -> gate3,
        # k=4 -> (0+4)%4=gate0, k=5 -> (0+5)%4=gate1
        # k=4: delta = gate0 - gate0 = [0,0,0]
        np.testing.assert_allclose(obs_flat[32:35], [0.0, 0.0, 0.0], atol=1e-5)
        # k=5: delta = gate1 - gate0 = [3,0,0]
        np.testing.assert_allclose(obs_flat[36:39], [3.0, 0.0, 0.0], atol=1e-5)

    def test_n_lookahead_gates_validation(self) -> None:
        """n_lookahead_gates < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n_lookahead_gates must be >= 1"):
            _make_env(n_lookahead_gates=0)

    def test_multi_env_different_tracks_lookahead(self) -> None:
        """Two envs with different tracks produce correct per-env lookahead."""
        from sim.tracks import Track
        track_a = Track([
            GateState(np.array([0., 0., 2.]), _quat_for_yaw(0.0)),
            GateState(np.array([5., 0., 2.]), _quat_for_yaw(0.0)),
        ])
        track_b = Track([
            GateState(np.array([0., 0., 2.]), _quat_for_yaw(0.0)),
            GateState(np.array([0., 8., 2.]), _quat_for_yaw(0.0)),
        ])
        env = GateRaceEnv(tracks=[track_a, track_b], n_envs=2, max_steps=10000, ceiling=50.0, n_lookahead_gates=2)
        env.reset(seed=0)
        obs = env._compute_obs_batched()
        # Env 0: gate1-gate0 = [5,0,0] in gate0 frame (yaw=0)
        np.testing.assert_allclose(obs[0, 20:23], [5.0, 0.0, 0.0], atol=1e-5)
        # Env 1: gate1-gate0 = [0,8,0] in gate0 frame (yaw=0)
        np.testing.assert_allclose(obs[1, 20:23], [0.0, 8.0, 0.0], atol=1e-5)
```

Also update the `_make_env` helper to accept `n_lookahead_gates`:

In `_make_env`, add `n_lookahead_gates: int = 1` parameter and pass it to `GateRaceEnv`:

```python
def _make_env(
    gate_positions: list[list[float]] | None = None,
    gate_yaws: list[float] | None = None,
    n_envs: int = 1,
    n_lookahead_gates: int = 1,
) -> GateRaceEnv:
    ...
    return GateRaceEnv(
        track=track,
        n_envs=n_envs,
        max_steps=10000,
        ceiling=50.0,
        n_lookahead_gates=n_lookahead_gates,
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_sim/test_obs_vector.py::TestObsLookaheadGates -v`
Expected: FAIL — `GateRaceEnv.__init__() got an unexpected keyword argument 'n_lookahead_gates'`

- [ ] **Step 3: Implement n_lookahead_gates in GateRaceEnv**

**3a.** Replace the module-level `OBS_DIM = 24` constant at line 50 with a function:

```python
# Default observation dimension (n_lookahead_gates=1):
#   gate_rel_pos(3) + vel(3) + roll_pitch(2) + yaw_rel(1) +
#   body_rates(3) + motor_speeds(4) + prev_action(4) +
#   lookahead_gates(4 * n_lookahead_gates) = 20 + 4*N
OBS_DIM = 24  # Keep for backwards compat (N=1 default)


def _compute_obs_dim(n_lookahead_gates: int) -> int:
    """Compute observation dimension: 20 base + 4 per lookahead gate."""
    return 20 + 4 * n_lookahead_gates
```

**3b.** Add `n_lookahead_gates: int = 1` parameter to `GateRaceEnv.__init__()` (after `tracks`):

```python
        tracks: list[Track] | None = None,
        n_lookahead_gates: int = 1,
    ) -> None:
```

Store it, validate, and compute obs dim:

```python
        if n_lookahead_gates < 1:
            raise ValueError(f"n_lookahead_gates must be >= 1, got {n_lookahead_gates}")
        self._n_lookahead_gates = n_lookahead_gates
        self._obs_dim = _compute_obs_dim(n_lookahead_gates)
```

**3c.** Update `observation_space` to use `self._obs_dim` instead of `OBS_DIM`:

```python
        obs_high = np.full(self._obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
```

**3d.** Refactor `_compute_obs_batched()`. Replace the entire method body. Key changes:
- Use `self._obs_dim` instead of `OBS_DIM` for array allocation
- Move prev_action to [16:20]
- Replace hardcoded next-gate block with a loop over lookahead gates

```python
    def _compute_obs_batched(self) -> NDArray[np.float32]:
        """Compute observation vectors for all environments.

        Observation layout (20 + 4*N dims, where N = n_lookahead_gates):
            [0:3]   position drone -> current gate  (gate-yaw-relative frame)
            [3:6]   velocity                        (gate-yaw-relative frame)
            [6:8]   roll, pitch                     (world-frame Euler angles)
            [8]     yaw relative to gate             (drone_yaw - gate_yaw, wrapped)
            [9:12]  body angular rates (p, q, r)     (body frame)
            [12:16] motor speeds                     (normalized [-1, 1])
            [16:20] previous action                  (normalized [-1, 1])
            [20:20+4*N] lookahead gates              (4 dims each: rel_pos(3) + yaw_delta(1))

        Gate-yaw-relative frame: XY rotated by negative gate yaw, Z unchanged.

        Returns:
            Observations (n_envs, obs_dim).
        """
        obs = np.zeros((self.n_envs, self._obs_dim), dtype=np.float32)

        for i in range(self.n_envs):
            state = self._states[i]
            gate_idx = int(self._gate_indices[i])
            track = self._tracks[i]
            n_gates = track.num_gates
            gate = track.gates[gate_idx % n_gates]

            # Gate yaw from its orientation quaternion
            gate_yaw = _quat_to_yaw(gate.orientation)
            cos_yaw = np.cos(gate_yaw)
            sin_yaw = np.sin(gate_yaw)

            # --- [0:3] Position drone -> current gate (gate-yaw frame) ---
            dpos = state[POS] - gate.position
            obs[i, 0:2] = _rotate_xy(dpos[:2], cos_yaw, sin_yaw)
            obs[i, 2] = dpos[2]

            # --- [3:6] Velocity (gate-yaw frame) ---
            vel_world = state[VEL]
            obs[i, 3:5] = _rotate_xy(vel_world[:2], cos_yaw, sin_yaw)
            obs[i, 5] = vel_world[2]

            # --- [6:8] Roll, Pitch (world frame Euler) ---
            drone_quat = state[QUAT]
            roll, pitch, drone_yaw = _quat_to_euler(drone_quat)
            obs[i, 6] = roll
            obs[i, 7] = pitch

            # --- [8] Yaw relative to gate ---
            obs[i, 8] = _wrap_angle(drone_yaw - gate_yaw)

            # --- [9:12] Body angular rates ---
            obs[i, 9:12] = state[OMEGA]

            # --- [12:16] Motor speeds normalized to [-1, 1] ---
            if self.dynamics._max_omega.size > i:
                max_omega_i = self.dynamics._max_omega[i]
            else:
                max_omega_i = self.params.max_omega
            obs[i, 12:16] = (state[MOTOR] / max(max_omega_i, 1e-10)) * 2.0 - 1.0

            # --- [16:20] Previous action (normalized [-1, 1]) ---
            obs[i, 16:20] = self._prev_actions[i]

            # --- [20:20+4*N] Lookahead gates ---
            for k in range(1, self._n_lookahead_gates + 1):
                lookahead_gate = track.gates[(gate_idx + k) % n_gates]
                # Relative position: lookahead gate - current gate, in current gate yaw frame
                dg = lookahead_gate.position - gate.position
                offset = 20 + 4 * (k - 1)
                obs[i, offset:offset + 2] = _rotate_xy(dg[:2], cos_yaw, sin_yaw)
                obs[i, offset + 2] = dg[2]
                # Yaw delta: lookahead gate yaw - current gate yaw
                lookahead_yaw = _quat_to_yaw(lookahead_gate.orientation)
                obs[i, offset + 3] = _wrap_angle(lookahead_yaw - gate_yaw)

        return obs
```

- [ ] **Step 4: Run the new lookahead tests**

Run: `python -m pytest tests/test_sim/test_obs_vector.py::TestObsLookaheadGates -v`
Expected: All PASS

- [ ] **Step 5: Update existing obs tests for reordered layout**

The obs layout reorder swaps prev_action ([20:24] → [16:20]) and next-gate ([16:20] → [20:24]). Update these existing test classes:

**`TestObsNextGatePosition`** — change index assertions from `obs_flat[16:19]` to `obs_flat[20:23]`:
```python
# In test_next_gate_no_rotation:
np.testing.assert_allclose(obs_flat[20:23], [10.0, 5.0, 1.0], ...)
# In test_next_gate_with_rotation:
np.testing.assert_allclose(obs_flat[20:23], [5.0, -10.0, 1.0], ...)
```

**`TestObsNextGateYaw`** — change from `obs_flat[19]` to `obs_flat[23]`:
```python
# In test_next_gate_yaw_simple:
np.testing.assert_allclose(obs_flat[23], np.pi / 2, ...)
# In test_next_gate_yaw_wrapping:
np.testing.assert_allclose(obs_flat[23], expected, ...)
```

**`TestObsActionHistoryInitial`** — change from `obs_flat[20:24]` to `obs_flat[16:20]`:
```python
prev_action = obs_flat[16:20]
```

**`TestObsActionHistoryAfterStep`** — change all `obs_flat[20:24]` to `obs_flat[16:20]`:
```python
prev_action = obs_flat[16:20]
```

**`TestObsAllDimsNonzero`** — update the dimension comments and index ranges:
```python
# Dims 16:20 (prev action) - should be nonzero since we stepped
for d in range(16, 20):
    assert obs_flat[d] != pytest.approx(0.0, abs=1e-2), ...

# Dims 20:23 (next gate relative pos)
assert obs_flat[20] != pytest.approx(0.0, abs=1e-3), ...

# Dim 23 (next gate yaw relative)
assert obs_flat[23] != pytest.approx(0.0, abs=1e-3), ...
```

**`TestObsShape`** — `test_obs_dim_constant_is_24` still passes since we keep `OBS_DIM = 24` for backwards compat. Update `test_single_env_shape` and `test_multi_env_shape` to use `env._obs_dim` instead of hardcoded 24:
```python
def test_single_env_shape(self) -> None:
    env = _make_env(n_envs=1)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (env._obs_dim,)

def test_multi_env_shape(self) -> None:
    n = 4
    env = _make_env(n_envs=n)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (n, env._obs_dim)
```

**`TestObsConsistencyAcrossEnvs`** — change `(n, 24)` to `(n, env._obs_dim)`:
```python
assert obs.shape == (n, env._obs_dim)
```

**`TestObsAllDimsNonzero`** — update shape check:
```python
assert obs_flat.shape == (env._obs_dim,), ...
```

Also update the module docstring at the top of the test file to reflect the new layout.

- [ ] **Step 6: Run full obs test suite**

Run: `python -m pytest tests/test_sim/test_obs_vector.py -v`
Expected: All PASS

- [ ] **Step 7: Update GateRaceEnv class docstring**

Update the docstring of `GateRaceEnv` (around line 207-220) to reflect the new layout:

```python
    """Gymnasium environment for quadrotor gate racing.

    Observation (20 + 4*N dims, where N = n_lookahead_gates, default 1):
        [0:3]   position to current gate (gate-yaw-relative frame)
        [3:6]   velocity (gate-yaw-relative frame)
        [6:8]   roll, pitch (world frame Euler angles)
        [8]     yaw relative to gate (drone_yaw - gate_yaw, wrapped [-pi, pi])
        [9:12]  body angular rates p, q, r (body frame)
        [12:16] motor speeds (normalized to [-1, 1]: (w / w_max) * 2 - 1)
        [16:20] previous action / motor commands (normalized [-1, 1])
        [20:20+4*N] lookahead gates (4 dims each: relative pos(3) + yaw delta(1),
                    all in current gate's yaw frame, wrapping via modulo)
    ...
    """
```

Also update the obs comment block at lines 46-50, and the `_compute_obs` squeeze wrapper docstring (line ~893) which references `OBS_DIM`:

```python
    def _compute_obs(self) -> NDArray[np.float32]:
        """Compute observation vectors, squeezing for single env.

        Returns:
            Observations (n_envs, obs_dim) or (obs_dim,) for single env.
        """
```

- [ ] **Step 8: Commit**

```bash
git add sim/envs/gate_race_env.py tests/test_sim/test_obs_vector.py
git commit -m "feat(sim): add configurable multi-gate lookahead to observation space

Adds n_lookahead_gates parameter (default 1 = current behavior).
Reorders obs: prev_action moves to [16:20], gate lookahead block
fills [20:20+4*N] with per-env modulo wrapping."
```

---

### Task 2: Wire n_lookahead_gates through factory and config

**Files:**
- Modify: `sim/envs/numpy_quad_factory.py:55-74` (constructor), `162-210` (`_build`)
- Modify: `configs/sim/numpy_quad.yaml`

- [ ] **Step 1: Write failing test**

Add to `tests/test_sim/test_procedural_training.py` (or a new focused test):

```python
def test_factory_passes_n_lookahead_gates():
    """Factory passes n_lookahead_gates to GateRaceEnv."""
    from omegaconf import OmegaConf
    from sim.envs.numpy_quad_factory import NumpyQuadEnvFactory

    factory = NumpyQuadEnvFactory(n_envs=2, n_lookahead_gates=3)
    dr_cfg = OmegaConf.create({"enabled": False})
    reward_cfg = OmegaConf.create({"weights": {"gate_passage": 1.0}, "v_max": 10.0})
    vec_env = factory.make_vec_env(dr_cfg, reward_cfg)
    raw_env = vec_env.env
    assert raw_env._n_lookahead_gates == 3
    assert raw_env._obs_dim == 20 + 4 * 3  # 32
    vec_env.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sim/test_procedural_training.py::test_factory_passes_n_lookahead_gates -v`
Expected: FAIL — `NumpyQuadEnvFactory.__init__() got an unexpected keyword argument 'n_lookahead_gates'`

- [ ] **Step 3: Add n_lookahead_gates to factory**

In `sim/envs/numpy_quad_factory.py`:

Add to constructor signature (after `seed: int = 42`):
```python
        n_lookahead_gates: int = 1,
```

Store it:
```python
        self._n_lookahead_gates = n_lookahead_gates
```

In `_build()`, pass to `GateRaceEnv`:
```python
        env = GateRaceEnv(
            ...
            tracks=tracks,
            n_lookahead_gates=self._n_lookahead_gates,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sim/test_procedural_training.py::test_factory_passes_n_lookahead_gates -v`
Expected: PASS

- [ ] **Step 5: Add config default**

Add to `configs/sim/numpy_quad.yaml` (after `arena_bounds` or at end before `params:`):
```yaml
# Multi-gate lookahead: number of future gates in observation (1 = current behavior)
n_lookahead_gates: 1
```

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest tests/test_sim/ -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add sim/envs/numpy_quad_factory.py configs/sim/numpy_quad.yaml tests/test_sim/test_procedural_training.py
git commit -m "feat(sim): wire n_lookahead_gates through factory and config"
```

---

### Task 3: Smoke test with n_lookahead_gates=2

**Files:**
- Modify: `tests/test_sim/test_procedural_training.py`

- [ ] **Step 1: Write integration smoke test**

```python
def test_training_smoke_with_lookahead():
    """Verify training runs with n_lookahead_gates=2 (28-dim obs)."""
    from omegaconf import OmegaConf
    from sim.envs.numpy_quad_factory import NumpyQuadEnvFactory

    factory = NumpyQuadEnvFactory(
        n_envs=2,
        n_lookahead_gates=2,
        max_steps=50,
    )
    dr_cfg = OmegaConf.create({"enabled": False})
    reward_cfg = OmegaConf.create({
        "weights": {"gate_passage": 1.0, "gate_progress": 1.0},
        "v_max": 10.0,
    })
    vec_env = factory.make_vec_env(dr_cfg, reward_cfg)

    # Verify obs space
    assert vec_env.observation_space.shape == (28,)

    # Step through a few iterations
    obs = vec_env.reset()
    assert obs.shape == (2, 28)
    for _ in range(10):
        action = np.zeros((2, 4), dtype=np.float32)
        obs, rewards, dones, infos = vec_env.step(action)
        assert obs.shape == (2, 28)
        assert np.all(np.isfinite(obs))

    vec_env.close()
```

- [ ] **Step 2: Run integration test**

Run: `python -m pytest tests/test_sim/test_procedural_training.py::test_training_smoke_with_lookahead -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_sim/test_procedural_training.py
git commit -m "test(sim): add smoke test for multi-gate lookahead training"
```
