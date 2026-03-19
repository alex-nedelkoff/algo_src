# Quick Wins: Boundary Safety + Gate Approach + Training Stability

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 19% OOB regression and 67% gate collision rate from the speed push, while adding entropy annealing and LR decay for more stable training.

**Architecture:** Two new reward functions (boundary proximity penalty, gate approach angle bonus), curriculum callback extension to support `ent_coef` updates, and LR schedule via SB3's native callable support. All compose into a single experiment config resuming from the speed_perception checkpoint.

**Tech Stack:** NumPy, SB3 PPO, Hydra/OmegaConf

---

## Chunk 1: Reward Functions + Curriculum Extension + Config

### File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `sim/rewards.py` | Add `boundary_penalty()` and `gate_approach_reward()` |
| Modify | `sim/envs/gate_race_env.py` | Wire two new reward components |
| Modify | `training/curriculum_callback.py` | Extend `_apply_stage` to set `ent_coef` on PPO model |
| Modify | `control/algorithms/ppo.py` | Accept `learning_rate` as schedule (callable) |
| Create | `configs/experiment/quick_wins.yaml` | Experiment config with all fixes |
| Modify | `tests/test_sim/test_rewards.py` | Tests for new rewards |
| Modify | `tests/test_sim/test_envs.py` | Tests for env wiring |

### Task 1: Boundary Penalty + Gate Approach Reward Functions

**Files:**
- Modify: `sim/rewards.py`
- Modify: `tests/test_sim/test_rewards.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_sim/test_rewards.py`:

```python
from sim.rewards import boundary_penalty, gate_approach_reward


class TestBoundaryPenalty:
    def test_center_of_arena_zero_penalty(self):
        """Far from walls → no penalty."""
        assert boundary_penalty(pos_xy=np.array([0.0, 0.0]), arena_bounds=10.0, margin=3.0) == pytest.approx(0.0)

    def test_at_margin_starts_penalty(self):
        """At margin distance from wall → small penalty."""
        p = boundary_penalty(pos_xy=np.array([7.0, 0.0]), arena_bounds=10.0, margin=3.0)
        assert p < 0.0  # penalty is negative

    def test_at_wall_max_penalty(self):
        """At arena edge → penalty = -1.0."""
        p = boundary_penalty(pos_xy=np.array([10.0, 0.0]), arena_bounds=10.0, margin=3.0)
        assert p == pytest.approx(-1.0)

    def test_outside_wall_capped(self):
        """Beyond arena edge → still -1.0 (capped)."""
        p = boundary_penalty(pos_xy=np.array([12.0, 0.0]), arena_bounds=10.0, margin=3.0)
        assert p == pytest.approx(-1.0)

    def test_y_axis_also_penalized(self):
        """Penalty applies to whichever axis is closer to boundary."""
        p = boundary_penalty(pos_xy=np.array([0.0, 9.0]), arena_bounds=10.0, margin=3.0)
        assert p < 0.0


class TestGateApproachReward:
    def test_aligned_velocity_max_reward(self):
        """Velocity perfectly aligned with gate normal → 1.0."""
        r = gate_approach_reward(
            velocity=np.array([1.0, 0.0, 0.0]),
            gate_normal=np.array([1.0, 0.0, 0.0]),
        )
        assert r == pytest.approx(1.0, abs=0.01)

    def test_perpendicular_velocity_zero(self):
        """Velocity perpendicular to gate normal → 0.0."""
        r = gate_approach_reward(
            velocity=np.array([0.0, 1.0, 0.0]),
            gate_normal=np.array([1.0, 0.0, 0.0]),
        )
        assert r == pytest.approx(0.0, abs=0.01)

    def test_opposite_velocity_zero(self):
        """Flying away from gate → 0.0 (not negative)."""
        r = gate_approach_reward(
            velocity=np.array([-1.0, 0.0, 0.0]),
            gate_normal=np.array([1.0, 0.0, 0.0]),
        )
        assert r == pytest.approx(0.0)

    def test_zero_velocity_zero_reward(self):
        """Stationary → 0.0."""
        r = gate_approach_reward(
            velocity=np.array([0.0, 0.0, 0.0]),
            gate_normal=np.array([1.0, 0.0, 0.0]),
        )
        assert r == pytest.approx(0.0)

    def test_diagonal_approach(self):
        """45 degree approach → cos(45) ≈ 0.707."""
        r = gate_approach_reward(
            velocity=np.array([1.0, 1.0, 0.0]),
            gate_normal=np.array([1.0, 0.0, 0.0]),
        )
        assert 0.6 < r < 0.8
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_sim/test_rewards.py::TestBoundaryPenalty tests/test_sim/test_rewards.py::TestGateApproachReward -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement both functions**

Append to `sim/rewards.py`:

```python
def boundary_penalty(
    pos_xy: NDArray[np.float64],
    arena_bounds: float,
    margin: float = 3.0,
) -> float:
    """Penalty for proximity to arena boundaries.

    Quadratic penalty that activates within `margin` meters of the arena edge.
    Returns 0 when safely inside, -1.0 at the wall.

    r = -max(0, 1 - d_wall / margin)^2

    where d_wall = arena_bounds - max(|x|, |y|).

    Args:
        pos_xy: [x, y] position (world frame).
        arena_bounds: Half-width of the square arena in meters.
        margin: Distance from wall where penalty starts.

    Returns:
        Penalty in [-1, 0].
    """
    d_wall = arena_bounds - max(abs(float(pos_xy[0])), abs(float(pos_xy[1])))
    if d_wall >= margin:
        return 0.0
    penetration = max(0.0, 1.0 - d_wall / margin)
    return -(penetration * penetration)


def gate_approach_reward(
    velocity: NDArray[np.float64],
    gate_normal: NDArray[np.float64],
) -> float:
    """Reward for approaching a gate with velocity aligned to its normal.

    r = max(0, cos(angle between velocity and gate_normal))

    Positive when flying through the gate (aligned), zero when perpendicular
    or flying away. Rewards clean gate entries.

    Args:
        velocity: [vx, vy, vz] drone velocity in world frame.
        gate_normal: [nx, ny, nz] gate forward-facing normal vector.

    Returns:
        Reward in [0, 1].
    """
    speed = np.linalg.norm(velocity)
    if speed < 1e-6:
        return 0.0
    cos_angle = np.dot(velocity, gate_normal) / (speed * max(np.linalg.norm(gate_normal), 1e-6))
    return max(0.0, float(cos_angle))
```

Also add to `DEFAULT_WEIGHTS`:
```python
"boundary_penalty": 0.0,
"gate_approach": 0.0,
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_sim/test_rewards.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add sim/rewards.py tests/test_sim/test_rewards.py
git commit -m "feat(sim): add boundary penalty and gate approach reward functions"
```

### Task 2: Wire Into GateRaceEnv

**Files:**
- Modify: `sim/envs/gate_race_env.py`
- Modify: `tests/test_sim/test_envs.py`

- [ ] **Step 1: Update reward component constants**

At the top of `gate_race_env.py`, update:

```python
RC_SPEED_BONUS = 8
RC_BOUNDARY_PENALTY = 9
RC_GATE_APPROACH = 10
NUM_REWARD_COMPONENTS = 11
REWARD_COMPONENT_NAMES = [
    "progress", "body_rate", "action_smooth",
    "gate_passage", "gate_offset", "crash_penalty",
    "spline_proximity", "heading_alignment", "speed_bonus",
    "boundary_penalty", "gate_approach",
]
```

Add imports:
```python
from sim.rewards import (
    gate_offset_penalty, monorace_reward,
    spline_proximity_reward, heading_alignment_reward,
    speed_bonus_reward, boundary_penalty, gate_approach_reward,
)
```

- [ ] **Step 2: Add boundary penalty computation**

In `step()`, after the speed bonus pre-compute block, add:

```python
        # Pre-compute boundary penalty (vectorized)
        boundary_weight = (self.reward_weights or {}).get("boundary_penalty", 0.0)
        if boundary_weight != 0.0:
            _boundary_penalties = np.array([
                boundary_weight * boundary_penalty(
                    self._states[i, :2], self.arena_bounds, margin=3.0
                ) for i in range(self.n_envs)
            ], dtype=np.float64)
        else:
            _boundary_penalties = np.zeros(self.n_envs, dtype=np.float64)
```

Then inside the per-env reward loop (after the speed bonus block), add:

```python
            # Boundary penalty (pre-computed)
            if boundary_weight != 0.0:
                rewards[i] += _boundary_penalties[i]
                self._step_reward_components[i, RC_BOUNDARY_PENALTY] = _boundary_penalties[i]
```

- [ ] **Step 3: Add gate approach reward**

Inside the per-env reward loop, after the boundary penalty, add:

```python
            # Gate approach reward (needs per-env gate normal)
            approach_weight = (self.reward_weights or {}).get("gate_approach", 0.0)
            if approach_weight != 0.0:
                gate = self._tracks[i].gates[int(self._gate_indices[i]) % self._tracks[i].num_gates]
                normal = _gate_normal(gate)
                vel = self._states[i, VEL]
                approach_val = approach_weight * gate_approach_reward(vel, normal)
                rewards[i] += approach_val
                self._step_reward_components[i, RC_GATE_APPROACH] = approach_val
```

- [ ] **Step 4: Write tests**

Add to `tests/test_sim/test_envs.py`:

```python
class TestBoundaryAndApproachRewards:
    def test_boundary_penalty_in_components(self):
        from sim.tracks import build_figure8_track
        env = GateRaceEnv(
            track=build_figure8_track(), n_envs=1, dt=0.01, max_steps=10,
            reward_weights={"boundary_penalty": 5.0, "gate_approach": 0.1,
                            "gate_progress": 1.0, "gate_passage": 1.5, "crash_penalty": 10.0},
        )
        env.reset()
        env.step(np.zeros((1, 4), dtype=np.float32))
        names, _ = env.get_step_reward_components(0)
        assert "boundary_penalty" in names
        assert "gate_approach" in names
```

- [ ] **Step 5: Run all tests**

Run: `pytest tests/test_sim/test_envs.py tests/test_sim/test_rewards.py -v --timeout=30`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add sim/envs/gate_race_env.py tests/test_sim/test_envs.py
git commit -m "feat(sim): wire boundary penalty and gate approach reward into env"
```

### Task 3: Curriculum Entropy Annealing

**Files:**
- Modify: `training/curriculum_callback.py`
- Modify: `tests/test_training/test_curriculum_callback.py`

- [ ] **Step 1: Extend CurriculumSB3Callback._apply_stage**

The curriculum config stages can now include an `ent_coef` field. When present, update the PPO model's entropy coefficient at runtime.

In `training/curriculum_callback.py`, modify `CurriculumSB3Callback.__init__` to also accept the PPO model:

```python
def __init__(self, curriculum, env, model=None, verbose=1):
    super().__init__(verbose)
    self.curriculum = curriculum
    self.env = env
    self._ppo_model = model
    self._applied_stage = -1
```

Extend `_apply_stage`:

```python
def _apply_stage(self):
    stage = self.curriculum.get_current_stage_config()
    self._applied_stage = self.curriculum.current_stage

    rw = stage.get("reward_weights")
    if rw is not None:
        self.env.set_reward_weights(rw)

    v_max = stage.get("v_max")
    if v_max is not None:
        self.env.set_v_max(v_max)

    # Entropy coefficient annealing
    ent_coef = stage.get("ent_coef")
    model = self._ppo_model or (self.model if hasattr(self, 'model') else None)
    if ent_coef is not None and model is not None:
        model.ent_coef = ent_coef
        log.info("Curriculum: set ent_coef=%.4f", ent_coef)
```

Note: `self.model` is set by SB3's `init_callback()` automatically, so we can fall back to it.

- [ ] **Step 2: Add test**

```python
# tests/test_training/test_curriculum_callback.py — add:
def test_ent_coef_in_stage_config(self):
    """ent_coef field should be accessible from stage config."""
    config = {
        "enabled": True,
        "stages": [
            {"timestep": 0, "ent_coef": 0.005, "reward_weights": {}, "v_max": 10.0, "trigger": None},
            {"timestep": 1000, "ent_coef": 0.001, "reward_weights": {}, "v_max": 10.0, "trigger": None},
        ],
    }
    cb = CurriculumCallback(config)
    assert cb.get_current_stage_config()["ent_coef"] == 0.005
    cb.advance()
    assert cb.get_current_stage_config()["ent_coef"] == 0.001
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_training/test_curriculum_callback.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add training/curriculum_callback.py tests/test_training/test_curriculum_callback.py
git commit -m "feat(training): extend curriculum callback to support ent_coef annealing"
```

### Task 4: LR Schedule + Experiment Config

SB3's PPO natively accepts `learning_rate` as a callable `schedule(progress_remaining) -> float` where `progress_remaining` goes from 1.0 to 0.0 over training. No code change needed — just pass a lambda or use a config-driven schedule.

**Files:**
- Modify: `control/algorithms/ppo.py` (accept schedule)
- Create: `configs/experiment/quick_wins.yaml`

- [ ] **Step 1: Support LR schedule in PPO wrapper**

In `control/algorithms/ppo.py`, the `learning_rate` is already passed directly to SB3 PPO which accepts callables. Just update the type hint:

```python
learning_rate: float | Callable[[float], float] = 3e-4,
```

And add a factory for cosine decay:

```python
@staticmethod
def cosine_lr_schedule(initial_lr: float, final_lr: float):
    """Create a cosine annealing LR schedule for SB3.

    Args:
        initial_lr: Starting learning rate.
        final_lr: Minimum learning rate at end of training.

    Returns:
        Callable that maps progress_remaining (1.0→0.0) to LR.
    """
    import math
    def schedule(progress_remaining: float) -> float:
        return final_lr + 0.5 * (initial_lr - final_lr) * (1 + math.cos(math.pi * (1 - progress_remaining)))
    return schedule
```

- [ ] **Step 2: Create experiment config**

```yaml
# configs/experiment/quick_wins.yaml
# @package _global_
# Quick wins: boundary penalty, gate approach reward, entropy annealing, LR decay.

defaults:
  - override /sim: numpy_quad
  - override /control: ppo
  - override /perception: corner_noise
  - override /domain_rand: uniform_30pct
  - override /logging: wandb
  - _self_

sim:
  action_mode: trpy
  n_lookahead_gates: 2
  max_steps: 3000
  arena_bounds: 10
  gate_passage_radius: 1.5

total_timesteps: 20_000_000

# Resume from speed+perception run
resume_checkpoint: null  # UPDATE with latest checkpoint path

track_gen:
  n_gates_min: 4
  n_gates_max: 8
  gate_spacing_min: 1.5
  gate_spacing_max: 4.0
  turn_angle_min: -120
  turn_angle_max: 120
  elevation_min: 1.0
  elevation_max: 3.5
  elevation_delta_max: 0.6
  closure_max_angle: 90
  closure_max_retries: 10

perception:
  noise_scale: 1.0
  dropout_onset: null
  dropout_continuation: 0.7

reward:
  weights:
    gate_passage: 50.0
    gate_progress: 0.5
    gate_offset: 2.0
    body_rate: 0.001
    action_smoothness: 0.0
    crash_penalty: 5.0
    spline_proximity: 0.0
    heading_alignment: 0.0
    speed_bonus: 0.3
    boundary_penalty: 5.0       # NEW: strong boundary avoidance
    gate_approach: 0.2          # NEW: reward aligned gate entry
  v_max: 6.0
  action_smoothness_threshold: 0.5

arpo:
  enabled: false

curriculum:
  enabled: true
  stages:
    - timestep: 0
      reward_weights:
        gate_passage: 50.0
        gate_progress: 0.5
        crash_penalty: 5.0
        speed_bonus: 0.3
        boundary_penalty: 5.0
        gate_approach: 0.2
      v_max: 6.0
      ent_coef: 0.005          # starting entropy
      trigger: null
    - timestep: 5_000_000
      reward_weights:
        gate_passage: 50.0
        gate_progress: 0.3
        crash_penalty: 5.0
        speed_bonus: 0.5
        boundary_penalty: 3.0
        gate_approach: 0.3
      v_max: 8.0
      ent_coef: 0.003          # reduce exploration
      trigger:
        metric: racing/gate_passage_rate
        threshold: 0.95
        window: 200
    - timestep: 12_000_000
      reward_weights:
        gate_passage: 50.0
        gate_progress: 0.0
        crash_penalty: 3.0
        speed_bonus: 0.8
        boundary_penalty: 2.0
        gate_approach: 0.3
      v_max: 12.0
      ent_coef: 0.001          # tight policy
      trigger:
        metric: racing/avg_speed
        threshold: 7.0
        window: 200

multi_scene:
  enabled: true
  n_scenes: 10

# LR cosine decay: 3e-4 → 5e-5
control:
  learning_rate: 0.0003  # will be overridden to schedule in training loop

logging:
  tags: ["trpy", "quick-wins", "boundary", "approach", "entropy-anneal", "20M"]
  group: quick_wins
  notes: "Quick wins: boundary penalty, gate approach, ent_coef anneal 0.005->0.001, LR cosine 3e-4->5e-5"
```

Note: The LR schedule needs to be applied programmatically since YAML can't represent callables. Add to `training/loops/rl.py` a check: if `cfg.control.get("lr_schedule")` is set, replace `learning_rate` with a schedule. OR simply hardcode the schedule in the experiment for now.

- [ ] **Step 3: Wire LR schedule in training loop**

In `training/loops/rl.py`, after building PPO but before training, add:

```python
# Apply LR schedule if configured
lr_schedule_cfg = cfg.get("lr_schedule")
if lr_schedule_cfg is not None and ppo._model is not None:
    from control.algorithms.ppo import PPO as PPOWrapper
    schedule = PPOWrapper.cosine_lr_schedule(
        initial_lr=lr_schedule_cfg.get("initial_lr", 3e-4),
        final_lr=lr_schedule_cfg.get("final_lr", 5e-5),
    )
    ppo._model.learning_rate = schedule
    ppo._model.lr_schedule = lambda _: schedule  # SB3 internal
```

Add to `configs/experiment/quick_wins.yaml`:
```yaml
lr_schedule:
  initial_lr: 0.0003
  final_lr: 0.00005
```

- [ ] **Step 4: Update reward configs with new defaults**

Add `boundary_penalty: 0.0` and `gate_approach: 0.0` to all reward yaml files for backwards compat.

- [ ] **Step 5: Run all tests**

Run: `pytest tests/test_sim/ tests/test_training/ tests/test_integration/ -v --timeout=60`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add control/algorithms/ppo.py training/loops/rl.py configs/experiment/quick_wins.yaml configs/reward/
git commit -m "feat: add quick wins config with boundary penalty, gate approach, entropy anneal, LR decay"
```

---

## Summary

| Task | What | Effort |
|------|------|--------|
| 1 | Two pure reward functions (boundary + approach) | 15 min |
| 2 | Wire into GateRaceEnv (2 new RC_ components) | 15 min |
| 3 | Extend curriculum to set ent_coef | 10 min |
| 4 | LR schedule + experiment config | 15 min |

**To launch after implementation:**
```bash
python -m training +experiment=quick_wins resume_checkpoint=<latest_checkpoint_path>
```

**Expected impact:**
- OOB: 19% → <5% (boundary penalty provides continuous gradient away from walls)
- Gate collisions: 67% → <40% (approach angle reward teaches clean entries)
- Training stability: entropy anneal 0.005→0.001 should close the train/eval gap
- Speed: LR decay prevents destabilizing the policy at high speeds
