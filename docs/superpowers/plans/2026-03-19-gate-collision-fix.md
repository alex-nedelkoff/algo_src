# Gate Collision Fix Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce gate collisions from 73.5% to <40% of training terminations by adding a continuous centering reward near gates and taming exploration noise.

**Architecture:** One new reward function (`gate_centering_reward`) that activates within a configurable radius of the gate plane, penalizing lateral offset with a strength that increases as the drone gets closer. Combined with forced low entropy (ent_coef=0.001) to reduce the exploration noise that causes most collisions.

**Tech Stack:** NumPy, SB3 PPO, Hydra/OmegaConf

**Root cause analysis:** The current system only signals centering quality at the binary pass/fail moment. The drone gets no gradient about *how centered* its approach is until it crosses the gate plane. At 5 m/s with std=3.1, a single noisy action can push the drone 10-20cm off course — enough to clip the 1.5m gate edge. The fix provides continuous centering feedback during approach.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `sim/rewards.py` | Add `gate_centering_reward()` function |
| Modify | `sim/envs/gate_race_env.py` | Add RC_GATE_CENTERING, compute in per-env loop |
| Create | `configs/experiment/gate_fix.yaml` | Experiment config with centering + low entropy |
| Modify | `tests/test_sim/test_rewards.py` | Tests for centering reward |
| Modify | `tests/test_sim/test_envs.py` | Test env wiring |

### Task 1: Gate Centering Reward Function

The reward gives continuous feedback about lateral offset from gate center as the drone approaches. Strength increases inversely with distance to the gate plane (stronger near the gate, zero far away).

**Formula:** `r = -lateral_offset / gate_radius * (1 / (1 + dist_to_plane²))`

This gives:
- Zero penalty when far from gate (dist_to_plane >> 1)
- Strong penalty when near gate plane AND off-center
- Normalized by gate_radius so reward scale is consistent

**Files:**
- Modify: `sim/rewards.py`
- Modify: `tests/test_sim/test_rewards.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_sim/test_rewards.py`:

```python
from sim.rewards import gate_centering_reward


class TestGateCenteringReward:
    def test_centered_at_gate_zero_penalty(self):
        """On-center at gate plane → zero penalty."""
        r = gate_centering_reward(
            lateral_offset=0.0, dist_to_plane=0.0, gate_radius=1.5
        )
        assert r == pytest.approx(0.0)

    def test_off_center_at_gate_max_penalty(self):
        """At gate edge on gate plane → penalty = -1.0."""
        r = gate_centering_reward(
            lateral_offset=1.5, dist_to_plane=0.0, gate_radius=1.5
        )
        assert r == pytest.approx(-1.0)

    def test_off_center_far_from_gate_small_penalty(self):
        """Off-center but far from gate → small penalty."""
        r = gate_centering_reward(
            lateral_offset=1.5, dist_to_plane=5.0, gate_radius=1.5
        )
        assert abs(r) < 0.1  # heavily attenuated by distance

    def test_half_offset_at_gate(self):
        """Half-offset at gate plane → penalty = -0.5."""
        r = gate_centering_reward(
            lateral_offset=0.75, dist_to_plane=0.0, gate_radius=1.5
        )
        assert r == pytest.approx(-0.5)

    def test_penalty_increases_as_approaching(self):
        """Penalty should increase as drone gets closer to gate plane."""
        far = gate_centering_reward(lateral_offset=1.0, dist_to_plane=3.0, gate_radius=1.5)
        near = gate_centering_reward(lateral_offset=1.0, dist_to_plane=0.5, gate_radius=1.5)
        assert near < far  # more negative = more penalty when closer

    def test_zero_radius_returns_zero(self):
        """Zero gate radius → no penalty (avoid division by zero)."""
        r = gate_centering_reward(lateral_offset=1.0, dist_to_plane=0.0, gate_radius=0.0)
        assert r == pytest.approx(0.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_sim/test_rewards.py::TestGateCenteringReward -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement**

Append to `sim/rewards.py`:

```python
def gate_centering_reward(
    lateral_offset: float,
    dist_to_plane: float,
    gate_radius: float,
) -> float:
    """Continuous centering reward that activates near gate plane.

    Penalizes lateral offset from gate center, with strength increasing
    as the drone approaches the gate plane. Inspired by Song et al. (2021)
    safety reward for drone racing.

    r = -(lateral_offset / gate_radius) * (1 / (1 + dist_to_plane^2))

    At the gate plane (dist=0): penalty proportional to how off-center.
    Far from gate (dist>>1): penalty attenuated to near zero.

    Args:
        lateral_offset: Distance from gate center perpendicular to gate
            normal, in meters. Always >= 0.
        dist_to_plane: Signed distance to gate plane along gate normal,
            in meters. Uses abs() internally.
        gate_radius: Gate passage radius in meters (normalizes penalty).

    Returns:
        Penalty in [-1, 0]. Zero when centered or far from gate.
    """
    if gate_radius <= 0.0:
        return 0.0
    proximity = 1.0 / (1.0 + dist_to_plane * dist_to_plane)
    return -(abs(lateral_offset) / gate_radius) * proximity
```

Add to `DEFAULT_WEIGHTS`:
```python
"gate_centering": 0.0,
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_sim/test_rewards.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add sim/rewards.py tests/test_sim/test_rewards.py
git commit -m "feat(sim): add gate_centering_reward for continuous approach centering"
```

### Task 2: Wire Gate Centering into GateRaceEnv

**Files:**
- Modify: `sim/envs/gate_race_env.py`
- Modify: `tests/test_sim/test_envs.py`

- [ ] **Step 1: Update reward component constants**

```python
RC_GATE_CENTERING = 11
NUM_REWARD_COMPONENTS = 12
REWARD_COMPONENT_NAMES = [
    "progress", "body_rate", "action_smooth",
    "gate_passage", "gate_offset", "crash_penalty",
    "spline_proximity", "heading_alignment", "speed_bonus",
    "boundary_penalty", "gate_approach", "gate_centering",
]
```

Add import: `gate_centering_reward` to the existing import line from `sim.rewards`.

- [ ] **Step 2: Add centering computation in per-env reward loop**

Inside the per-env loop, after the gate_approach block and before the plane-crossing detection section, add:

```python
            # Gate centering reward (continuous, distance-attenuated)
            centering_weight = (self.reward_weights or {}).get("gate_centering", 0.0)
            if centering_weight != 0.0:
                gate_normal = _gate_normal(gate)
                rel_pos = self._states[i, POS] - gate.position
                # Distance to gate plane (signed, along normal)
                dist_to_plane = float(np.dot(rel_pos, gate_normal))
                # Lateral offset (perpendicular to normal)
                lateral_vec = rel_pos - dist_to_plane * gate_normal
                lateral_offset = float(np.linalg.norm(lateral_vec))
                centering_val = centering_weight * gate_centering_reward(
                    lateral_offset, dist_to_plane, self.gate_passage_radius
                )
                rewards[i] += centering_val
                self._step_reward_components[i, RC_GATE_CENTERING] = centering_val
```

- [ ] **Step 3: Write env test**

Add to `tests/test_sim/test_envs.py`:

```python
class TestGateCenteringReward:
    def test_centering_component_present(self):
        from sim.tracks import build_figure8_track
        env = GateRaceEnv(
            track=build_figure8_track(), n_envs=1, dt=0.01, max_steps=10,
            reward_weights={"gate_centering": 3.0, "gate_progress": 1.0,
                            "gate_passage": 1.5, "crash_penalty": 10.0},
        )
        env.reset()
        env.step(np.zeros((1, 4), dtype=np.float32))
        names, components = env.get_step_reward_components(0)
        assert "gate_centering" in names
        # Should have some nonzero value (drone starts offset from gate center)
        centering_idx = names.index("gate_centering")
        assert components[centering_idx] != 0.0
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/test_sim/test_envs.py tests/test_sim/test_rewards.py -v --timeout=30`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add sim/envs/gate_race_env.py tests/test_sim/test_envs.py
git commit -m "feat(sim): wire gate centering reward into GateRaceEnv"
```

### Task 3: Experiment Config — Gate Fix

**Files:**
- Create: `configs/experiment/gate_fix.yaml`
- Update reward configs with `gate_centering: 0.0` default

- [ ] **Step 1: Add defaults to reward configs**

Add `gate_centering: 0.0` to monorace.yaml, monorace_safe.yaml, perception_aware.yaml, spline_guided.yaml (after `gate_approach`).

- [ ] **Step 2: Create experiment config**

```yaml
# configs/experiment/gate_fix.yaml
# @package _global_
# Fix gate collisions: centering reward + forced low entropy.
# Resumes from quick_wins checkpoint.

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
resume_checkpoint: outputs/2026-03-19/01-17-07/final_model.zip

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
    boundary_penalty: 5.0
    gate_approach: 0.2
    gate_centering: 3.0        # NEW: continuous centering near gates
  v_max: 6.0
  action_smoothness_threshold: 0.5

# Force low entropy from start (don't wait for curriculum)
control:
  ent_coef: 0.001             # was 0.005 — tame exploration immediately

arpo:
  enabled: false

# Curriculum with already-low entropy
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
        gate_centering: 3.0
      v_max: 6.0
      ent_coef: 0.001
      trigger: null
    - timestep: 5_000_000
      reward_weights:
        gate_passage: 50.0
        gate_progress: 0.3
        crash_penalty: 5.0
        speed_bonus: 0.5
        boundary_penalty: 3.0
        gate_approach: 0.3
        gate_centering: 2.0
      v_max: 8.0
      ent_coef: 0.0008
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
        gate_centering: 1.0
      v_max: 12.0
      ent_coef: 0.0005
      trigger:
        metric: racing/avg_speed
        threshold: 6.0
        window: 200

multi_scene:
  enabled: true
  n_scenes: 10

lr_schedule:
  initial_lr: 0.0001          # lower starting LR for fine-tune
  final_lr: 0.00003

logging:
  tags: ["trpy", "gate-fix", "centering", "low-entropy", "20M"]
  group: gate_collision_fix
  notes: "Gate collision fix: gate_centering=3.0, ent_coef=0.001, LR 1e-4->3e-5"
```

- [ ] **Step 3: Run all tests**

Run: `pytest tests/test_sim/ tests/test_training/ tests/test_integration/ -v --timeout=60`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add configs/experiment/gate_fix.yaml configs/reward/
git commit -m "config: add gate_fix experiment with centering reward and low entropy"
```

---

## Summary

| Task | What | Key Change |
|------|------|------------|
| 1 | `gate_centering_reward()` function | Continuous lateral offset penalty, distance-attenuated |
| 2 | Wire into GateRaceEnv | RC_GATE_CENTERING component (12th) |
| 3 | `gate_fix.yaml` config | centering=3.0, ent_coef=0.001, LR=1e-4→3e-5 |

**To launch:** `python -m training +experiment=gate_fix`

**Expected impact:**
- Gate collisions: 73.5% → <40% (centering reward gives continuous gradient)
- Train/eval gap: should close (ent_coef 0.001 reduces exploration noise)
- train/std: 3.1 → ~1.0-1.5 (much tighter policy)
- Speed: may temporarily dip then recover as policy learns centered approaches
