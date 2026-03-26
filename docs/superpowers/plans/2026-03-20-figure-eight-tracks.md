# Figure-Eight Track Training Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add randomized figure-eight tracks to the training distribution to close the generalization gap (55% gate passage on figure-8 vs 100% on procedural loops).

**Architecture:** A new `Figure8TrackGenerator` creates randomized figure-eights (varying size, gate count per loop, crossing offset, elevation). A `MixedTrackGenerator` wraps both it and the existing `ProceduralTrackGenerator`, selecting between them with a configurable probability. The factory's `_make_track_generator` is updated to return the mixed generator. Also bump `n_lookahead_gates` from 2 to 3 for better crossing-point disambiguation.

**Tech Stack:** NumPy, Hydra/OmegaConf

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `sim/figure8_tracks.py` | Randomized figure-eight track generator |
| Modify | `sim/procedural_tracks.py` | Add `MixedTrackGenerator` |
| Modify | `sim/envs/numpy_quad_factory.py` | Support mixed generator from config |
| Create | `configs/experiment/figure8_training.yaml` | Experiment config |
| Create | `tests/test_sim/test_figure8_tracks.py` | Tests for figure-8 generator |
| Modify | `tests/test_sim/test_procedural_tracks.py` | Tests for mixed generator |

### Task 1: Randomized Figure-Eight Track Generator

The generator creates two elliptical loops connected at a crossing point, with randomized:
- Loop radius (1.5–4m)
- Gates per loop (3–5, so 6–10 total)
- Crossing offset (0.3–0.8m to avoid gate overlap)
- Elevation (1.0–3.5m with gentle variation)
- Overall rotation (random yaw)

**Files:**
- Create: `sim/figure8_tracks.py`
- Create: `tests/test_sim/test_figure8_tracks.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_sim/test_figure8_tracks.py
"""Tests for randomized figure-eight track generation."""
import numpy as np
import pytest

from sim.figure8_tracks import Figure8TrackGenerator


class TestFigure8TrackGenerator:
    @pytest.fixture
    def gen(self):
        return Figure8TrackGenerator()

    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    def test_generates_track(self, gen, rng):
        """Should produce a valid Track object."""
        track = gen.generate(rng)
        assert track.num_gates >= 6  # min 3 per loop

    def test_gate_count_in_range(self, gen, rng):
        """Gate count should be between 2*gates_per_loop_min and 2*gates_per_loop_max."""
        for _ in range(20):
            track = gen.generate(rng)
            assert 6 <= track.num_gates <= 10

    def test_crossing_gates_are_offset(self, gen, rng):
        """The two crossing gates should not overlap spatially."""
        track = gen.generate(rng)
        n = track.num_gates
        half = n // 2
        # Crossing gates are at indices half-1 and n-1 (last gate of each loop)
        # Actually the crossing is between the two loops
        # Just check no two gates are closer than 0.2m
        positions = np.array([g.position for g in track.gates])
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(positions[i] - positions[j])
                assert dist > 0.2, f"Gates {i} and {j} too close: {dist:.3f}m"

    def test_track_is_closed(self, gen, rng):
        """Last gate should be within reasonable distance of first gate."""
        track = gen.generate(rng)
        pos_first = track.gates[0].position
        pos_last = track.gates[-1].position
        dist = np.linalg.norm(pos_first - pos_last)
        assert dist < 10.0  # should close within reasonable distance

    def test_elevation_in_bounds(self, gen, rng):
        """All gates should be within elevation bounds."""
        for _ in range(10):
            track = gen.generate(rng)
            for gate in track.gates:
                assert 0.5 <= gate.position[2] <= 5.0

    def test_deterministic_with_seed(self):
        """Same seed should produce same track."""
        gen = Figure8TrackGenerator()
        t1 = gen.generate(np.random.default_rng(123))
        t2 = gen.generate(np.random.default_rng(123))
        assert t1.num_gates == t2.num_gates
        for g1, g2 in zip(t1.gates, t2.gates):
            np.testing.assert_allclose(g1.position, g2.position)

    def test_tracks_vary_with_different_seeds(self, gen):
        """Different seeds should produce different tracks."""
        t1 = gen.generate(np.random.default_rng(1))
        t2 = gen.generate(np.random.default_rng(2))
        positions_differ = any(
            not np.allclose(g1.position, g2.position)
            for g1, g2 in zip(t1.gates, t2.gates)
        ) or t1.num_gates != t2.num_gates
        assert positions_differ
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_sim/test_figure8_tracks.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: Implement**

```python
# sim/figure8_tracks.py
"""Randomized figure-eight track generator.

Creates two connected elliptical loops with a crossing point,
randomizing size, gate count, offset, elevation, and rotation.
"""

from __future__ import annotations

import math

import numpy as np

from sim.tracks import Track, _yaw_to_quat
from sim.types import GateState


class Figure8TrackGenerator:
    """Generate randomized figure-eight racing tracks.

    Creates two elliptical loops connected at a central crossing point.
    The crossing gates are offset slightly so they don't physically overlap.

    Args:
        loop_radius_min: Minimum loop radius in meters.
        loop_radius_max: Maximum loop radius in meters.
        gates_per_loop_min: Minimum gates per loop (total = 2x this).
        gates_per_loop_max: Maximum gates per loop.
        crossing_offset_min: Minimum lateral offset at crossing (meters).
        crossing_offset_max: Maximum lateral offset at crossing (meters).
        elevation_min: Minimum gate height (meters).
        elevation_max: Maximum gate height (meters).
        elevation_delta_max: Max height change between adjacent gates.
    """

    def __init__(
        self,
        *,
        loop_radius_min: float = 1.5,
        loop_radius_max: float = 4.0,
        gates_per_loop_min: int = 3,
        gates_per_loop_max: int = 5,
        crossing_offset_min: float = 0.3,
        crossing_offset_max: float = 0.8,
        elevation_min: float = 1.0,
        elevation_max: float = 3.5,
        elevation_delta_max: float = 0.6,
    ) -> None:
        self.loop_radius_min = loop_radius_min
        self.loop_radius_max = loop_radius_max
        self.gates_per_loop_min = gates_per_loop_min
        self.gates_per_loop_max = gates_per_loop_max
        self.crossing_offset_min = crossing_offset_min
        self.crossing_offset_max = crossing_offset_max
        self.elevation_min = elevation_min
        self.elevation_max = elevation_max
        self.elevation_delta_max = elevation_delta_max

    def generate(self, rng: np.random.Generator) -> Track:
        """Generate a randomized figure-eight track.

        Layout (top view):
            Loop A (top):  gates around an ellipse centered at (0, +radius)
            Crossing A→B:  gate near origin, offset to (-dx, 0)
            Loop B (bottom): gates around an ellipse centered at (0, -radius)
            Crossing B→A:  gate near origin, offset to (+dx, 0)

        The whole track is then rotated by a random yaw and translated
        to a random center position.
        """
        n_per_loop = int(rng.integers(self.gates_per_loop_min, self.gates_per_loop_max + 1))
        radius = rng.uniform(self.loop_radius_min, self.loop_radius_max)
        crossing_offset = rng.uniform(self.crossing_offset_min, self.crossing_offset_max)
        base_z = rng.uniform(self.elevation_min, self.elevation_max)

        # Generate positions for loop A (top, counter-clockwise)
        loop_a = self._make_loop(n_per_loop, radius, center_y=radius, rng=rng)
        # Generate positions for loop B (bottom, clockwise)
        loop_b = self._make_loop(n_per_loop, radius, center_y=-radius, rng=rng, clockwise=True)

        # Build the full sequence:
        # [loop_a gates] → [crossing gate A→B] → [loop_b gates] → [crossing gate B→A]
        positions_2d = []

        # Loop A: skip the last point (that's near the crossing, we'll add crossing gates)
        for p in loop_a[:-1]:
            positions_2d.append(p)

        # Crossing gate A→B (offset left)
        positions_2d.append(np.array([-crossing_offset, 0.0]))

        # Loop B: skip the first point (near crossing)
        for p in loop_b[1:]:
            positions_2d.append(p)

        # Crossing gate B→A (offset right)
        positions_2d.append(np.array([crossing_offset, 0.0]))

        # Add elevation with gentle variation
        positions_3d = []
        z = base_z
        for p2d in positions_2d:
            z_delta = rng.uniform(-self.elevation_delta_max * 0.3, self.elevation_delta_max * 0.3)
            z = np.clip(z + z_delta, self.elevation_min, self.elevation_max)
            positions_3d.append(np.array([p2d[0], p2d[1], z]))

        # Random rotation and translation
        yaw_offset = rng.uniform(-math.pi, math.pi)
        cos_y, sin_y = math.cos(yaw_offset), math.sin(yaw_offset)
        tx = rng.uniform(-2.0, 2.0)
        ty = rng.uniform(-2.0, 2.0)

        transformed = []
        for p in positions_3d:
            rx = cos_y * p[0] - sin_y * p[1] + tx
            ry = sin_y * p[0] + cos_y * p[1] + ty
            transformed.append(np.array([rx, ry, p[2]]))

        # Build gates with yaw pointing toward next gate
        n = len(transformed)
        gates: list[GateState] = []
        for i, pos in enumerate(transformed):
            next_pos = transformed[(i + 1) % n]
            dx = next_pos[0] - pos[0]
            dy = next_pos[1] - pos[1]
            yaw = math.atan2(dy, dx)
            gates.append(GateState(position=pos, orientation=_yaw_to_quat(yaw)))

        return Track(gates)

    def _make_loop(
        self,
        n_gates: int,
        radius: float,
        center_y: float,
        rng: np.random.Generator,
        clockwise: bool = False,
    ) -> list[np.ndarray]:
        """Generate 2D points around an ellipse.

        Returns n_gates+1 points (includes the start/end near crossing).
        """
        # Slight ellipticity for variety
        aspect = rng.uniform(0.8, 1.2)
        rx = radius * aspect
        ry = radius

        # Distribute gates evenly around the ellipse
        # Start from bottom of loop (near crossing point)
        n_total = n_gates + 1  # include endpoint near crossing
        angles = np.linspace(0, 2 * math.pi, n_total, endpoint=False)

        if clockwise:
            angles = -angles

        # Start angle: bottom of loop (pointing toward crossing)
        start_angle = -math.pi / 2 if not clockwise else math.pi / 2
        angles = angles + start_angle

        points = []
        for a in angles:
            x = rx * math.cos(a)
            y = center_y + ry * math.sin(a)
            # Add small random perturbation
            x += rng.normal(0, radius * 0.05)
            y += rng.normal(0, radius * 0.05)
            points.append(np.array([x, y]))

        return points
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_sim/test_figure8_tracks.py -v`
Expected: All 7 PASS

- [ ] **Step 5: Commit**

```bash
git add sim/figure8_tracks.py tests/test_sim/test_figure8_tracks.py
git commit -m "feat(sim): add randomized figure-eight track generator"
```

### Task 2: Mixed Track Generator

Wraps both generators and selects between them with a configurable probability.

**Files:**
- Modify: `sim/procedural_tracks.py`
- Modify: `tests/test_sim/test_procedural_tracks.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_sim/test_procedural_tracks.py`:

```python
from sim.procedural_tracks import MixedTrackGenerator, ProceduralTrackGenerator
from sim.figure8_tracks import Figure8TrackGenerator


class TestMixedTrackGenerator:
    def test_all_procedural(self):
        """figure8_ratio=0 should produce only procedural tracks."""
        gen = MixedTrackGenerator(
            procedural=ProceduralTrackGenerator(n_gates_min=4, n_gates_max=4),
            figure8=Figure8TrackGenerator(),
            figure8_ratio=0.0,
        )
        rng = np.random.default_rng(42)
        for _ in range(10):
            track = gen.generate(rng)
            assert track.num_gates == 4  # procedural always 4

    def test_all_figure8(self):
        """figure8_ratio=1 should produce only figure-8 tracks."""
        gen = MixedTrackGenerator(
            procedural=ProceduralTrackGenerator(),
            figure8=Figure8TrackGenerator(gates_per_loop_min=4, gates_per_loop_max=4),
            figure8_ratio=1.0,
        )
        rng = np.random.default_rng(42)
        for _ in range(10):
            track = gen.generate(rng)
            # Figure-8 with 4 per loop = 8 gates + 2 crossing = ~8-10 gates
            assert track.num_gates >= 6

    def test_mixed_ratio(self):
        """figure8_ratio=0.5 should produce a mix."""
        gen = MixedTrackGenerator(
            procedural=ProceduralTrackGenerator(n_gates_min=4, n_gates_max=4),
            figure8=Figure8TrackGenerator(gates_per_loop_min=5, gates_per_loop_max=5),
            figure8_ratio=0.5,
        )
        rng = np.random.default_rng(42)
        gate_counts = set()
        for _ in range(50):
            track = gen.generate(rng)
            gate_counts.add(track.num_gates)
        # Should see both 4-gate (procedural) and larger (figure-8) tracks
        assert len(gate_counts) > 1

    def test_generate_interface_compatible(self):
        """Should have the same generate(rng) -> Track interface."""
        gen = MixedTrackGenerator(
            procedural=ProceduralTrackGenerator(),
            figure8=Figure8TrackGenerator(),
            figure8_ratio=0.3,
        )
        rng = np.random.default_rng(42)
        track = gen.generate(rng)
        assert hasattr(track, 'gates')
        assert hasattr(track, 'num_gates')
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_sim/test_procedural_tracks.py::TestMixedTrackGenerator -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Add MixedTrackGenerator to sim/procedural_tracks.py**

Append at the end of `sim/procedural_tracks.py`:

```python
class MixedTrackGenerator:
    """Randomly selects between procedural loop and figure-eight tracks.

    Args:
        procedural: ProceduralTrackGenerator for closed-loop tracks.
        figure8: Figure8TrackGenerator for figure-eight tracks.
        figure8_ratio: Probability of generating a figure-eight (0.0–1.0).
    """

    def __init__(
        self,
        procedural: ProceduralTrackGenerator,
        figure8: "Figure8TrackGenerator",
        figure8_ratio: float = 0.3,
    ) -> None:
        self.procedural = procedural
        self.figure8 = figure8
        self.figure8_ratio = figure8_ratio

    def generate(self, rng: np.random.Generator) -> Track:
        """Generate a track, randomly selecting the type."""
        if rng.random() < self.figure8_ratio:
            return self.figure8.generate(rng)
        return self.procedural.generate(rng)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_sim/test_procedural_tracks.py -v`
Expected: All PASS (old + new)

- [ ] **Step 5: Commit**

```bash
git add sim/procedural_tracks.py tests/test_sim/test_procedural_tracks.py
git commit -m "feat(sim): add MixedTrackGenerator for figure-8 + loop mix"
```

### Task 3: Wire Into Factory + Config

**Files:**
- Modify: `sim/envs/numpy_quad_factory.py`
- Create: `configs/experiment/figure8_training.yaml`

- [ ] **Step 1: Update factory to support mixed generator**

In `sim/envs/numpy_quad_factory.py`, modify `_make_track_generator`:

```python
def _make_track_generator(self):
    """Construct a track generator from config, or None."""
    if self._track_gen_cfg is None:
        return None
    from sim.procedural_tracks import ProceduralTrackGenerator

    tg_dict = OmegaConf.to_container(self._track_gen_cfg, resolve=True)

    # Check for figure-8 mix config
    figure8_cfg = tg_dict.pop("figure8", None)
    figure8_ratio = tg_dict.pop("figure8_ratio", 0.0)

    procedural = ProceduralTrackGenerator(
        arena_half_width=self.arena_bounds,
        **tg_dict,
    )

    if figure8_cfg is not None and figure8_ratio > 0:
        from sim.figure8_tracks import Figure8TrackGenerator
        from sim.procedural_tracks import MixedTrackGenerator

        fig8_gen = Figure8TrackGenerator(**figure8_cfg)
        return MixedTrackGenerator(procedural, fig8_gen, figure8_ratio)

    return procedural
```

- [ ] **Step 2: Create experiment config**

```yaml
# configs/experiment/figure8_training.yaml
# @package _global_
# Train on mixed figure-8 + procedural tracks.
# Resumes from gate_fix checkpoint.

defaults:
  - override /sim: numpy_quad
  - override /control: ppo
  - override /perception: corner_noise
  - override /domain_rand: uniform_30pct
  - override /logging: wandb
  - _self_

sim:
  action_mode: trpy
  n_lookahead_gates: 3        # INCREASED from 2 — helps at crossing points
  max_steps: 3000
  arena_bounds: 10
  gate_passage_radius: 1.5

total_timesteps: 20_000_000
resume_checkpoint: outputs/2026-03-19/17-06-24/final_model.zip

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
  # Figure-eight mixing
  figure8_ratio: 0.3         # 30% figure-eights, 70% procedural loops
  figure8:
    loop_radius_min: 1.5
    loop_radius_max: 4.0
    gates_per_loop_min: 3
    gates_per_loop_max: 5
    crossing_offset_min: 0.3
    crossing_offset_max: 0.8
    elevation_min: 1.0
    elevation_max: 3.5
    elevation_delta_max: 0.6

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
    gate_centering: 3.0
  v_max: 6.0
  action_smoothness_threshold: 0.5

control:
  ent_coef: 0.001

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
        threshold: 0.90
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
        threshold: 5.0
        window: 200

multi_scene:
  enabled: true
  n_scenes: 10

lr_schedule:
  initial_lr: 0.0001
  final_lr: 0.00003

logging:
  tags: ["trpy", "figure8", "mixed-tracks", "lookahead-3", "20M"]
  group: figure8_training
  notes: "Mixed training: 30% figure-8, 70% procedural. n_lookahead=3. Resume from gate_fix."
```

**IMPORTANT NOTE on n_lookahead_gates change:** The observation space dimension changes from `20 + 4*2 = 28` to `20 + 4*3 = 32`. When resuming from a checkpoint trained with 28-dim obs, SB3 will raise a shape mismatch error. Two options:

1. **Don't resume** — train from scratch with the new obs dim (slower but clean)
2. **Keep n_lookahead_gates=2** for this run and change it in a future from-scratch run

The config above uses option 1 approach with the understanding that `resume_checkpoint` may need to be set to `null` if the obs dim mismatch prevents loading. If so, remove the resume line and accept a fresh training start.

- [ ] **Step 3: Run all tests**

Run: `pytest tests/test_sim/ -v --timeout=30`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add sim/envs/numpy_quad_factory.py configs/experiment/figure8_training.yaml
git commit -m "feat(sim): wire figure-8 mixed generator into factory and config"
```

---

## Summary

| Task | What | Key Change |
|------|------|------------|
| 1 | `Figure8TrackGenerator` | Two randomized elliptical loops with crossing |
| 2 | `MixedTrackGenerator` | Probability-based selection between loop and figure-8 |
| 3 | Factory + config wiring | `figure8_ratio: 0.3` in track_gen config, `n_lookahead_gates: 3` |

**To launch:** `python -m training +experiment=figure8_training`

**Note on resume:** If the checkpoint has 28-dim obs (n_lookahead=2) and the new config uses 32-dim (n_lookahead=3), set `resume_checkpoint: null` to train from scratch. Alternatively, keep `n_lookahead_gates: 2` to resume and benefit from the figure-8 tracks without the obs change.

**Expected impact:**
- Figure-8 gate passage: 55% → 85%+ (direct exposure to crossing geometry)
- Procedural track performance: should hold (still 70% of training)
- Crossing-point disambiguation: improved by 3-gate lookahead (if training from scratch)
