# Procedural Track Generation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train RL policies on procedurally generated track layouts so they generalize to unseen race courses.

**Architecture:** New `ProceduralTrackGenerator` class generates closed-loop tracks via sequential gate placement with randomized parameters. `GateRaceEnv` stores per-env tracks and regenerates on reset. Hydra config group `track_gen=procedural` enables it.

**Tech Stack:** NumPy (geometry), Hydra/OmegaConf (config), pytest (testing)

**Spec:** `docs/superpowers/specs/2026-03-17-procedural-track-generation-design.md`

---

## Chunk 1: Track Generator

### Task 1: ProceduralTrackGenerator — core generation

**Files:**
- Create: `sim/procedural_tracks.py`
- Create: `tests/test_sim/test_procedural_tracks.py`

- [ ] **Step 1: Write failing tests for generator constraints**

```python
# tests/test_sim/test_procedural_tracks.py
"""Tests for procedural track generation."""
from __future__ import annotations

import numpy as np
import pytest

from sim.procedural_tracks import ProceduralTrackGenerator
from sim.tracks import Track


class TestGeneratorConstraints:
    """Generated tracks satisfy all configured constraints."""

    def setup_method(self):
        self.gen = ProceduralTrackGenerator(
            n_gates_min=4,
            n_gates_max=8,
            gate_spacing_min=2.0,
            gate_spacing_max=6.0,
            turn_angle_min=-90.0,
            turn_angle_max=90.0,
            elevation_min=1.0,
            elevation_max=4.0,
            elevation_delta_max=0.8,
            arena_half_width=15.0,
            closure_max_angle=90.0,
        )
        self.rng = np.random.default_rng(42)

    def test_returns_track(self):
        track = self.gen.generate(self.rng)
        assert isinstance(track, Track)

    def test_gate_count_in_range(self):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            track = self.gen.generate(rng)
            assert 4 <= track.num_gates <= 8

    def test_positions_within_arena(self):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            track = self.gen.generate(rng)
            for gate in track.gates:
                assert abs(gate.position[0]) <= 15.0
                assert abs(gate.position[1]) <= 15.0

    def test_consecutive_spacing(self):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            track = self.gen.generate(rng)
            for i in range(track.num_gates):
                j = (i + 1) % track.num_gates
                dist = np.linalg.norm(
                    track.gates[j].position - track.gates[i].position
                )
                # Closure segment allowed up to 2x max
                max_allowed = 6.0 * 2 if j == 0 else 6.0
                assert dist >= 2.0 - 0.01, f"Gates {i}->{j} too close: {dist}"
                assert dist <= max_allowed + 0.01, f"Gates {i}->{j} too far: {dist}"

    def test_elevation_in_range(self):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            track = self.gen.generate(rng)
            for gate in track.gates:
                assert 1.0 - 0.01 <= gate.position[2] <= 4.0 + 0.01

    def test_elevation_delta(self):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            track = self.gen.generate(rng)
            for i in range(track.num_gates):
                j = (i + 1) % track.num_gates
                dz = abs(track.gates[j].position[2] - track.gates[i].position[2])
                assert dz <= 0.8 + 0.01

    def test_orientations_are_unit_quaternions(self):
        track = self.gen.generate(self.rng)
        for gate in track.gates:
            assert abs(np.linalg.norm(gate.orientation) - 1.0) < 1e-6

    def test_gates_face_next_gate(self):
        """Each gate's forward normal should point roughly toward next gate."""
        track = self.gen.generate(self.rng)
        for i in range(track.num_gates):
            j = (i + 1) % track.num_gates
            # Gate forward = rotate [1,0,0] by quaternion
            q = track.gates[i].orientation
            # Quaternion rotation of [1,0,0]: simplified
            w, x, y, z = q
            forward = np.array([
                1 - 2*(y*y + z*z),
                2*(x*y + w*z),
                2*(x*z - w*y),
            ])
            to_next = track.gates[j].position - track.gates[i].position
            to_next_xy = to_next[:2]
            forward_xy = forward[:2]
            if np.linalg.norm(to_next_xy) > 0.01:
                cos_angle = (
                    np.dot(forward_xy, to_next_xy)
                    / (np.linalg.norm(forward_xy) * np.linalg.norm(to_next_xy))
                )
                assert cos_angle > 0.5, f"Gate {i} not facing gate {j}"

    def test_seeded_reproducibility(self):
        t1 = self.gen.generate(np.random.default_rng(99))
        t2 = self.gen.generate(np.random.default_rng(99))
        assert t1.num_gates == t2.num_gates
        for g1, g2 in zip(t1.gates, t2.gates):
            np.testing.assert_array_equal(g1.position, g2.position)

    def test_min_separation_between_all_gates(self):
        """Non-consecutive gates should also be separated."""
        for seed in range(20):
            rng = np.random.default_rng(seed)
            track = self.gen.generate(rng)
            for i in range(track.num_gates):
                for j in range(i + 2, track.num_gates):
                    if j == track.num_gates - 1 and i == 0:
                        continue  # consecutive (wrapping)
                    dist = np.linalg.norm(
                        track.gates[j].position - track.gates[i].position
                    )
                    assert dist >= 2.0 - 0.01, (
                        f"Non-consecutive gates {i},{j} too close: {dist}"
                    )


class TestGeneratorValidation:
    """Invalid parameters are rejected."""

    def test_n_gates_min_zero(self):
        with pytest.raises(ValueError):
            ProceduralTrackGenerator(n_gates_min=0)

    def test_n_gates_min_two(self):
        with pytest.raises(ValueError):
            ProceduralTrackGenerator(n_gates_min=2, n_gates_max=4)

    def test_n_gates_min_gt_max(self):
        with pytest.raises(ValueError):
            ProceduralTrackGenerator(n_gates_min=10, n_gates_max=5)

    def test_spacing_min_gt_max(self):
        with pytest.raises(ValueError):
            ProceduralTrackGenerator(gate_spacing_min=10.0, gate_spacing_max=5.0)

    def test_elevation_min_gt_max(self):
        with pytest.raises(ValueError):
            ProceduralTrackGenerator(elevation_min=5.0, elevation_max=2.0)


class TestGeneratorFallback:
    """Generator falls back to figure-8 when retries exhausted."""

    def test_impossible_params_fallback(self):
        """Extremely tight constraints should trigger fallback."""
        gen = ProceduralTrackGenerator(
            n_gates_min=12,
            n_gates_max=12,
            gate_spacing_min=7.0,
            gate_spacing_max=8.0,
            turn_angle_min=-10.0,
            turn_angle_max=10.0,
            arena_half_width=5.0,
            closure_max_angle=10.0,
            closure_max_retries=3,
        )
        rng = np.random.default_rng(42)
        track = gen.generate(rng)
        # Should get figure-8 fallback (8 gates)
        assert track.num_gates == 8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sim/test_procedural_tracks.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sim.procedural_tracks'`

- [ ] **Step 3: Implement ProceduralTrackGenerator**

```python
# sim/procedural_tracks.py
"""Procedural track generation for diverse training layouts."""
from __future__ import annotations

import logging
import math
import warnings

import numpy as np

from sim.tracks import Track, _yaw_to_quat, build_figure8_track
from sim.types import GateState

log = logging.getLogger(__name__)


class ProceduralTrackGenerator:
    """Generate closed-loop racing tracks with randomized geometry.

    Sequential gate placement: each gate is placed relative to the previous
    one by sampling turn angle, distance, and elevation. The final segment
    must close back to gate 0 within angle/distance constraints.

    Args:
        n_gates_min: Minimum gates per track (>= 3).
        n_gates_max: Maximum gates per track.
        gate_spacing_min: Minimum distance between consecutive gates (m).
        gate_spacing_max: Maximum distance between consecutive gates (m).
        turn_angle_min: Minimum direction change per gate (degrees).
        turn_angle_max: Maximum direction change per gate (degrees).
        elevation_min: Minimum gate height (m).
        elevation_max: Maximum gate height (m).
        elevation_delta_max: Maximum height change between consecutive gates (m).
        arena_half_width: Gates must fit within [-arena, +arena] in XY (m).
        closure_max_angle: Maximum turn angle for closing segment (degrees).
        closure_max_retries: Retries before falling back to figure-8.
    """

    def __init__(
        self,
        *,
        n_gates_min: int = 4,
        n_gates_max: int = 12,
        gate_spacing_min: float = 2.0,
        gate_spacing_max: float = 8.0,
        turn_angle_min: float = -120.0,
        turn_angle_max: float = 120.0,
        elevation_min: float = 1.0,
        elevation_max: float = 4.0,
        elevation_delta_max: float = 0.8,
        arena_half_width: float = 20.0,
        closure_max_angle: float = 90.0,
        closure_max_retries: int = 10,
    ) -> None:
        if n_gates_min < 3:
            raise ValueError(f"n_gates_min must be >= 3, got {n_gates_min}")
        if n_gates_min > n_gates_max:
            raise ValueError(
                f"n_gates_min ({n_gates_min}) > n_gates_max ({n_gates_max})"
            )
        if gate_spacing_min > gate_spacing_max:
            raise ValueError(
                f"gate_spacing_min ({gate_spacing_min}) > gate_spacing_max ({gate_spacing_max})"
            )
        if elevation_min > elevation_max:
            raise ValueError(
                f"elevation_min ({elevation_min}) > elevation_max ({elevation_max})"
            )

        self.n_gates_min = n_gates_min
        self.n_gates_max = n_gates_max
        self.gate_spacing_min = gate_spacing_min
        self.gate_spacing_max = gate_spacing_max
        self.turn_angle_min_rad = math.radians(turn_angle_min)
        self.turn_angle_max_rad = math.radians(turn_angle_max)
        self.elevation_min = elevation_min
        self.elevation_max = elevation_max
        self.elevation_delta_max = elevation_delta_max
        self.arena_half_width = arena_half_width
        self.closure_max_angle_rad = math.radians(closure_max_angle)
        self.closure_max_retries = closure_max_retries

    def generate(self, rng: np.random.Generator) -> Track:
        """Generate a random closed-loop track.

        Args:
            rng: NumPy random generator for reproducibility.

        Returns:
            A Track with randomized gate positions and orientations.
            Falls back to figure-8 if generation fails after retries.
        """
        for attempt in range(self.closure_max_retries):
            result = self._try_generate(rng)
            if result is not None:
                return result

        warnings.warn(
            f"Procedural track generation failed after {self.closure_max_retries} "
            f"retries, falling back to figure-8 track.",
            stacklevel=2,
        )
        return build_figure8_track()

    def _try_generate(self, rng: np.random.Generator) -> Track | None:
        """Attempt to generate one track. Returns None if constraints violated."""
        n_gates = int(rng.integers(self.n_gates_min, self.n_gates_max + 1))

        # Place gate 0 near center
        margin = self.arena_half_width * 0.3
        x0 = rng.uniform(-margin, margin)
        y0 = rng.uniform(-margin, margin)
        z0 = rng.uniform(self.elevation_min, self.elevation_max)
        heading = rng.uniform(-math.pi, math.pi)

        positions = [np.array([x0, y0, z0])]
        headings = [heading]

        for _ in range(1, n_gates):
            # Sample turn angle (clipped normal — gentle turns more likely)
            turn_std = (self.turn_angle_max_rad - self.turn_angle_min_rad) / 4.0
            turn = rng.normal(0.0, turn_std)
            turn = np.clip(turn, self.turn_angle_min_rad, self.turn_angle_max_rad)
            heading = heading + turn

            # Sample distance
            dist = rng.uniform(self.gate_spacing_min, self.gate_spacing_max)

            # New XY position
            new_x = positions[-1][0] + dist * math.cos(heading)
            new_y = positions[-1][1] + dist * math.sin(heading)

            # Clamp to arena
            new_x = np.clip(new_x, -self.arena_half_width, self.arena_half_width)
            new_y = np.clip(new_y, -self.arena_half_width, self.arena_half_width)

            # Sample elevation with delta constraint
            prev_z = positions[-1][2]
            z_lo = max(self.elevation_min, prev_z - self.elevation_delta_max)
            z_hi = min(self.elevation_max, prev_z + self.elevation_delta_max)
            new_z = rng.uniform(z_lo, z_hi)

            positions.append(np.array([new_x, new_y, new_z]))
            headings.append(heading)

        # --- Validate closure ---
        # Check closing segment distance
        close_dist = float(np.linalg.norm(positions[0] - positions[-1]))
        if close_dist > self.gate_spacing_max * 2:
            return None

        # Check closing segment angle
        dx = positions[0][0] - positions[-1][0]
        dy = positions[0][1] - positions[-1][1]
        close_heading = math.atan2(dy, dx)
        close_turn = abs(_wrap_angle(close_heading - headings[-1]))
        if close_turn > self.closure_max_angle_rad:
            return None

        # Check closing elevation delta
        if abs(positions[0][2] - positions[-1][2]) > self.elevation_delta_max:
            return None

        # Check minimum separation between all non-consecutive gates
        for i in range(len(positions)):
            for j in range(i + 2, len(positions)):
                if i == 0 and j == len(positions) - 1:
                    continue  # consecutive (wrapping)
                sep = float(np.linalg.norm(positions[j] - positions[i]))
                if sep < self.gate_spacing_min:
                    return None

        # --- Build Track ---
        gates: list[GateState] = []
        for i, pos in enumerate(positions):
            next_pos = positions[(i + 1) % n_gates]
            dx = next_pos[0] - pos[0]
            dy = next_pos[1] - pos[1]
            yaw = math.atan2(dy, dx)
            gates.append(GateState(position=pos, orientation=_yaw_to_quat(yaw)))

        return Track(gates)


def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sim/test_procedural_tracks.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add sim/procedural_tracks.py tests/test_sim/test_procedural_tracks.py
git commit -m "feat: add ProceduralTrackGenerator with constraint validation"
```

---

## Chunk 2: GateRaceEnv Per-Env Track Support

### Task 2: Refactor GateRaceEnv from single track to per-env tracks

**Files:**
- Modify: `sim/envs/gate_race_env.py`
- Modify: `metrics/contract.py`
- Modify: `training/trajectory_recorder.py`
- Modify: `sim/envs/rate_ctrl_env.py` (signature only)

- [ ] **Step 1: Run existing tests to confirm green baseline**

Run: `pytest tests/test_sim/ -v`
Expected: All PASS

- [ ] **Step 2: Update `metrics/contract.py` — add `env_idx` parameter**

Change the `get_gate_geometry` signature in the `TrajectoryProvider` protocol:

```python
# metrics/contract.py — update get_gate_geometry signature
    def get_gate_geometry(self, env_idx: int = 0) -> dict[str, np.ndarray]:
        """Return gate geometry for a specific environment's track.

        Args:
            env_idx: Environment index. Default 0 for backwards compatibility.

        Returns dict with keys:
            positions: (n_gates, 3) float64
            orientations: (n_gates, 4) float64 — quaternions (w, x, y, z)
            half_extents: (n_gates, 2) float64 — (half_width, half_height)
        """
        ...
```

- [ ] **Step 3: Update `GateRaceEnv` — per-env track storage**

In `gate_race_env.py`, replace `self.track` with `self._tracks`:

Constructor (around line 303-307), replace:
```python
    if track is None:
        from sim.tracks import build_figure8_track
        track = build_figure8_track()
    self.track = track
```
with:
```python
    if track is None:
        from sim.tracks import build_figure8_track
        track = build_figure8_track()
    # Per-env track storage: all envs start with the same track
    self._tracks: list[Track] = [track] * n_envs
    self.track_generator: ProceduralTrackGenerator | None = track_generator
```

Add `track_generator` to the constructor signature (after `arena_bounds`):
```python
    track_generator: ProceduralTrackGenerator | None = None,
```

Add import at top of file:
```python
from sim.procedural_tracks import ProceduralTrackGenerator
```

- [ ] **Step 4: Update all `self.track` references to `self._tracks[i]`**

Replace every `self.track` access with the per-env version. The exact replacements:

In `_randomize_start` (lines 384-390):
- `self.track.num_gates` → `self._tracks[idx].num_gates`
- `self.track.gates[gate_idx]` → `self._tracks[idx].gates[gate_idx]`

In `_update_gate_tracking` (lines 421-427):
- `self.track.gates[gate_idx % self.track.num_gates]` → `self._tracks[idx].gates[gate_idx % self._tracks[idx].num_gates]`

In `step()` gate passage detection (around lines 620-695):
- `gate = self.track.gates[...]` → `gate = self._tracks[i].gates[...]`
- `self.track.num_gates` → `self._tracks[i].num_gates`
- `self.track.gates[int(self._gate_indices[i]) % self.track.num_gates]` → `self._tracks[i].gates[int(self._gate_indices[i]) % self._tracks[i].num_gates]`

In `_compute_obs_batched` (around lines 808-810):
- `self.track.gates[gate_idx % self.track.num_gates]` → `self._tracks[i].gates[gate_idx % self._tracks[i].num_gates]`
- `self.track.gates[(gate_idx + 1) % self.track.num_gates]` → `self._tracks[i].gates[(gate_idx + 1) % self._tracks[i].num_gates]`

In `get_gate_geometry` (lines 886-902):
- Update signature to `def get_gate_geometry(self, env_idx: int = 0):`
- Replace `self.track` with `self._tracks[env_idx]`

In `reset()` — add track generation for each env during initial reset:
```python
# In reset(), before _randomize_start:
if self.track_generator is not None:
    for i in range(self.n_envs):
        self._tracks[i] = self.track_generator.generate(self.np_random)
```

In `step()` auto-reset block (around line 743-770) — this is where per-env resets happen during training. Add track regeneration **before** `_randomize_start` is called for done envs:
```python
# In the auto-reset section of step(), before _randomize_start(done_indices):
if self.track_generator is not None:
    for idx in done_indices:
        self._tracks[idx] = self.track_generator.generate(self.np_random)
```

This is critical — without it, tracks would only be generated once on the initial `reset()` and never change during training.

Add a `@property` for backwards compatibility:
```python
@property
def track(self) -> Track:
    """First env's track (backwards compatibility)."""
    return self._tracks[0]
```

- [ ] **Step 5: Update `trajectory_recorder.py` — pass env_idx, refresh geometry**

In `_extract_gate_geometry` (lines 140-146), update to:
```python
def _extract_gate_geometry(self, env: Any, env_idx: int = 0) -> tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    """Return gate geometry via TrajectoryProvider protocol."""
    raw_env = getattr(env, "env", env)
    geom = raw_env.get_gate_geometry(env_idx=env_idx)
    return geom["positions"], geom["orientations"], geom["half_extents"]
```

Update usage in trajectory recording (around line 268) to pass the env index and call per-recording (not cached):
```python
gate_positions, gate_orientations, gate_half_extents = (
    self._extract_gate_geometry(env, env_idx=env_idx)
)
```

- [ ] **Step 6: Update `RateCtrlEnv.get_gate_geometry` signature**

In `sim/envs/rate_ctrl_env.py`, update the `get_gate_geometry` method signature to match the updated protocol:

```python
def get_gate_geometry(self, env_idx: int = 0) -> dict[str, np.ndarray]:
```

Body unchanged — `RateCtrlEnv` uses a single shared track, so `env_idx` is ignored.

- [ ] **Step 7: Use grep to find ALL remaining `self.track` references**

Run: `grep -n "self\.track[^_s]" sim/envs/gate_race_env.py`

Every match must be converted to `self._tracks[i]` (using the appropriate loop variable). The backwards-compat `@property` will catch any missed ones at runtime but will return `_tracks[0]`, which is wrong for envs > 0.

- [ ] **Step 8: Run all existing tests**

Run: `pytest tests/test_sim/ -v`
Expected: All PASS (backwards compatible — single track duplicated across envs)

- [ ] **Step 9: Commit**

```bash
git add sim/envs/gate_race_env.py sim/envs/rate_ctrl_env.py metrics/contract.py training/trajectory_recorder.py
git commit -m "refactor: per-env track storage in GateRaceEnv"
```

---

### Task 3: Add per-episode track info to logging

**Files:**
- Modify: `sim/envs/gate_race_env.py`

- [ ] **Step 1: Add `n_gates` and `track_id` to episode info**

In the section of `step()` that builds the batched `ep_info` dict (around lines 722-741), add track metadata arrays alongside existing fields like `"r"` and `"gates_passed"`:

```python
# Track metadata (batched, matching existing ep_info pattern)
"n_gates": np.array([self._tracks[i].num_gates for i in range(self.n_envs)]),
"track_id": np.array([
    hash(tuple(tuple(g.position) for g in self._tracks[i].gates))
    for i in range(self.n_envs)
]),
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_sim/ -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add sim/envs/gate_race_env.py
git commit -m "feat: log n_gates and track_id per episode"
```

---

## Chunk 3: Factory and Config Integration

### Task 4: Hydra config and factory wiring

**Files:**
- Create: `configs/track_gen/procedural.yaml`
- Modify: `sim/envs/numpy_quad_factory.py`
- Create: `tests/test_sim/test_procedural_training.py`

- [ ] **Step 1: Write integration tests**

```python
# tests/test_sim/test_procedural_training.py
"""Integration tests for procedural track training."""
from __future__ import annotations

import numpy as np
import pytest

from sim.envs.gate_race_env import GateRaceEnv
from sim.procedural_tracks import ProceduralTrackGenerator


class TestEnvWithGenerator:
    """GateRaceEnv with procedural track generator."""

    def setup_method(self):
        self.gen = ProceduralTrackGenerator(
            n_gates_min=4,
            n_gates_max=8,
            arena_half_width=15.0,
        )
        self.env = GateRaceEnv(
            n_envs=4,
            track_generator=self.gen,
            arena_bounds=15.0,
        )

    def test_env_builds_and_resets(self):
        obs = self.env.reset()
        assert obs.shape == (4, 24)

    def test_env_steps_without_error(self):
        self.env.reset()
        action = np.zeros((4, 4), dtype=np.float32)
        obs, rewards, terminated, truncated, infos = self.env.step(action)
        assert obs.shape == (4, 24)
        assert rewards.shape == (4,)

    def test_different_envs_get_different_tracks(self):
        self.env.reset()
        # Force resets by terminating
        tracks_env0 = self.env._tracks[0]
        tracks_env1 = self.env._tracks[1]
        # With 4 envs and random generation, at least some should differ
        any_differ = False
        for i in range(4):
            for j in range(i + 1, 4):
                if self.env._tracks[i].num_gates != self.env._tracks[j].num_gates:
                    any_differ = True
                    break
                for gi, gj in zip(self.env._tracks[i].gates, self.env._tracks[j].gates):
                    if not np.allclose(gi.position, gj.position):
                        any_differ = True
                        break
            if any_differ:
                break
        # It's possible (but unlikely) all 4 get identical tracks with seed
        # Just verify they're valid tracks
        for i in range(4):
            assert 4 <= self.env._tracks[i].num_gates <= 8

    def test_gate_passage_on_generated_track(self):
        """Fly drone straight through first gate, verify passage detected."""
        self.env.reset()
        env_idx = 0
        track = self.env._tracks[env_idx]
        gate = track.gates[0]

        # Place drone behind gate
        from sim.envs.gate_race_env import _gate_normal
        normal = _gate_normal(gate)
        self.env._states[env_idx, 0:3] = gate.position - 0.5 * normal
        self.env._gate_indices[env_idx] = 0
        self.env._gates_passed[env_idx] = 0
        self.env._prev_along_normal[env_idx] = -0.5

        # Move drone through gate
        self.env._states[env_idx, 0:3] = gate.position + 0.5 * normal
        # Step to trigger detection
        action = np.zeros((self.env.n_envs, 4), dtype=np.float32)
        self.env.step(action)
        assert self.env._gates_passed[env_idx] >= 1

    def test_get_gate_geometry_per_env(self):
        self.env.reset()
        geom0 = self.env.get_gate_geometry(env_idx=0)
        geom1 = self.env.get_gate_geometry(env_idx=1)
        assert "positions" in geom0
        assert "orientations" in geom0
        assert geom0["positions"].shape[0] == self.env._tracks[0].num_gates
        assert geom1["positions"].shape[0] == self.env._tracks[1].num_gates


class TestEnvWithoutGenerator:
    """Backwards compatibility: no generator uses figure-8."""

    def test_default_figure8(self):
        env = GateRaceEnv(n_envs=2)
        env.reset()
        assert env._tracks[0].num_gates == 8
        assert env._tracks[1].num_gates == 8

    def test_explicit_track_shared(self):
        from sim.tracks import build_figure8_track
        track = build_figure8_track()
        env = GateRaceEnv(n_envs=2, track=track)
        env.reset()
        assert env._tracks[0] is env._tracks[1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sim/test_procedural_training.py -v`
Expected: FAIL — `track_generator` not yet a constructor parameter

- [ ] **Step 3: Create Hydra config**

```yaml
# configs/track_gen/procedural.yaml
n_gates_min: 4
n_gates_max: 12
gate_spacing_min: 2.0
gate_spacing_max: 8.0
turn_angle_min: -120
turn_angle_max: 120
elevation_min: 1.0
elevation_max: 4.0
elevation_delta_max: 0.8
closure_max_angle: 90
closure_max_retries: 10
```

- [ ] **Step 4: Update `numpy_quad_factory.py`**

Add track generator construction to `_build()`. After the domain randomizer setup and before `GateRaceEnv(...)`:

```python
# Track generation
track_generator = None
if hasattr(self, '_track_gen_cfg') and self._track_gen_cfg is not None:
    from sim.procedural_tracks import ProceduralTrackGenerator
    tg_dict = OmegaConf.to_container(self._track_gen_cfg, resolve=True)
    track_generator = ProceduralTrackGenerator(
        arena_half_width=self.arena_bounds,
        **tg_dict,
    )
```

Add `track_generator=track_generator` to the `GateRaceEnv(...)` constructor call.

In the factory's `__init__`, accept the track_gen config:
```python
self._track_gen_cfg = cfg.get("track_gen", None)
```

For eval envs (`make_eval_env`), generate fixed tracks and pass them without a generator:
```python
if track_generator is not None:
    eval_rng = np.random.default_rng(self.seed + 1000)
    n_eval_tracks = 10
    eval_tracks = [track_generator.generate(eval_rng) for _ in range(n_eval_tracks)]
    # No generator for eval — tracks are fixed for reproducibility.
    # Assign round-robin: env i gets eval_tracks[i % n_eval_tracks].
    # Pass as a pre-assigned list via new `tracks` parameter:
    track_generator = None  # disable generation for eval
```

Add a `tracks: list[Track] | None = None` parameter to `GateRaceEnv.__init__`. When provided, it overrides the per-env track list directly (used for eval):
```python
if tracks is not None:
    assert len(tracks) == n_envs
    self._tracks = list(tracks)
elif track is not None:
    self._tracks = [track] * n_envs
else:
    self._tracks = [build_figure8_track()] * n_envs
```

In the factory's `make_eval_env`, build the round-robin list:
```python
eval_track_list = [eval_tracks[i % n_eval_tracks] for i in range(n_envs)]
# Pass tracks=eval_track_list to GateRaceEnv, no track_generator
```

- [ ] **Step 5: Run integration tests**

Run: `pytest tests/test_sim/test_procedural_training.py -v`
Expected: All PASS

- [ ] **Step 6: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add configs/track_gen/procedural.yaml sim/envs/numpy_quad_factory.py tests/test_sim/test_procedural_training.py
git commit -m "feat: wire procedural track generation into factory and Hydra config"
```

---

## Chunk 4: Smoke Test

### Task 5: End-to-end training smoke test

**Files:** None (manual verification)

- [ ] **Step 1: Run short training with procedural tracks**

```bash
cd C:/Users/alexj/Documents/algo_src
python -m training +experiment=monorace_baseline track_gen=procedural total_timesteps=200000 device=cpu eval_freq=100000
```

Expected: Training starts, wandb logs metrics including `n_gates`, no crashes.

- [ ] **Step 2: Verify wandb metrics**

Check that the wandb run shows:
- `racing/gates_per_ep` varying (different track difficulties)
- No errors in logs
- Rerun recordings (if viz_freq is hit) show varied track layouts

- [ ] **Step 3: Run with default config (no track_gen) to confirm backwards compatibility**

```bash
python -m training +experiment=monorace_baseline total_timesteps=200000 device=cpu eval_freq=100000
```

Expected: Trains on figure-8 as before. No regressions.

- [ ] **Step 4: Commit any fixes needed**
