# Auto-Research MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 1 MVP of the auto-research system — config-only hyperparameter mutations with a MAP-Elites archive, interactive mode, git branch isolation, and team coordination.

**Architecture:** A Python package (`autoresearch/`) provides infrastructure (archive, descriptors, constraints, coordination, hypothesis schema). A Claude Code plugin (`.claude/plugins/autoresearch/`) provides skills and hooks that orchestrate the research loop. State is stored in git-synced JSON files.

**Tech Stack:** Python 3.10+, numpy, wandb, pytest, Hydra/OmegaConf, git

**Spec:** `docs/superpowers/specs/2026-03-19-auto-research-design.md`

---

## File Map

### New Files — `autoresearch/` Python Package

| File | Responsibility |
|------|---------------|
| `autoresearch/__init__.py` | Package init |
| `autoresearch/config.py` | Config dataclass loaded from plugin settings |
| `autoresearch/hypothesis/schema.py` | `Hypothesis`, `HydraOverride`, `Change` dataclasses + content hashing |
| `autoresearch/hypothesis/prompt.py` | Deferred to Phase 2 — MVP prompt assembly is handled inline by the `/auto-research` skill |
| `autoresearch/descriptors/actuator.py` | Axis 1: actuator utilization from .npz trajectory |
| `autoresearch/descriptors/smoothness.py` | Axis 2: control smoothness from .npz trajectory |
| `autoresearch/descriptors/aero_regime.py` | Axis 3: speed × angle-of-attack from .npz trajectory |
| `autoresearch/descriptors/compute.py` | Orchestrates all 3 descriptors, returns `DescriptorVector` |
| `autoresearch/archive/map_elites.py` | `MapElitesArchive` class: grid, cell insertion, querying |
| `autoresearch/archive/serialization.py` | Load/save `archive.json`, conflict resolution |
| `autoresearch/constraints/validator.py` | `ConstraintValidator` + `ValidationResult` dataclass |
| `autoresearch/constraints/physics.py` | Physics plausibility checks |
| `autoresearch/constraints/racing.py` | Gate passage quality + behavioral sanity checks |
| `autoresearch/coordination/claims.py` | Claim read/write/expire/refresh on `claims.json` |
| `autoresearch/coordination/dedup.py` | Hypothesis deduplication against claims + tree |
| `autoresearch/tree/research_tree.py` | `ResearchTree` class: node CRUD, baseline chain |
| `autoresearch/tree/serialization.py` | Load/save `tree.json` |
| `autoresearch/analysis/wandb_pull.py` | Deferred to Phase 2 — MVP W&B access is handled inline by the `/auto-research` skill via `wandb.Api()` |
| `autoresearch/analysis/trajectory.py` | Load .npz files, extract fields as numpy arrays |

### New Files — State

| File | Responsibility |
|------|---------------|
| `autoresearch/state/archive.json` | MAP-Elites grid (initialized empty) |
| `autoresearch/state/tree.json` | Research tree (initialized with baseline root) |
| `autoresearch/state/claims.json` | Active experiment claims (initialized empty) |

### New Files — Tests

| File | Responsibility |
|------|---------------|
| `tests/test_autoresearch/__init__.py` | Test package init |
| `tests/test_autoresearch/conftest.py` | Shared fixtures (fake .npz data, configs) |
| `tests/test_autoresearch/test_descriptors.py` | All 3 descriptor axes + compute orchestration |
| `tests/test_autoresearch/test_archive.py` | MAP-Elites grid, insertion, promotion, serialization |
| `tests/test_autoresearch/test_constraints.py` | Physics, racing, and behavioral sanity validators |
| `tests/test_autoresearch/test_coordination.py` | Claims, expiry, deduplication |
| `tests/test_autoresearch/test_hypothesis.py` | Hypothesis schema, hashing, novelty scoring |
| `tests/test_autoresearch/test_tree.py` | Research tree CRUD, baseline chain |
| `tests/test_autoresearch/test_trajectory.py` | .npz loading and field extraction |

### New Files — Plugin

| File | Responsibility |
|------|---------------|
| `.claude/plugins/autoresearch/plugin.json` | Plugin manifest |
| `.claude/plugins/autoresearch/skills/auto-research.md` | Main research loop skill |
| `.claude/plugins/autoresearch/skills/ar-status.md` | Archive status display skill |
| `.claude/plugins/autoresearch/skills/ar-review.md` | Experiment review skill |
| `.claude/plugins/autoresearch/settings.yaml` | Default configuration |

Note: `ar-branch.md` and `ar-promote.md` skills are deferred to Phase 2+ when code-level mutations and baseline promotion are enabled.

---

## Task 1: Package Skeleton + Config

**Files:**
- Create: `autoresearch/__init__.py`
- Create: `autoresearch/config.py`
- Test: `tests/test_autoresearch/__init__.py`

- [ ] **Step 1: Create package directory structure**

```bash
mkdir -p autoresearch/{archive,descriptors,constraints,coordination,analysis,tree,hypothesis,state}
mkdir -p tests/test_autoresearch
```

- [ ] **Step 2: Create `autoresearch/__init__.py`**

```python
"""Auto-research system for autonomous drone racing exploration."""
```

- [ ] **Step 3: Create subpackage `__init__.py` files**

Create empty `__init__.py` in: `autoresearch/archive/`, `autoresearch/descriptors/`, `autoresearch/constraints/`, `autoresearch/coordination/`, `autoresearch/analysis/`, `autoresearch/tree/`, `autoresearch/hypothesis/`

- [ ] **Step 4: Create `tests/test_autoresearch/__init__.py`**

Empty file.

- [ ] **Step 5: Write config test**

File: `tests/test_autoresearch/test_config.py`

```python
"""Tests for autoresearch configuration."""

from autoresearch.config import AutoResearchConfig


def test_default_config():
    cfg = AutoResearchConfig()
    assert cfg.wandb_project == "corvidx-drone-racing"
    assert cfg.mode == "interactive"
    assert cfg.budgets["hyperparameter"] == 5_000_000
    assert cfg.constraints["min_success_rate"] == 0.8
    assert cfg.branch_selection["exploit_weight"] == 1.0
    assert cfg.coordination["claim_timeout_hours"] == 4


def test_config_override():
    cfg = AutoResearchConfig(mode="yolo", budgets={"hyperparameter": 1_000_000})
    assert cfg.mode == "yolo"
    assert cfg.budgets["hyperparameter"] == 1_000_000
```

- [ ] **Step 6: Run test to verify it fails**

Run: `python -m pytest tests/test_autoresearch/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'autoresearch.config'`

- [ ] **Step 7: Implement `autoresearch/config.py`**

```python
"""Configuration for the auto-research system."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AutoResearchConfig:
    """Top-level auto-research configuration.

    Mirrors the plugin settings.yaml schema.
    """

    wandb_project: str = "corvidx-drone-racing"
    mode: str = "interactive"  # interactive | autonomous | yolo

    budgets: dict[str, int] = field(default_factory=lambda: {
        "hyperparameter": 5_000_000,
        "algorithm": 15_000_000,
        "architecture": 30_000_000,
        "system": 50_000_000,
    })

    early_stop: dict[str, object] = field(default_factory=lambda: {
        "baseline_threshold": 0.7,
        "archive_redundancy": True,
        "poll_interval_seconds": 60,
    })

    constraints: dict[str, float] = field(default_factory=lambda: {
        "min_success_rate": 0.8,
        "min_avg_speed": 2.0,
        "max_gate_offset_ratio": 0.8,
        "max_acceleration_g": 4.0,
    })

    diff_policy: dict[str, dict[str, int]] = field(default_factory=lambda: {
        "max_diff_lines": {
            "algorithm": 500,
            "architecture": 1000,
            "system": 2000,
        },
    })

    branch_selection: dict[str, float] = field(default_factory=lambda: {
        "exploit_weight": 1.0,
        "explore_weight": 1.0,
        "gap_weight": 0.5,
        "diversity_weight": 0.3,
        "random_restart_pct": 0.2,
        "min_branch_experiments": 3,
        "tie_threshold": 0.05,
    })

    coordination: dict[str, object] = field(default_factory=lambda: {
        "claim_timeout_hours": 4,
        "claim_refresh_minutes": 15,
    })
```

- [ ] **Step 8: Run test to verify it passes**

Run: `python -m pytest tests/test_autoresearch/test_config.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add autoresearch/ tests/test_autoresearch/
git commit -m "feat(autoresearch): package skeleton and config dataclass"
```

---

## Task 2: Trajectory Loading

**Files:**
- Create: `autoresearch/analysis/trajectory.py`
- Create: `tests/test_autoresearch/conftest.py`
- Test: `tests/test_autoresearch/test_trajectory.py`

- [ ] **Step 1: Create shared test fixtures**

File: `tests/test_autoresearch/conftest.py`

```python
"""Shared fixtures for autoresearch tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def sample_npz(tmp_path: Path) -> Path:
    """Create a minimal .npz trajectory file matching schema_version=2."""
    rng = np.random.default_rng(42)
    T = 100  # timesteps
    n_gates = 4
    dt = 0.01

    positions = np.cumsum(rng.normal(0, 0.1, (T, 3)), axis=0)
    positions[:, 2] = np.abs(positions[:, 2]) + 1.0  # keep z > 0

    quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (T, 1))
    # Add small perturbations
    quaternions += rng.normal(0, 0.01, (T, 4))
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)

    # Physically realistic: smooth velocities with bounded acceleration
    # Start at 5 m/s forward, add small random walk (keeps accel < 4g)
    vel_base = np.tile([5.0, 0.0, 0.0], (T, 1))
    vel_noise = np.cumsum(rng.normal(0, 0.1, (T, 3)), axis=0)
    velocities = vel_base + vel_noise
    body_rates = rng.normal(0, 0.5, (T, 3))
    motor_rpms = rng.uniform(5000, 25000, (T, 4))
    actions = rng.uniform(-1, 1, (T, 4))
    rewards = rng.normal(1.0, 0.5, (T,))

    n_components = 6
    reward_components = rng.normal(0, 1, (T, n_components))
    reward_component_names = np.array([
        "progress", "body_rate", "action_smooth",
        "gate_passage", "gate_offset", "crash_penalty",
    ])

    gate_events = np.array([[25, 0], [50, 1], [75, 2]], dtype=np.int64)
    gate_positions = rng.uniform(-5, 5, (n_gates, 3))
    gate_orientations = np.tile([1.0, 0.0, 0.0, 0.0], (n_gates, 1))
    gate_half_extents = np.full((n_gates, 2), 0.5)

    path = tmp_path / "episode_0.npz"
    np.savez_compressed(
        path,
        schema_version=np.int64(2),
        positions=positions,
        quaternions=quaternions,
        velocities=velocities,
        body_rates=body_rates,
        motor_rpms=motor_rpms,
        actions=actions,
        rewards=rewards,
        reward_components=reward_components,
        reward_component_names=reward_component_names,
        gate_events=gate_events,
        gate_positions=gate_positions,
        gate_orientations=gate_orientations,
        gate_half_extents=gate_half_extents,
        dt=np.float64(dt),
    )
    return path


@pytest.fixture
def max_rpm() -> float:
    """Racing quad max RPM from configs/sim/numpy_quad.yaml."""
    return 31470.0


@pytest.fixture
def control_freq() -> float:
    """Control frequency = 1/dt."""
    return 100.0  # 1 / 0.01
```

- [ ] **Step 2: Write trajectory loading test**

File: `tests/test_autoresearch/test_trajectory.py`

```python
"""Tests for trajectory .npz loading."""

import numpy as np

from autoresearch.analysis.trajectory import load_trajectory, TrajectoryData


def test_load_trajectory_fields(sample_npz):
    traj = load_trajectory(sample_npz)
    assert isinstance(traj, TrajectoryData)
    assert traj.positions.shape == (100, 3)
    assert traj.quaternions.shape == (100, 4)
    assert traj.velocities.shape == (100, 3)
    assert traj.body_rates.shape == (100, 3)
    assert traj.motor_rpms.shape == (100, 4)
    assert traj.gate_events.shape == (3, 2)
    assert traj.gate_positions.shape == (4, 3)
    assert traj.dt == 0.01


def test_load_trajectory_dtypes(sample_npz):
    traj = load_trajectory(sample_npz)
    assert traj.positions.dtype == np.float64
    assert traj.motor_rpms.dtype == np.float64
    assert traj.gate_events.dtype == np.int64


def test_load_trajectory_schema_version(sample_npz):
    traj = load_trajectory(sample_npz)
    assert traj.schema_version == 2
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_autoresearch/test_trajectory.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Implement `autoresearch/analysis/trajectory.py`**

```python
"""Load and parse .npz trajectory files (schema_version=2)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TrajectoryData:
    """Parsed trajectory from a .npz file."""

    schema_version: int
    positions: NDArray[np.float64]       # (T, 3)
    quaternions: NDArray[np.float64]     # (T, 4)
    velocities: NDArray[np.float64]     # (T, 3)
    body_rates: NDArray[np.float64]     # (T, 3)
    motor_rpms: NDArray[np.float64]     # (T, 4)
    actions: NDArray[np.float64]        # (T, 4)
    rewards: NDArray[np.float64]        # (T,)
    reward_components: NDArray[np.float64]  # (T, n_components)
    reward_component_names: list[str]       # (n_components,)
    gate_events: NDArray[np.int64]      # (N_events, 2)
    gate_positions: NDArray[np.float64] # (n_gates, 3)
    gate_orientations: NDArray[np.float64]  # (n_gates, 4)
    gate_half_extents: NDArray[np.float64]  # (n_gates, 2)
    dt: float

    @property
    def n_timesteps(self) -> int:
        return self.positions.shape[0]

    @property
    def n_gates(self) -> int:
        return self.gate_positions.shape[0]


def load_trajectory(path: Path | str) -> TrajectoryData:
    """Load a .npz trajectory file and return parsed TrajectoryData."""
    path = Path(path)
    data = np.load(path, allow_pickle=False)

    return TrajectoryData(
        schema_version=int(data["schema_version"]),
        positions=data["positions"].astype(np.float64),
        quaternions=data["quaternions"].astype(np.float64),
        velocities=data["velocities"].astype(np.float64),
        body_rates=data["body_rates"].astype(np.float64),
        motor_rpms=data["motor_rpms"].astype(np.float64),
        actions=data["actions"].astype(np.float64),
        rewards=data["rewards"].astype(np.float64),
        reward_components=data["reward_components"].astype(np.float64),
        reward_component_names=list(data["reward_component_names"]),
        gate_events=data["gate_events"].astype(np.int64),
        gate_positions=data["gate_positions"].astype(np.float64),
        gate_orientations=data["gate_orientations"].astype(np.float64),
        gate_half_extents=data["gate_half_extents"].astype(np.float64),
        dt=float(data["dt"]),
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_autoresearch/test_trajectory.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add autoresearch/analysis/ tests/test_autoresearch/
git commit -m "feat(autoresearch): trajectory .npz loading"
```

---

## Task 3: Behavioral Descriptors

**Files:**
- Create: `autoresearch/descriptors/actuator.py`
- Create: `autoresearch/descriptors/smoothness.py`
- Create: `autoresearch/descriptors/aero_regime.py`
- Create: `autoresearch/descriptors/compute.py`
- Test: `tests/test_autoresearch/test_descriptors.py`

- [ ] **Step 1: Write descriptor tests**

File: `tests/test_autoresearch/test_descriptors.py`

```python
"""Tests for behavioral descriptor computation."""

import numpy as np
import pytest

from autoresearch.analysis.trajectory import load_trajectory
from autoresearch.descriptors.actuator import actuator_utilization
from autoresearch.descriptors.smoothness import control_smoothness
from autoresearch.descriptors.aero_regime import aero_regime_index
from autoresearch.descriptors.compute import compute_descriptors, DescriptorVector


class TestActuatorUtilization:
    def test_hovering_is_low(self, max_rpm):
        """Hovering (~25% throttle) should give low utilization."""
        # 4 motors at ~25% of max to hover (TWR ~4)
        motor_rpms = np.full((100, 4), max_rpm * 0.25)
        result = actuator_utilization(motor_rpms, max_rpm)
        assert 0.2 < result < 0.3

    def test_full_throttle_is_near_one(self, max_rpm):
        motor_rpms = np.full((100, 4), max_rpm * 0.99)
        result = actuator_utilization(motor_rpms, max_rpm)
        assert result > 0.95

    def test_range_zero_to_one(self, max_rpm):
        motor_rpms = np.random.default_rng(42).uniform(0, max_rpm, (100, 4))
        result = actuator_utilization(motor_rpms, max_rpm)
        assert 0.0 <= result <= 1.0


class TestControlSmoothness:
    def test_constant_commands_are_smooth(self, max_rpm, control_freq):
        motor_rpms = np.full((100, 4), max_rpm * 0.5)
        result = control_smoothness(motor_rpms, max_rpm, control_freq)
        assert result < 0.01  # near zero rate of change

    def test_oscillating_commands_are_rough(self, max_rpm, control_freq):
        """Alternating between 0 and max is maximally rough."""
        motor_rpms = np.zeros((100, 4))
        motor_rpms[::2] = max_rpm
        result = control_smoothness(motor_rpms, max_rpm, control_freq)
        assert result > 0.5

    def test_range_zero_to_one(self, max_rpm, control_freq):
        motor_rpms = np.random.default_rng(42).uniform(0, max_rpm, (100, 4))
        result = control_smoothness(motor_rpms, max_rpm, control_freq)
        assert 0.0 <= result <= 1.0


class TestAeroRegimeIndex:
    def test_stationary_is_zero(self):
        velocities = np.zeros((100, 3))
        quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (100, 1))
        result = aero_regime_index(velocities, quaternions)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_forward_flight_level_is_low(self):
        """Flying forward with body z aligned to velocity → alpha ≈ 0."""
        velocities = np.tile([10.0, 0.0, 0.0], (100, 1))
        # Identity quat = body z points up, velocity is horizontal
        # alpha = angle between [10,0,0] and body z [0,0,1] = 90 degrees
        # so this is actually high aero regime
        quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (100, 1))
        result = aero_regime_index(velocities, quaternions)
        # speed=10, sin(90°)=1 → index ≈ 10
        assert result > 5.0

    def test_hovering_is_zero(self):
        velocities = np.tile([0.0, 0.0, 0.1], (100, 1))  # near-zero speed
        quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (100, 1))
        result = aero_regime_index(velocities, quaternions)
        assert result < 0.2


class TestComputeDescriptors:
    def test_returns_descriptor_vector(self, sample_npz, max_rpm, control_freq):
        traj = load_trajectory(sample_npz)
        desc = compute_descriptors(traj, max_rpm, control_freq)
        assert isinstance(desc, DescriptorVector)
        assert 0.0 <= desc.actuator_utilization <= 1.0
        assert 0.0 <= desc.control_smoothness <= 1.0
        assert desc.aero_regime >= 0.0

    def test_descriptor_vector_to_tuple(self, sample_npz, max_rpm, control_freq):
        traj = load_trajectory(sample_npz)
        desc = compute_descriptors(traj, max_rpm, control_freq)
        t = desc.to_tuple()
        assert len(t) == 3
        assert t == (desc.actuator_utilization, desc.control_smoothness, desc.aero_regime)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_autoresearch/test_descriptors.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `autoresearch/descriptors/actuator.py`**

```python
"""Axis 1: Actuator utilization (aggressiveness) descriptor."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def actuator_utilization(motor_rpms: NDArray[np.float64], max_rpm: float) -> float:
    """Compute mean actuator utilization across a trajectory.

    Returns the mean L2-norm of motor RPMs divided by the L2-norm of max RPMs.
    Range: [0, 1]. Values near 1 indicate near-saturation operation.
    """
    # L2 norm of each timestep's 4-motor vector
    norms = np.linalg.norm(motor_rpms, axis=1)
    max_norm = np.linalg.norm(np.full(4, max_rpm))
    return float(np.mean(norms / max_norm))
```

- [ ] **Step 4: Implement `autoresearch/descriptors/smoothness.py`**

```python
"""Axis 2: Control smoothness (command rate of change) descriptor."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def control_smoothness(
    motor_rpms: NDArray[np.float64],
    max_rpm: float,
    control_freq: float,
) -> float:
    """Compute mean normalized rate of change of motor commands.

    Returns the mean L2-norm of d(motor_RPMs)/dt, normalized by
    max_rpm * control_freq to produce a dimensionless [0, 1] value.
    """
    if motor_rpms.shape[0] < 2:
        return 0.0

    # Finite difference: d(rpm)/dt
    d_rpm = np.diff(motor_rpms, axis=0) * control_freq  # (T-1, 4), units RPM/s
    d_rpm_norms = np.linalg.norm(d_rpm, axis=1)  # (T-1,)

    # Normalize: max possible rate = going from 0 to max_rpm in one step
    max_rate_norm = np.linalg.norm(np.full(4, max_rpm)) * control_freq

    return float(np.clip(np.mean(d_rpm_norms / max_rate_norm), 0.0, 1.0))
```

- [ ] **Step 5: Implement `autoresearch/descriptors/aero_regime.py`**

```python
"""Axis 3: Aerodynamic regime (speed × angle-of-attack) descriptor."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _body_z_from_quaternion(quaternions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Extract body z-axis direction from quaternions (w,x,y,z convention).

    Returns (T, 3) array of unit vectors.
    """
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    # Body z-axis in world frame from rotation matrix third column
    bz_x = 2.0 * (x * z + w * y)
    bz_y = 2.0 * (y * z - w * x)
    bz_z = 1.0 - 2.0 * (x * x + y * y)
    return np.column_stack([bz_x, bz_y, bz_z])


def aero_regime_index(
    velocities: NDArray[np.float64],
    quaternions: NDArray[np.float64],
) -> float:
    """Compute mean(||v|| * sin(alpha)) where alpha is angle between velocity and body z.

    Higher values indicate more time spent in aerodynamically complex regimes.
    """
    speeds = np.linalg.norm(velocities, axis=1)  # (T,)

    # Angle between velocity and body z-axis
    body_z = _body_z_from_quaternion(quaternions)  # (T, 3)

    # cos(alpha) = dot(v_hat, body_z) for nonzero velocity
    nonzero = speeds > 1e-6
    cos_alpha = np.zeros_like(speeds)
    if np.any(nonzero):
        v_hat = velocities[nonzero] / speeds[nonzero, np.newaxis]
        cos_alpha[nonzero] = np.sum(v_hat * body_z[nonzero], axis=1)
        cos_alpha[nonzero] = np.clip(cos_alpha[nonzero], -1.0, 1.0)

    sin_alpha = np.sqrt(1.0 - cos_alpha ** 2)
    return float(np.mean(speeds * sin_alpha))
```

- [ ] **Step 6: Implement `autoresearch/descriptors/compute.py`**

```python
"""Orchestrate all behavioral descriptor computations."""

from __future__ import annotations

from dataclasses import dataclass

from autoresearch.analysis.trajectory import TrajectoryData
from autoresearch.descriptors.actuator import actuator_utilization
from autoresearch.descriptors.smoothness import control_smoothness
from autoresearch.descriptors.aero_regime import aero_regime_index


@dataclass(frozen=True)
class DescriptorVector:
    """Three-axis behavioral descriptor for MAP-Elites archive placement."""

    actuator_utilization: float  # [0, 1]
    control_smoothness: float   # [0, 1]
    aero_regime: float          # [0, v_max]

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.actuator_utilization, self.control_smoothness, self.aero_regime)


def compute_descriptors(
    traj: TrajectoryData,
    max_rpm: float,
    control_freq: float,
) -> DescriptorVector:
    """Compute all three behavioral descriptors from trajectory data."""
    return DescriptorVector(
        actuator_utilization=actuator_utilization(traj.motor_rpms, max_rpm),
        control_smoothness=control_smoothness(traj.motor_rpms, max_rpm, control_freq),
        aero_regime=aero_regime_index(traj.velocities, traj.quaternions),
    )
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_autoresearch/test_descriptors.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add autoresearch/descriptors/ tests/test_autoresearch/test_descriptors.py
git commit -m "feat(autoresearch): behavioral descriptor computation (3 axes)"
```

---

## Task 4: MAP-Elites Archive

**Files:**
- Create: `autoresearch/archive/map_elites.py`
- Create: `autoresearch/archive/serialization.py`
- Test: `tests/test_autoresearch/test_archive.py`

- [ ] **Step 1: Write archive tests**

File: `tests/test_autoresearch/test_archive.py`

```python
"""Tests for MAP-Elites archive."""

import json
from pathlib import Path

import pytest

from autoresearch.archive.map_elites import MapElitesArchive, CellEntry
from autoresearch.archive.serialization import save_archive, load_archive
from autoresearch.descriptors.compute import DescriptorVector


@pytest.fixture
def archive():
    """A default 5x5x5 archive with racing quad v_max."""
    return MapElitesArchive(
        bins_per_axis=5,
        axis_ranges=[(0.25, 1.0), (0.0, 1.0), (0.0, 24.0)],
    )


@pytest.fixture
def sample_entry():
    return CellEntry(
        fitness=5.2,
        status="candidate",
        wandb_run_id="run_abc",
        git_commit="abc123",
        descriptors=DescriptorVector(0.5, 0.3, 10.0),
        constraint_results={"passed": True},
        budget_spent=5_000_000,
        hypothesis_id="hyp_001",
    )


class TestGridBinning:
    def test_descriptor_to_cell(self, archive):
        desc = DescriptorVector(0.5, 0.3, 10.0)
        cell = archive.descriptor_to_cell(desc)
        assert len(cell) == 3
        assert all(0 <= c < 5 for c in cell)

    def test_min_descriptor_maps_to_zero(self, archive):
        desc = DescriptorVector(0.25, 0.0, 0.0)
        cell = archive.descriptor_to_cell(desc)
        assert cell == (0, 0, 0)

    def test_max_descriptor_maps_to_last_bin(self, archive):
        desc = DescriptorVector(1.0, 1.0, 24.0)
        cell = archive.descriptor_to_cell(desc)
        assert cell == (4, 4, 4)

    def test_out_of_range_clamps(self, archive):
        desc = DescriptorVector(0.1, -0.5, 30.0)
        cell = archive.descriptor_to_cell(desc)
        assert cell == (0, 0, 4)


class TestInsertion:
    def test_insert_into_empty_cell(self, archive, sample_entry):
        inserted = archive.try_insert(sample_entry)
        assert inserted is True
        cell = archive.descriptor_to_cell(sample_entry.descriptors)
        assert archive.get(cell) == sample_entry

    def test_better_fitness_replaces(self, archive, sample_entry):
        archive.try_insert(sample_entry)
        better = CellEntry(
            fitness=4.0,
            status="candidate",
            wandb_run_id="run_def",
            git_commit="def456",
            descriptors=sample_entry.descriptors,
            constraint_results={"passed": True},
            budget_spent=5_000_000,
            hypothesis_id="hyp_002",
        )
        inserted = archive.try_insert(better)
        assert inserted is True
        cell = archive.descriptor_to_cell(sample_entry.descriptors)
        assert archive.get(cell).fitness == 4.0

    def test_worse_fitness_rejected(self, archive, sample_entry):
        archive.try_insert(sample_entry)
        worse = CellEntry(
            fitness=10.0,
            status="candidate",
            wandb_run_id="run_ghi",
            git_commit="ghi789",
            descriptors=sample_entry.descriptors,
            constraint_results={"passed": True},
            budget_spent=5_000_000,
            hypothesis_id="hyp_003",
        )
        inserted = archive.try_insert(worse)
        assert inserted is False

    def test_empty_cells_count(self, archive, sample_entry):
        assert archive.n_occupied == 0
        archive.try_insert(sample_entry)
        assert archive.n_occupied == 1
        assert archive.n_empty == 124


class TestPromotion:
    def test_initial_status_is_candidate(self, archive, sample_entry):
        archive.try_insert(sample_entry)
        cell = archive.descriptor_to_cell(sample_entry.descriptors)
        assert archive.get(cell).status == "candidate"

    def test_approve_candidate(self, archive, sample_entry):
        archive.try_insert(sample_entry)
        cell = archive.descriptor_to_cell(sample_entry.descriptors)
        archive.set_status(cell, "approved")
        assert archive.get(cell).status == "approved"

    def test_reject_candidate(self, archive, sample_entry):
        archive.try_insert(sample_entry)
        cell = archive.descriptor_to_cell(sample_entry.descriptors)
        archive.set_status(cell, "rejected")
        assert archive.get(cell).status == "rejected"


class TestSerialization:
    def test_round_trip(self, archive, sample_entry, tmp_path):
        archive.try_insert(sample_entry)
        path = tmp_path / "archive.json"
        save_archive(archive, path)
        loaded = load_archive(path)
        cell = archive.descriptor_to_cell(sample_entry.descriptors)
        assert loaded.get(cell).fitness == sample_entry.fitness
        assert loaded.get(cell).wandb_run_id == sample_entry.wandb_run_id

    def test_empty_archive_round_trip(self, archive, tmp_path):
        path = tmp_path / "archive.json"
        save_archive(archive, path)
        loaded = load_archive(path)
        assert loaded.n_occupied == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_autoresearch/test_archive.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `autoresearch/archive/map_elites.py`**

```python
"""MAP-Elites archive with fixed grid and promotion states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from autoresearch.descriptors.compute import DescriptorVector


@dataclass
class CellEntry:
    """An entry in a MAP-Elites archive cell."""

    fitness: float  # lap time (lower is better)
    status: Literal["candidate", "approved", "rejected"]
    wandb_run_id: str
    git_commit: str
    descriptors: DescriptorVector
    constraint_results: dict
    budget_spent: int
    hypothesis_id: str


class MapElitesArchive:
    """Fixed-grid MAP-Elites archive."""

    def __init__(
        self,
        bins_per_axis: int = 5,
        axis_ranges: list[tuple[float, float]] | None = None,
    ) -> None:
        self.bins_per_axis = bins_per_axis
        self.axis_ranges = axis_ranges or [
            (0.25, 1.0),   # actuator utilization
            (0.0, 1.0),    # control smoothness
            (0.0, 24.0),   # aero regime index
        ]
        self._grid: dict[tuple[int, int, int], CellEntry] = {}

    @property
    def n_cells(self) -> int:
        return self.bins_per_axis ** 3

    @property
    def n_occupied(self) -> int:
        return len(self._grid)

    @property
    def n_empty(self) -> int:
        return self.n_cells - self.n_occupied

    def descriptor_to_cell(self, desc: DescriptorVector) -> tuple[int, int, int]:
        """Map a descriptor vector to a grid cell index."""
        values = desc.to_tuple()
        indices = []
        for val, (lo, hi) in zip(values, self.axis_ranges):
            clamped = max(lo, min(hi, val))
            # Normalize to [0, 1), then map to bin index
            normalized = (clamped - lo) / (hi - lo)
            idx = min(int(normalized * self.bins_per_axis), self.bins_per_axis - 1)
            indices.append(idx)
        return (indices[0], indices[1], indices[2])

    def get(self, cell: tuple[int, int, int]) -> CellEntry | None:
        """Get the entry in a cell, or None if empty."""
        return self._grid.get(cell)

    def try_insert(self, entry: CellEntry) -> bool:
        """Try to insert an entry. Returns True if inserted (cell was empty or fitness improved)."""
        cell = self.descriptor_to_cell(entry.descriptors)
        existing = self._grid.get(cell)
        if existing is None or entry.fitness < existing.fitness:
            self._grid[cell] = entry
            return True
        return False

    def set_status(
        self, cell: tuple[int, int, int], status: Literal["candidate", "approved", "rejected"]
    ) -> None:
        """Update the promotion status of a cell entry."""
        entry = self._grid.get(cell)
        if entry is None:
            raise KeyError(f"Cell {cell} is empty")
        entry.status = status

    def occupied_cells(self) -> dict[tuple[int, int, int], CellEntry]:
        """Return all occupied cells."""
        return dict(self._grid)

    def approved_cells(self) -> dict[tuple[int, int, int], CellEntry]:
        """Return only cells with approved status."""
        return {k: v for k, v in self._grid.items() if v.status == "approved"}
```

- [ ] **Step 4: Implement `autoresearch/archive/serialization.py`**

```python
"""Serialize/deserialize MAP-Elites archive to/from JSON."""

from __future__ import annotations

import json
from pathlib import Path

from autoresearch.archive.map_elites import MapElitesArchive, CellEntry
from autoresearch.descriptors.compute import DescriptorVector


def save_archive(archive: MapElitesArchive, path: Path | str) -> None:
    """Save archive to JSON file."""
    path = Path(path)
    data = {
        "bins_per_axis": archive.bins_per_axis,
        "axis_ranges": archive.axis_ranges,
        "cells": {},
    }
    for cell, entry in archive.occupied_cells().items():
        key = f"{cell[0]},{cell[1]},{cell[2]}"
        data["cells"][key] = {
            "fitness": entry.fitness,
            "status": entry.status,
            "wandb_run_id": entry.wandb_run_id,
            "git_commit": entry.git_commit,
            "descriptors": list(entry.descriptors.to_tuple()),
            "constraint_results": entry.constraint_results,
            "budget_spent": entry.budget_spent,
            "hypothesis_id": entry.hypothesis_id,
        }
    path.write_text(json.dumps(data, indent=2) + "\n")


def load_archive(path: Path | str) -> MapElitesArchive:
    """Load archive from JSON file."""
    path = Path(path)
    data = json.loads(path.read_text())
    archive = MapElitesArchive(
        bins_per_axis=data["bins_per_axis"],
        axis_ranges=[tuple(r) for r in data["axis_ranges"]],
    )
    for key, entry_data in data.get("cells", {}).items():
        desc_vals = entry_data["descriptors"]
        entry = CellEntry(
            fitness=entry_data["fitness"],
            status=entry_data["status"],
            wandb_run_id=entry_data["wandb_run_id"],
            git_commit=entry_data["git_commit"],
            descriptors=DescriptorVector(*desc_vals),
            constraint_results=entry_data["constraint_results"],
            budget_spent=entry_data["budget_spent"],
            hypothesis_id=entry_data["hypothesis_id"],
        )
        archive.try_insert(entry)
    return archive
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_autoresearch/test_archive.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add autoresearch/archive/ tests/test_autoresearch/test_archive.py
git commit -m "feat(autoresearch): MAP-Elites archive with promotion states"
```

---

## Task 5: Constraint Validation

**Files:**
- Create: `autoresearch/constraints/validator.py`
- Create: `autoresearch/constraints/physics.py`
- Create: `autoresearch/constraints/racing.py`
- Test: `tests/test_autoresearch/test_constraints.py`

- [ ] **Step 1: Write constraint tests**

File: `tests/test_autoresearch/test_constraints.py`

```python
"""Tests for constraint validation."""

import numpy as np
import pytest

from autoresearch.analysis.trajectory import TrajectoryData
from autoresearch.constraints.physics import check_physics
from autoresearch.constraints.racing import check_gate_passage, check_behavioral_sanity
from autoresearch.constraints.validator import ConstraintValidator, ValidationResult


def _make_trajectory(**overrides) -> TrajectoryData:
    """Helper to create a TrajectoryData with sensible defaults."""
    T = overrides.pop("n_timesteps", 100)
    defaults = dict(
        schema_version=2,
        positions=np.column_stack([
            np.linspace(0, 10, T),
            np.zeros(T),
            np.full(T, 2.0),
        ]),
        quaternions=np.tile([1.0, 0.0, 0.0, 0.0], (T, 1)),
        velocities=np.tile([5.0, 0.0, 0.0], (T, 1)),
        body_rates=np.zeros((T, 3)),
        motor_rpms=np.full((T, 4), 15000.0),
        actions=np.zeros((T, 4)),
        rewards=np.ones(T),
        gate_events=np.array([[25, 0], [50, 1], [75, 2]], dtype=np.int64),
        gate_positions=np.array([[2.5, 0, 2], [5.0, 0, 2], [7.5, 0, 2]], dtype=np.float64),
        gate_orientations=np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
        gate_half_extents=np.full((3, 2), 0.5),
        dt=0.01,
    )
    defaults.update(overrides)
    return TrajectoryData(**defaults)


class TestPhysicsConstraints:
    def test_valid_trajectory_passes(self):
        traj = _make_trajectory()
        result = check_physics(traj, max_rpm=31470.0, max_accel_g=4.0)
        assert result.passed is True

    def test_underground_flight_fails(self):
        traj = _make_trajectory()
        positions = traj.positions.copy()
        positions[:, 2] = -1.0  # all underground
        traj = _make_trajectory(positions=positions)
        result = check_physics(traj, max_rpm=31470.0, max_accel_g=4.0)
        assert result.passed is False
        assert "ground" in result.reason.lower()

    def test_motor_over_limit_fails(self):
        traj = _make_trajectory(motor_rpms=np.full((100, 4), 40000.0))
        result = check_physics(traj, max_rpm=31470.0, max_accel_g=4.0)
        assert result.passed is False
        assert "motor" in result.reason.lower()

    def test_motor_slightly_over_passes_with_tolerance(self):
        # 5% tolerance: 31470 * 1.05 = 33043.5
        traj = _make_trajectory(motor_rpms=np.full((100, 4), 33000.0))
        result = check_physics(traj, max_rpm=31470.0, max_accel_g=4.0)
        assert result.passed is True

    def test_excessive_acceleration_fails(self):
        """Velocity step change creates huge acceleration."""
        vels = np.tile([0.0, 0.0, 0.0], (100, 1))
        vels[50:] = [100.0, 0.0, 0.0]  # instant jump at step 50
        traj = _make_trajectory(velocities=vels)
        result = check_physics(traj, max_rpm=31470.0, max_accel_g=4.0)
        assert result.passed is False
        assert "acceleration" in result.reason.lower()

    def test_smooth_acceleration_passes(self):
        """Gradual velocity ramp stays within limits."""
        vels = np.column_stack([
            np.linspace(0, 5, 100),  # gentle ramp: 5 m/s over 1s = 5 m/s^2 < 4g
            np.zeros(100),
            np.zeros(100),
        ])
        traj = _make_trajectory(velocities=vels)
        result = check_physics(traj, max_rpm=31470.0, max_accel_g=4.0)
        assert result.passed is True


class TestGatePassageConstraints:
    def test_sequential_passage_passes(self):
        traj = _make_trajectory()
        result = check_gate_passage(traj, max_offset_ratio=0.8)
        assert result.passed is True

    def test_skipped_gate_fails(self):
        # Gate events skip gate 1
        gate_events = np.array([[25, 0], [50, 2]], dtype=np.int64)
        traj = _make_trajectory(gate_events=gate_events)
        result = check_gate_passage(traj, max_offset_ratio=0.8)
        assert result.passed is False
        assert "sequential" in result.reason.lower() or "skip" in result.reason.lower()


class TestBehavioralSanity:
    def test_normal_trajectory_passes(self):
        traj = _make_trajectory()
        result = check_behavioral_sanity(
            traj, min_avg_speed=2.0, dr_active=True,
        )
        assert result.passed is True

    def test_too_slow_fails(self):
        traj = _make_trajectory(velocities=np.tile([0.5, 0.0, 0.0], (100, 1)))
        result = check_behavioral_sanity(
            traj, min_avg_speed=2.0, dr_active=True,
        )
        assert result.passed is False
        assert "speed" in result.reason.lower()


class TestConstraintValidator:
    def test_all_pass(self):
        traj = _make_trajectory()
        metrics = {
            "success_rate": 0.9,
            "lap_times": [5.0, 5.1, 5.2, 4.9, 5.0] * 4,
        }
        validator = ConstraintValidator(
            max_rpm=31470.0, max_accel_g=4.0, min_success_rate=0.8,
            min_avg_speed=2.0, max_gate_offset_ratio=0.8,
        )
        result = validator.validate(traj, metrics, dr_active=True)
        assert isinstance(result, ValidationResult)
        assert result.passed is True

    def test_low_success_rate_fails(self):
        traj = _make_trajectory()
        metrics = {
            "success_rate": 0.5,
            "lap_times": [5.0, 5.1, 5.2, 4.9, 5.0] * 4,
        }
        validator = ConstraintValidator(
            max_rpm=31470.0, max_accel_g=4.0, min_success_rate=0.8,
            min_avg_speed=2.0, max_gate_offset_ratio=0.8,
        )
        result = validator.validate(traj, metrics, dr_active=True)
        assert result.passed is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_autoresearch/test_constraints.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `autoresearch/constraints/physics.py`**

```python
"""Physics plausibility constraint checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autoresearch.analysis.trajectory import TrajectoryData

G = 9.81  # m/s^2


@dataclass
class ConstraintCheck:
    passed: bool
    reason: str


def check_physics(
    traj: TrajectoryData,
    max_rpm: float,
    max_accel_g: float = 4.0,
) -> ConstraintCheck:
    """Check physics plausibility of a trajectory."""
    # Ground/ceiling: reject if >1% of timesteps violate
    z = traj.positions[:, 2]
    underground = np.mean(z <= 0.0)
    if underground > 0.01:
        return ConstraintCheck(False, f"Ground violation: {underground:.1%} of timesteps below ground")

    above_ceiling = np.mean(z > 10.0)
    if above_ceiling > 0.01:
        return ConstraintCheck(False, f"Ceiling violation: {above_ceiling:.1%} of timesteps above ceiling")

    # Motor limits: 5% tolerance
    max_motor = np.max(traj.motor_rpms)
    if max_motor > max_rpm * 1.05:
        return ConstraintCheck(False, f"Motor limit exceeded: {max_motor:.0f} > {max_rpm * 1.05:.0f} RPM")

    # Acceleration check
    if traj.velocities.shape[0] >= 2:
        accel = np.diff(traj.velocities, axis=0) / traj.dt
        accel_mag = np.linalg.norm(accel, axis=1)
        max_accel = np.max(accel_mag)
        if max_accel > max_accel_g * G:
            return ConstraintCheck(
                False,
                f"Acceleration exceeded: {max_accel / G:.1f}g > {max_accel_g}g limit",
            )

    # Quaternion norm
    quat_norms = np.linalg.norm(traj.quaternions, axis=1)
    max_deviation = np.max(np.abs(quat_norms - 1.0))
    if max_deviation > 0.01:
        return ConstraintCheck(False, f"Quaternion norm deviation: {max_deviation:.4f} > 0.01")

    return ConstraintCheck(True, "Physics plausibility: all checks passed")
```

- [ ] **Step 4: Implement `autoresearch/constraints/racing.py`**

```python
"""Gate passage quality and behavioral sanity constraint checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autoresearch.analysis.trajectory import TrajectoryData
from autoresearch.constraints.physics import ConstraintCheck


def check_gate_passage(
    traj: TrajectoryData,
    max_offset_ratio: float = 0.8,
) -> ConstraintCheck:
    """Check gate passage quality: sequential order and clearance."""
    if traj.gate_events.shape[0] == 0:
        return ConstraintCheck(True, "No gate events to validate")

    # Check sequential ordering
    gate_indices = traj.gate_events[:, 1]
    for i in range(1, len(gate_indices)):
        expected_next = (gate_indices[i - 1] + 1) % traj.n_gates
        if gate_indices[i] != expected_next:
            return ConstraintCheck(
                False,
                f"Gate skip detected: gate {gate_indices[i-1]} → {gate_indices[i]}, "
                f"expected {expected_next}",
            )

    # Check gate offset at passage timesteps
    offsets = []
    for event_idx in range(traj.gate_events.shape[0]):
        timestep = traj.gate_events[event_idx, 0]
        gate_idx = traj.gate_events[event_idx, 1]

        if timestep >= traj.n_timesteps:
            continue

        drone_pos = traj.positions[timestep]
        gate_pos = traj.gate_positions[gate_idx]
        half_ext = traj.gate_half_extents[gate_idx]
        max_half = np.max(half_ext)

        offset_dist = np.linalg.norm(drone_pos - gate_pos)
        offsets.append(offset_dist / max_half if max_half > 0 else 0.0)

    if offsets:
        mean_ratio = np.mean(offsets)
        max_ratio = np.max(offsets)
        if mean_ratio > max_offset_ratio:
            return ConstraintCheck(
                False, f"Gate offset too large: mean ratio {mean_ratio:.2f} > {max_offset_ratio}"
            )
        if max_ratio > 0.95:
            return ConstraintCheck(
                False, f"Gate clearance too tight: max ratio {max_ratio:.2f} > 0.95"
            )

    return ConstraintCheck(True, "Gate passage quality: all checks passed")


def check_behavioral_sanity(
    traj: TrajectoryData,
    min_avg_speed: float = 2.0,
    dr_active: bool = True,
) -> ConstraintCheck:
    """Check behavioral sanity: forward progress, speed, diversity."""
    # Minimum average speed
    speeds = np.linalg.norm(traj.velocities, axis=1)
    avg_speed = float(np.mean(speeds))
    if avg_speed < min_avg_speed:
        return ConstraintCheck(
            False, f"Average speed too low: {avg_speed:.2f} m/s < {min_avg_speed} m/s"
        )

    # Forward progress: check 3-second sliding window
    window_steps = int(3.0 / traj.dt)
    if traj.gate_events.shape[0] > 0 and traj.n_timesteps > window_steps:
        # Compute distance to current target gate over time
        # Simplified: check that position moves forward overall
        for start in range(0, traj.n_timesteps - window_steps):
            end = start + window_steps
            displacement = np.linalg.norm(
                traj.positions[end] - traj.positions[start]
            )
            if displacement < 0.1:  # effectively stationary for 3 seconds
                return ConstraintCheck(
                    False,
                    f"No forward progress: displacement {displacement:.2f}m "
                    f"over 3s window at step {start}",
                )

    return ConstraintCheck(True, "Behavioral sanity: all checks passed")
```

- [ ] **Step 5: Implement `autoresearch/constraints/validator.py`**

```python
"""Top-level constraint validator that combines all checks."""

from __future__ import annotations

from dataclasses import dataclass, field

from autoresearch.analysis.trajectory import TrajectoryData
from autoresearch.constraints.physics import check_physics, ConstraintCheck
from autoresearch.constraints.racing import check_gate_passage, check_behavioral_sanity


@dataclass
class ValidationResult:
    """Result of running all constraint validations."""

    passed: bool
    constraint_results: dict[str, ConstraintCheck] = field(default_factory=dict)
    reasoning: str = ""
    soft_signals: dict[str, float] = field(default_factory=dict)


class ConstraintValidator:
    """Runs all hard constraint checks on a trajectory + metrics."""

    def __init__(
        self,
        max_rpm: float = 31470.0,
        max_accel_g: float = 4.0,
        min_success_rate: float = 0.8,
        min_avg_speed: float = 2.0,
        max_gate_offset_ratio: float = 0.8,
    ) -> None:
        self.max_rpm = max_rpm
        self.max_accel_g = max_accel_g
        self.min_success_rate = min_success_rate
        self.min_avg_speed = min_avg_speed
        self.max_gate_offset_ratio = max_gate_offset_ratio

    def validate(
        self,
        traj: TrajectoryData,
        metrics: dict,
        dr_active: bool = True,
    ) -> ValidationResult:
        """Run all constraint checks and return combined result.

        Args:
            traj: Parsed trajectory data from .npz file.
            metrics: Pre-aggregated metrics dict with keys:
                - "success_rate" (float): fraction of eval episodes that succeeded
                - "lap_times" (list[float]): per-episode lap times
                Caller is responsible for aggregating raw EpisodeMetrics
                into this format (typically from the final 200 eval episodes).
            dr_active: Whether domain randomization was active during eval.
        """
        results: dict[str, ConstraintCheck] = {}

        # Success rate
        success_rate = metrics.get("success_rate", 0.0)
        if success_rate < self.min_success_rate:
            results["success_rate"] = ConstraintCheck(
                False, f"Success rate {success_rate:.1%} < {self.min_success_rate:.1%}"
            )
        else:
            results["success_rate"] = ConstraintCheck(True, f"Success rate: {success_rate:.1%}")

        # Physics plausibility
        results["physics"] = check_physics(traj, self.max_rpm, self.max_accel_g)

        # Gate passage quality
        results["gate_passage"] = check_gate_passage(traj, self.max_gate_offset_ratio)

        # Behavioral sanity
        results["behavioral_sanity"] = check_behavioral_sanity(
            traj, self.min_avg_speed, dr_active
        )

        all_passed = all(r.passed for r in results.values())
        failed = [f"{k}: {v.reason}" for k, v in results.items() if not v.passed]
        reasoning = "All constraints passed" if all_passed else "; ".join(failed)

        return ValidationResult(
            passed=all_passed,
            constraint_results=results,
            reasoning=reasoning,
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_autoresearch/test_constraints.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add autoresearch/constraints/ tests/test_autoresearch/test_constraints.py
git commit -m "feat(autoresearch): constraint validation (physics, racing, behavioral)"
```

---

## Task 6: Hypothesis Schema

**Files:**
- Create: `autoresearch/hypothesis/schema.py`
- Test: `tests/test_autoresearch/test_hypothesis.py`

- [ ] **Step 1: Write hypothesis tests**

File: `tests/test_autoresearch/test_hypothesis.py`

```python
"""Tests for hypothesis schema and content hashing."""

from autoresearch.hypothesis.schema import (
    Hypothesis, HydraOverride, compute_hypothesis_id,
)


def test_hydra_override_to_string():
    override = HydraOverride(key="control.learning_rate", value="1e-4")
    assert override.to_cli_arg() == "control.learning_rate=1e-4"


def test_hypothesis_id_deterministic():
    h1 = Hypothesis.create(
        scope="hyperparameter",
        description="Increase learning rate",
        changes=[HydraOverride("control.learning_rate", "3e-4")],
        rationale="Current LR may be too conservative",
    )
    h2 = Hypothesis.create(
        scope="hyperparameter",
        description="Different description",  # description doesn't affect ID
        changes=[HydraOverride("control.learning_rate", "3e-4")],
        rationale="Different rationale",
    )
    assert h1.id == h2.id  # same scope + changes + targets = same ID


def test_different_targets_different_id():
    h1 = Hypothesis.create(
        scope="hyperparameter",
        description="LR 3e-4",
        changes=[HydraOverride("control.learning_rate", "3e-4")],
        target_cells=[(1, 2, 3)],
        rationale="test",
    )
    h2 = Hypothesis.create(
        scope="hyperparameter",
        description="LR 3e-4",
        changes=[HydraOverride("control.learning_rate", "3e-4")],
        target_cells=[(4, 4, 4)],
        rationale="test",
    )
    assert h1.id != h2.id


def test_different_changes_different_id():
    h1 = Hypothesis.create(
        scope="hyperparameter",
        description="LR 3e-4",
        changes=[HydraOverride("control.learning_rate", "3e-4")],
        rationale="test",
    )
    h2 = Hypothesis.create(
        scope="hyperparameter",
        description="LR 1e-3",
        changes=[HydraOverride("control.learning_rate", "1e-3")],
        rationale="test",
    )
    assert h1.id != h2.id


def test_hypothesis_cli_overrides():
    h = Hypothesis.create(
        scope="hyperparameter",
        description="Multi-param change",
        changes=[
            HydraOverride("control.learning_rate", "3e-4"),
            HydraOverride("reward.gate_passage", "2.0"),
        ],
        rationale="test",
    )
    args = h.to_cli_overrides()
    assert "control.learning_rate=3e-4" in args
    assert "reward.gate_passage=2.0" in args


def test_hypothesis_default_budget():
    h = Hypothesis.create(
        scope="hyperparameter",
        description="test",
        changes=[HydraOverride("control.learning_rate", "3e-4")],
        rationale="test",
    )
    assert h.estimated_budget == 5_000_000

    h2 = Hypothesis.create(
        scope="algorithm",
        description="test",
        changes=[HydraOverride("control.n_steps", "2000")],
        rationale="test",
    )
    assert h2.estimated_budget == 15_000_000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_autoresearch/test_hypothesis.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `autoresearch/hypothesis/schema.py`**

```python
"""Hypothesis data contract and content hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal

SCOPE_BUDGETS = {
    "hyperparameter": 5_000_000,
    "algorithm": 15_000_000,
    "architecture": 30_000_000,
    "system": 50_000_000,
}


@dataclass(frozen=True)
class HydraOverride:
    """A single Hydra config override."""

    key: str
    value: str

    def to_cli_arg(self) -> str:
        return f"{self.key}={self.value}"


def compute_hypothesis_id(
    scope: str,
    changes: list[HydraOverride],
    target_cells: list[tuple[int, int, int]] | None = None,
) -> str:
    """Compute a deterministic content hash for deduplication.

    Hash includes scope + changes + target_cells per spec.
    """
    content = {
        "scope": scope,
        "changes": sorted([(c.key, c.value) for c in changes]),
        "target_cells": sorted([list(c) for c in (target_cells or [])]),
    }
    blob = json.dumps(content, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


@dataclass
class Hypothesis:
    """A research hypothesis specifying what to change and why."""

    id: str
    parent_id: str | None
    inspired_by: str | None
    scope: Literal["hyperparameter", "algorithm", "architecture", "system"]
    description: str
    changes: list[HydraOverride]
    target_cells: list[tuple[int, int, int]]
    predicted_descriptor_range: dict
    rationale: str
    estimated_budget: int
    risk_level: Literal["low", "medium", "high"]
    novelty_score: float

    @classmethod
    def create(
        cls,
        scope: Literal["hyperparameter", "algorithm", "architecture", "system"],
        description: str,
        changes: list[HydraOverride],
        rationale: str,
        parent_id: str | None = None,
        inspired_by: str | None = None,
        target_cells: list[tuple[int, int, int]] | None = None,
        predicted_descriptor_range: dict | None = None,
        estimated_budget: int | None = None,
        risk_level: Literal["low", "medium", "high"] = "low",
        novelty_score: float = 1.0,
    ) -> Hypothesis:
        """Create a hypothesis with auto-computed ID and defaults."""
        return cls(
            id=compute_hypothesis_id(scope, changes, target_cells),
            parent_id=parent_id,
            inspired_by=inspired_by,
            scope=scope,
            description=description,
            changes=changes,
            target_cells=target_cells or [],
            predicted_descriptor_range=predicted_descriptor_range or {},
            rationale=rationale,
            estimated_budget=estimated_budget or SCOPE_BUDGETS[scope],
            risk_level=risk_level,
            novelty_score=novelty_score,
        )

    def to_cli_overrides(self) -> list[str]:
        """Generate Hydra CLI override arguments."""
        return [c.to_cli_arg() for c in self.changes]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_autoresearch/test_hypothesis.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autoresearch/hypothesis/ tests/test_autoresearch/test_hypothesis.py
git commit -m "feat(autoresearch): hypothesis schema with content hashing"
```

---

## Task 7: Research Tree

**Files:**
- Create: `autoresearch/tree/research_tree.py`
- Create: `autoresearch/tree/serialization.py`
- Test: `tests/test_autoresearch/test_tree.py`

- [ ] **Step 1: Write research tree tests**

File: `tests/test_autoresearch/test_tree.py`

```python
"""Tests for research tree."""

from autoresearch.tree.research_tree import ResearchTree, TreeNode
from autoresearch.tree.serialization import save_tree, load_tree


def test_create_baseline_root():
    tree = ResearchTree.create_with_baseline("monorace_baseline", "abc123")
    root = tree.root
    assert root.hypothesis_id == "baseline_v1"
    assert root.git_commit == "abc123"
    assert root.parent_id is None
    assert tree.total_experiments == 0


def test_add_experiment():
    tree = ResearchTree.create_with_baseline("monorace_baseline", "abc123")
    tree.add_experiment(
        hypothesis_id="hyp_001",
        parent_id="baseline_v1",
        scope="hyperparameter",
        description="Increase LR",
        status="running",
    )
    assert tree.total_experiments == 1
    node = tree.get_node("hyp_001")
    assert node.parent_id == "baseline_v1"
    assert node.status == "running"


def test_complete_experiment():
    tree = ResearchTree.create_with_baseline("monorace_baseline", "abc123")
    tree.add_experiment(
        hypothesis_id="hyp_001",
        parent_id="baseline_v1",
        scope="hyperparameter",
        description="Increase LR",
        status="running",
    )
    tree.complete_experiment("hyp_001", fitness=5.0, wandb_run_id="run_abc")
    node = tree.get_node("hyp_001")
    assert node.status == "completed"
    assert node.fitness == 5.0


def test_fail_experiment():
    tree = ResearchTree.create_with_baseline("monorace_baseline", "abc123")
    tree.add_experiment(
        hypothesis_id="hyp_001",
        parent_id="baseline_v1",
        scope="hyperparameter",
        description="Increase LR",
        status="running",
    )
    tree.fail_experiment("hyp_001", reason="NaN at step 50k")
    node = tree.get_node("hyp_001")
    assert node.status == "failed"
    assert "NaN" in node.failure_reason


def test_serialization_round_trip(tmp_path):
    tree = ResearchTree.create_with_baseline("monorace_baseline", "abc123")
    tree.add_experiment(
        hypothesis_id="hyp_001",
        parent_id="baseline_v1",
        scope="hyperparameter",
        description="Increase LR",
        status="completed",
    )
    path = tmp_path / "tree.json"
    save_tree(tree, path)
    loaded = load_tree(path)
    assert loaded.total_experiments == 1
    assert loaded.get_node("hyp_001").description == "Increase LR"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_autoresearch/test_tree.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `autoresearch/tree/research_tree.py`**

```python
"""Research tree for tracking experiment lineage."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TreeNode:
    """A node in the research tree."""

    hypothesis_id: str
    parent_id: str | None
    inspired_by: str | None = None
    scope: str = "baseline"
    description: str = ""
    status: str = "pending"  # pending | running | completed | failed
    fitness: float | None = None
    wandb_run_id: str | None = None
    git_commit: str | None = None
    failure_reason: str | None = None
    archive_cell: tuple[int, int, int] | None = None


class ResearchTree:
    """Tree structure tracking all experiments and their lineage."""

    def __init__(self) -> None:
        self._nodes: dict[str, TreeNode] = {}
        self._root_id: str | None = None

    @classmethod
    def create_with_baseline(cls, baseline_name: str, git_commit: str) -> ResearchTree:
        """Create a new tree with a baseline root node."""
        tree = cls()
        root = TreeNode(
            hypothesis_id="baseline_v1",
            parent_id=None,
            scope="baseline",
            description=baseline_name,
            status="baseline",
            git_commit=git_commit,
        )
        tree._nodes[root.hypothesis_id] = root
        tree._root_id = root.hypothesis_id
        return tree

    @property
    def root(self) -> TreeNode:
        assert self._root_id is not None
        return self._nodes[self._root_id]

    @property
    def total_experiments(self) -> int:
        """Count non-baseline nodes."""
        return sum(1 for n in self._nodes.values() if n.scope != "baseline")

    def get_node(self, hypothesis_id: str) -> TreeNode:
        return self._nodes[hypothesis_id]

    def add_experiment(
        self,
        hypothesis_id: str,
        parent_id: str,
        scope: str,
        description: str,
        status: str = "pending",
        inspired_by: str | None = None,
        git_commit: str | None = None,
    ) -> TreeNode:
        """Add a new experiment node."""
        node = TreeNode(
            hypothesis_id=hypothesis_id,
            parent_id=parent_id,
            inspired_by=inspired_by,
            scope=scope,
            description=description,
            status=status,
            git_commit=git_commit,
        )
        self._nodes[hypothesis_id] = node
        return node

    def complete_experiment(
        self,
        hypothesis_id: str,
        fitness: float,
        wandb_run_id: str | None = None,
        archive_cell: tuple[int, int, int] | None = None,
    ) -> None:
        node = self._nodes[hypothesis_id]
        node.status = "completed"
        node.fitness = fitness
        node.wandb_run_id = wandb_run_id
        node.archive_cell = archive_cell

    def fail_experiment(self, hypothesis_id: str, reason: str) -> None:
        node = self._nodes[hypothesis_id]
        node.status = "failed"
        node.failure_reason = reason

    def all_nodes(self) -> list[TreeNode]:
        return list(self._nodes.values())
```

- [ ] **Step 4: Implement `autoresearch/tree/serialization.py`**

```python
"""Serialize/deserialize research tree to/from JSON."""

from __future__ import annotations

import json
from pathlib import Path

from autoresearch.tree.research_tree import ResearchTree, TreeNode


def save_tree(tree: ResearchTree, path: Path | str) -> None:
    """Save research tree to JSON."""
    path = Path(path)
    data = {
        "root_id": tree._root_id,
        "nodes": {},
    }
    for node in tree.all_nodes():
        data["nodes"][node.hypothesis_id] = {
            "parent_id": node.parent_id,
            "inspired_by": node.inspired_by,
            "scope": node.scope,
            "description": node.description,
            "status": node.status,
            "fitness": node.fitness,
            "wandb_run_id": node.wandb_run_id,
            "git_commit": node.git_commit,
            "failure_reason": node.failure_reason,
            "archive_cell": list(node.archive_cell) if node.archive_cell else None,
        }
    path.write_text(json.dumps(data, indent=2) + "\n")


def load_tree(path: Path | str) -> ResearchTree:
    """Load research tree from JSON."""
    path = Path(path)
    data = json.loads(path.read_text())
    tree = ResearchTree()
    tree._root_id = data["root_id"]
    for hyp_id, node_data in data["nodes"].items():
        cell = node_data.get("archive_cell")
        tree._nodes[hyp_id] = TreeNode(
            hypothesis_id=hyp_id,
            parent_id=node_data["parent_id"],
            inspired_by=node_data.get("inspired_by"),
            scope=node_data["scope"],
            description=node_data["description"],
            status=node_data["status"],
            fitness=node_data.get("fitness"),
            wandb_run_id=node_data.get("wandb_run_id"),
            git_commit=node_data.get("git_commit"),
            failure_reason=node_data.get("failure_reason"),
            archive_cell=tuple(cell) if cell else None,
        )
    return tree
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_autoresearch/test_tree.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add autoresearch/tree/ tests/test_autoresearch/test_tree.py
git commit -m "feat(autoresearch): research tree with baseline chain"
```

---

## Task 8: Coordination (Claims + Dedup)

**Files:**
- Create: `autoresearch/coordination/claims.py`
- Create: `autoresearch/coordination/dedup.py`
- Test: `tests/test_autoresearch/test_coordination.py`

- [ ] **Step 1: Write coordination tests**

File: `tests/test_autoresearch/test_coordination.py`

```python
"""Tests for experiment coordination and deduplication."""

from datetime import datetime, timezone, timedelta
from pathlib import Path

from autoresearch.coordination.claims import (
    Claim, load_claims, save_claims, add_claim, expire_claims, release_claim,
)
from autoresearch.coordination.dedup import is_duplicate
from autoresearch.hypothesis.schema import Hypothesis, HydraOverride


def test_add_and_load_claim(tmp_path):
    path = tmp_path / "claims.json"
    save_claims([], path)

    claim = Claim(
        hypothesis_id="hyp_001",
        researcher="shaan@laptop",
        branch="ar/exp-hyp001",
        target_cells=[(2, 1, 3)],
        scope="hyperparameter",
        timestamp=datetime.now(timezone.utc),
        wandb_run_id="run_abc",
        status="running",
    )
    claims = load_claims(path)
    claims = add_claim(claims, claim)
    save_claims(claims, path)

    loaded = load_claims(path)
    assert len(loaded) == 1
    assert loaded[0].hypothesis_id == "hyp_001"


def test_expire_old_claims():
    old = Claim(
        hypothesis_id="old",
        researcher="test",
        branch="ar/old",
        target_cells=[],
        scope="hyperparameter",
        timestamp=datetime.now(timezone.utc) - timedelta(hours=5),
        wandb_run_id=None,
        status="running",
    )
    fresh = Claim(
        hypothesis_id="fresh",
        researcher="test",
        branch="ar/fresh",
        target_cells=[],
        scope="hyperparameter",
        timestamp=datetime.now(timezone.utc),
        wandb_run_id=None,
        status="running",
    )
    result = expire_claims([old, fresh], timeout_hours=4)
    assert len(result) == 1
    assert result[0].hypothesis_id == "fresh"


def test_release_claim():
    claim = Claim(
        hypothesis_id="hyp_001",
        researcher="test",
        branch="ar/test",
        target_cells=[],
        scope="hyperparameter",
        timestamp=datetime.now(timezone.utc),
        wandb_run_id=None,
        status="running",
    )
    claims = release_claim([claim], "hyp_001", new_status="completed")
    assert claims[0].status == "completed"


def test_is_duplicate_exact_match():
    h = Hypothesis.create(
        scope="hyperparameter",
        description="LR 3e-4",
        changes=[HydraOverride("control.learning_rate", "3e-4")],
        rationale="test",
    )
    completed_ids = {h.id}
    active_ids: set[str] = set()
    assert is_duplicate(h, completed_ids, active_ids) is True


def test_is_not_duplicate():
    h = Hypothesis.create(
        scope="hyperparameter",
        description="LR 3e-4",
        changes=[HydraOverride("control.learning_rate", "3e-4")],
        rationale="test",
    )
    assert is_duplicate(h, set(), set()) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_autoresearch/test_coordination.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `autoresearch/coordination/claims.py`**

```python
"""Experiment claiming and coordination via git-synced claims.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path


@dataclass
class Claim:
    """An active experiment claim."""

    hypothesis_id: str
    researcher: str
    branch: str
    target_cells: list[tuple[int, ...]]
    scope: str
    timestamp: datetime
    wandb_run_id: str | None
    status: str  # running | completed | failed


def save_claims(claims: list[Claim], path: Path | str) -> None:
    path = Path(path)
    data = {"claims": [_claim_to_dict(c) for c in claims]}
    path.write_text(json.dumps(data, indent=2) + "\n")


def load_claims(path: Path | str) -> list[Claim]:
    path = Path(path)
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [_dict_to_claim(d) for d in data.get("claims", [])]


def add_claim(claims: list[Claim], claim: Claim) -> list[Claim]:
    return claims + [claim]


def expire_claims(claims: list[Claim], timeout_hours: int = 4) -> list[Claim]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=timeout_hours)
    return [c for c in claims if c.status != "running" or c.timestamp > cutoff]


def release_claim(
    claims: list[Claim], hypothesis_id: str, new_status: str
) -> list[Claim]:
    for c in claims:
        if c.hypothesis_id == hypothesis_id:
            c.status = new_status
    return claims


def _claim_to_dict(c: Claim) -> dict:
    return {
        "hypothesis_id": c.hypothesis_id,
        "researcher": c.researcher,
        "branch": c.branch,
        "target_cells": [list(t) for t in c.target_cells],
        "scope": c.scope,
        "timestamp": c.timestamp.isoformat(),
        "wandb_run_id": c.wandb_run_id,
        "status": c.status,
    }


def _dict_to_claim(d: dict) -> Claim:
    return Claim(
        hypothesis_id=d["hypothesis_id"],
        researcher=d["researcher"],
        branch=d["branch"],
        target_cells=[tuple(t) for t in d["target_cells"]],
        scope=d["scope"],
        timestamp=datetime.fromisoformat(d["timestamp"]),
        wandb_run_id=d.get("wandb_run_id"),
        status=d["status"],
    )
```

- [ ] **Step 4: Implement `autoresearch/coordination/dedup.py`**

```python
"""Hypothesis deduplication against claims and completed experiments."""

from __future__ import annotations

from autoresearch.hypothesis.schema import Hypothesis


def is_duplicate(
    hypothesis: Hypothesis,
    completed_ids: set[str],
    active_claim_ids: set[str],
) -> bool:
    """Check if a hypothesis is a duplicate of completed or active work."""
    return hypothesis.id in completed_ids or hypothesis.id in active_claim_ids
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_autoresearch/test_coordination.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add autoresearch/coordination/ tests/test_autoresearch/test_coordination.py
git commit -m "feat(autoresearch): coordination claims and deduplication"
```

---

## Task 9: State File Initialization

**Files:**
- Create: `autoresearch/state/archive.json`
- Create: `autoresearch/state/tree.json`
- Create: `autoresearch/state/claims.json`

- [ ] **Step 1: Initialize empty state files**

`autoresearch/state/archive.json`:
```json
{
  "bins_per_axis": 5,
  "axis_ranges": [[0.25, 1.0], [0.0, 1.0], [0.0, 24.0]],
  "cells": {}
}
```

`autoresearch/state/tree.json`:
```json
{
  "root_id": "baseline_v1",
  "nodes": {
    "baseline_v1": {
      "parent_id": null,
      "inspired_by": null,
      "scope": "baseline",
      "description": "monorace_baseline",
      "status": "baseline",
      "fitness": null,
      "wandb_run_id": null,
      "git_commit": null,
      "failure_reason": null,
      "archive_cell": null
    }
  }
}
```

`autoresearch/state/claims.json`:
```json
{
  "claims": []
}
```

- [ ] **Step 2: Note on committed state files**

These are the git-synced state files that the team shares. They start empty and accumulate data as experiments run.

Note: the spec also mentions `state/config.json` for shared config overrides — this is deferred to Phase 2 when per-session config overrides become necessary. MVP uses the plugin `settings.yaml` defaults.

- [ ] **Step 3: Commit**

```bash
git add autoresearch/state/
git commit -m "feat(autoresearch): initialize empty state files"
```

---

## Task 10: Plugin Skeleton

**Files:**
- Create: `.claude/plugins/autoresearch/plugin.json`
- Create: `.claude/plugins/autoresearch/settings.yaml`
- Create: `.claude/plugins/autoresearch/skills/auto-research.md`
- Create: `.claude/plugins/autoresearch/skills/ar-status.md`
- Create: `.claude/plugins/autoresearch/skills/ar-review.md`

- [ ] **Step 1: Create plugin directory structure**

```bash
mkdir -p .claude/plugins/autoresearch/skills
```

- [ ] **Step 2: Create `plugin.json`**

```json
{
  "name": "autoresearch",
  "version": "0.1.0",
  "description": "Automated research exploration for drone racing algorithms using MAP-Elites"
}
```

- [ ] **Step 3: Create `settings.yaml`**

This is the default config that maps to `AutoResearchConfig`:

```yaml
autoresearch:
  wandb_project: corvidx-drone-racing
  mode: interactive
  budgets:
    hyperparameter: 5_000_000
    algorithm: 15_000_000
    architecture: 30_000_000
    system: 50_000_000
  early_stop:
    baseline_threshold: 0.7
    archive_redundancy: true
    poll_interval_seconds: 60
  constraints:
    min_success_rate: 0.8
    min_avg_speed: 2.0
    max_gate_offset_ratio: 0.8
    max_acceleration_g: 4.0
  branch_selection:
    exploit_weight: 1.0
    explore_weight: 1.0
    gap_weight: 0.5
    diversity_weight: 0.3
    random_restart_pct: 0.2
    min_branch_experiments: 3
    tie_threshold: 0.05
  coordination:
    claim_timeout_hours: 4
    claim_refresh_minutes: 15
```

- [ ] **Step 4: Create `/auto-research` skill**

File: `.claude/plugins/autoresearch/skills/auto-research.md`

Write the main research loop skill that instructs Claude Code how to:
1. Pull latest state (`git pull`, load archive/tree/claims)
2. Check W&B for completed/orphaned runs
3. Analyze archive gaps and trajectory data
4. Generate 3-5 hypotheses (hyperparameter scope only for MVP)
5. Present top candidates to human for approval (interactive mode)
6. Create git branch, build Hydra override command
7. Write claim, launch training
8. Monitor via polling loop
9. On completion: compute descriptors, validate constraints, insert as candidate
10. Present results for human review/approval

- [ ] **Step 5: Create `/ar-status` skill**

File: `.claude/plugins/autoresearch/skills/ar-status.md`

Skill that displays:
- Archive grid summary (occupied/empty cells, best fitness)
- Active claims (who is running what)
- Research tree summary (branches, depth, recent experiments)
- Pending candidates awaiting review

- [ ] **Step 6: Create `/ar-review` skill**

File: `.claude/plugins/autoresearch/skills/ar-review.md`

Skill that:
- Lists candidate entries pending approval
- Shows metrics comparison, descriptor values, constraint results
- Provides W&B and Rerun links
- Allows human to approve/reject candidates

- [ ] **Step 7: Commit**

```bash
git add .claude/plugins/autoresearch/
git commit -m "feat(autoresearch): Claude Code plugin skeleton with skills"
```

---

## Task 11: Integration Test

**Files:**
- Test: `tests/test_autoresearch/test_integration.py`

- [ ] **Step 1: Write end-to-end integration test**

File: `tests/test_autoresearch/test_integration.py`

```python
"""Integration test: full pipeline from trajectory to archive insertion."""

import numpy as np
from pathlib import Path

from autoresearch.analysis.trajectory import load_trajectory
from autoresearch.descriptors.compute import compute_descriptors
from autoresearch.constraints.validator import ConstraintValidator
from autoresearch.archive.map_elites import MapElitesArchive, CellEntry
from autoresearch.archive.serialization import save_archive, load_archive
from autoresearch.hypothesis.schema import Hypothesis, HydraOverride
from autoresearch.coordination.claims import Claim, save_claims, load_claims, add_claim
from autoresearch.coordination.dedup import is_duplicate
from autoresearch.tree.research_tree import ResearchTree
from autoresearch.tree.serialization import save_tree, load_tree

from datetime import datetime, timezone


def test_full_pipeline(sample_npz, tmp_path):
    """Simulate the complete MVP research loop (minus actual training)."""
    max_rpm = 31470.0
    control_freq = 100.0

    # 1. Load trajectory
    traj = load_trajectory(sample_npz)
    assert traj.n_timesteps == 100

    # 2. Compute descriptors
    desc = compute_descriptors(traj, max_rpm, control_freq)
    assert 0 <= desc.actuator_utilization <= 1.0

    # 3. Validate constraints
    validator = ConstraintValidator(max_rpm=max_rpm)
    metrics = {"success_rate": 0.9, "lap_times": [5.0] * 20}
    result = validator.validate(traj, metrics, dr_active=True)
    # Result may or may not pass depending on fixture data — that's fine
    # The point is it runs without error

    # 4. Create hypothesis
    hyp = Hypothesis.create(
        scope="hyperparameter",
        description="Increase learning rate to 3e-4",
        changes=[HydraOverride("control.learning_rate", "3e-4")],
        rationale="Baseline LR may be too conservative",
    )
    assert hyp.id is not None
    assert hyp.to_cli_overrides() == ["control.learning_rate=3e-4"]

    # 5. Check deduplication
    assert is_duplicate(hyp, set(), set()) is False

    # 6. Create claim
    claims_path = tmp_path / "claims.json"
    save_claims([], claims_path)
    claim = Claim(
        hypothesis_id=hyp.id,
        researcher="test@laptop",
        branch=f"ar/exp-{hyp.id}",
        target_cells=[],
        scope="hyperparameter",
        timestamp=datetime.now(timezone.utc),
        wandb_run_id="run_test",
        status="running",
    )
    claims = add_claim(load_claims(claims_path), claim)
    save_claims(claims, claims_path)
    assert len(load_claims(claims_path)) == 1

    # 7. Archive insertion (if constraints passed)
    archive_path = tmp_path / "archive.json"
    archive = MapElitesArchive()
    entry = CellEntry(
        fitness=5.2,
        status="candidate",
        wandb_run_id="run_test",
        git_commit="test123",
        descriptors=desc,
        constraint_results={"passed": result.passed},
        budget_spent=5_000_000,
        hypothesis_id=hyp.id,
    )
    archive.try_insert(entry)
    save_archive(archive, archive_path)
    loaded_archive = load_archive(archive_path)
    assert loaded_archive.n_occupied == 1

    # 8. Research tree update
    tree_path = tmp_path / "tree.json"
    tree = ResearchTree.create_with_baseline("monorace_baseline", "abc123")
    tree.add_experiment(
        hypothesis_id=hyp.id,
        parent_id="baseline_v1",
        scope="hyperparameter",
        description=hyp.description,
        status="completed",
    )
    tree.complete_experiment(hyp.id, fitness=5.2, wandb_run_id="run_test")
    save_tree(tree, tree_path)
    loaded_tree = load_tree(tree_path)
    assert loaded_tree.total_experiments == 1
```

- [ ] **Step 2: Run integration test**

Run: `python -m pytest tests/test_autoresearch/test_integration.py -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/test_autoresearch/ -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_autoresearch/test_integration.py
git commit -m "test(autoresearch): end-to-end integration test for MVP pipeline"
```

---

## Summary

| Task | Component | Estimated Steps |
|------|-----------|----------------|
| 1 | Package skeleton + config | 9 |
| 2 | Trajectory loading | 6 |
| 3 | Behavioral descriptors (3 axes) | 8 |
| 4 | MAP-Elites archive | 6 |
| 5 | Constraint validation | 7 |
| 6 | Hypothesis schema | 5 |
| 7 | Research tree | 6 |
| 8 | Coordination (claims + dedup) | 6 |
| 9 | State file initialization | 3 |
| 10 | Plugin skeleton | 7 |
| 11 | Integration test | 4 |
| **Total** | | **67 steps** |

Tasks 1-9 are the Python package (`autoresearch/`). Task 10 is the Claude Code plugin. Task 11 validates the full pipeline end-to-end. All tasks follow TDD: write failing test → implement → verify → commit.
