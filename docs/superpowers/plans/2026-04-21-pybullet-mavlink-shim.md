# PyBullet MAVLink Shim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable UDP MAVLink server that wraps a drone dynamics backend, exposing SET_ATTITUDE_TARGET in / ATTITUDE+ODOMETRY+HIGHRES_IMU+HEARTBEAT out. Phase 1 ships the core shim plus a NumpyQuadBackend over the existing `sim/dynamics/numpy_quad.py`.

**Architecture:** Single Python package (`sim/pybullet/mavlink_shim/`) with a `DroneBackend` Protocol, a single-loop attitude PD controller that feeds the existing TRPY mixer, a per-message rate scheduler, and a UDP server using pymavlink. Two run modes: free-running (worker thread on wall-clock timer) and lockstep (caller drives `shim.step()`). Clients (pymavlink, QGroundControl, our policy code) connect via UDP — same code path local or remote.

**Tech Stack:** Python 3.11 (monorace conda env), pymavlink, numpy, pytest. Reuses existing `sim/dynamics/numpy_quad.py`, `sim/dynamics/trpy_mixer.py`, `sim/dynamics/params.py`, `sim/sensors/imu_model.py` (if present), and `sim/pybullet/coords.py`.

**Spec:** `docs/superpowers/specs/2026-04-21-pybullet-mavlink-shim-design.md`

**Working directory:** `C:\Users\alexj\Documents\algo_src\.claude\worktrees\warehouse-tsdf-pybullet-mvp`

**Python environment:** Always use `conda run -n monorace python ...` and `conda run -n monorace pytest ...`. The monorace env has Python 3.11 with all required deps including pymavlink.

---

## File Structure

**New files:**

| Path | Responsibility |
|------|----------------|
| `sim/pybullet/mavlink_shim/__init__.py` | Public API re-exports |
| `sim/pybullet/mavlink_shim/backend.py` | `DroneState`, `ImuSample` dataclasses; `DroneBackend` Protocol |
| `sim/pybullet/mavlink_shim/numpy_quad_backend.py` | `NumpyQuadBackend` wrapping `sim/dynamics/numpy_quad.py` |
| `sim/pybullet/mavlink_shim/coords_mavlink.py` | NED↔ENU adapters for MAVLink message conventions (positions, velocities, quaternions, Euler angles) |
| `sim/pybullet/mavlink_shim/attitude_controller.py` | Single-loop attitude P controller + TRPY mixer call |
| `sim/pybullet/mavlink_shim/rate_scheduler.py` | Per-message tick due-time accumulator |
| `sim/pybullet/mavlink_shim/server.py` | `MavlinkServer` — pymavlink UDP I/O wrapper (recv/send only, no logic) |
| `sim/pybullet/mavlink_shim/shim.py` | `MavlinkShim` top-level orchestrator + lifecycle |
| `scripts/mavlink/__init__.py` | Empty package marker |
| `scripts/mavlink/smoke_test.py` | Manual demo: free-running shim + pymavlink client + attitude tracking PNG |
| `tests/test_mavlink_shim/__init__.py` | Test package marker |
| `tests/test_mavlink_shim/test_coords_mavlink.py` | Coord adapter unit tests |
| `tests/test_mavlink_shim/test_attitude_controller.py` | Controller PD unit tests |
| `tests/test_mavlink_shim/test_rate_scheduler.py` | Scheduler timing unit tests |
| `tests/test_mavlink_shim/test_numpy_quad_backend.py` | Backend wrapper unit tests |
| `tests/test_mavlink_shim/test_server.py` | UDP loopback server unit test |
| `tests/test_mavlink_shim/test_backend_protocol.py` | Shim ↔ backend interface verification with MockBackend |
| `tests/test_mavlink_shim/test_hover_hold.py` | Integration: identity attitude + hover thrust → drone stays put |
| `tests/test_mavlink_shim/test_attitude_step_response.py` | Integration: 30° pitch step → drone reaches target |
| `tests/test_mavlink_shim/test_reset_returns_to_initial.py` | Integration: reset semantics |
| `tests/test_mavlink_shim/test_message_rates_match_config.py` | Integration: free-running rate verification |

**Modified files:** none. The shim is purely additive.

---

## Task 1: Package skeleton + dataclasses + Protocol

**Files:**
- Create: `sim/pybullet/mavlink_shim/__init__.py`
- Create: `sim/pybullet/mavlink_shim/backend.py`
- Create: `tests/test_mavlink_shim/__init__.py`

This task establishes the dataclass types and Protocol interface — no behavior, just types. No tests in this task; later tasks test the Protocol via concrete impls.

- [ ] **Step 1: Create the package directories and empty markers**

```bash
mkdir -p sim/pybullet/mavlink_shim
mkdir -p tests/test_mavlink_shim
touch sim/pybullet/mavlink_shim/__init__.py
touch tests/test_mavlink_shim/__init__.py
```

- [ ] **Step 2: Create `sim/pybullet/mavlink_shim/backend.py`**

```python
"""DroneBackend Protocol and shared state dataclasses.

The Protocol isolates the MAVLink shim from any specific physics engine.
Phase 1 ships NumpyQuadBackend; future backends (PyBullet warehouse
collision, real drone passthrough) plug in by implementing this Protocol.

All state is in ENU world frame; the MAVLink server handles ENU↔NED
conversion at the message boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass
class DroneState:
    """Full kinematic state of a single drone in ENU world frame."""
    pos_enu: NDArray[np.float64]            # (3,) world position, ENU
    vel_enu: NDArray[np.float64]            # (3,) world velocity, ENU
    quat_wxyz: NDArray[np.float64]          # (4,) attitude in ENU world frame
    angular_vel_body: NDArray[np.float64]   # (3,) body-frame rotation rates rad/s
    motor_speed: NDArray[np.float64]        # (4,) per-motor angular velocity rad/s
    timestamp_us: int                       # monotonic microseconds since shim start


@dataclass
class ImuSample:
    """Single IMU reading in body frame."""
    accel_body: NDArray[np.float64]         # (3,) body-frame accel including gravity, m/s²
    gyro_body: NDArray[np.float64]          # (3,) body-frame angular rate, rad/s
    timestamp_us: int


class DroneBackend(Protocol):
    """Interface for any physics engine driving a single quadrotor.

    motor_commands convention: per-motor angular velocity in rad/s
    (matches the output of sim/dynamics/trpy_mixer.py).
    """

    def reset(self, initial_state: DroneState) -> None:
        """Reset internal state to the given snapshot."""
        ...

    def step(self, motor_commands: NDArray[np.float64], dt: float) -> DroneState:
        """Advance physics by dt seconds and return the new state."""
        ...

    def get_imu(self) -> ImuSample:
        """Return the latest IMU sample derived from current state."""
        ...
```

- [ ] **Step 3: Populate `sim/pybullet/mavlink_shim/__init__.py`**

```python
"""Reusable MAVLink server for drone dynamics backends."""
from sim.pybullet.mavlink_shim.backend import (
    DroneBackend,
    DroneState,
    ImuSample,
)

__all__ = ["DroneBackend", "DroneState", "ImuSample"]
```

- [ ] **Step 4: Verify the package imports cleanly**

```bash
conda run -n monorace python -c "from sim.pybullet.mavlink_shim import DroneBackend, DroneState, ImuSample; print('OK')"
```

Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add sim/pybullet/mavlink_shim/__init__.py \
        sim/pybullet/mavlink_shim/backend.py \
        tests/test_mavlink_shim/__init__.py
git commit -m "feat(mavlink-shim): package skeleton + DroneBackend Protocol"
```

---

## Task 2: NED↔ENU coord adapters for MAVLink messages

**Files:**
- Create: `sim/pybullet/mavlink_shim/coords_mavlink.py`
- Create: `tests/test_mavlink_shim/test_coords_mavlink.py`

The existing `sim/pybullet/coords.py` has position and quaternion conversions. This task adds MAVLink-specific helpers: ATTITUDE message Euler angles (NED roll/pitch/yaw from ENU quaternion), and ODOMETRY-friendly NED quaternion order.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mavlink_shim/test_coords_mavlink.py`:

```python
"""Tests for MAVLink-specific coordinate adapters."""
import numpy as np
import pytest

from sim.pybullet.mavlink_shim.coords_mavlink import (
    enu_quat_to_ned_euler,
    enu_quat_to_ned_quat_xyzw,
    ned_quat_wxyz_to_enu_quat,
)


def test_enu_quat_identity_to_ned_euler_is_zero():
    q_enu_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
    roll, pitch, yaw = enu_quat_to_ned_euler(q_enu_wxyz)
    assert abs(roll) < 1e-9
    assert abs(pitch) < 1e-9
    assert abs(yaw) < 1e-9


def test_enu_quat_yaw_90deg_to_ned_euler_yaw_negative_90():
    # 90° yaw around ENU +Z axis (East→North) corresponds to -90° yaw in NED
    # (because NED yaw is around +Z_NED = -Z_ENU, so signs flip).
    half = np.sqrt(0.5)
    q_enu_wxyz = np.array([half, 0.0, 0.0, half])  # 90° rotation about +Z_ENU
    roll, pitch, yaw = enu_quat_to_ned_euler(q_enu_wxyz)
    assert abs(roll) < 1e-9
    assert abs(pitch) < 1e-9
    assert abs(yaw - (-np.pi / 2)) < 1e-6


def test_enu_quat_to_ned_quat_xyzw_round_trip():
    # Take a non-trivial ENU quaternion, convert to NED xyzw, convert back; same.
    q_enu_wxyz = np.array([0.5, 0.5, 0.5, 0.5])  # arbitrary
    q_enu_wxyz /= np.linalg.norm(q_enu_wxyz)

    q_ned_xyzw = enu_quat_to_ned_quat_xyzw(q_enu_wxyz)

    # Reorder NED xyzw → wxyz, then NED→ENU
    q_ned_wxyz = np.array([q_ned_xyzw[3], q_ned_xyzw[0], q_ned_xyzw[1], q_ned_xyzw[2]])
    q_back_enu_wxyz = ned_quat_wxyz_to_enu_quat(q_ned_wxyz)

    np.testing.assert_array_almost_equal(q_back_enu_wxyz, q_enu_wxyz)


def test_enu_to_ned_quat_xyzw_returns_unit_quaternion():
    q_enu_wxyz = np.array([0.7071, 0.0, 0.7071, 0.0])
    q_ned_xyzw = enu_quat_to_ned_quat_xyzw(q_enu_wxyz)
    assert abs(np.linalg.norm(q_ned_xyzw) - 1.0) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n monorace pytest tests/test_mavlink_shim/test_coords_mavlink.py -v
```

Expected: ImportError on `sim.pybullet.mavlink_shim.coords_mavlink`.

- [ ] **Step 3: Implement the adapters**

Create `sim/pybullet/mavlink_shim/coords_mavlink.py`:

```python
"""MAVLink-specific coordinate adapters (ENU↔NED for MAVLink message fields).

Builds on sim/pybullet/coords.py for the basic flips. Adds:
  - enu_quat_to_ned_euler: ATTITUDE message uses (roll, pitch, yaw)
    Euler angles in NED frame.
  - enu_quat_to_ned_quat_xyzw: ODOMETRY uses NED quaternion in xyzw
    order (MAVLink convention).
  - ned_quat_wxyz_to_enu_quat: inverse, used for round-trip tests.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sim.pybullet.coords import ned_to_enu_quaternion


def _quat_to_euler_zyx(q_wxyz: NDArray[np.float64]) -> tuple[float, float, float]:
    """Convert quaternion (wxyz) to (roll, pitch, yaw) using Z-Y-X intrinsic."""
    w, x, y, z = q_wxyz
    # roll (X axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = float(np.arctan2(sinr_cosp, cosr_cosp))
    # pitch (Y axis)
    sinp = 2.0 * (w * y - z * x)
    sinp = float(np.clip(sinp, -1.0, 1.0))
    pitch = float(np.arcsin(sinp))
    # yaw (Z axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = float(np.arctan2(siny_cosp, cosy_cosp))
    return roll, pitch, yaw


def enu_quat_to_ned_euler(q_enu_wxyz: NDArray[np.float64]) -> tuple[float, float, float]:
    """ENU quaternion → NED Euler (roll, pitch, yaw) in radians."""
    # Step 1: re-express the same physical rotation in NED basis.
    # Inverse of ned_to_enu_quaternion: (w,x,y,z)_ned = (w,y,x,-z)_enu
    w, x, y, z = q_enu_wxyz
    q_ned_wxyz = np.array([w, y, x, -z])
    return _quat_to_euler_zyx(q_ned_wxyz)


def enu_quat_to_ned_quat_xyzw(q_enu_wxyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """ENU quaternion (wxyz) → NED quaternion (xyzw) for MAVLink ODOMETRY."""
    w, x, y, z = q_enu_wxyz
    # NED wxyz, then reorder to xyzw
    return np.array([y, x, -z, w])


def ned_quat_wxyz_to_enu_quat(q_ned_wxyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """NED quaternion (wxyz) → ENU quaternion (wxyz). Same math as ned_to_enu_quaternion."""
    return ned_to_enu_quaternion(q_ned_wxyz)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n monorace pytest tests/test_mavlink_shim/test_coords_mavlink.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add sim/pybullet/mavlink_shim/coords_mavlink.py \
        tests/test_mavlink_shim/test_coords_mavlink.py
git commit -m "feat(mavlink-shim): NED↔ENU adapters for MAVLink message conventions"
```

---

## Task 3: RateScheduler

**Files:**
- Create: `sim/pybullet/mavlink_shim/rate_scheduler.py`
- Create: `tests/test_mavlink_shim/test_rate_scheduler.py`

A small per-message tick scheduler. Given a configured Hz for each message name, `due()` returns the list of names that should fire on this tick. Uses accumulator math (no drift).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mavlink_shim/test_rate_scheduler.py`:

```python
"""Tests for RateScheduler — per-message tick due-time accumulator."""
import pytest

from sim.pybullet.mavlink_shim.rate_scheduler import RateScheduler


def test_messages_due_at_their_configured_period():
    sched = RateScheduler(rates_hz={"a": 100.0, "b": 50.0}, tick_hz=200.0)
    # Tick 0: nothing due (initial state).
    assert set(sched.due(t_us=0)) == {"a", "b"}  # both due at t=0 (initial)
    # Next tick at 5 ms (one period of "a") — "a" due, "b" not due yet.
    assert set(sched.due(t_us=5_000)) == {"a"}
    # At 10 ms — "a" due (second period), "b" due (first period).
    assert set(sched.due(t_us=10_000)) == {"a", "b"}


def test_message_count_matches_rate_over_n_seconds():
    sched = RateScheduler(rates_hz={"x": 100.0, "y": 1.0}, tick_hz=200.0)
    counts = {"x": 0, "y": 0}
    # Simulate 5 seconds at 200 Hz tick rate.
    n_ticks = 5 * 200
    for tick in range(n_ticks):
        t_us = int(tick * 1_000_000 / 200)
        for name in sched.due(t_us):
            counts[name] += 1
    # Expected: ~500 "x" (100 Hz × 5 s), ~5 "y" (1 Hz × 5 s).
    assert 495 <= counts["x"] <= 505
    assert 4 <= counts["y"] <= 6


def test_rate_zero_never_fires():
    sched = RateScheduler(rates_hz={"silent": 0.0, "loud": 1000.0}, tick_hz=1000.0)
    counts = {"silent": 0, "loud": 0}
    for tick in range(1000):
        t_us = int(tick * 1_000)
        for name in sched.due(t_us):
            counts[name] += 1
    assert counts["silent"] == 0
    assert counts["loud"] >= 990
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n monorace pytest tests/test_mavlink_shim/test_rate_scheduler.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement the scheduler**

Create `sim/pybullet/mavlink_shim/rate_scheduler.py`:

```python
"""Per-message rate scheduler for MAVLink outbound messages."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RateScheduler:
    """Returns which messages should fire on each tick.

    rates_hz: {message_name: target_rate_hz}.  Rate of 0 = never fires.
    tick_hz: the highest-frequency tick (sets the scheduling resolution).

    Each message has a "next_due_us" timestamp. On each tick, messages
    whose next_due_us is <= the current tick time fire and have their
    next_due_us advanced by their period. Drift-free.
    """
    rates_hz: dict[str, float]
    tick_hz: float
    _next_due_us: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        # Initialize all enabled messages to fire immediately at t=0.
        self._next_due_us = {
            name: 0 for name, hz in self.rates_hz.items() if hz > 0
        }

    def due(self, t_us: int) -> list[str]:
        """Return names of messages due to fire at or before t_us."""
        due_now = []
        for name, next_due in list(self._next_due_us.items()):
            if next_due <= t_us:
                due_now.append(name)
                period_us = int(1_000_000 / self.rates_hz[name])
                self._next_due_us[name] = next_due + period_us
        return due_now
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n monorace pytest tests/test_mavlink_shim/test_rate_scheduler.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add sim/pybullet/mavlink_shim/rate_scheduler.py \
        tests/test_mavlink_shim/test_rate_scheduler.py
git commit -m "feat(mavlink-shim): RateScheduler with drift-free per-message tick"
```

---

## Task 4: AttitudeController

**Files:**
- Create: `sim/pybullet/mavlink_shim/attitude_controller.py`
- Create: `tests/test_mavlink_shim/test_attitude_controller.py`

Single-loop P controller on attitude error → desired body rates → TRPY mixer → motor speeds.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mavlink_shim/test_attitude_controller.py`:

```python
"""Tests for AttitudeController — quat error → motor speeds via TRPY mixer."""
import numpy as np
import pytest

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim.attitude_controller import AttitudeController


def _hover_thrust_normalized(params: VehicleParams) -> float:
    """Compute the normalized [0,1] thrust that gives hover with these params."""
    weight_n = params.mass_kg * 9.81
    return weight_n / (4.0 * params.max_thrust_per_motor_n)


def test_identity_attitude_error_produces_equal_motor_speeds_at_hover_thrust():
    params = VehicleParams.default()
    ctrl = AttitudeController(params=params)

    q_target = np.array([1.0, 0.0, 0.0, 0.0])     # identity ENU
    q_current = np.array([1.0, 0.0, 0.0, 0.0])    # identity ENU
    omega_current = np.zeros(3)
    thrust_norm = _hover_thrust_normalized(params)

    motor_speeds = ctrl.compute(
        q_target_enu_wxyz=q_target,
        thrust_normalized=thrust_norm,
        q_current_enu_wxyz=q_current,
        omega_current_body=omega_current,
    )
    # All four motor speeds should be (approximately) equal at hover.
    assert motor_speeds.shape == (4,)
    spread = motor_speeds.max() - motor_speeds.min()
    assert spread < 1e-6, f"motors not equal at hover: {motor_speeds}"


def test_pitch_attitude_error_produces_pitch_torque_signed_correctly():
    """Target pitch +30° while current is identity → desired pitch rate > 0."""
    params = VehicleParams.default()
    ctrl = AttitudeController(params=params, k_att=4.0)

    angle = np.deg2rad(30.0)
    q_target = np.array([np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0])  # +30° about Y
    q_current = np.array([1.0, 0.0, 0.0, 0.0])
    omega_current = np.zeros(3)
    thrust_norm = _hover_thrust_normalized(params)

    motor_speeds = ctrl.compute(
        q_target_enu_wxyz=q_target,
        thrust_normalized=thrust_norm,
        q_current_enu_wxyz=q_current,
        omega_current_body=omega_current,
    )
    # In X-config (FR-CW, RR-CCW, RL-CW, FL-CCW), positive pitch torque means
    # rear motors spin faster than front. Index map per sim/dynamics/numpy_quad.py:
    # 0=FR, 1=RR, 2=RL, 3=FL → rear pair = {1, 2}, front pair = {0, 3}.
    rear_avg = (motor_speeds[1] + motor_speeds[2]) / 2.0
    front_avg = (motor_speeds[0] + motor_speeds[3]) / 2.0
    assert rear_avg > front_avg, f"rear should be faster for +pitch: {motor_speeds}"


def test_quaternion_sign_handling_takes_shorter_path():
    """When q_error.w < 0, the controller should flip sign so we go the short way."""
    params = VehicleParams.default()
    ctrl = AttitudeController(params=params, k_att=4.0)

    # Target = identity, but represented with negated quaternion (same rotation).
    q_target = np.array([-1.0, 0.0, 0.0, 0.0])
    q_current = np.array([1.0, 0.0, 0.0, 0.0])
    motor_speeds = ctrl.compute(
        q_target_enu_wxyz=q_target,
        thrust_normalized=_hover_thrust_normalized(params),
        q_current_enu_wxyz=q_current,
        omega_current_body=np.zeros(3),
    )
    # No actual rotation needed → motors equal (within hover tolerance).
    spread = motor_speeds.max() - motor_speeds.min()
    assert spread < 1e-6, f"sign-flipped identity should yield zero rates: {motor_speeds}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n monorace pytest tests/test_mavlink_shim/test_attitude_controller.py -v
```

Expected: ImportError on `attitude_controller` AND likely on `VehicleParams.default()` if that classmethod doesn't exist. If `VehicleParams.default()` is missing, peek at `sim/dynamics/params.py` and adjust the test to construct `VehicleParams(...)` directly with explicit fields. The intent is "use the default project drone parameters for tests."

- [ ] **Step 3: Verify what `VehicleParams` exposes (one-time exploration)**

Read `sim/dynamics/params.py` to confirm field names and construction. If a default-factory classmethod doesn't exist, add a small helper at the top of the test file:

```python
def _default_params() -> "VehicleParams":
    return VehicleParams(
        mass_kg=0.5,
        # ...other fields with project default values from sim/dynamics/params.py
    )
```

Replace `VehicleParams.default()` with `_default_params()` in the three test cases.

- [ ] **Step 4: Implement the controller**

Inspect `sim/dynamics/trpy_mixer.py` to confirm the `TRPYMixer.compute()` (or similar) method signature. The plan assumes it exposes a method that takes `(thrust_n, omega_body)` and returns motor speeds.

Create `sim/pybullet/mavlink_shim/attitude_controller.py`:

```python
"""Single-loop attitude P controller feeding the TRPY mixer.

For SET_ATTITUDE_TARGET (q_target, thrust_normalized):
  1. Compute quaternion error in body frame.
  2. P controller on error vector → desired body rates ω_des.
  3. Hand (thrust_n, ω_des) to the existing TRPY mixer for motor speeds.

No inner rate PD: the TRPY mixer's inverted allocation matrix maps desired
rates directly to motor speeds. For a perfect-model sim this is sufficient.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sim.dynamics.params import VehicleParams
from sim.dynamics.trpy_mixer import TRPYMixer


def _quat_multiply(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Hamilton product of two wxyz quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _quat_inverse_unit(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse of a unit quaternion (= conjugate)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


@dataclass
class AttitudeController:
    """Quat-error P controller. Outputs motor speeds via TRPYMixer."""
    params: VehicleParams
    k_att: float = 6.0       # attitude P gain (rad/s per rad of error)

    def __post_init__(self) -> None:
        self._mixer = TRPYMixer(self.params)
        # Total max collective thrust (all 4 motors at max).
        self._max_thrust_n = 4.0 * self.params.max_thrust_per_motor_n

    def compute(
        self,
        *,
        q_target_enu_wxyz: NDArray[np.float64],
        thrust_normalized: float,
        q_current_enu_wxyz: NDArray[np.float64],
        omega_current_body: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return 4-vector of motor speeds (rad/s)."""
        # 1. Quaternion error in body frame: q_err = q_target ⊗ q_current⁻¹
        q_err = _quat_multiply(q_target_enu_wxyz, _quat_inverse_unit(q_current_enu_wxyz))

        # 2. Take the shorter rotation path.
        sign = 1.0 if q_err[0] >= 0.0 else -1.0

        # 3. P controller: ω_desired = k_att * 2 * sign(w) * (xyz components).
        omega_desired = self.k_att * 2.0 * sign * q_err[1:4]

        # 4. Map normalized thrust [0,1] to physical thrust force in Newtons.
        thrust_n = float(np.clip(thrust_normalized, 0.0, 1.0)) * self._max_thrust_n

        # 5. TRPY mixer takes (thrust_n, ω_desired) and returns motor speeds.
        return self._mixer.compute(thrust_n, omega_desired)
```

If the existing TRPYMixer method is named differently (e.g., `__call__`, `mix`, `solve`), match it. If TRPYMixer takes a 4-vector `[T, ωx, ωy, ωz]` instead of two args, pack accordingly:

```python
return self._mixer.compute(np.array([thrust_n, *omega_desired]))
```

Verify by reading `sim/dynamics/trpy_mixer.py` lines 30+ before writing the call.

- [ ] **Step 5: Run tests to verify they pass**

```bash
conda run -n monorace pytest tests/test_mavlink_shim/test_attitude_controller.py -v
```

Expected: 3 passed. If the rear-vs-front motor index check (`test_pitch_attitude_error_produces_pitch_torque_signed_correctly`) fails, re-check the X-config motor index map at the top of `sim/dynamics/numpy_quad.py` and update the test's `rear`/`front` slices accordingly. The implementation is correct; the test reflects the project's motor convention.

- [ ] **Step 6: Commit**

```bash
git add sim/pybullet/mavlink_shim/attitude_controller.py \
        tests/test_mavlink_shim/test_attitude_controller.py
git commit -m "feat(mavlink-shim): single-loop attitude P controller + TRPY mixer call"
```

---

## Task 5: NumpyQuadBackend

**Files:**
- Create: `sim/pybullet/mavlink_shim/numpy_quad_backend.py`
- Create: `tests/test_mavlink_shim/test_numpy_quad_backend.py`

Wraps `sim/dynamics/numpy_quad.py` (vectorized N-drone implementation) for a single-vehicle (N=1) DroneBackend.

- [ ] **Step 1: Inspect existing dynamics signatures**

Read `sim/dynamics/numpy_quad.py` lines 1-80 to confirm:
- The state vector layout (already documented in the file: pos/vel/quat/omega/motor 17 elements).
- The step function name and signature (likely `step(state, motor_commands, dt) -> new_state` or a class method).
- Whether there's a class wrapping it or it's pure functions.

Adjust the implementation in Step 4 below to match the actual API.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_mavlink_shim/test_numpy_quad_backend.py`:

```python
"""Tests for NumpyQuadBackend — single-vehicle wrapper of numpy_quad dynamics."""
import numpy as np
import pytest

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim.backend import DroneState
from sim.pybullet.mavlink_shim.numpy_quad_backend import NumpyQuadBackend


def _initial_state(timestamp_us: int = 0) -> DroneState:
    return DroneState(
        pos_enu=np.zeros(3),
        vel_enu=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3),
        motor_speed=np.zeros(4),
        timestamp_us=timestamp_us,
    )


def test_reset_then_step_zero_motor_drops_drone_due_to_gravity():
    backend = NumpyQuadBackend(params=VehicleParams.default())
    backend.reset(_initial_state())

    # Zero motor speeds, dt = 0.1s. Drone should fall ~ 0.5 * g * dt² ≈ 0.049 m.
    state = backend.step(motor_commands=np.zeros(4), dt=0.1)
    assert state.pos_enu[2] < -0.04, f"drone did not fall: z={state.pos_enu[2]}"
    assert state.vel_enu[2] < 0.0, f"drone z-velocity should be negative: {state.vel_enu}"


def test_reset_returns_state_to_initial():
    backend = NumpyQuadBackend(params=VehicleParams.default())
    initial = _initial_state()
    backend.reset(initial)

    # Step a few times to disturb state.
    for _ in range(10):
        backend.step(motor_commands=np.array([100.0, 100.0, 100.0, 100.0]), dt=0.01)

    backend.reset(initial)
    state = backend.step(motor_commands=np.zeros(4), dt=0.0)  # zero-time step to read state
    np.testing.assert_array_almost_equal(state.pos_enu, initial.pos_enu)
    np.testing.assert_array_almost_equal(state.vel_enu, initial.vel_enu)
    np.testing.assert_array_almost_equal(state.quat_wxyz, initial.quat_wxyz)


def test_get_imu_returns_gravity_when_at_rest():
    """Stationary drone should sense ~9.81 m/s² downward in body frame.
    
    Body frame matches world ENU at identity attitude, so accel_body[2] ≈ +9.81
    (the proper acceleration sensed by an IMU at rest = -gravity vector = up).
    """
    backend = NumpyQuadBackend(params=VehicleParams.default())
    backend.reset(_initial_state())

    imu = backend.get_imu()
    assert imu.accel_body.shape == (3,)
    assert imu.gyro_body.shape == (3,)
    # IMU at rest measures acceleration opposing gravity → ~+9.81 in body Z.
    assert abs(imu.accel_body[2] - 9.81) < 0.5
    np.testing.assert_array_almost_equal(imu.gyro_body, np.zeros(3))
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
conda run -n monorace pytest tests/test_mavlink_shim/test_numpy_quad_backend.py -v
```

Expected: ImportError on `NumpyQuadBackend`.

- [ ] **Step 4: Implement the backend**

Create `sim/pybullet/mavlink_shim/numpy_quad_backend.py`. Use the existing numpy_quad API — adapt this template to the actual function names you find in `sim/dynamics/numpy_quad.py`:

```python
"""Single-vehicle DroneBackend wrapping sim/dynamics/numpy_quad.py."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sim.dynamics.params import VehicleParams
# Note: import the actual step function or class from numpy_quad. Adjust names below
# to match what the file exposes (look at the bottom of numpy_quad.py for a
# `step()` function or a `NumpyQuadDynamics` class).
from sim.dynamics import numpy_quad as nq

from sim.pybullet.mavlink_shim.backend import DroneState, ImuSample

# State vector indices (mirrors sim/dynamics/numpy_quad.py)
POS = slice(0, 3)
VEL = slice(3, 6)
QUAT = slice(6, 10)   # wxyz
OMEGA = slice(10, 13)
MOTOR = slice(13, 17)
GRAVITY = 9.81


@dataclass
class NumpyQuadBackend:
    params: VehicleParams

    def __post_init__(self) -> None:
        # Single-vehicle state, shape (1, 17).
        self._state = np.zeros((1, 17), dtype=np.float64)
        self._state[0, QUAT] = np.array([1.0, 0.0, 0.0, 0.0])
        self._t0_ns = time.monotonic_ns()
        self._last_accel_body = np.array([0.0, 0.0, GRAVITY])
        self._last_omega_body = np.zeros(3)

    def _now_us(self) -> int:
        return (time.monotonic_ns() - self._t0_ns) // 1000

    def reset(self, initial_state: DroneState) -> None:
        self._state[0, POS] = initial_state.pos_enu
        self._state[0, VEL] = initial_state.vel_enu
        self._state[0, QUAT] = initial_state.quat_wxyz
        self._state[0, OMEGA] = initial_state.angular_vel_body
        self._state[0, MOTOR] = initial_state.motor_speed
        self._t0_ns = time.monotonic_ns()
        self._last_accel_body = np.array([0.0, 0.0, GRAVITY])
        self._last_omega_body = initial_state.angular_vel_body.copy()

    def step(self, motor_commands: NDArray[np.float64], dt: float) -> DroneState:
        # Pre-step velocity for accel computation.
        vel_before = self._state[0, VEL].copy()

        if dt > 0.0:
            # Adapt this call to the actual numpy_quad API. Many vectorized
            # implementations have:
            #   nq.step(state[N,17], motor_cmd[N,4], dt, params) → new_state[N,17]
            # If the function name or signature differs, adjust here.
            self._state = nq.step(
                state=self._state,
                motor_commands=motor_commands.reshape(1, 4),
                dt=dt,
                params=self.params,
            )

        # Body-frame linear acceleration: world accel rotated into body, plus gravity sensed.
        vel_after = self._state[0, VEL]
        if dt > 0.0:
            world_accel = (vel_after - vel_before) / dt
        else:
            world_accel = np.zeros(3)

        # Rotate world_accel into body frame, then add +gravity (proper acceleration).
        q = self._state[0, QUAT]
        accel_body_kinematic = self._rotate_world_to_body(world_accel, q)
        # Sensed acceleration = kinematic + gravity reaction (-gravity in world → in body).
        gravity_body = self._rotate_world_to_body(np.array([0.0, 0.0, -GRAVITY]), q)
        self._last_accel_body = accel_body_kinematic - gravity_body

        self._last_omega_body = self._state[0, OMEGA].copy()

        return DroneState(
            pos_enu=self._state[0, POS].copy(),
            vel_enu=self._state[0, VEL].copy(),
            quat_wxyz=self._state[0, QUAT].copy(),
            angular_vel_body=self._state[0, OMEGA].copy(),
            motor_speed=self._state[0, MOTOR].copy(),
            timestamp_us=self._now_us(),
        )

    def get_imu(self) -> ImuSample:
        return ImuSample(
            accel_body=self._last_accel_body.copy(),
            gyro_body=self._last_omega_body.copy(),
            timestamp_us=self._now_us(),
        )

    @staticmethod
    def _rotate_world_to_body(v_world: NDArray[np.float64], q_wxyz: NDArray[np.float64]) -> NDArray[np.float64]:
        """Rotate world-frame vector v into body frame using quaternion q (wxyz)."""
        w, x, y, z = q_wxyz
        # Rotation matrix R_body_from_world = R_world_from_body.T
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y + w*z),     2*(x*z - w*y)],
            [2*(x*y - w*z),     1 - 2*(x*x + z*z), 2*(y*z + w*x)],
            [2*(x*z + w*y),     2*(y*z - w*x),     1 - 2*(x*x + y*y)],
        ])
        return R @ v_world
```

If `numpy_quad.py` doesn't expose a `step()` function — look for what it does expose (e.g., a class) and call accordingly. The intent is "advance the 17-element state by dt using motor commands."

- [ ] **Step 5: Run tests to verify they pass**

```bash
conda run -n monorace pytest tests/test_mavlink_shim/test_numpy_quad_backend.py -v
```

Expected: 3 passed. If `test_get_imu_returns_gravity_when_at_rest` is off by sign or factor of 2, fix the gravity bookkeeping in `step()` and `get_imu()` (this is the trickiest part — gravity convention varies).

- [ ] **Step 6: Commit**

```bash
git add sim/pybullet/mavlink_shim/numpy_quad_backend.py \
        tests/test_mavlink_shim/test_numpy_quad_backend.py
git commit -m "feat(mavlink-shim): NumpyQuadBackend wrapping numpy_quad dynamics"
```

---

## Task 6: MavlinkServer (UDP I/O wrapper)

**Files:**
- Create: `sim/pybullet/mavlink_shim/server.py`
- Create: `tests/test_mavlink_shim/test_server.py`

Thin pymavlink wrapper. Owns a UDP socket, exposes `recv_pending()` (drains all queued inbound) and message-encoding helpers for ATTITUDE/ODOMETRY/HIGHRES_IMU/HEARTBEAT.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mavlink_shim/test_server.py`:

```python
"""Tests for MavlinkServer — UDP loopback I/O via pymavlink."""
import socket
import time

import numpy as np
import pytest
from pymavlink import mavutil

from sim.pybullet.mavlink_shim.server import MavlinkServer


def _free_port() -> int:
    """Find an unused UDP port for test isolation."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_server_sends_heartbeat_received_by_pymavlink_client():
    port = _free_port()
    server = MavlinkServer(host="127.0.0.1", port=port, sysid=1, compid=1)
    client = mavutil.mavlink_connection(f"udpout:127.0.0.1:{port}",
                                         source_system=255, source_component=0)
    # Client must send something first so server learns its address.
    client.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
    )
    # Give server a chance to receive client heartbeat.
    time.sleep(0.05)
    server.recv_pending()
    server.send_heartbeat()

    # Client should receive the heartbeat back.
    msg = client.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
    assert msg is not None, "client did not receive heartbeat from server"
    assert msg.get_srcSystem() == 1


def test_server_recv_pending_returns_set_attitude_target_payload():
    port = _free_port()
    server = MavlinkServer(host="127.0.0.1", port=port, sysid=1, compid=1)
    client = mavutil.mavlink_connection(f"udpout:127.0.0.1:{port}",
                                         source_system=255, source_component=0)
    # Send a SET_ATTITUDE_TARGET from client.
    client.mav.set_attitude_target_send(
        time_boot_ms=0, target_system=1, target_component=1,
        type_mask=0,
        q=[1.0, 0.0, 0.0, 0.0],
        body_roll_rate=0.0, body_pitch_rate=0.0, body_yaw_rate=0.0,
        thrust=0.5,
    )
    time.sleep(0.05)
    msgs = server.recv_pending()
    assert any(m.get_type() == "SET_ATTITUDE_TARGET" for m in msgs), \
        f"got {[m.get_type() for m in msgs]}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n monorace pytest tests/test_mavlink_shim/test_server.py -v
```

Expected: ImportError on `MavlinkServer`.

- [ ] **Step 3: Implement the server**

Create `sim/pybullet/mavlink_shim/server.py`:

```python
"""MavlinkServer — UDP MAVLink I/O wrapper using pymavlink.

Owns the socket, encodes/decodes messages, tracks the last-seen client
address (for the udpin-with-broadcast-back pattern). No control logic;
the MavlinkShim wires this to the AttitudeController + Backend.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from pymavlink import mavutil


@dataclass
class MavlinkServer:
    host: str = "0.0.0.0"
    port: int = 14550
    sysid: int = 1
    compid: int = 1
    _conn: object = field(init=False)
    _t0_us: int = field(init=False)

    def __post_init__(self) -> None:
        # "udpin" binds and waits; first incoming packet's address is captured
        # by pymavlink and outbound messages go back to that address.
        self._conn = mavutil.mavlink_connection(
            f"udpin:{self.host}:{self.port}",
            source_system=self.sysid,
            source_component=self.compid,
        )
        self._t0_us = int(time.monotonic() * 1_000_000)

    def close(self) -> None:
        if hasattr(self._conn, "close"):
            self._conn.close()

    def recv_pending(self) -> list:
        """Drain all messages currently queued on the socket. Non-blocking."""
        out = []
        while True:
            msg = self._conn.recv_match(blocking=False)
            if msg is None:
                break
            out.append(msg)
        return out

    def _t_boot_ms(self) -> int:
        return (int(time.monotonic() * 1_000_000) - self._t0_us) // 1000

    def _t_usec(self) -> int:
        return int(time.monotonic() * 1_000_000) - self._t0_us

    def send_heartbeat(self) -> None:
        self._conn.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_QUADROTOR,
            mavutil.mavlink.MAV_AUTOPILOT_GENERIC,
            base_mode=0,
            custom_mode=0,
            system_status=mavutil.mavlink.MAV_STATE_ACTIVE,
        )

    def send_attitude(
        self,
        roll: float, pitch: float, yaw: float,
        rollspeed: float, pitchspeed: float, yawspeed: float,
    ) -> None:
        self._conn.mav.attitude_send(
            self._t_boot_ms(),
            roll, pitch, yaw, rollspeed, pitchspeed, yawspeed,
        )

    def send_odometry(
        self,
        pos_ned: NDArray[np.float64],
        vel_ned: NDArray[np.float64],
        quat_ned_xyzw: NDArray[np.float64],
        angular_vel_body: NDArray[np.float64],
    ) -> None:
        zero_cov = [float("nan")] * 21  # mark covariance unknown
        self._conn.mav.odometry_send(
            time_usec=self._t_usec(),
            frame_id=mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            child_frame_id=mavutil.mavlink.MAV_FRAME_BODY_FRD,
            x=float(pos_ned[0]), y=float(pos_ned[1]), z=float(pos_ned[2]),
            q=[float(quat_ned_xyzw[3]), float(quat_ned_xyzw[0]),
               float(quat_ned_xyzw[1]), float(quat_ned_xyzw[2])],  # MAVLink wants wxyz here
            vx=float(vel_ned[0]), vy=float(vel_ned[1]), vz=float(vel_ned[2]),
            rollspeed=float(angular_vel_body[0]),
            pitchspeed=float(angular_vel_body[1]),
            yawspeed=float(angular_vel_body[2]),
            pose_covariance=zero_cov,
            velocity_covariance=zero_cov,
            reset_counter=0,
            estimator_type=mavutil.mavlink.MAV_ESTIMATOR_TYPE_NAIVE,
            quality=100,
        )

    def send_highres_imu(
        self,
        accel_body: NDArray[np.float64],
        gyro_body: NDArray[np.float64],
    ) -> None:
        accel_gyro_bits = (
            mavutil.mavlink.HIGHRES_IMU_UPDATED_XACC
            | mavutil.mavlink.HIGHRES_IMU_UPDATED_YACC
            | mavutil.mavlink.HIGHRES_IMU_UPDATED_ZACC
            | mavutil.mavlink.HIGHRES_IMU_UPDATED_XGYRO
            | mavutil.mavlink.HIGHRES_IMU_UPDATED_YGYRO
            | mavutil.mavlink.HIGHRES_IMU_UPDATED_ZGYRO
        )
        self._conn.mav.highres_imu_send(
            time_usec=self._t_usec(),
            xacc=float(accel_body[0]), yacc=float(accel_body[1]), zacc=float(accel_body[2]),
            xgyro=float(gyro_body[0]), ygyro=float(gyro_body[1]), zgyro=float(gyro_body[2]),
            xmag=0.0, ymag=0.0, zmag=0.0,
            abs_pressure=0.0, diff_pressure=0.0, pressure_alt=0.0, temperature=0.0,
            fields_updated=accel_gyro_bits,
        )
```

Note: pymavlink's `mavutil.mavlink.HIGHRES_IMU_UPDATED_*` constants might be named differently across pymavlink versions. If the constant lookup fails at import time, replace with literal bit values (`1 << 0`, `1 << 1`, etc.) per the MAVLink spec. Run `conda run -n monorace python -c "from pymavlink import mavutil; print([x for x in dir(mavutil.mavlink) if 'HIGHRES_IMU' in x])"` to confirm.

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n monorace pytest tests/test_mavlink_shim/test_server.py -v
```

Expected: 2 passed.

If `send_odometry` errors on the `q` parameter shape — different pymavlink versions accept different conventions. Check what your version expects with `help(client.mav.odometry_send)` and adjust the `q` element ordering.

- [ ] **Step 5: Commit**

```bash
git add sim/pybullet/mavlink_shim/server.py \
        tests/test_mavlink_shim/test_server.py
git commit -m "feat(mavlink-shim): MavlinkServer UDP I/O wrapper around pymavlink"
```

---

## Task 7: MavlinkShim orchestrator + lifecycle

**Files:**
- Create: `sim/pybullet/mavlink_shim/shim.py`
- Modify: `sim/pybullet/mavlink_shim/__init__.py` (export MavlinkShim)
- Create: `tests/test_mavlink_shim/test_backend_protocol.py`

The top-level class. Holds the backend, controller, scheduler, server. Provides `start/stop/reset/step` and a context-manager interface.

- [ ] **Step 1: Write the failing test**

Create `tests/test_mavlink_shim/test_backend_protocol.py`:

```python
"""Tests for the MavlinkShim ↔ DroneBackend interface contract using a MockBackend."""
import socket
from dataclasses import dataclass, field

import numpy as np
import pytest
from pymavlink import mavutil

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim import (
    DroneBackend,
    DroneState,
    ImuSample,
    MavlinkShim,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class MockBackend:
    """Records every call and returns canned state."""
    reset_calls: list = field(default_factory=list)
    step_calls: list = field(default_factory=list)
    imu_calls: int = 0
    _state: DroneState = field(default_factory=lambda: DroneState(
        pos_enu=np.zeros(3), vel_enu=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3), motor_speed=np.zeros(4),
        timestamp_us=0,
    ))

    def reset(self, initial_state: DroneState) -> None:
        self.reset_calls.append(initial_state)
        self._state = initial_state

    def step(self, motor_commands, dt: float) -> DroneState:
        self.step_calls.append((motor_commands.copy(), dt))
        return self._state

    def get_imu(self) -> ImuSample:
        self.imu_calls += 1
        return ImuSample(
            accel_body=np.array([0.0, 0.0, 9.81]),
            gyro_body=np.zeros(3),
            timestamp_us=0,
        )


def test_shim_step_calls_backend_step_with_motor_commands_from_controller():
    port = _free_port()
    backend = MockBackend()
    shim = MavlinkShim(
        backend=backend,
        params=VehicleParams.default(),
        host="127.0.0.1", port=port,
        lockstep=True,
        rates_hz={"heartbeat": 1, "attitude": 100, "odometry": 100, "highres_imu": 200},
    )
    initial = DroneState(
        pos_enu=np.zeros(3), vel_enu=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3), motor_speed=np.zeros(4),
        timestamp_us=0,
    )

    try:
        shim.start()
        shim.reset(initial)
        # No client command yet — step with timeout returns False.
        stepped = shim.step(timeout_s=0.05)
        assert stepped is False
        assert len(backend.step_calls) == 0

        # Send a SET_ATTITUDE_TARGET → step should advance and call backend.step.
        client = mavutil.mavlink_connection(
            f"udpout:127.0.0.1:{port}", source_system=255, source_component=0,
        )
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0, 0, 0,
        )
        client.mav.set_attitude_target_send(
            time_boot_ms=0, target_system=1, target_component=1, type_mask=0,
            q=[1.0, 0.0, 0.0, 0.0],
            body_roll_rate=0.0, body_pitch_rate=0.0, body_yaw_rate=0.0,
            thrust=0.5,
        )
        stepped = shim.step(timeout_s=0.5)
        assert stepped is True
        assert len(backend.step_calls) == 1
        cmd, dt = backend.step_calls[0]
        assert cmd.shape == (4,)
        assert dt > 0.0
    finally:
        shim.stop()


def test_shim_reset_propagates_to_backend():
    port = _free_port()
    backend = MockBackend()
    shim = MavlinkShim(
        backend=backend, params=VehicleParams.default(),
        host="127.0.0.1", port=port, lockstep=True,
        rates_hz={"heartbeat": 1, "attitude": 100, "odometry": 100, "highres_imu": 200},
    )
    initial = DroneState(
        pos_enu=np.array([1.0, 2.0, 3.0]),
        vel_enu=np.zeros(3), quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3), motor_speed=np.zeros(4),
        timestamp_us=0,
    )
    try:
        shim.start()
        shim.reset(initial)
        assert len(backend.reset_calls) == 1
        np.testing.assert_array_equal(backend.reset_calls[0].pos_enu, [1.0, 2.0, 3.0])
    finally:
        shim.stop()


def test_shim_context_manager_starts_and_stops():
    port = _free_port()
    backend = MockBackend()
    shim = MavlinkShim(
        backend=backend, params=VehicleParams.default(),
        host="127.0.0.1", port=port, lockstep=True,
        rates_hz={"heartbeat": 1, "attitude": 100, "odometry": 100, "highres_imu": 200},
    )
    with shim:
        assert shim.is_running()
    assert not shim.is_running()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n monorace pytest tests/test_mavlink_shim/test_backend_protocol.py -v
```

Expected: ImportError on `MavlinkShim`.

- [ ] **Step 3: Implement the shim**

Create `sim/pybullet/mavlink_shim/shim.py`:

```python
"""MavlinkShim — top-level orchestrator wiring server + controller + backend."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from sim.dynamics.params import VehicleParams
from sim.pybullet.coords import ned_to_enu_position, ned_to_enu_quaternion
from sim.pybullet.mavlink_shim.attitude_controller import AttitudeController
from sim.pybullet.mavlink_shim.backend import DroneBackend, DroneState
from sim.pybullet.mavlink_shim.coords_mavlink import (
    enu_quat_to_ned_euler,
    enu_quat_to_ned_quat_xyzw,
)
from sim.pybullet.mavlink_shim.rate_scheduler import RateScheduler
from sim.pybullet.mavlink_shim.server import MavlinkServer


@dataclass
class MavlinkShim:
    backend: DroneBackend
    params: VehicleParams
    host: str = "0.0.0.0"
    port: int = 14550
    sysid: int = 1
    compid: int = 1
    lockstep: bool = False
    rates_hz: dict = field(default_factory=lambda: {
        "heartbeat": 1.0,
        "attitude": 100.0,
        "odometry": 100.0,
        "highres_imu": 200.0,
    })
    controller_k_att: float = 6.0

    _server: MavlinkServer = field(init=False, default=None)
    _controller: AttitudeController = field(init=False, default=None)
    _scheduler: RateScheduler = field(init=False, default=None)
    _thread: threading.Thread = field(init=False, default=None)
    _stop_event: threading.Event = field(init=False, default_factory=threading.Event)
    _last_target_q: NDArray[np.float64] = field(init=False, default=None)
    _last_target_thrust: float = field(init=False, default=0.0)
    _last_state: DroneState = field(init=False, default=None)
    _last_step_us: int = field(init=False, default=0)
    _t0_us: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._controller = AttitudeController(params=self.params, k_att=self.controller_k_att)
        # tick rate = highest configured outbound rate, default 200 Hz.
        tick_hz = max(self.rates_hz.values()) if self.rates_hz else 200.0
        self._scheduler = RateScheduler(rates_hz=dict(self.rates_hz), tick_hz=tick_hz)
        self._last_target_q = np.array([1.0, 0.0, 0.0, 0.0])

    # ---- lifecycle ----

    def start(self) -> None:
        if self._server is not None:
            return  # already started
        self._server = MavlinkServer(
            host=self.host, port=self.port, sysid=self.sysid, compid=self.compid,
        )
        self._t0_us = int(time.monotonic() * 1_000_000)
        self._stop_event.clear()
        if not self.lockstep:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._server is not None:
            self._server.close()
            self._server = None

    def is_running(self) -> bool:
        return self._server is not None

    def __enter__(self) -> "MavlinkShim":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def reset(self, initial_state: DroneState) -> None:
        self.backend.reset(initial_state)
        self._last_state = initial_state
        self._last_target_q = np.array([1.0, 0.0, 0.0, 0.0])
        self._last_target_thrust = 0.0
        self._last_step_us = 0
        # Re-init scheduler so all messages are due immediately.
        tick_hz = max(self.rates_hz.values()) if self.rates_hz else 200.0
        self._scheduler = RateScheduler(rates_hz=dict(self.rates_hz), tick_hz=tick_hz)
        if self._server is not None:
            self._server.send_heartbeat()

    def wait_for_shutdown(self, timeout_s: float | None = None) -> None:
        if self._thread is None:
            return
        self._thread.join(timeout=timeout_s)

    # ---- lockstep step ----

    def step(self, timeout_s: float = 0.1) -> bool:
        """Lockstep mode: block until a SET_ATTITUDE_TARGET arrives or timeout.
        Returns True if a step happened, False on timeout.
        """
        deadline = time.monotonic() + timeout_s
        got_command = False
        while time.monotonic() < deadline:
            msgs = self._server.recv_pending()
            for m in msgs:
                if m.get_type() == "SET_ATTITUDE_TARGET":
                    self._last_target_q = np.array(m.q, dtype=np.float64)
                    self._last_target_thrust = float(m.thrust)
                    got_command = True
            if got_command:
                break
            time.sleep(0.001)
        if not got_command:
            return False
        self._do_one_step()
        return True

    # ---- free-running loop ----

    def _run_loop(self) -> None:
        tick_period_s = 1.0 / max(self.rates_hz.values())
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            # Drain inbound.
            msgs = self._server.recv_pending()
            for m in msgs:
                if m.get_type() == "SET_ATTITUDE_TARGET":
                    self._last_target_q = np.array(m.q, dtype=np.float64)
                    self._last_target_thrust = float(m.thrust)
            # Step physics.
            self._do_one_step()
            # Sleep until next tick.
            next_tick += tick_period_s
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()  # catch up if we fell behind

    def _do_one_step(self) -> None:
        # Compute dt from monotonic time.
        now_us = int(time.monotonic() * 1_000_000) - self._t0_us
        dt = (now_us - self._last_step_us) / 1_000_000.0
        if dt <= 0:
            dt = 1.0 / max(self.rates_hz.values())
        self._last_step_us = now_us

        # NED quaternion from MAVLink → ENU for the controller.
        from sim.pybullet.mavlink_shim.coords_mavlink import ned_quat_wxyz_to_enu_quat
        q_target_enu = ned_quat_wxyz_to_enu_quat(self._last_target_q)

        if self._last_state is None:
            # Hold motors at zero until first step's state read.
            motor_speeds = np.zeros(4)
            self._last_state = self.backend.step(motor_speeds, dt=0.0)

        motor_speeds = self._controller.compute(
            q_target_enu_wxyz=q_target_enu,
            thrust_normalized=self._last_target_thrust,
            q_current_enu_wxyz=self._last_state.quat_wxyz,
            omega_current_body=self._last_state.angular_vel_body,
        )
        if not np.all(np.isfinite(motor_speeds)):
            motor_speeds = np.zeros(4)
        self._last_state = self.backend.step(motor_speeds, dt=dt)

        # Emit due outbound messages.
        for name in self._scheduler.due(now_us):
            self._send_message(name)

    def _send_message(self, name: str) -> None:
        s = self._last_state
        if name == "heartbeat":
            self._server.send_heartbeat()
        elif name == "attitude":
            roll, pitch, yaw = enu_quat_to_ned_euler(s.quat_wxyz)
            # body rates are body-frame; same in NED and ENU body frames since
            # body axes themselves are the same convention.
            self._server.send_attitude(
                roll, pitch, yaw,
                float(s.angular_vel_body[0]),
                float(s.angular_vel_body[1]),
                float(s.angular_vel_body[2]),
            )
        elif name == "odometry":
            pos_ned = np.array([s.pos_enu[1], s.pos_enu[0], -s.pos_enu[2]])
            vel_ned = np.array([s.vel_enu[1], s.vel_enu[0], -s.vel_enu[2]])
            quat_ned_xyzw = enu_quat_to_ned_quat_xyzw(s.quat_wxyz)
            self._server.send_odometry(pos_ned, vel_ned, quat_ned_xyzw, s.angular_vel_body)
        elif name == "highres_imu":
            imu = self.backend.get_imu()
            self._server.send_highres_imu(imu.accel_body, imu.gyro_body)
```

- [ ] **Step 4: Update `__init__.py` to export the shim**

Edit `sim/pybullet/mavlink_shim/__init__.py`:

```python
"""Reusable MAVLink server for drone dynamics backends."""
from sim.pybullet.mavlink_shim.backend import (
    DroneBackend,
    DroneState,
    ImuSample,
)
from sim.pybullet.mavlink_shim.shim import MavlinkShim

__all__ = ["DroneBackend", "DroneState", "ImuSample", "MavlinkShim"]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
conda run -n monorace pytest tests/test_mavlink_shim/test_backend_protocol.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add sim/pybullet/mavlink_shim/shim.py \
        sim/pybullet/mavlink_shim/__init__.py \
        tests/test_mavlink_shim/test_backend_protocol.py
git commit -m "feat(mavlink-shim): MavlinkShim orchestrator + lifecycle"
```

---

## Task 8: Integration test — hover_hold

**Files:**
- Create: `tests/test_mavlink_shim/test_hover_hold.py`

End-to-end lockstep test. Identity attitude + hover thrust commanded for 2.5 seconds (500 steps at 200 Hz). Drone should stay at initial position within 5 cm and within 1° of identity attitude.

- [ ] **Step 1: Write the test**

Create `tests/test_mavlink_shim/test_hover_hold.py`:

```python
"""Integration test: hover hold with identity attitude + hover thrust."""
import socket
import time

import numpy as np
import pytest
from pymavlink import mavutil

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim import DroneState, MavlinkShim
from sim.pybullet.mavlink_shim.numpy_quad_backend import NumpyQuadBackend


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _hover_thrust(params: VehicleParams) -> float:
    weight_n = params.mass_kg * 9.81
    return weight_n / (4.0 * params.max_thrust_per_motor_n)


def test_hover_hold_keeps_drone_within_tolerance():
    port = _free_port()
    params = VehicleParams.default()
    backend = NumpyQuadBackend(params=params)
    shim = MavlinkShim(
        backend=backend, params=params,
        host="127.0.0.1", port=port, lockstep=True,
        rates_hz={"heartbeat": 1, "attitude": 100, "odometry": 100, "highres_imu": 200},
    )

    initial = DroneState(
        pos_enu=np.zeros(3), vel_enu=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3), motor_speed=np.zeros(4),
        timestamp_us=0,
    )

    try:
        shim.start()
        shim.reset(initial)
        client = mavutil.mavlink_connection(
            f"udpout:127.0.0.1:{port}", source_system=255, source_component=0,
        )
        # Send a heartbeat first so the shim sees us.
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )

        thrust = _hover_thrust(params)
        for _ in range(500):
            client.mav.set_attitude_target_send(
                time_boot_ms=0, target_system=1, target_component=1, type_mask=0,
                q=[1.0, 0.0, 0.0, 0.0],
                body_roll_rate=0.0, body_pitch_rate=0.0, body_yaw_rate=0.0,
                thrust=float(thrust),
            )
            stepped = shim.step(timeout_s=0.5)
            assert stepped, "shim.step did not advance"

        final = shim._last_state  # internal state for assertion
        pos_err = np.linalg.norm(final.pos_enu - initial.pos_enu)
        att_err_deg = np.degrees(2.0 * np.arccos(abs(np.clip(final.quat_wxyz[0], -1.0, 1.0))))
        assert pos_err < 0.05, f"position drifted {pos_err:.3f} m"
        assert att_err_deg < 1.0, f"attitude drifted {att_err_deg:.2f}°"
    finally:
        shim.stop()
```

- [ ] **Step 2: Run test**

```bash
conda run -n monorace pytest tests/test_mavlink_shim/test_hover_hold.py -v
```

Expected: 1 passed. If the test FAILS with "position drifted N m", the hover-thrust calibration is off — check the `_hover_thrust` formula matches `params.max_thrust_per_motor_n` field name. If the test FAILS with attitude drift, gain `controller_k_att` may be too aggressive — try `controller_k_att=4.0` in shim construction.

- [ ] **Step 3: Commit**

```bash
git add tests/test_mavlink_shim/test_hover_hold.py
git commit -m "test(mavlink-shim): integration — hover hold within 5cm + 1°"
```

---

## Task 9: Integration test — attitude_step_response

**Files:**
- Create: `tests/test_mavlink_shim/test_attitude_step_response.py`

Command 30° pitch step. Within 0.5 s the drone should reach 25-30° pitch.

- [ ] **Step 1: Write the test**

Create `tests/test_mavlink_shim/test_attitude_step_response.py`:

```python
"""Integration test: 30° pitch step → drone tracks it within 0.5s."""
import socket

import numpy as np
import pytest
from pymavlink import mavutil

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim import DroneState, MavlinkShim
from sim.pybullet.mavlink_shim.numpy_quad_backend import NumpyQuadBackend
from sim.pybullet.mavlink_shim.coords_mavlink import enu_quat_to_ned_euler


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _hover_thrust(params: VehicleParams) -> float:
    weight_n = params.mass_kg * 9.81
    return weight_n / (4.0 * params.max_thrust_per_motor_n)


def test_30deg_pitch_step_reaches_target_within_0_5s():
    port = _free_port()
    params = VehicleParams.default()
    backend = NumpyQuadBackend(params=params)
    shim = MavlinkShim(
        backend=backend, params=params,
        host="127.0.0.1", port=port, lockstep=True,
        rates_hz={"heartbeat": 1, "attitude": 100, "odometry": 100, "highres_imu": 200},
    )

    initial = DroneState(
        pos_enu=np.zeros(3), vel_enu=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3), motor_speed=np.zeros(4),
        timestamp_us=0,
    )

    angle = np.deg2rad(30.0)
    # Pitch +30° in NED frame: rotation about NED +Y axis (East).
    q_target_ned = np.array([np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0])

    try:
        shim.start()
        shim.reset(initial)
        client = mavutil.mavlink_connection(
            f"udpout:127.0.0.1:{port}", source_system=255, source_component=0,
        )
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )

        thrust = _hover_thrust(params)
        # Run for 0.5 s = 100 steps at 200 Hz.
        for _ in range(100):
            client.mav.set_attitude_target_send(
                time_boot_ms=0, target_system=1, target_component=1, type_mask=0,
                q=list(q_target_ned),
                body_roll_rate=0.0, body_pitch_rate=0.0, body_yaw_rate=0.0,
                thrust=float(thrust),
            )
            assert shim.step(timeout_s=0.5)

        final = shim._last_state
        roll, pitch, yaw = enu_quat_to_ned_euler(final.quat_wxyz)
        pitch_deg = np.degrees(pitch)
        assert 25.0 <= pitch_deg <= 30.0, f"pitch={pitch_deg:.2f}°, expected 25-30"
    finally:
        shim.stop()
```

- [ ] **Step 2: Run test**

```bash
conda run -n monorace pytest tests/test_mavlink_shim/test_attitude_step_response.py -v
```

Expected: 1 passed. If pitch undershoots (< 25°), bump `controller_k_att` higher (e.g., 8.0). If pitch overshoots (> 30°), reduce `k_att`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_mavlink_shim/test_attitude_step_response.py
git commit -m "test(mavlink-shim): integration — 30° pitch step response"
```

---

## Task 10: Integration test — reset_returns_to_initial

**Files:**
- Create: `tests/test_mavlink_shim/test_reset_returns_to_initial.py`

After running 100 commands then calling reset, drone state == specified initial.

- [ ] **Step 1: Write the test**

Create `tests/test_mavlink_shim/test_reset_returns_to_initial.py`:

```python
"""Integration test: reset returns drone to specified initial state exactly."""
import socket

import numpy as np
import pytest
from pymavlink import mavutil

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim import DroneState, MavlinkShim
from sim.pybullet.mavlink_shim.numpy_quad_backend import NumpyQuadBackend


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_reset_after_disturbance_returns_state_to_initial():
    port = _free_port()
    params = VehicleParams.default()
    backend = NumpyQuadBackend(params=params)
    shim = MavlinkShim(
        backend=backend, params=params,
        host="127.0.0.1", port=port, lockstep=True,
        rates_hz={"heartbeat": 1, "attitude": 100, "odometry": 100, "highres_imu": 200},
    )

    initial = DroneState(
        pos_enu=np.array([1.0, 2.0, 3.0]),
        vel_enu=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3),
        motor_speed=np.zeros(4),
        timestamp_us=0,
    )

    try:
        shim.start()
        shim.reset(initial)
        client = mavutil.mavlink_connection(
            f"udpout:127.0.0.1:{port}", source_system=255, source_component=0,
        )
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        # Disturb: 100 commands at full thrust + 30° pitch.
        for _ in range(100):
            client.mav.set_attitude_target_send(
                time_boot_ms=0, target_system=1, target_component=1, type_mask=0,
                q=[0.966, 0.0, 0.259, 0.0],  # ~30° pitch
                body_roll_rate=0.0, body_pitch_rate=0.0, body_yaw_rate=0.0,
                thrust=1.0,
            )
            assert shim.step(timeout_s=0.5)
        # State has now diverged. Reset → should be back to initial.
        shim.reset(initial)
        np.testing.assert_array_equal(shim._last_state.pos_enu, initial.pos_enu)
        np.testing.assert_array_equal(shim._last_state.vel_enu, initial.vel_enu)
        np.testing.assert_array_equal(shim._last_state.quat_wxyz, initial.quat_wxyz)
    finally:
        shim.stop()
```

- [ ] **Step 2: Run test**

```bash
conda run -n monorace pytest tests/test_mavlink_shim/test_reset_returns_to_initial.py -v
```

Expected: 1 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_mavlink_shim/test_reset_returns_to_initial.py
git commit -m "test(mavlink-shim): integration — reset returns to initial state"
```

---

## Task 11: Integration test — message_rates_match_config

**Files:**
- Create: `tests/test_mavlink_shim/test_message_rates_match_config.py`

Run free-running for 5 s. Capture inbound message counts on a pymavlink client. Verify within ±5% of expected.

- [ ] **Step 1: Write the test**

Create `tests/test_mavlink_shim/test_message_rates_match_config.py`:

```python
"""Integration test: message rates match configured Hz over 5s window."""
import socket
import time

import pytest
from pymavlink import mavutil

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim import DroneState, MavlinkShim
from sim.pybullet.mavlink_shim.numpy_quad_backend import NumpyQuadBackend
import numpy as np


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_message_rates_within_5_percent_of_config():
    port = _free_port()
    params = VehicleParams.default()
    backend = NumpyQuadBackend(params=params)
    rates = {"heartbeat": 1, "attitude": 100, "odometry": 100, "highres_imu": 200}
    shim = MavlinkShim(
        backend=backend, params=params,
        host="127.0.0.1", port=port, lockstep=False,
        rates_hz=rates,
    )

    initial = DroneState(
        pos_enu=np.zeros(3), vel_enu=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3), motor_speed=np.zeros(4),
        timestamp_us=0,
    )

    counts = {"HEARTBEAT": 0, "ATTITUDE": 0, "ODOMETRY": 0, "HIGHRES_IMU": 0}

    try:
        with shim:
            shim.reset(initial)
            client = mavutil.mavlink_connection(
                f"udpout:127.0.0.1:{port}", source_system=255, source_component=0,
            )
            # First heartbeat from client so shim learns where to send.
            client.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
            )

            t_end = time.monotonic() + 5.0
            while time.monotonic() < t_end:
                msg = client.recv_match(blocking=True, timeout=0.5)
                if msg is None:
                    continue
                t = msg.get_type()
                if t in counts:
                    counts[t] += 1

        for name_count, name_rate in [
            ("HEARTBEAT", "heartbeat"), ("ATTITUDE", "attitude"),
            ("ODOMETRY", "odometry"), ("HIGHRES_IMU", "highres_imu"),
        ]:
            expected = rates[name_rate] * 5  # over 5 seconds
            actual = counts[name_count]
            tol = max(1, expected * 0.05)
            assert expected - tol <= actual <= expected + tol, (
                f"{name_count}: got {actual}, expected {expected} ± {tol}"
            )
    finally:
        # context manager already calls stop, but be defensive.
        if shim.is_running():
            shim.stop()
```

- [ ] **Step 2: Run test**

```bash
conda run -n monorace pytest tests/test_mavlink_shim/test_message_rates_match_config.py -v
```

Expected: 1 passed. This test takes ~5 seconds to run.

If counts come up consistently low (e.g., 80% of expected), the worker thread is missing deadlines — likely on Windows where `time.sleep` resolution is ~16ms. Mitigation: lower the test rates (e.g., `attitude=50`, `imu=100`) and adjust expectations, OR increase the assertion tolerance to ±10%.

- [ ] **Step 3: Commit**

```bash
git add tests/test_mavlink_shim/test_message_rates_match_config.py
git commit -m "test(mavlink-shim): integration — message rates match config ±5%"
```

---

## Task 12: Smoke test script with attitude tracking plot

**Files:**
- Create: `scripts/mavlink/__init__.py`
- Create: `scripts/mavlink/smoke_test.py`

Manual demo. Free-running shim + pymavlink client streams a sequence of attitude targets. Records ATTITUDE responses. Plots commanded vs achieved.

- [ ] **Step 1: Create the empty package marker**

```bash
mkdir -p scripts/mavlink
touch scripts/mavlink/__init__.py
```

- [ ] **Step 2: Create the smoke test**

Create `scripts/mavlink/smoke_test.py`:

```python
"""Manual smoke test for the MAVLink shim.

Spins up MavlinkShim in free-running mode, connects a pymavlink client,
streams a sequence of attitude targets, and plots commanded vs achieved
attitude over time.

Usage:
    conda run -n monorace python -m scripts.mavlink.smoke_test \\
        [--port 14550] [--out outputs/mavlink_smoke/attitude.png]

Output: PNG with two subplots (pitch / roll) showing commanded vs
achieved trajectories.

Bonus manual check: while this script is running (or with --hold to keep
the shim alive after plotting), point QGroundControl at udp:127.0.0.1:14550
and confirm the vehicle appears with non-zero telemetry.
"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim import DroneState, MavlinkShim
from sim.pybullet.mavlink_shim.numpy_quad_backend import NumpyQuadBackend
from sim.pybullet.mavlink_shim.coords_mavlink import enu_quat_to_ned_euler

from pymavlink import mavutil


def _hover_thrust(params: VehicleParams) -> float:
    weight_n = params.mass_kg * 9.81
    return weight_n / (4.0 * params.max_thrust_per_motor_n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=14550)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/mavlink_smoke/attitude.png"))
    ap.add_argument("--hold", action="store_true",
                    help="Keep shim alive after plotting for QGC manual check")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    params = VehicleParams.default()
    backend = NumpyQuadBackend(params=params)
    shim = MavlinkShim(
        backend=backend, params=params,
        host="127.0.0.1", port=args.port, lockstep=False,
        rates_hz={"heartbeat": 1, "attitude": 100, "odometry": 100, "highres_imu": 200},
    )

    initial = DroneState(
        pos_enu=np.zeros(3), vel_enu=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3), motor_speed=np.zeros(4),
        timestamp_us=0,
    )

    thrust = _hover_thrust(params)

    sequence = [
        # (q_target_ned_wxyz, duration_s, label)
        ((1.0, 0.0, 0.0, 0.0), 1.0, "hover"),
        ((np.cos(np.pi/12), 0.0, np.sin(np.pi/12), 0.0), 1.0, "+30 pitch"),
        ((1.0, 0.0, 0.0, 0.0), 1.0, "hover"),
        ((np.cos(np.pi/12), 0.0, -np.sin(np.pi/12), 0.0), 1.0, "-30 pitch"),
        ((1.0, 0.0, 0.0, 0.0), 1.0, "hover"),
    ]

    cmd_pitch_log: list[tuple[float, float]] = []
    ach_pitch_log: list[tuple[float, float]] = []

    with shim:
        shim.reset(initial)
        client = mavutil.mavlink_connection(
            f"udpout:127.0.0.1:{args.port}",
            source_system=255, source_component=0,
        )
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )

        # Background thread: receive ATTITUDE messages.
        stop_recv = threading.Event()
        t_start = time.monotonic()

        def _recv_loop() -> None:
            while not stop_recv.is_set():
                msg = client.recv_match(type="ATTITUDE", blocking=True, timeout=0.05)
                if msg is None:
                    continue
                t = time.monotonic() - t_start
                ach_pitch_log.append((t, np.degrees(msg.pitch)))

        recv_thread = threading.Thread(target=_recv_loop, daemon=True)
        recv_thread.start()

        for q_target, dur, _label in sequence:
            cmd_pitch_deg = float(np.degrees(2.0 * np.arctan2(q_target[2], q_target[0])))
            t_seg_end = time.monotonic() + dur
            while time.monotonic() < t_seg_end:
                client.mav.set_attitude_target_send(
                    time_boot_ms=0, target_system=1, target_component=1, type_mask=0,
                    q=list(q_target),
                    body_roll_rate=0.0, body_pitch_rate=0.0, body_yaw_rate=0.0,
                    thrust=float(thrust),
                )
                cmd_pitch_log.append((time.monotonic() - t_start, cmd_pitch_deg))
                time.sleep(0.01)

        stop_recv.set()
        recv_thread.join(timeout=1.0)

        if args.hold:
            print(f"Holding shim alive on UDP port {args.port}. "
                  f"Connect QGroundControl to udp:127.0.0.1:{args.port}.")
            print("Press Ctrl-C to exit.")
            try:
                shim.wait_for_shutdown()
            except KeyboardInterrupt:
                pass

    # Plot.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmd_t, cmd_p = zip(*cmd_pitch_log) if cmd_pitch_log else ([], [])
    ach_t, ach_p = zip(*ach_pitch_log) if ach_pitch_log else ([], [])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cmd_t, cmd_p, "k--", label="commanded pitch")
    ax.plot(ach_t, ach_p, "b-", label="achieved pitch")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("pitch (deg)")
    ax.set_title("MAVLink shim — attitude tracking smoke test")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"Wrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run the smoke test**

```bash
conda run -n monorace python -m scripts.mavlink.smoke_test
```

Expected: prints `Wrote outputs/mavlink_smoke/attitude.png`. Open the PNG and confirm the achieved pitch (blue) tracks the commanded pitch (dashed black) within ~5° steady-state error during the +30 / -30 segments and returns to zero during the hover segments.

- [ ] **Step 4: Bonus manual check (optional, not required for spec exit)**

Run with `--hold`:

```bash
conda run -n monorace python -m scripts.mavlink.smoke_test --hold
```

Open QGroundControl, add a UDP link to `127.0.0.1:14550`, confirm a vehicle appears with non-zero telemetry. Press Ctrl-C in the terminal when done.

Note this in the implementation issue or session notes: "QGC handshake confirmed" or "QGC handshake skipped (QGC not installed)."

- [ ] **Step 5: Commit**

```bash
git add scripts/mavlink/__init__.py scripts/mavlink/smoke_test.py
git commit -m "feat(mavlink-shim): smoke test script with attitude tracking plot"
```

---

## Self-Review Notes

- **Spec coverage:** every spec requirement has a task —
  - DroneBackend Protocol + dataclasses → T1
  - NumpyQuadBackend → T5
  - AttitudeController (single-loop P + TRPY mixer) → T4
  - MAVLink message set (HEARTBEAT, ATTITUDE, ODOMETRY, HIGHRES_IMU, SET_ATTITUDE_TARGET) → T6
  - RateScheduler → T3
  - Free-running + lockstep modes → T7
  - ENU↔NED conversion at boundary → T2 (helpers) + T7 (use)
  - Lifecycle (start/stop/reset/step + context manager) → T7
  - Unit tests (4) → T2/T3/T4/T7
  - Integration tests (4) → T8/T9/T10/T11
  - Smoke test + plot + QGC bonus → T12

- **Placeholder scan:** none. Two task steps include "if X happens, do Y" debugging hints (T5 step 4, T8 step 2, T9 step 2, T11 step 2) — these are intentional resilience guidance for the implementer, not unspecified work.

- **Type consistency:**
  - `DroneState` and `ImuSample` field names defined in T1 and used identically in T5, T7, T8-11.
  - `DroneBackend.reset/step/get_imu` signatures consistent across T5 (impl), T7 (caller), T7 test (mock).
  - `MavlinkShim` constructor params match across T7 implementation and T8-11 callers.
  - `AttitudeController.compute(...)` keyword-only signature consistent between T4 (impl) and T7 (caller).
  - `RateScheduler.due(t_us)` consistent across T3 and T7.

- **External-API risk areas flagged inline:**
  - T5: `numpy_quad.py` API (function or class) — instructed implementer to inspect first.
  - T6: pymavlink `HIGHRES_IMU_UPDATED_*` constant naming — instructed to verify and fall back to literals.
  - T6: `odometry_send` quaternion order — instructed to verify with `help()`.
  - T11: Windows `time.sleep` resolution may force tolerance bump — noted.

- **No half-finished implementations:** every task ends with green tests and a commit. Smoke test (T12) is the only non-CI step but produces a concrete PNG artifact.
