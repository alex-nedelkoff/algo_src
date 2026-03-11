"""Dimension-by-dimension verification tests for the MonoRace observation vector.

Validates that _compute_obs() in GateRaceEnv produces a 24-dim observation
matching the MonoRace paper spec:

| Idx   | Content                          | Dim | Frame           |
|-------|----------------------------------|-----|-----------------|
| 0-2   | Position -> current gate         | 3   | Gate-yaw frame  |
| 3-5   | Velocity                         | 3   | Gate-yaw frame  |
| 6-7   | Roll, Pitch                      | 2   | World frame     |
| 8     | Yaw (gate-relative)              | 1   | drone - gate    |
| 9-11  | Body angular rates               | 3   | Body frame      |
| 12-15 | Motor speeds                     | 4   | Normalized [-1,1] |
| 16-18 | Pos -> next gate (from cur gate) | 3   | Gate-yaw frame  |
| 19    | Relative yaw next->current gate  | 1   | Wrapped [-pi,pi]|
| 20-23 | Previous action                  | 4   | Normalized [-1,1] |

Gate-yaw transform:
    cos_y, sin_y = cos(gate_yaw), sin(gate_yaw)
    rotated_x =  cos_y * dx + sin_y * dy
    rotated_y = -sin_y * dx + cos_y * dy
    z = dz  (unchanged)
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.dynamics.numpy_quad import MOTOR, OMEGA, POS, QUAT, VEL
from sim.dynamics.params import VehicleParams
from sim.envs.gate_race_env import GateRaceEnv, OBS_DIM
from sim.tracks import Track
from sim.types import GateState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Construct a [w, x, y, z] quaternion from Euler angles (ZYX convention).

    Rotation order: yaw (Z) -> pitch (Y) -> roll (X).
    """
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=np.float64)


def _yaw_from_quat(q: np.ndarray) -> float:
    """Extract yaw angle from [w, x, y, z] quaternion (ZYX convention)."""
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _quat_for_yaw(yaw: float) -> np.ndarray:
    """Return [w, x, y, z] quaternion for pure yaw rotation about Z."""
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float64)


def _gate_yaw_rotate(dx: float, dy: float, dz: float, gate_yaw: float) -> np.ndarray:
    """Apply the 2D gate-yaw rotation to a delta vector.

    Returns [rotated_x, rotated_y, dz].
    """
    cos_y = np.cos(gate_yaw)
    sin_y = np.sin(gate_yaw)
    rx = cos_y * dx + sin_y * dy
    ry = -sin_y * dx + cos_y * dy
    return np.array([rx, ry, dz], dtype=np.float64)


def _wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _make_env(
    gate_positions: list[list[float]] | None = None,
    gate_yaws: list[float] | None = None,
    n_envs: int = 1,
) -> GateRaceEnv:
    """Create a GateRaceEnv with gates at known positions and yaw angles.

    Args:
        gate_positions: List of [x, y, z] gate positions. Default 3 gates.
        gate_yaws: List of yaw angles (radians) for each gate. Default 0s.
        n_envs: Number of parallel environments.

    Returns:
        A freshly-constructed (but not yet reset) GateRaceEnv.
    """
    if gate_positions is None:
        gate_positions = [
            [5.0, 0.0, 2.0],
            [10.0, 5.0, 2.0],
            [5.0, 10.0, 2.0],
        ]
    if gate_yaws is None:
        gate_yaws = [0.0] * len(gate_positions)

    gates = [
        GateState(
            position=np.array(p, dtype=np.float64),
            orientation=_quat_for_yaw(yaw),
        )
        for p, yaw in zip(gate_positions, gate_yaws)
    ]
    track = Track(gates)
    return GateRaceEnv(
        track=track,
        n_envs=n_envs,
        max_steps=10000,
        ceiling=50.0,
    )


def _set_state(
    env: GateRaceEnv,
    env_idx: int,
    pos: np.ndarray | None = None,
    vel: np.ndarray | None = None,
    quat: np.ndarray | None = None,
    omega: np.ndarray | None = None,
    motor: np.ndarray | None = None,
) -> None:
    """Overwrite specific fields in the env's internal state for env_idx.

    Fields that are None are left unchanged.
    """
    if pos is not None:
        env._states[env_idx, POS] = pos
    if vel is not None:
        env._states[env_idx, VEL] = vel
    if quat is not None:
        env._states[env_idx, QUAT] = quat
    if omega is not None:
        env._states[env_idx, OMEGA] = omega
    if motor is not None:
        env._states[env_idx, MOTOR] = motor


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestObsShape:
    """Observation shape must be (24,) for single env, (N,24) for multi-env."""

    def test_single_env_shape(self) -> None:
        env = _make_env(n_envs=1)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (24,), f"Expected (24,), got {obs.shape}"

    def test_multi_env_shape(self) -> None:
        n = 4
        env = _make_env(n_envs=n)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (n, 24), f"Expected ({n}, 24), got {obs.shape}"

    def test_obs_dim_constant_is_24(self) -> None:
        assert OBS_DIM == 24, f"OBS_DIM should be 24, got {OBS_DIM}"


class TestObsGatePositionIdentityGate:
    """Gate at origin with yaw=0.  Drone at [3,4,5].

    dpos = drone - gate = [3,4,5].  With gate yaw=0 the rotation is identity,
    so obs[0:3] should be [3, 4, 5].
    """

    def test_identity_gate_position(self) -> None:
        env = _make_env(
            gate_positions=[[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
            gate_yaws=[0.0, 0.0],
        )
        env.reset(seed=0)
        _set_state(env, 0, pos=np.array([3.0, 4.0, 5.0]))
        obs = env._compute_obs()
        obs_flat = obs.flatten() if obs.ndim > 1 else obs
        np.testing.assert_allclose(
            obs_flat[0:3], [3.0, 4.0, 5.0], atol=1e-5,
            err_msg="obs[0:3] gate-rel position with identity gate yaw",
        )


class TestObsGatePositionRotatedGate:
    """Gate at [10,0,2] with 90-deg yaw.  Drone at [10,5,4].

    dpos = [0, 5, 2].
    Gate yaw = pi/2 => cos=0, sin=1.
    rotated_x =  0*0 + 1*5 = 5
    rotated_y = -1*0 + 0*5 = 0
    obs[0:3] = [5, 0, 2].
    """

    def test_rotated_gate_position(self) -> None:
        env = _make_env(
            gate_positions=[[10.0, 0.0, 2.0], [20.0, 0.0, 2.0]],
            gate_yaws=[np.pi / 2, 0.0],
        )
        env.reset(seed=0)
        _set_state(
            env, 0,
            pos=np.array([10.0, 5.0, 4.0]),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        obs = env._compute_obs()
        obs_flat = obs.flatten() if obs.ndim > 1 else obs
        np.testing.assert_allclose(
            obs_flat[0:3], [5.0, 0.0, 2.0], atol=1e-5,
            err_msg="obs[0:3] with 90-deg gate yaw",
        )


class TestObsVelocityGateFrame:
    """Velocity [10, 0, 5] with gate yaw=pi/2.

    vx_gate =  cos(pi/2)*10 + sin(pi/2)*0 = 0
    vy_gate = -sin(pi/2)*10 + cos(pi/2)*0 = -10
    obs[3:6] = [0, -10, 5].
    """

    def test_velocity_rotated_by_gate_yaw(self) -> None:
        env = _make_env(
            gate_positions=[[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
            gate_yaws=[np.pi / 2, 0.0],
        )
        env.reset(seed=0)
        _set_state(
            env, 0,
            vel=np.array([10.0, 0.0, 5.0]),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        obs = env._compute_obs()
        obs_flat = obs.flatten() if obs.ndim > 1 else obs
        np.testing.assert_allclose(
            obs_flat[3:6], [0.0, -10.0, 5.0], atol=1e-5,
            err_msg="obs[3:6] velocity in gate-yaw frame",
        )


class TestObsEulerAngles:
    """Set known roll=0.1, pitch=0.2; obs[6]=roll, obs[7]=pitch."""

    def test_euler_roll_pitch(self) -> None:
        roll, pitch, yaw = 0.1, 0.2, 0.0
        env = _make_env(
            gate_positions=[[5.0, 0.0, 2.0], [15.0, 0.0, 2.0]],
            gate_yaws=[0.0, 0.0],
        )
        env.reset(seed=0)
        q = _quat_from_euler(roll, pitch, yaw)
        _set_state(env, 0, quat=q)
        obs = env._compute_obs()
        obs_flat = obs.flatten() if obs.ndim > 1 else obs
        np.testing.assert_allclose(
            obs_flat[6], roll, atol=1e-4,
            err_msg="obs[6] should be roll",
        )
        np.testing.assert_allclose(
            obs_flat[7], pitch, atol=1e-4,
            err_msg="obs[7] should be pitch",
        )

    def test_euler_angles_larger_values(self) -> None:
        """Test with larger roll/pitch to ensure extraction is accurate."""
        roll, pitch = 0.5, -0.3
        env = _make_env(
            gate_positions=[[5.0, 0.0, 2.0], [15.0, 0.0, 2.0]],
            gate_yaws=[0.0, 0.0],
        )
        env.reset(seed=0)
        q = _quat_from_euler(roll, pitch, 0.0)
        _set_state(env, 0, quat=q)
        obs = env._compute_obs()
        obs_flat = obs.flatten() if obs.ndim > 1 else obs
        np.testing.assert_allclose(obs_flat[6], roll, atol=1e-4)
        np.testing.assert_allclose(obs_flat[7], pitch, atol=1e-4)


class TestObsYawGateRelative:
    """Drone yaw=pi/4, gate yaw=pi/2 => obs[8] = pi/4 - pi/2 = -pi/4."""

    def test_yaw_gate_relative(self) -> None:
        drone_yaw = np.pi / 4
        gate_yaw = np.pi / 2
        env = _make_env(
            gate_positions=[[5.0, 0.0, 2.0], [15.0, 0.0, 2.0]],
            gate_yaws=[gate_yaw, 0.0],
        )
        env.reset(seed=0)
        q = _quat_from_euler(0.0, 0.0, drone_yaw)
        _set_state(env, 0, quat=q)
        obs = env._compute_obs()
        obs_flat = obs.flatten() if obs.ndim > 1 else obs
        expected = _wrap_angle(drone_yaw - gate_yaw)  # -pi/4
        np.testing.assert_allclose(
            obs_flat[8], expected, atol=1e-4,
            err_msg="obs[8] gate-relative yaw",
        )


class TestObsYawWrapping:
    """Drone yaw=3pi/4, gate yaw=-3pi/4.

    Raw diff = 3pi/4 - (-3pi/4) = 3pi/2.
    Wrapped to [-pi, pi] => -pi/2.
    """

    def test_yaw_wrapping(self) -> None:
        drone_yaw = 3 * np.pi / 4
        gate_yaw = -3 * np.pi / 4
        env = _make_env(
            gate_positions=[[5.0, 0.0, 2.0], [15.0, 0.0, 2.0]],
            gate_yaws=[gate_yaw, 0.0],
        )
        env.reset(seed=0)
        q = _quat_from_euler(0.0, 0.0, drone_yaw)
        _set_state(env, 0, quat=q)
        obs = env._compute_obs()
        obs_flat = obs.flatten() if obs.ndim > 1 else obs
        expected = _wrap_angle(drone_yaw - gate_yaw)  # -pi/2
        np.testing.assert_allclose(
            obs_flat[8], expected, atol=1e-4,
            err_msg="obs[8] wrapped yaw difference",
        )


class TestObsAngularRates:
    """Body angular rates omega=[1,2,3] => obs[9:12] = [1,2,3]."""

    def test_angular_rates_direct(self) -> None:
        env = _make_env(
            gate_positions=[[5.0, 0.0, 2.0], [15.0, 0.0, 2.0]],
            gate_yaws=[0.0, 0.0],
        )
        env.reset(seed=0)
        _set_state(env, 0, omega=np.array([1.0, 2.0, 3.0]))
        obs = env._compute_obs()
        obs_flat = obs.flatten() if obs.ndim > 1 else obs
        np.testing.assert_allclose(
            obs_flat[9:12], [1.0, 2.0, 3.0], atol=1e-5,
            err_msg="obs[9:12] should equal state angular rates",
        )


class TestObsMotorNormalization:
    """Motor speed normalization: (motor / max_omega) * 2 - 1.

    motors at max_omega => [1, 1, 1, 1]
    motors at 0         => [-1, -1, -1, -1]
    motors at max_omega/2 => [0, 0, 0, 0]
    """

    def test_motors_at_max(self) -> None:
        env = _make_env(
            gate_positions=[[5.0, 0.0, 2.0], [15.0, 0.0, 2.0]],
            gate_yaws=[0.0, 0.0],
        )
        env.reset(seed=0)
        max_omega = env.params.max_omega
        _set_state(env, 0, motor=np.full(4, max_omega))
        obs = env._compute_obs()
        obs_flat = obs.flatten() if obs.ndim > 1 else obs
        np.testing.assert_allclose(
            obs_flat[12:16], [1.0, 1.0, 1.0, 1.0], atol=1e-5,
            err_msg="Motors at max should normalize to +1",
        )

    def test_motors_at_zero(self) -> None:
        env = _make_env(
            gate_positions=[[5.0, 0.0, 2.0], [15.0, 0.0, 2.0]],
            gate_yaws=[0.0, 0.0],
        )
        env.reset(seed=0)
        _set_state(env, 0, motor=np.zeros(4))
        obs = env._compute_obs()
        obs_flat = obs.flatten() if obs.ndim > 1 else obs
        np.testing.assert_allclose(
            obs_flat[12:16], [-1.0, -1.0, -1.0, -1.0], atol=1e-5,
            err_msg="Motors at zero should normalize to -1",
        )

    def test_motors_at_half(self) -> None:
        env = _make_env(
            gate_positions=[[5.0, 0.0, 2.0], [15.0, 0.0, 2.0]],
            gate_yaws=[0.0, 0.0],
        )
        env.reset(seed=0)
        max_omega = env.params.max_omega
        _set_state(env, 0, motor=np.full(4, max_omega / 2))
        obs = env._compute_obs()
        obs_flat = obs.flatten() if obs.ndim > 1 else obs
        np.testing.assert_allclose(
            obs_flat[12:16], [0.0, 0.0, 0.0, 0.0], atol=1e-5,
            err_msg="Motors at half max should normalize to 0",
        )


class TestObsNextGatePosition:
    """Relative position from current gate to next gate, in current gate yaw frame.

    obs[16:19] = gate_yaw_rotate(next_gate_pos - cur_gate_pos, cur_gate_yaw).
    """

    def test_next_gate_no_rotation(self) -> None:
        """current gate at [0,0,2], next at [10,5,3], yaw=0 => [10, 5, 1]."""
        env = _make_env(
            gate_positions=[[0.0, 0.0, 2.0], [10.0, 5.0, 3.0]],
            gate_yaws=[0.0, 0.0],
        )
        env.reset(seed=0)
        obs = env._compute_obs()
        obs_flat = obs.flatten() if obs.ndim > 1 else obs
        np.testing.assert_allclose(
            obs_flat[16:19], [10.0, 5.0, 1.0], atol=1e-5,
            err_msg="obs[16:19] next gate relative, no rotation",
        )

    def test_next_gate_with_rotation(self) -> None:
        """current gate at [0,0,2], next at [10,5,3], yaw=pi/2.

        d = [10, 5, 1]. With yaw=pi/2 (cos=0, sin=1):
        rx =  0*10 + 1*5 = 5
        ry = -1*10 + 0*5 = -10
        obs[16:19] = [5, -10, 1].
        """
        env = _make_env(
            gate_positions=[[0.0, 0.0, 2.0], [10.0, 5.0, 3.0]],
            gate_yaws=[np.pi / 2, 0.0],
        )
        env.reset(seed=0)
        obs = env._compute_obs()
        obs_flat = obs.flatten() if obs.ndim > 1 else obs
        np.testing.assert_allclose(
            obs_flat[16:19], [5.0, -10.0, 1.0], atol=1e-5,
            err_msg="obs[16:19] next gate relative, 90-deg gate yaw",
        )


class TestObsNextGateYaw:
    """Relative yaw between next gate and current gate.

    obs[19] = wrap(next_gate_yaw - current_gate_yaw).
    """

    def test_next_gate_yaw_simple(self) -> None:
        """cur yaw=0, next yaw=pi/2 => obs[19] = pi/2."""
        env = _make_env(
            gate_positions=[[0.0, 0.0, 2.0], [10.0, 0.0, 2.0]],
            gate_yaws=[0.0, np.pi / 2],
        )
        env.reset(seed=0)
        obs = env._compute_obs()
        obs_flat = obs.flatten() if obs.ndim > 1 else obs
        np.testing.assert_allclose(
            obs_flat[19], np.pi / 2, atol=1e-4,
            err_msg="obs[19] next gate relative yaw",
        )

    def test_next_gate_yaw_wrapping(self) -> None:
        """cur yaw=-3pi/4, next yaw=3pi/4 => raw diff=3pi/2, wrapped=-pi/2."""
        env = _make_env(
            gate_positions=[[0.0, 0.0, 2.0], [10.0, 0.0, 2.0]],
            gate_yaws=[-3 * np.pi / 4, 3 * np.pi / 4],
        )
        env.reset(seed=0)
        obs = env._compute_obs()
        obs_flat = obs.flatten() if obs.ndim > 1 else obs
        expected = _wrap_angle(3 * np.pi / 4 - (-3 * np.pi / 4))  # -pi/2
        np.testing.assert_allclose(
            obs_flat[19], expected, atol=1e-4,
            err_msg="obs[19] wrapped next-gate yaw difference",
        )


class TestObsActionHistoryInitial:
    """After reset (no prior action), obs[20:24] should be zeros (no previous command).

    The prev_action is stored directly as the normalized [-1, 1] action input.
    On reset, prev_actions are initialized to zeros.
    """

    def test_action_history_after_reset(self) -> None:
        env = _make_env(
            gate_positions=[[5.0, 0.0, 2.0], [15.0, 0.0, 2.0]],
            gate_yaws=[0.0, 0.0],
        )
        obs, _ = env.reset(seed=0)
        obs_flat = obs.flatten() if obs.ndim > 1 else obs
        prev_action = obs_flat[20:24]
        # All four values should be zero after reset
        np.testing.assert_allclose(
            prev_action, [0.0, 0.0, 0.0, 0.0], atol=1e-5,
            err_msg="prev_action after reset should be 0 (unset)",
        )


class TestObsActionHistoryAfterStep:
    """After one step with a known action, obs[20:24] should reflect it directly.

    Action space is [-1, 1] (normalized ESC commands); prev_action in obs
    stores these values directly without further transformation.
    """

    def test_action_reflected_after_step(self) -> None:
        env = _make_env(
            gate_positions=[[5.0, 0.0, 2.0], [15.0, 0.0, 2.0]],
            gate_yaws=[0.0, 0.0],
        )
        env.reset(seed=0)

        # Action space is now [-1, 1]; use mid-range (0.0)
        action = np.full(4, 0.0, dtype=np.float32)
        obs, _, _, _, _ = env.step(action)

        obs_flat = obs.flatten() if obs.ndim > 1 else obs
        prev_action = obs_flat[20:24]

        # prev_action in obs should directly reflect the normalized action
        np.testing.assert_allclose(
            prev_action, [0.0, 0.0, 0.0, 0.0], atol=0.05,
            err_msg="prev_action at u=0 should be 0",
        )

    def test_action_at_max(self) -> None:
        env = _make_env(
            gate_positions=[[5.0, 0.0, 2.0], [15.0, 0.0, 2.0]],
            gate_yaws=[0.0, 0.0],
        )
        env.reset(seed=0)

        action = np.full(4, 1.0, dtype=np.float32)
        obs, _, _, _, _ = env.step(action)

        obs_flat = obs.flatten() if obs.ndim > 1 else obs
        prev_action = obs_flat[20:24]
        np.testing.assert_allclose(
            prev_action, [1.0, 1.0, 1.0, 1.0], atol=0.05,
            err_msg="prev_action at u=+1 should be +1",
        )

    def test_action_at_min(self) -> None:
        env = _make_env(
            gate_positions=[[5.0, 0.0, 2.0], [15.0, 0.0, 2.0]],
            gate_yaws=[0.0, 0.0],
        )
        env.reset(seed=0)

        action = -np.ones(4, dtype=np.float32)
        obs, _, terminated, _, _ = env.step(action)

        term_val = terminated.item() if hasattr(terminated, "item") else terminated
        if not term_val:
            obs_flat = obs.flatten() if obs.ndim > 1 else obs
            prev_action = obs_flat[20:24]
            np.testing.assert_allclose(
                prev_action, [-1.0, -1.0, -1.0, -1.0], atol=0.05,
                err_msg="prev_action at u=-1 should be -1",
            )


class TestObsAllDimsNonzero:
    """In a general scenario, no dimension should be accidentally zero.

    Construct a setup where every observation dimension has a non-trivial
    value and verify none of them are zero (except those that should be,
    like padding).
    """

    def test_general_nonzero(self) -> None:
        # Gate 0 at [5, 3, 2] with yaw=0.3
        # Gate 1 at [12, -2, 3] with yaw=0.8
        env = _make_env(
            gate_positions=[[5.0, 3.0, 2.0], [12.0, -2.0, 3.0]],
            gate_yaws=[0.3, 0.8],
        )
        env.reset(seed=0)

        # Set a rich state
        drone_yaw = 0.6
        _set_state(
            env, 0,
            pos=np.array([3.0, 1.0, 4.0]),
            vel=np.array([2.0, -1.5, 0.5]),
            quat=_quat_from_euler(0.15, -0.1, drone_yaw),
            omega=np.array([0.3, -0.4, 0.2]),
            motor=np.full(4, env.params.max_omega * 0.7),
        )

        # Provide a previous action by stepping first (normalized [-1, 1])
        action = np.full(4, 0.3, dtype=np.float32)
        obs, _, terminated, _, _ = env.step(action)

        term_val = terminated.item() if hasattr(terminated, "item") else terminated
        if term_val:
            pytest.skip("Env terminated before we could check obs")

        obs_flat = obs.flatten() if obs.ndim > 1 else obs

        # Check obs dimensions 0-23 all exist
        assert obs_flat.shape == (24,), f"Expected 24 dims, got {obs_flat.shape}"
        assert np.all(np.isfinite(obs_flat)), f"Non-finite values: {obs_flat}"

        # Dims 0:2 (gate-rel pos XY) - should be nonzero because drone is offset
        assert obs_flat[0] != pytest.approx(0.0, abs=1e-3), \
            f"obs[0] gate_rel_x unexpectedly zero"
        assert obs_flat[1] != pytest.approx(0.0, abs=1e-3), \
            f"obs[1] gate_rel_y unexpectedly zero"

        # Dims 3:5 (velocity) - should be nonzero due to set velocity
        assert obs_flat[3] != pytest.approx(0.0, abs=1e-3), \
            f"obs[3] vel_x unexpectedly zero"

        # Dims 6:8 (roll, pitch, yaw_rel) - set to nonzero values
        # Note: dynamics step may perturb these, so just check finiteness
        assert obs_flat[6] != pytest.approx(0.0, abs=1e-3) or \
               obs_flat[7] != pytest.approx(0.0, abs=1e-3), \
            "At least one of roll/pitch should be nonzero"

        # Dim 8 (yaw relative) - drone yaw != gate yaw
        assert obs_flat[8] != pytest.approx(0.0, abs=1e-3), \
            f"obs[8] relative yaw unexpectedly zero"

        # Dims 9:12 (angular rates)
        for d in range(9, 12):
            assert obs_flat[d] != pytest.approx(0.0, abs=1e-2), \
                f"obs[{d}] angular rate unexpectedly zero"

        # Dims 12:16 (motors normalized) - at 0.7*max, should be nonzero
        for d in range(12, 16):
            assert obs_flat[d] != pytest.approx(0.0, abs=1e-2), \
                f"obs[{d}] motor unexpectedly zero"

        # Dims 16:19 (next gate relative pos)
        assert obs_flat[16] != pytest.approx(0.0, abs=1e-3), \
            f"obs[16] next_gate_x unexpectedly zero"

        # Dim 19 (next gate yaw relative)
        assert obs_flat[19] != pytest.approx(0.0, abs=1e-3), \
            f"obs[19] next gate yaw diff unexpectedly zero"

        # Dims 20:24 (prev action) - should be nonzero since we stepped
        for d in range(20, 24):
            assert obs_flat[d] != pytest.approx(0.0, abs=1e-2), \
                f"obs[{d}] prev action unexpectedly zero"


class TestObsGatePositionSignConvention:
    """Verify the sign convention: obs[0:3] = drone_pos - gate_pos in gate frame.

    When the drone is in front of the gate (closer to the gate along its
    forward axis), the along-gate-forward component should be negative
    if dpos is drone-gate and the gate faces +x. But with the gate-yaw
    rotation, the interpretation changes. We just verify consistency.
    """

    def test_drone_behind_gate(self) -> None:
        """Drone at [2,0,2], gate at [5,0,2] with yaw=0.

        dpos = [-3, 0, 0]. With yaw=0: obs[0:3] = [-3, 0, 0].
        """
        env = _make_env(
            gate_positions=[[5.0, 0.0, 2.0], [15.0, 0.0, 2.0]],
            gate_yaws=[0.0, 0.0],
        )
        env.reset(seed=0)
        _set_state(env, 0, pos=np.array([2.0, 0.0, 2.0]))
        obs = env._compute_obs()
        obs_flat = obs.flatten() if obs.ndim > 1 else obs
        np.testing.assert_allclose(
            obs_flat[0:3], [-3.0, 0.0, 0.0], atol=1e-5,
            err_msg="Drone behind gate: obs[0] should be negative",
        )

    def test_drone_above_gate(self) -> None:
        """Drone at [5,0,5], gate at [5,0,2] with yaw=0.

        dpos = [0, 0, 3]. With yaw=0: obs[0:3] = [0, 0, 3].
        """
        env = _make_env(
            gate_positions=[[5.0, 0.0, 2.0], [15.0, 0.0, 2.0]],
            gate_yaws=[0.0, 0.0],
        )
        env.reset(seed=0)
        _set_state(env, 0, pos=np.array([5.0, 0.0, 5.0]))
        obs = env._compute_obs()
        obs_flat = obs.flatten() if obs.ndim > 1 else obs
        np.testing.assert_allclose(
            obs_flat[0:3], [0.0, 0.0, 3.0], atol=1e-5,
            err_msg="Drone above gate: obs[2] should be +3",
        )


class TestObsVelocityZeroGateYaw:
    """With gate yaw=0, velocity should pass through unrotated."""

    def test_velocity_identity_rotation(self) -> None:
        env = _make_env(
            gate_positions=[[5.0, 0.0, 2.0], [15.0, 0.0, 2.0]],
            gate_yaws=[0.0, 0.0],
        )
        env.reset(seed=0)
        _set_state(env, 0, vel=np.array([3.0, -2.0, 1.0]))
        obs = env._compute_obs()
        obs_flat = obs.flatten() if obs.ndim > 1 else obs
        np.testing.assert_allclose(
            obs_flat[3:6], [3.0, -2.0, 1.0], atol=1e-5,
            err_msg="Velocity with zero gate yaw should be unrotated",
        )


class TestObsVelocityZUnchanged:
    """Vertical velocity component should always pass through unchanged,
    regardless of gate yaw.
    """

    def test_vz_unchanged_with_rotation(self) -> None:
        vz = 7.77
        env = _make_env(
            gate_positions=[[5.0, 0.0, 2.0], [15.0, 0.0, 2.0]],
            gate_yaws=[1.23, 0.0],  # arbitrary nonzero yaw
        )
        env.reset(seed=0)
        _set_state(env, 0, vel=np.array([0.0, 0.0, vz]))
        obs = env._compute_obs()
        obs_flat = obs.flatten() if obs.ndim > 1 else obs
        np.testing.assert_allclose(
            obs_flat[5], vz, atol=1e-5,
            err_msg="obs[5] (vz) should be unchanged by gate yaw rotation",
        )


class TestObsGatePositionNonTrivialYaw:
    """Test with a 45-degree gate yaw for a less trivial rotation.

    Gate at [0,0,0], yaw=pi/4.  Drone at [1,1,0].
    dpos = [1, 1, 0].
    cos(pi/4) = sin(pi/4) = sqrt(2)/2 ~ 0.7071.
    rx = 0.7071*1 + 0.7071*1 = sqrt(2) ~ 1.4142
    ry = -0.7071*1 + 0.7071*1 = 0
    obs[0:3] = [sqrt(2), 0, 0].
    """

    def test_45_deg_gate_yaw(self) -> None:
        env = _make_env(
            gate_positions=[[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
            gate_yaws=[np.pi / 4, 0.0],
        )
        env.reset(seed=0)
        _set_state(env, 0, pos=np.array([1.0, 1.0, 0.0]))
        obs = env._compute_obs()
        obs_flat = obs.flatten() if obs.ndim > 1 else obs
        np.testing.assert_allclose(
            obs_flat[0:3], [np.sqrt(2), 0.0, 0.0], atol=1e-4,
            err_msg="obs[0:3] with 45-deg gate yaw",
        )


class TestObsConsistencyAcrossEnvs:
    """Parallel environments with same state should produce identical obs."""

    def test_parallel_obs_identical(self) -> None:
        n = 3
        env = _make_env(
            gate_positions=[[5.0, 0.0, 2.0], [15.0, 0.0, 2.0]],
            gate_yaws=[0.3, 0.7],
            n_envs=n,
        )
        env.reset(seed=0)

        # Set identical state for all envs
        for i in range(n):
            _set_state(
                env, i,
                pos=np.array([3.0, 1.0, 4.0]),
                vel=np.array([2.0, -1.0, 0.5]),
                quat=_quat_from_euler(0.1, 0.2, 0.3),
                omega=np.array([0.5, -0.3, 0.1]),
                motor=np.full(4, env.params.max_omega * 0.6),
            )
            env._gate_indices[i] = 0

        obs = env._compute_obs()
        assert obs.shape == (n, 24)
        np.testing.assert_allclose(
            obs[0], obs[1], atol=1e-10,
            err_msg="Parallel envs with same state should have same obs",
        )
        np.testing.assert_allclose(
            obs[1], obs[2], atol=1e-10,
            err_msg="Parallel envs with same state should have same obs",
        )


class TestObsDtype:
    """Observation should be float32."""

    def test_obs_is_float32(self) -> None:
        env = _make_env()
        obs, _ = env.reset(seed=0)
        assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"


class TestObsFinite:
    """All observation values should be finite after reset and after step."""

    def test_finite_after_reset(self) -> None:
        env = _make_env()
        obs, _ = env.reset(seed=0)
        assert np.all(np.isfinite(obs)), f"Non-finite obs after reset: {obs}"

    def test_finite_after_step(self) -> None:
        env = _make_env()
        env.reset(seed=0)
        action = np.full(4, 0.0, dtype=np.float32)  # mid-range normalized action
        obs, _, _, _, _ = env.step(action)
        obs_flat = obs.flatten() if obs.ndim > 1 else obs
        assert np.all(np.isfinite(obs_flat)), f"Non-finite obs after step: {obs_flat}"
