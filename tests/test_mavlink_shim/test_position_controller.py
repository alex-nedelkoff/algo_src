"""SE(3) PositionController unit tests — pure math, no shim/UDP."""
from __future__ import annotations

import numpy as np

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim.backend import DroneState
from sim.pybullet.mavlink_shim.position_controller import PositionController
from sim.pybullet.mavlink_shim.position_target import PositionTarget


def _identity_state(pos=(0.0, 0.0, 0.0)) -> DroneState:
    return DroneState(
        pos_enu=np.asarray(pos, dtype=np.float64),
        vel_enu=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3),
        motor_speed=np.zeros(4),
        timestamp_us=0,
    )


def _hover_target(pos_enu) -> PositionTarget:
    return PositionTarget(
        pos_enu=np.asarray(pos_enu, dtype=np.float64),
        vel_enu=np.zeros(3), accel_enu=np.zeros(3), yaw_enu=0.0,
        use_position=True, use_velocity=True, use_accel=True, use_yaw=True,
    )


def test_at_target_hovering_outputs_hover_thrust_identity_attitude():
    """When pos error = 0 and zero velocity, target attitude = identity and
    thrust normalised equals m·g / max_thrust."""
    params = VehicleParams()  # CrazyFlie defaults
    ctrl = PositionController(params=params)
    target = _hover_target([0.0, 0.0, 1.0])
    state = _identity_state(pos=(0.0, 0.0, 1.0))

    q_des, thrust_norm = ctrl.compute(target=target, state=state)

    # Identity quaternion is wxyz=(1,0,0,0).
    np.testing.assert_allclose(q_des, [1.0, 0.0, 0.0, 0.0], atol=1e-6)
    max_thrust = 4 * params.k_thrust * params.max_omega ** 2
    expected = params.mass * 9.81 / max_thrust
    assert abs(thrust_norm - expected) < 1e-4


def test_horizontal_error_tilts_in_correct_direction():
    """Drone at origin, target at +x: should tilt nose-forward (pitch + about Y)
    so the thrust vector pushes the drone toward +x. In wxyz: pitch + means q_y > 0."""
    params = VehicleParams()
    ctrl = PositionController(params=params)
    target = _hover_target([2.0, 0.0, 1.0])
    state = _identity_state(pos=(0.0, 0.0, 1.0))

    q_des, _ = ctrl.compute(target=target, state=state)
    # Tilting forward (about +Y body, i.e. pitch +): q_des should have q_y > 0.
    # We don't assert magnitude — just sign of the pitch component.
    assert q_des[2] > 0.01, f"expected forward tilt (q_y>0), got {q_des}"


def test_target_below_reduces_thrust():
    """Target z=0.5 below current z=1.0 → desired upward thrust < hover (we want to fall)."""
    params = VehicleParams()
    ctrl = PositionController(params=params)
    target = _hover_target([0.0, 0.0, 0.5])
    state = _identity_state(pos=(0.0, 0.0, 1.0))

    _, thrust_norm = ctrl.compute(target=target, state=state)
    max_thrust = 4 * params.k_thrust * params.max_omega ** 2
    hover = params.mass * 9.81 / max_thrust
    assert thrust_norm < hover, "expected below-hover thrust to descend"
    assert thrust_norm >= 0.0, "thrust normalised must be non-negative"
