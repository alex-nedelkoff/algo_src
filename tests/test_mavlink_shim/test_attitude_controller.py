"""Tests for AttitudeController — quat error → motor speeds via TRPY mixer."""
import numpy as np
import pytest

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim.attitude_controller import AttitudeController


def _hover_thrust_normalized(params: VehicleParams) -> float:
    """Compute the normalized [0,1] thrust that gives hover with these params.

    VehicleParams has no max_thrust_per_motor_n field; compute it from
    k_thrust and max_omega: max_thrust_per_motor = k_thrust * max_omega^2.
    """
    max_thrust_per_motor = params.k_thrust * params.max_omega ** 2
    weight_n = params.mass * 9.81
    return weight_n / (4.0 * max_thrust_per_motor)


def test_identity_attitude_error_produces_equal_motor_speeds_at_hover_thrust():
    params = VehicleParams()
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
    params = VehicleParams()
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
    # In X-config the allocation matrix (trpy_mixer.py lines 58-61, numpy_quad.py lines 302-305):
    #   τ_pitch = k_t * L/√2 * (-w0² - w1² + w2² + w3²)
    # Index map per sim/dynamics/numpy_quad.py: 0=FR, 1=RR, 2=RL, 3=FL.
    # For +pitch torque (positive ωy_cmd), motors 2 and 3 (RL, FL = left side) spin
    # faster than motors 0 and 1 (FR, RR = right side).
    left_avg = (motor_speeds[2] + motor_speeds[3]) / 2.0
    right_avg = (motor_speeds[0] + motor_speeds[1]) / 2.0
    assert left_avg > right_avg, f"left (RL,FL) should be faster for +pitch about Y: {motor_speeds}"


def test_quaternion_sign_handling_takes_shorter_path():
    """When q_error.w < 0, the controller should flip sign so we go the short way."""
    params = VehicleParams()
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
