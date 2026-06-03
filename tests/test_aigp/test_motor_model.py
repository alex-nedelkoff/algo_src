import numpy as np
from aigp.motor_model import motor_outputs_to_wrench

# synthetic quad-X geometry (signs only matter for the math test; real signs pinned in calibration)
P = dict(k_f=5.0, k_q=0.1, L=0.14, form="quadratic",
         sx=[+1, -1, +1, -1], sy=[+1, +1, -1, -1], sz=[+1, -1, -1, +1])


def test_equal_outputs_pure_thrust():
    T, tau = motor_outputs_to_wrench([0.5, 0.5, 0.5, 0.5], P)
    assert np.isclose(T, 4 * 5.0 * 0.25)        # 4 * k_f * 0.5^2
    assert np.allclose(tau, 0.0, atol=1e-9)     # symmetric -> no torque


def test_roll_from_thrust_differential():
    # raise sx=+1 motors (0,2), lower sx=-1 (1,3) -> +roll, balanced pitch/yaw
    T, tau = motor_outputs_to_wrench([0.6, 0.4, 0.6, 0.4], P)
    assert tau[0] > 0
    assert np.isclose(tau[1], 0.0, atol=1e-9)
    assert np.isclose(tau[2], 0.0, atol=1e-9)


def test_yaw_from_reaction_torque():
    # raise sz=+1 motors (0,3), lower sz=-1 (1,2) -> +yaw, balanced roll/pitch
    T, tau = motor_outputs_to_wrench([0.6, 0.4, 0.4, 0.6], P)
    assert tau[2] > 0
    assert np.isclose(tau[0], 0.0, atol=1e-9)
    assert np.isclose(tau[1], 0.0, atol=1e-9)


def test_thrust_scales_with_collective():
    T_lo, _ = motor_outputs_to_wrench([0.3, 0.3, 0.3, 0.3], P)
    T_hi, _ = motor_outputs_to_wrench([0.6, 0.6, 0.6, 0.6], P)
    assert T_hi > T_lo


def test_roll_arm_scaling():
    # tau_roll is proportional to L
    Pl = dict(P, L=0.28)
    _, tau1 = motor_outputs_to_wrench([0.6, 0.4, 0.6, 0.4], P)
    _, tau2 = motor_outputs_to_wrench([0.6, 0.4, 0.6, 0.4], Pl)
    assert np.isclose(tau2[0], 2.0 * tau1[0])
