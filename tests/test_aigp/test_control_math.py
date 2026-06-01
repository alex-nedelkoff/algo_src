import numpy as np
from aigp.control_math import G, desired_accel, collective_accel, accel_to_thrust_norm

IDENT = np.array([1.0, 0.0, 0.0, 0.0])


def test_desired_accel_zero_at_setpoint():
    a = desired_accel(pos=np.zeros(3), vel=np.zeros(3),
                      pos_sp=np.zeros(3), vel_sp=np.zeros(3),
                      kp_pos=np.array([6, 6, 6]), kd_pos=np.array([4, 4, 4]))
    assert np.allclose(a, 0.0)


def test_desired_accel_points_to_setpoint():
    # setpoint 1 m "up" in NED (z more negative) -> a_des z negative (upward accel)
    a = desired_accel(np.zeros(3), np.zeros(3), np.array([0, 0, -1.0]), np.zeros(3),
                      np.array([6, 6, 6]), np.array([4, 4, 4]))
    assert a[2] < 0.0


def test_collective_accel_is_g_at_hover_level():
    # zero desired accel, level attitude -> collective magnitude ~ g
    c = collective_accel(np.zeros(3), IDENT)
    assert np.isclose(c, G)


def test_accel_to_thrust_norm_hover_and_climb():
    assert np.isclose(accel_to_thrust_norm(G, hover_thrust=0.5, k_a=20.0), 0.5)
    # one extra unit of k_a worth of accel -> +1.0 normalized (then clipped)
    assert np.isclose(accel_to_thrust_norm(G + 20.0, 0.5, 20.0), 1.0)  # 0.5+1.0 clipped to 1
    assert np.isclose(accel_to_thrust_norm(G + 4.0, 0.5, 20.0), 0.7)


from aigp.control_math import desired_attitude, attitude_error, body_rate_cmd


def _Rx(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def test_desired_attitude_level_north_is_identity():
    R = desired_attitude(np.zeros(3), yaw_sp=0.0)
    assert np.allclose(R, np.eye(3), atol=1e-9)


def test_desired_attitude_yaw_east_points_forward_east():
    R = desired_attitude(np.zeros(3), yaw_sp=np.pi / 2)
    # body x-axis (forward) should point east = world (0,1,0)
    assert np.allclose(R[:, 0], [0, 1, 0], atol=1e-9)


def test_attitude_error_zero_when_aligned():
    R = _Rx(0.3)
    assert np.allclose(attitude_error(R, R), 0.0, atol=1e-9)


def test_body_rate_cmd_corrects_roll():
    R_cur = np.eye(3)
    R_des = _Rx(0.2)            # desired rolled +0.2 about body-x
    w = body_rate_cmd(R_cur, R_des, kp_att=8.0)
    assert w[0] > 0.0 and abs(w[1]) < 1e-9 and abs(w[2]) < 1e-9


def test_body_rate_cmd_clamps():
    R_des = _Rx(1.0)
    w = body_rate_cmd(np.eye(3), R_des, kp_att=50.0, max_rate=2.0)
    assert np.all(np.abs(w) <= 2.0 + 1e-9)
