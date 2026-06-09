import numpy as np
from corner_ff import body_vel, weathervane_ff


def test_body_vel_identity():
    assert np.allclose(body_vel(np.eye(3), [1.0, 2.0, 3.0]), [1.0, 2.0, 3.0])


def test_body_vel_yaw90():
    th = np.pi / 2
    R = np.array([[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
    # R is body->world (+90deg about z): a body +x velocity shows as world +y.
    # body_vel(R, world=+y) must recover body +x.
    assert np.allclose(body_vel(R, [0.0, 1.0, 0.0]), [1.0, 0.0, 0.0], atol=1e-9)


def test_weathervane_ff_formula():
    out = weathervane_ff(vbx=-7.0, vby=2.0, roll_wv0=-0.105, roll_wv1=-0.019,
                         yaw_wv=-0.149, g_roll=-2.54, g_yaw=-2.29, gain=1.0, sign=1.0)
    exp_roll = -1.0 * 1.0 * (-0.105 + -0.019 * -7.0) * 2.0 / -2.54
    exp_yaw = -1.0 * 1.0 * -0.149 * 2.0 / -2.29
    assert abs(out[0] - exp_roll) < 1e-12 and abs(out[1] - exp_yaw) < 1e-12


def test_sign_flips():
    a = weathervane_ff(-5.0, 1.5, -0.105, -0.019, -0.149, -2.54, -2.29, sign=1.0)
    b = weathervane_ff(-5.0, 1.5, -0.105, -0.019, -0.149, -2.54, -2.29, sign=-1.0)
    assert np.allclose(a, [-x for x in b])


def test_zero_sideslip_zero_ff():
    assert weathervane_ff(-5.0, 0.0, -0.105, -0.019, -0.149, -2.54, -2.29) == (0.0, 0.0)
