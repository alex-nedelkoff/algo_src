import numpy as np
from aigp.attitude_control import BodyRateController

IDENT = np.array([1.0, 0.0, 0.0, 0.0])


def _ctl():
    return BodyRateController(hover_thrust=0.5, k_a=20.0, kp_pos=[6, 6, 6],
                             kd_pos=[4, 4, 4], kp_att=8.0, max_rate=4.0)


def test_hover_equilibrium_gives_hover_thrust_zero_rates():
    w, thrust = _ctl().update(
        pos=np.zeros(3), vel=np.zeros(3), quat=IDENT, omega=np.zeros(3),
        pos_sp=np.zeros(3), vel_sp=np.zeros(3), yaw_sp=0.0,
    )
    assert np.isclose(thrust, 0.5, atol=1e-6)
    assert np.allclose(w, 0.0, atol=1e-6)


def test_below_setpoint_increases_thrust():
    # drone at z=0, setpoint 1 m up (z=-1) -> climb -> thrust > hover
    w, thrust = _ctl().update(
        np.zeros(3), np.zeros(3), IDENT, np.zeros(3),
        pos_sp=np.array([0, 0, -1.0]), vel_sp=np.zeros(3), yaw_sp=0.0,
    )
    assert thrust > 0.5


def test_rates_clamped():
    w, _ = _ctl().update(
        np.zeros(3), np.zeros(3), IDENT, np.zeros(3),
        pos_sp=np.array([20.0, 0, 0]), vel_sp=np.zeros(3), yaw_sp=np.pi,
    )
    assert np.all(np.abs(w) <= 4.0 + 1e-9)
