import numpy as np
from aigp.dyn_fit import fit_gain, fit_specific_thrust, select_power


def test_fit_gain_through_origin():
    x = np.array([0.0, 1, 2, 3])
    y = 5.0 * x
    assert np.isclose(fit_gain(x, y), 5.0)


def test_fit_gain_zero_input():
    assert fit_gain(np.zeros(3), np.zeros(3)) == 0.0


def test_fit_specific_thrust():
    psum = np.array([1.0, 2, 3, 4])
    up = 3.5 * psum
    assert np.isclose(fit_specific_thrust(psum, up), 3.5)


def test_select_power_picks_quadratic():
    u = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    u_sum = 4 * u
    u_sq_sum = 4 * u ** 2
    up_accel = 6.0 * u_sq_sum
    power, c_T = select_power(u_sum, u_sq_sum, up_accel)
    assert power == 2
    assert np.isclose(c_T, 6.0, atol=1e-6)


from aigp.dyn_fit import angular_accel, fit_axis_torque


def test_angular_accel_linear_gyro():
    t = np.linspace(0, 1, 11)
    gyro = np.stack([2.0 * t, -3.0 * t, np.zeros_like(t)], axis=1)
    aa = angular_accel(t, gyro)
    assert np.allclose(aa[:, 0], 2.0, atol=1e-6)
    assert np.allclose(aa[:, 1], -3.0, atol=1e-6)


def test_fit_axis_torque():
    mix_reg = np.array([0.1, 0.2, -0.1, -0.2])
    ang = 40.0 * mix_reg
    assert np.isclose(fit_axis_torque(mix_reg, ang), 40.0)


from aigp.dyn_fit import fit_motor_lag, fit_drag


def test_fit_motor_lag_recovers_tau():
    t = np.linspace(0, 0.3, 200)
    tau_true = 0.025
    resp = 9.0 * (1 - np.exp(-t / tau_true))
    tau = fit_motor_lag(t, resp)
    assert abs(tau - tau_true) < 0.005


def test_fit_drag():
    vel = np.array([0.0, 1, 2, 3, 4])
    resid_accel = -0.5 * vel
    assert np.isclose(fit_drag(vel, resid_accel), 0.5)


from aigp.dyn_probe import build_campaign


def test_build_campaign_segments():
    seg = build_campaign(n_motors=4, hover_guess=0.3, levels=3, axes=("roll", "pitch", "yaw"))
    names = [s["name"] for s in seg]
    assert "collective" in names
    assert "bump_m0" in names and "bump_m3" in names
    assert "diff_roll" in names and "diff_yaw" in names
    for s in seg:
        for u in s["commands"]:
            assert len(u) == 4 and all(0.0 <= v <= 1.0 for v in u)


from aigp.dyn_fit import fit_from_log


def _coll_sample(seg, t, u, accz):
    return {"segment": seg, "t": t, "u": list(u),
            "imu_acc": [0.0, 0.0, accz], "imu_gyro": [0.0, 0.0, 0.0],
            "vel_ned": [0.0, 0.0, 0.0], "omega": [0.0, 0.0, 0.0]}


def test_fit_from_log_thrust():
    samples = []
    t = 0.0
    for lvl in [0.2, 0.3, 0.4, 0.5]:
        for _ in range(4):
            up = 6.0 * 4 * lvl ** 2
            samples.append(_coll_sample("collective", t, [lvl] * 4, -up))
            t += 0.01
    out = fit_from_log(samples, n_motors=4)
    assert out["power"] == 2
    assert np.isclose(out["c_T"], 6.0, atol=0.3)


def test_fit_from_log_detects_swapped_axis():
    # diff_roll mix produces response only on body gyro axis 2 (the swap)
    samples = []
    t = 0.0
    for k in range(8):
        samples.append({"segment": "diff_roll", "t": t, "u": [0.3, 0.4, 0.4, 0.3],
                        "imu_gyro": [0.0, 0.0, float(k)], "imu_acc": [0, 0, -9]})
        t += 0.01
    out = fit_from_log(samples, n_motors=4)
    assert out["c_L_axis"] == 2
