"""Per-axis yaw moment regression (wv_z weathervane) — decoupled from the joint
3-axis kappa fit that washes wv_z out via kappa/damping collinearity.

Yaw (z) moment balance:
    tau_motor_z = kappa*inertia_z + d_z*w_z - wv_z*v_y
    inertia_z   = wd_z + (I_ratio_y - I_ratio_x)*w_x*w_y
"""
import numpy as np
from aigp.aero_fit import fit_yaw_axis

I_RATIO = np.array([3.7, 1.0, 1.0])


def _synth_yaw(n=500, seed=1, kappa=1.0, d_z=0.05, wv_z=0.08, noise=0.0):
    rng = np.random.default_rng(seed)
    W = rng.uniform(-2.5, 2.5, (n, 3))
    WD = rng.uniform(-4, 4, (n, 3))
    V = rng.uniform(-6, 6, (n, 3))
    inertia_z = WD[:, 2] + (I_RATIO[1] - I_RATIO[0]) * W[:, 0] * W[:, 1]
    tau_z = kappa * inertia_z + d_z * W[:, 2] - wv_z * V[:, 1]
    if noise:
        tau_z = tau_z + rng.normal(0, noise, n)
    return W, WD, V, tau_z


def test_fit_yaw_axis_free_kappa_recovers():
    W, WD, V, tau_z = _synth_yaw(kappa=0.9, d_z=0.04, wv_z=0.08)
    res = fit_yaw_axis(W, WD, V, tau_z, I_RATIO, kappa=None)
    assert set(res.keys()) >= {"kappa", "d_z", "wv_z", "r2", "cond", "n"}
    assert abs(res["kappa"] - 0.9) < 1e-6, res["kappa"]
    assert abs(res["d_z"] - 0.04) < 1e-6, res["d_z"]
    assert abs(res["wv_z"] - 0.08) < 1e-6, res["wv_z"]
    assert res["r2"] > 0.999, res["r2"]


def test_fit_yaw_axis_fixed_kappa_recovers():
    W, WD, V, tau_z = _synth_yaw(kappa=0.9, d_z=0.04, wv_z=0.08)
    res = fit_yaw_axis(W, WD, V, tau_z, I_RATIO, kappa=0.9)
    assert res["kappa"] == 0.9
    assert abs(res["d_z"] - 0.04) < 1e-6, res["d_z"]
    assert abs(res["wv_z"] - 0.08) < 1e-6, res["wv_z"]
    assert res["r2"] > 0.999, res["r2"]


def test_fit_yaw_axis_wv_z_positive_under_noise():
    # fixed-kappa decoupling should recover a positive wv_z even with measurement noise
    W, WD, V, tau_z = _synth_yaw(n=800, seed=3, kappa=1.0, d_z=0.05, wv_z=0.08, noise=0.02)
    res = fit_yaw_axis(W, WD, V, tau_z, I_RATIO, kappa=1.0)
    assert res["wv_z"] > 0, res["wv_z"]
    assert abs(res["wv_z"] - 0.08) < 0.02, res["wv_z"]
