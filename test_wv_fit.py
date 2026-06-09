import numpy as np
from wv_speed_fit import fit_wv_axis


def test_recovers_known_weathervane():
    rng = np.random.default_rng(0)
    n = 5000
    wcmd = rng.normal(0, 0.3, n); omega = rng.normal(0, 0.5, n); vby = rng.normal(0, 1.5, n)
    a, b, wv_true = 0.8, 1.2, -0.15
    dW = a * wcmd - b * omega + wv_true * vby + rng.normal(0, 0.01, n)
    out = fit_wv_axis(dW, wcmd, omega, vby)
    assert abs(out["wv"] - wv_true) < 0.02
    assert out["r2"] > 0.95


def test_zero_weathervane():
    rng = np.random.default_rng(1)
    n = 3000
    wcmd = rng.normal(0, 0.3, n); omega = rng.normal(0, 0.5, n); vby = rng.normal(0, 1.5, n)
    dW = 0.8 * wcmd - 1.2 * omega + rng.normal(0, 0.01, n)
    assert abs(fit_wv_axis(dW, wcmd, omega, vby)["wv"]) < 0.02
