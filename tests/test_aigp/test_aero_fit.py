import numpy as np
from aigp.aero_model import force_features, moment_features, FORCE_COLS, MOMENT_COLS, predict
from aigp.aero_fit import fit_parametric


def _make_synth(n=400, seed=0):
    rng = np.random.default_rng(seed)
    thF = np.zeros(len(FORCE_COLS)); thF[FORCE_COLS.index("C_x")] = 0.05; thF[FORCE_COLS.index("D_y")] = 0.2
    thM = np.zeros(len(MOMENT_COLS)); thM[MOMENT_COLS.index("d_z")] = 0.04; thM[MOMENT_COLS.index("wv_z")] = 0.01
    V = rng.uniform(-20, 20, (n, 3)); W = rng.uniform(-3, 3, (n, 3))
    aF = np.array([force_features(V[i]) @ thF for i in range(n)])
    aM = np.array([moment_features(V[i], W[i]) @ thM for i in range(n)])
    return V, W, aF, aM, thF, thM


def test_recovers_force_and_moment_params():
    V, W, aF, aM, thF, thM = _make_synth()
    res = fit_parametric(V, W, aF, aM)
    assert np.allclose(res["theta_F"], thF, atol=1e-6)
    assert np.allclose(res["theta_M"], thM, atol=1e-6)
    assert res["r2_force"] > 0.999 and res["r2_moment"] > 0.999


def test_robust_to_noise():
    V, W, aF, aM, thF, thM = _make_synth(n=2000)
    aF = aF + np.random.default_rng(1).normal(0, 0.05, aF.shape)
    res = fit_parametric(V, W, aF, aM)
    assert np.allclose(res["theta_F"], thF, atol=0.02)
