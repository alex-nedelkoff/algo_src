import numpy as np
from aigp.aero_model import force_features_cl, moment_features_cl, FORCE_COLS_CL, MOMENT_COLS_CL
from aigp.aero_fit import fit_parametric_cl


def _make_synth_cl(n=600, seed=42):
    rng = np.random.default_rng(seed)
    thF = np.zeros(len(FORCE_COLS_CL))
    thF[FORCE_COLS_CL.index("D0_y")] = 0.18
    thF[FORCE_COLS_CL.index("DT_x")] = 0.04
    thF[FORCE_COLS_CL.index("C_z")] = 0.06
    thM = np.zeros(len(MOMENT_COLS_CL))
    thM[MOMENT_COLS_CL.index("d_z")] = 0.05
    thM[MOMENT_COLS_CL.index("wv_z")] = 0.02
    V = rng.uniform(-15, 15, (n, 3))
    W = rng.uniform(-3, 3, (n, 3))
    T = rng.uniform(5, 12, (n,))
    aF = np.array([force_features_cl(V[i], T[i]) @ thF for i in range(n)])
    aM = np.array([moment_features_cl(V[i], W[i]) @ thM for i in range(n)])
    return V, W, T, aF, aM, thF, thM


def test_fit_cl_recovers_params():
    V, W, T, aF, aM, thF, thM = _make_synth_cl()
    res = fit_parametric_cl(V, W, T, aF, aM)
    assert set(res.keys()) >= {"theta_F", "theta_M", "force_cols", "moment_cols", "r2_force", "r2_moment"}
    assert np.allclose(res["theta_F"], thF, atol=1e-6), f"thF mismatch: {res['theta_F']} vs {thF}"
    assert np.allclose(res["theta_M"], thM, atol=1e-6), f"thM mismatch: {res['theta_M']} vs {thM}"
    assert res["r2_force"] > 0.999, f"r2_force={res['r2_force']}"
    assert res["r2_moment"] > 0.999, f"r2_moment={res['r2_moment']}"


def test_fit_cl_robust_to_noise():
    V, W, T, aF, aM, thF, thM = _make_synth_cl(n=600)
    aF_noisy = aF + np.random.default_rng(7).normal(0, 0.05, aF.shape)
    res = fit_parametric_cl(V, W, T, aF_noisy, aM)
    assert np.allclose(res["theta_F"], thF, atol=0.03), f"noisy thF mismatch: {res['theta_F']} vs {thF}"
