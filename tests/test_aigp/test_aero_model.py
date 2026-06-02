import numpy as np
from aigp.aero_model import force_features, moment_features, FORCE_COLS, MOMENT_COLS, predict


def test_force_features_shape_and_values():
    v = np.array([2.0, -3.0, 1.0])
    phi = force_features(v)
    assert phi.shape == (3, len(FORCE_COLS))
    iD = FORCE_COLS.index("D_x"); iC = FORCE_COLS.index("C_x")
    assert np.isclose(phi[0, iD], -2.0)
    assert np.isclose(phi[0, iC], -2.0 * 2.0)


def test_force_features_are_linear_in_theta():
    v = np.array([1.0, 0.0, -2.0])
    theta = np.zeros(len(FORCE_COLS)); theta[FORCE_COLS.index("C_z")] = 0.5
    a = force_features(v) @ theta
    assert np.isclose(a[2], 0.5 * (-(-2.0) * 2.0))


def test_moment_features_damping_and_weathervane():
    v = np.array([5.0, 0.0, 0.0]); w = np.array([0.0, 0.0, 0.3])
    phi = moment_features(v, w)
    assert phi.shape == (3, len(MOMENT_COLS))
    idw = MOMENT_COLS.index("d_z")
    assert np.isclose(phi[2, idw], -0.3)
    assert "w_x" in MOMENT_COLS and "w_z" in MOMENT_COLS


def test_predict_roundtrip():
    v = np.array([3.0, 1.0, 0.0]); w = np.array([0.1, 0.0, 0.0])
    thF = np.zeros(len(FORCE_COLS)); thF[FORCE_COLS.index("D_x")] = 0.2
    thM = np.zeros(len(MOMENT_COLS)); thM[MOMENT_COLS.index("d_x")] = 0.05
    aF, aM = predict(v, w, thF, thM)
    assert np.isclose(aF[0], -0.2 * 3.0)
    assert np.isclose(aM[0], -0.05 * 0.1)
