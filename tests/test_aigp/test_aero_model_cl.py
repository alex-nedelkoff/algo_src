import numpy as np
from aigp.aero_model import (
    FORCE_COLS_CL, MOMENT_COLS_CL,
    force_features_cl, moment_features_cl, predict_cl,
)


def test_force_features_cl_thrust_coupling():
    v = np.array([2.0, -3.0, 1.0])
    T = 5.0
    phi = force_features_cl(v, T)
    assert phi.shape == (3, len(FORCE_COLS_CL))

    idx = FORCE_COLS_CL.index
    # D0_x term: -v[0] = -2
    assert np.isclose(phi[0, idx("D0_x")], -2.0)
    # DT_x term: -T * v[0] = -10
    assert np.isclose(phi[0, idx("DT_x")], -10.0)
    # C_x term: -v[0]*|v[0]| = -2*2 = -4
    assert np.isclose(phi[0, idx("C_x")], -4.0)


def test_moment_features_cl_weathervane():
    v = np.array([4.0, 1.0, 0.0])
    w = np.array([0.0, 0.0, 0.3])
    phi = moment_features_cl(v, w)
    assert phi.shape == (3, len(MOMENT_COLS_CL))

    idx = MOMENT_COLS_CL.index
    # yaw damping: -omega[2] = -0.3
    assert np.isclose(phi[2, idx("d_z")], -0.3)
    # yaw weathervane from v_y: phi[2, wv_z] = v[1] = 1
    assert np.isclose(phi[2, idx("wv_z")], v[1])


def test_predict_cl_roundtrip():
    v = np.array([3.0, 1.0, -1.0])
    w = np.array([0.1, -0.2, 0.3])
    T = 8.0
    thF = np.zeros(len(FORCE_COLS_CL))
    thF[FORCE_COLS_CL.index("D0_x")] = 0.3
    thF[FORCE_COLS_CL.index("DT_z")] = 0.05
    thM = np.zeros(len(MOMENT_COLS_CL))
    thM[MOMENT_COLS_CL.index("d_y")] = 0.07
    thM[MOMENT_COLS_CL.index("wv_x")] = 0.02

    aF, aM = predict_cl(v, w, T, thF, thM)
    expected_F = force_features_cl(v, T) @ thF
    expected_M = moment_features_cl(v, w) @ thM
    assert np.allclose(aF, expected_F)
    assert np.allclose(aM, expected_M)
