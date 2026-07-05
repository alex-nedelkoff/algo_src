"""Unit tests for the VQ2 position/velocity KF."""
import math

import numpy as np

from vq2.eskf import PosVelKF, accel_level, GRAVITY


def test_accel_level_at_rest_level():
    a = accel_level((0, 0, -GRAVITY), 0.0, 0.0)
    assert np.allclose(a, 0, atol=1e-9)


def test_accel_level_at_rest_spawn_tilt():
    # measured VQ2 spawn: xacc -3.00, zacc -9.34 at pitch -17.8 deg
    p = math.radians(-17.8)
    a = accel_level((-3.0, 0.0, -9.34), 0.0, p)
    assert np.linalg.norm(a) < 0.05


def test_predict_integrates_thrust():
    kf = PosVelKF()
    # 1 m/s^2 upward (level frame z down => a_z = -1)
    for _ in range(100):
        kf.predict(np.array([0, 0, -1.0]), 0.01)
    assert abs(kf.v[2] - (-1.0)) < 1e-6
    assert abs(kf.p[2] - (-0.5)) < 0.01
    assert kf.P[3, 3] > 0.01 * 0  # covariance grew


def test_update_pulls_position_and_velocity():
    kf = PosVelKF()
    kf.P[3:, 3:] = np.eye(3) * 1.0   # unknown initial velocity (post-maneuver)
    for _ in range(200):
        kf.predict(np.zeros(3), 0.01)
    # two position fixes 1 s apart at the true positions
    ok1 = kf.update_position(np.array([2.0, 0, 0]), rng=8.0)
    for _ in range(100):
        kf.predict(np.zeros(3), 0.01)
    ok2 = kf.update_position(np.array([3.0, 0, 0]), rng=8.0)
    assert ok1 and ok2
    assert kf.p[0] > 2.0          # pulled toward the fixes
    assert kf.v[0] > 0.15         # velocity inferred from successive fixes


def test_mahalanobis_rejects_wild_fix():
    kf = PosVelKF()
    for _ in range(50):
        kf.predict(np.zeros(3), 0.01)
    assert not kf.update_position(np.array([50.0, 0, 0]), rng=8.0)
    assert abs(kf.p[0]) < 1e-6


def test_covariance_symmetric_and_bounded():
    kf = PosVelKF()
    rng = np.random.default_rng(0)
    for i in range(500):
        kf.predict(rng.normal(0, 0.5, 3), 0.007)
        if i % 40 == 0:
            kf.update_position(kf.p + rng.normal(0, 0.3, 3), rng=10.0)
    assert np.allclose(kf.P, kf.P.T, atol=1e-9)
    assert float(np.trace(kf.P)) < 50.0
