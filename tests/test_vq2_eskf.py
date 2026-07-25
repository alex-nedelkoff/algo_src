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


HUBER_DELTA = 1.5   # vq2wp: r_scale = miss / HUBER_DELTA above this


def _r_scale(miss):
    """The call pattern vq2wp actually ships (vq2wp.py:1010)."""
    return 1.0 if miss <= HUBER_DELTA else miss / HUBER_DELTA


def test_wild_fix_rejected_under_shipped_r_scale():
    """COR-147: the pre-fix gate was undefeatable because r_scale was derived
    from the same innovation it tested. test_mahalanobis_rejects_wild_fix only
    passed because it used the DEFAULT r_scale=1.0 -- never the shipped one."""
    kf = PosVelKF()
    for _ in range(50):
        kf.predict(np.zeros(3), 0.01)
    miss = 50.0
    assert not kf.update_position(np.array([miss, 0, 0]), rng=8.0,
                                  r_scale=_r_scale(miss))
    assert abs(kf.p[0]) < 1e-6


def test_legacy_gate_reproduces_the_bug():
    """Guard the A/B path: the old behaviour must still be reachable, and it
    must still accept the wild fix (otherwise the fix proves nothing)."""
    kf = PosVelKF(legacy_gate=True)
    for _ in range(50):
        kf.predict(np.zeros(3), 0.01)
    miss = 50.0
    assert kf.update_position(np.array([miss, 0, 0]), rng=8.0,
                              r_scale=_r_scale(miss))
    assert kf.last_nis < kf.maha_gate      # saturated well under the gate
    assert kf.p[0] > 1e-6                  # and it moved the state


def test_good_fix_still_accepted_after_fix():
    """The gate must not become so tight that legitimate fixes are lost."""
    kf = PosVelKF()
    for _ in range(200):
        kf.predict(np.zeros(3), 0.01)
    miss = 0.8                              # inside HUBER_DELTA
    assert kf.update_position(np.array([miss, 0, 0]), rng=8.0,
                              r_scale=_r_scale(miss))
    assert kf.p[0] > 0.0


def test_joseph_keeps_covariance_positive_definite():
    """The short form (I-KH)P loses positive-definiteness under the extreme
    gain ratios r_scale produces -- that is what manufactured the negative-NIS
    events (70 across 70 banked corpora, worst -133.44)."""
    rng = np.random.default_rng(7)
    for legacy in (True, False):
        kf = PosVelKF(legacy_gate=legacy)
        kf.P = np.diag([2.0, 2.0, 2.0, 5.0, 5.0, 5.0])
        worst = 1.0
        for i in range(60):
            kf.predict(rng.normal(0, 2.0, 3), 0.02)
            miss = 3.0 + 8.0 * (i % 5)
            kf.update_position(kf.p + np.array([miss, 0, 0]), rng=25.0,
                               r_scale=_r_scale(miss))
            worst = min(worst, float(np.linalg.eigvalsh(kf.P)[0]))
        if legacy:
            legacy_worst = worst
        else:
            assert worst > -1e-9, f"Joseph form went indefinite: {worst}"
    # the short form is the one that degrades; keep the contrast honest
    assert legacy_worst <= worst + 1e-12


def test_negative_nis_is_rejected_not_applied():
    """A negative d2 is only reachable with an indefinite P. Previously it
    passed the gate (d2 < 0 < maha_gate) and the update was APPLIED."""
    kf = PosVelKF()
    # the negative eigenvalue must exceed R_true along the innovation
    # direction, else S stays PD and the gate rejects on magnitude instead
    kf.P[:3, :3] = np.diag([-2.0, 1.0, 1.0])   # forced indefinite
    p0 = kf.p.copy()
    assert not kf.update_position(np.array([30.0, 0, 0]), rng=10.0,
                                  r_scale=_r_scale(30.0))
    assert kf.last_nis < 0                      # the diagnostic still reports
    assert np.allclose(kf.p, p0)                # and the state is untouched


def test_covariance_symmetric_and_bounded():
    kf = PosVelKF()
    rng = np.random.default_rng(0)
    for i in range(500):
        kf.predict(rng.normal(0, 0.5, 3), 0.007)
        if i % 40 == 0:
            kf.update_position(kf.p + rng.normal(0, 0.3, 3), rng=10.0)
    assert np.allclose(kf.P, kf.P.T, atol=1e-9)
    assert float(np.trace(kf.P)) < 50.0
