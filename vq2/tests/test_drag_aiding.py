import math
import os

import numpy as np
import pytest

from vq2.camera import R_world_body
from vq2.corpus import ImuSample
from vq2.eskf import PosVelKF
from vq2.fusion import FusionConfig, drag_velocity_update, run_fusion

REC = os.path.expanduser("~/Documents/drone-ai-grand-prix/vq2_data/vq2_rec")


def _sample(acc, gyr=(0.02, 0.0, 0.0), t_us=0):
    # default gyr is non-zero so is_at_rest() sees "flying"
    return ImuSample(t_us=t_us, acc=acc, gyr=gyr, rx_wall=0.0)


def test_drag_converges_to_true_velocity_without_vision():
    """Steady 2 m/s forward flight, level attitude: f_bx = -D*v -> KF v
    must converge to the true velocity from drag updates alone."""
    cfg = FusionConfig()
    kf = PosVelKF()
    kf.reset_at_rest()
    R_wb = R_world_body(0.0, 0.0, 0.0)
    acc = (-cfg.drag_d * 2.0, 0.0, -9.81)  # drag of v_bx = +2 m/s
    for _ in range(300):  # ~2 s at 144 Hz
        # predict with zero net accel (steady cruise: thrust balances drag);
        # without predict the covariance never grows and the gain collapses
        kf.predict(np.zeros(3), 1.0 / 144.0)
        applied = drag_velocity_update(kf, _sample(acc), R_wb, cfg)
        assert applied
    assert abs(kf.v[0] - 2.0) < 0.15
    assert abs(kf.v[1]) < 0.15


def test_drag_measurement_rotates_with_yaw():
    """Same body-frame drag, drone yawed 90 deg: velocity lands on world y."""
    cfg = FusionConfig()
    kf = PosVelKF()
    kf.reset_at_rest()
    R_wb = R_world_body(0.0, 0.0, math.pi / 2)
    acc = (-cfg.drag_d * 1.5, 0.0, -9.81)
    for _ in range(300):
        kf.predict(np.zeros(3), 1.0 / 144.0)
        drag_velocity_update(kf, _sample(acc), R_wb, cfg)
    assert abs(kf.v[1] - 1.5) < 0.15
    assert abs(kf.v[0]) < 0.15


def test_drag_skipped_at_rest():
    """On the (tilted) pad, contact forces break the drag model: the rest
    detector must gate the update off. At rest on the 18-deg pad the accel
    xy reads ~ -g*sin(18) = -3 m/s^2 == a false 30 m/s."""
    cfg = FusionConfig()
    kf = PosVelKF()
    kf.reset_at_rest()
    tilt = math.radians(18.0)
    acc = (-9.81 * math.sin(tilt), 0.0, -9.81 * math.cos(tilt))
    s = _sample(acc, gyr=(0.0, 0.0, 0.0))  # quiet gyro + |f|~g = at rest
    applied = drag_velocity_update(kf, s, R_world_body(0.0, tilt, 0.0), cfg)
    assert not applied
    assert np.allclose(kf.v, 0.0)


def test_drag_skipped_on_insane_magnitude():
    cfg = FusionConfig()
    kf = PosVelKF()
    kf.reset_at_rest()
    acc = (-cfg.drag_d * 40.0, 0.0, -9.81)  # implies 40 m/s: garbage
    applied = drag_velocity_update(kf, _sample(acc), R_world_body(0, 0, 0), cfg)
    assert not applied


@pytest.mark.skipif(not os.path.isdir(REC), reason="vq2_rec corpus not present")
def test_at_rest_corpus_unchanged_by_drag():
    """Whole-rest corpus: rest gate keeps drag updates ~off; final velocity
    stays pinned near zero exactly as without drag."""
    res = run_fusion(REC, FusionConfig(use_flow=True, use_drag=True))
    assert float(np.linalg.norm(res.v[-1])) < 0.10
