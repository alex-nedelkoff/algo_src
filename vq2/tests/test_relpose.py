"""Contract tests for vq2.relpose — the shared measurement interface for
the pillar-pnp and vo-loop branches. Fail-closed on malformed inputs."""
import numpy as np
import pytest

from vq2.relpose import LandmarkRelPose, OdomDelta
from vq2.window_smoother import OdomFactor


def test_landmark_relpose_roundtrip():
    m = LandmarkRelPose(t=12.5, source="pillar_pnp", number="22",
                        p_lm_level=[3.0, -1.0, -7.16], sigma_p=0.5)
    assert m.p_lm_level.dtype == np.float64
    assert m.number == "22"
    assert m.yaw_rel is None            # yaw optional: PnL may not observe it


def test_landmark_relpose_unread_number_ok():
    # quad solved but digits unread — association handles candidate gating
    m = LandmarkRelPose(t=1.0, source="pillar_pnl", number=None,
                        p_lm_level=np.zeros(3))
    assert m.number is None


def test_landmark_relpose_fails_closed():
    with pytest.raises(ValueError):
        LandmarkRelPose(t=0.0, source="magnetometer", number=None,
                        p_lm_level=np.zeros(3))
    with pytest.raises(ValueError):
        LandmarkRelPose(t=0.0, source="pillar_pnp", number="22",
                        p_lm_level=[1.0, np.nan, 0.0])


def test_odom_delta_lowers_to_factor():
    d = OdomDelta(t0=10.0, t1=10.5, dp_local=[0.3, 0.0, -0.05], dyaw=0.01,
                  scale_locked=True)
    f = d.to_factor(4, 5)
    assert isinstance(f, OdomFactor)
    assert (f.i, f.j) == (4, 5)
    np.testing.assert_array_equal(f.dp_local, d.dp_local)
    assert f.sigma_p == d.sigma_p


def test_odom_delta_fails_closed():
    with pytest.raises(ValueError):
        OdomDelta(t0=2.0, t1=2.0, dp_local=np.zeros(3), dyaw=0.0)  # t1 !> t0
    with pytest.raises(ValueError):
        OdomDelta(t0=0.0, t1=1.0, dp_local=np.zeros(3), dyaw=0.0,
                  source="wheel_odom")
    d = OdomDelta(t0=0.0, t1=1.0, dp_local=np.zeros(3), dyaw=0.0)
    with pytest.raises(ValueError):
        d.to_factor(3, 3)                                          # j !> i
