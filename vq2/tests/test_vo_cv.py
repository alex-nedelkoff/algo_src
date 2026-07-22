"""Tests for the in-VM monocular VO (COR-147).

- frame-convention math (cam->body, OdomDelta lowering) on synthetic input
- keyframe-by-parallax behaviour on synthetic translating dots
- (opt) real-corpus health check, skipped when no corpus is present
"""
import math
import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from vq2.live.vo_cv import MonoVO, VoStep, step_to_odom_delta
from vq2.relpose import OdomDelta


def test_step_to_odom_delta_frame_and_contract():
    # pure forward body motion, no rotation
    s = VoStep(t0=1.0, t1=1.5, R_body=np.eye(3),
               t_body_unit=np.array([1.0, 0.0, 0.0]), dyaw=0.0,
               n_inliers=100, n_tracked=300, median_flow_px=10.0)
    d = step_to_odom_delta(s, yaw0=0.0)
    assert isinstance(d, OdomDelta)
    assert d.source == "vo_generic" and d.scale_locked is False
    np.testing.assert_allclose(d.dp_local, [1.0, 0.0, 0.0], atol=1e-9)
    assert abs(np.linalg.norm(d.dp_local) - 1.0) < 1e-9   # unit (monocular)

    # yaw0=+90deg rotates the forward body vector into the state's -y (right)
    d90 = step_to_odom_delta(s, yaw0=math.pi / 2)
    np.testing.assert_allclose(d90.dp_local, [0.0, -1.0, 0.0], atol=1e-6)


def _synthetic_pair(dx_px, n=400, seed=0):
    """Two frames: random dots translated by dx_px (pure image shift)."""
    rng = np.random.default_rng(seed)
    a = np.zeros((360, 640), np.uint8)
    pts = rng.integers([40, 40], [600, 320], size=(n, 2))
    for x, y in pts:
        a[y - 1:y + 2, x - 1:x + 2] = 255
    b = np.zeros((360, 640), np.uint8)
    for x, y in pts:
        x2 = x + dx_px
        if 2 <= x2 < 638:
            b[y - 1:y + 2, x2 - 1:x2 + 2] = 255
    return a, b


def test_keyframe_by_parallax_holds_until_baseline():
    """Sub-pixel motion must NOT close a keyframe; real motion must."""
    vo = MonoVO(kf_flow_px=9.0, min_inliers=10)
    a, _ = _synthetic_pair(0)
    assert vo.step(0.0, a) is None            # first frame => set keyframe
    # a near-duplicate (1 px shift) is below the 9px threshold -> no keyframe
    _, tiny = _synthetic_pair(1, seed=0)
    assert vo.step(0.1, tiny) is None
    # a 15px shift crosses the parallax threshold -> keyframe closes
    _, big = _synthetic_pair(15, seed=0)
    out = vo.step(0.2, big)
    assert out is not None
    assert out.median_flow_px >= 9.0
    assert out.n_inliers >= 10


REAL = r"C:\Users\Administrator\vq2_test46"


@pytest.mark.skipif(not os.path.isdir(os.path.join(REAL, "frames")),
                    reason="real corpus not present")
def test_real_corpus_motion_window_is_healthy():
    """On the flight window (frames ~380+), keyframes close with healthy
    inliers. Guards against regressions in the KLT/essential-matrix path."""
    import glob
    ns = sorted(int(os.path.splitext(os.path.basename(p))[0])
                for p in glob.glob(os.path.join(REAL, "frames", "*.jpg")))
    files = [os.path.join(REAL, "frames", f"{n}.jpg") for n in ns][380:900]
    vo = MonoVO(kf_flow_px=9.0)
    steps = []
    for i, f in enumerate(files):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        s = vo.step(i * 0.02, img)
        if s is not None:
            steps.append(s)
    assert len(steps) >= 5, f"too few keyframes: {len(steps)}"
    med_inl = np.median([s.n_inliers for s in steps])
    assert med_inl >= 30, f"unhealthy inliers: {med_inl}"
