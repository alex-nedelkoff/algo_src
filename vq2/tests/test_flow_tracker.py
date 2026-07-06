import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from vq2 import corpus as corpus_mod
from vq2.flow_vel import FlowVelocity

REC = os.path.expanduser("~/Documents/drone-ai-grand-prix/vq2_data/vq2_rec")


def test_synthetic_translating_texture_gives_forward_velocity():
    """Render a textured floor plane, translate camera, expect v_x > 0."""
    rng = np.random.default_rng(7)
    from vq2 import camera
    from vq2.tests.test_flow_geometry import _floor_points, _project

    # build two synthetic frames by splatting bright dots at projected floor pts
    pts = _floor_points()
    dt = 1.0 / 30.0
    pos0 = np.array([0.0, 0.0, -1.3])
    pos1 = pos0 + np.array([1.5, 0.0, 0.0]) * dt
    att = (0.0, 0.0, 0.0)
    frames = []
    for pos in (pos0, pos1):
        uv, ok = _project(pts, pos, att)
        img = np.zeros((camera.H, camera.W), np.uint8)
        for u, v in uv[ok]:
            cv2.circle(img, (int(round(u)), int(round(v))), 3, 255, -1)
        frames.append(cv2.GaussianBlur(img, (5, 5), 1.0))
    flow = FlowVelocity()
    assert flow.process(frames[0], 0.0, att, 1.3) is None  # first frame
    res = flow.process(frames[1], dt, att, 1.3)
    assert res is not None
    v_w, sigma, ninl, ntr = res
    assert v_w[0] > 1.0  # forward motion detected, right order of magnitude
    assert abs(v_w[1]) < 0.4


@pytest.mark.skipif(not os.path.isdir(REC), reason="vq2_rec corpus not present")
def test_at_rest_corpus_reports_near_zero_velocity():
    c = corpus_mod.load(REC)
    frames = [fr for fr in c.frames if os.path.exists(fr.path)][:120]
    assert len(frames) > 60, "corpus should have frames"
    flow = FlowVelocity()
    att = (0.0, 0.0, 0.0)  # at rest on the pad; spawn tilt irrelevant for |v|
    vs = []
    for fr in frames:
        gray = cv2.imread(fr.path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        res = flow.process(gray, fr.sim_ns / 1e9, att, 0.35)
        if res is not None:
            vs.append(np.linalg.norm(res[0]))
    assert len(vs) > 30, "flow should produce estimates on most frames"
    assert np.median(vs) < 0.05  # at rest => ~zero velocity
