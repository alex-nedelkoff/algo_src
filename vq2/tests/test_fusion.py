import os

import numpy as np
import pytest

from vq2.fusion import FusionConfig, run_fusion

REC = os.path.expanduser("~/Documents/drone-ai-grand-prix/vq2_data/vq2_rec")
MOT = os.path.expanduser("~/Documents/drone-ai-grand-prix/vq2_data/vq2_motion")


@pytest.mark.skipif(not os.path.isdir(REC), reason="vq2_rec corpus not present")
def test_at_rest_flow_bounds_velocity_drift():
    res_imu = run_fusion(REC, FusionConfig(use_flow=False))
    res_flow = run_fusion(REC, FusionConfig(use_flow=True))
    v_end_imu = np.linalg.norm(res_imu.v[-1])
    v_end_flow = np.linalg.norm(res_flow.v[-1])
    # pure IMU integration drifts; flow must pin an at-rest drone near zero
    assert v_end_flow < 0.10
    assert v_end_flow <= v_end_imu + 1e-9
    assert res_flow.flow_updates > 50


@pytest.mark.skipif(not os.path.isdir(MOT), reason="vq2_motion corpus not present")
def test_motion_corpus_produces_anchors_and_flow():
    res = run_fusion(MOT, FusionConfig(use_flow=True))
    assert res.flow_updates > 100, "flow should track through the flight"
    assert len(res.anchors) > 20, "GateNet locks should anchor the run"
    # distance actually traveled per vision: last anchor should be well
    # downcourse of the pad (the flight approached gate 1 at ~11 m)
    rngs = [a["range"] for a in res.anchors]
    assert min(rngs) < 8.0
