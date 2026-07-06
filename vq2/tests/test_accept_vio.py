import os

import pytest

from vq2.accept_vio import evaluate, sweep_z_floor

MOT = os.path.expanduser("~/Documents/drone-ai-grand-prix/vq2_data/vq2_motion")


@pytest.mark.skipif(not os.path.isdir(MOT), reason="vq2_motion corpus not present")
def test_evaluate_reports_complete_verdict():
    r = evaluate(MOT, z_floor=0.15)
    for key in ("max_err_xy_12m", "p90_err_xy", "dist_traveled_m",
                "n_anchors", "flow_updates", "passed"):
        assert key in r
    assert r["n_anchors"] > 20
    assert r["dist_traveled_m"] > 5.0


@pytest.mark.skipif(not os.path.isdir(MOT), reason="vq2_motion corpus not present")
def test_sweep_orders_candidates_by_error():
    rows = sweep_z_floor(MOT, lo=0.05, hi=0.30, step=0.125)
    assert len(rows) == 3
    assert all(len(row) == 2 for row in rows)
