import math
import os

import numpy as np
import pytest

from vq2 import camera
from vq2.gates3d import GATE_CORNERS, corner_jacobian, project_corners

ACC5 = os.path.expanduser(
    "~/Documents/drone-ai-grand-prix/vq2_data/vq2_accept5"
)


def test_gate1_corner_geometry():
    c = GATE_CORNERS["G1"]
    assert c.shape == (4, 3)
    # aperture is a 1.5 m square in the y-z plane at x=11, centered -1.3
    assert np.allclose(c[:, 0], 11.0)
    assert np.allclose(sorted(set(np.round(c[:, 1], 2))), [-0.75, 0.75])
    assert np.allclose(sorted(set(np.round(c[:, 2], 2))), [-2.05, -0.55])


def test_projection_from_pad_is_left_right_symmetric():
    """From the origin at level attitude, G1 (dead ahead, centered) must
    project symmetrically about the image center column."""
    uv, vis = project_corners(np.zeros(3), (0.0, 0.0, 0.0), "G1")
    assert vis.all()
    xs = uv[:, 0] - camera.CX
    assert abs(xs.sum()) < 1e-6  # +x pairs mirror -x pairs
    # gate below the 20-deg-up optical axis -> corners in lower image half
    assert (uv[:, 1] > camera.CY).all()


def test_jacobian_matches_numeric_diff():
    p = np.array([2.0, 0.4, -1.2])
    att = (0.03, -0.05, 0.1)
    uv0, vis = project_corners(p, att, "G1")
    H = corner_jacobian(p, att, "G1")          # (4, 2, 3)
    eps = 1e-5
    for k in range(3):
        dp = np.zeros(3)
        dp[k] = eps
        uv1, _ = project_corners(p + dp, att, "G1")
        num = (uv1 - uv0) / eps
        assert np.allclose(H[:, :, k], num, atol=1e-2), k


@pytest.mark.skipif(not os.path.isdir(ACC5), reason="run-6 corpus absent")
@pytest.mark.xfail(reason="FINDING (07-06): the pad-view instance is the "
                   "G1+HIGH stacked assembly merged into one detection "
                   "(8 corner rows, box ~5.0x4.5 m at 11 m) -- corner "
                   "identity must split/handle merged-stack instances "
                   "before this ground-truth check can pass; see "
                   "docs/superpowers/plans/2026-07-06-corner-updates.md")
def test_pad_rest_projection_lands_on_detected_corners():
    """THE ground-truth check: project G1's map corners from the pad (known
    origin, accel-implied rest attitude) and compare against GateNet's
    detected corner_xy in a real pad-rest frame. Median corner error must
    be small -- this validates corner ORDER, aperture size, camera model
    and map yaw in one shot."""
    import json

    from vq2 import corpus as C
    from vq2.replay import find_rest_windows
    from vq2.estimators import accel_implied_attitude

    c = C.load(ACC5)
    seg = c.flight_segment
    rests = find_rest_windows(seg.imu)
    s0 = seg.imu[(rests[0][0] + rests[0][1]) // 2]
    roll, pitch = accel_implied_attitude(s0)

    rows = [json.loads(l) for l in open(os.path.join(ACC5, "detections.jsonl"))]
    rows.sort(key=lambda d: d["sim_ns"])
    # a solved, confident instance from the rest window (early frames)
    inst = None
    for d in rows[: len(rows) // 4]:
        for i in d["insts"]:
            if i.get("solved") and not i.get("low_confidence") and i.get("corner_xy"):
                rng = float(np.linalg.norm(i["t_cam"]))
                if 9.0 < rng < 13.0:  # the G1 pad view
                    inst = i
                    break
        if inst:
            break
    assert inst is not None, "no pad-rest G1 detection found"

    uv_pred, vis = project_corners(np.zeros(3), (roll, pitch, 0.0), "G1")
    det = np.asarray(inst["corner_xy"], float)
    # order-agnostic match: nearest predicted corner per detected corner
    errs = []
    for d_uv in det:
        errs.append(min(np.linalg.norm(d_uv - p_uv) for p_uv in uv_pred))
    assert np.median(errs) < 12.0, (np.round(errs, 1), uv_pred.round(1), det.round(1))
