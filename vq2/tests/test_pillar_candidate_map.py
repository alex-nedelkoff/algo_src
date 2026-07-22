import numpy as np
import pytest

from vq2.corpus import ImuSample
from vq2.pillar_candidate_map import (
    ApprovedCameraModel, GateAnchor, TextPanelObservation, anchor_camera_pose,
    build_candidate, robust_fuse,
)


def _anchor():
    return GateAnchor(
        p_world_gate=np.array([10.595, 0.051, -0.2472]),
        R_world_gate=np.eye(3), R_cam_gate=np.eye(3),
        t_cam_gate=np.array([0.0, 0.0, 10.595]), t_rx_wall=10.0,
        translation_sigma_m=0.25,
    )


def _camera(approved=True):
    return ApprovedCameraModel("g0-manual-fit", np.eye(3), approved)


def _imu():
    return [ImuSample(int(t * 1e6), (0, 0, -9.81), (0, 0, 0.1), t)
            for t in (10.0, 10.1, 10.2, 10.3, 10.4)]


def _obs(t, x, confidence=0.99, rms=0.5, edge=1.0):
    return TextPanelObservation(f"{t}.jpg", "22", confidence, t,
                                np.array([x, 0.0, 5.0]), rms,
                                300.0, 63.0, edge)


def test_anchor_pose_uses_opencv_object_to_camera_rotation():
    anchor = _anchor()
    anchor = GateAnchor(anchor.p_world_gate, np.eye(3),
                        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]]),
                        anchor.t_cam_gate, anchor.t_rx_wall, anchor.translation_sigma_m)
    R_wc, p_wc = anchor_camera_pose(anchor)
    assert np.allclose(R_wc, anchor.R_cam_gate.T)
    assert np.allclose(p_wc, anchor.p_world_gate - R_wc @ anchor.t_cam_gate)


def test_builder_fuses_text_panels_and_quarantines_axis():
    out = build_candidate(_anchor(), _camera(), _imu(), [_obs(10.2, 1.0), _obs(10.4, 1.1)])
    panel = out["text_panels"][0]
    assert out["frame"] == "reset-NED"
    assert panel["number"] == "22"
    assert len(panel["observations"]) == 2
    assert all(row["accepted"] for row in panel["observations"])
    assert out["quarantined"][0]["human_approved"] is False
    assert "pillar-axis" in out["quarantined"][0]["id"]


def test_builder_refuses_unapproved_calibration():
    with pytest.raises(ValueError, match="not approved"):
        build_candidate(_anchor(), _camera(False), _imu(), [_obs(10.2, 1), _obs(10.4, 1.1)])


def test_bad_observations_are_audited_and_not_fused():
    out = build_candidate(_anchor(), _camera(), _imu(), [
        _obs(10.2, 1.0), _obs(10.4, 1.1), _obs(10.3, 4.0, confidence=.1),
    ])
    rejected = out["text_panels"][0]["observations"][2]
    assert not rejected["accepted"]
    assert rejected["rejection"] == "low_ocr_confidence"


def test_robust_fuse_inflates_covariance_for_hover_translation():
    center, cov = robust_fuse([np.array([0., 0., 0.]), np.array([.1, 0., 0.])], .25)
    assert np.allclose(center, [.05, 0, 0])
    assert cov[0, 0] >= .25 ** 2
