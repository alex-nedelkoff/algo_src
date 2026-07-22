import cv2
import numpy as np
import pytest

from vq2.g0_calibration import _OBJECT_3D, compare_hypotheses, fit_g0_aperture


def _project(K, rvec, tvec):
    image, _ = cv2.projectPoints(_OBJECT_3D, rvec, tvec, K, None)
    return image.reshape(4, 2)


def test_fits_focal_lengths_when_principal_point_is_declared():
    K = np.array([[228., 0., 319.5], [0., 221., 179.5], [0., 0., 1.]])
    image = _project(K, np.array([.12, -.18, .04]), np.array([.3, -.2, 10.595]))
    fit = fit_g0_aperture(image)
    assert np.allclose([fit.K[0, 0], fit.K[1, 1]], [228., 221.], atol=1e-2)
    assert np.allclose(fit.t_cam_gate, [.3, -.2, 10.595], atol=1e-3)
    assert fit.reproj_rms_px < 1e-4


def test_compares_historic_models_without_approving_a_model():
    K = np.array([[226., 0., 319.5], [0., 226., 179.5], [0., 0., 1.]])
    report = compare_hypotheses(_project(K, np.array([.1, .05, 0.]), np.array([0., 0., 10.])),
                                certified_depth_m=10.)
    assert [row["id"] for row in report["models"]] == ["fitted", "mapping-226", "legacy-320"]
    assert report["approved"] is False
    assert abs(report["models"][1]["depth_delta_m"]) < 1e-5


def test_frontoparallel_aperture_refuses_free_focal_fit_but_compares_models():
    K = np.array([[320., 0., 319.5], [0., 320., 179.5], [0., 0., 1.]])
    image = _project(K, np.zeros(3), np.array([0., 0., 10.595]))
    with pytest.raises(ValueError, match="ill-conditioned"):
        fit_g0_aperture(image)
    report = compare_hypotheses(image, certified_depth_m=10.595)
    assert report["models"][0]["available"] is False
    assert report["models"][1]["id"] == "mapping-226"
