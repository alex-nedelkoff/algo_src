import numpy as np
import cv2

from vq2.pillars import (GpuPillarReader, PillarRead, classify_top_reads,
                         fit_pillar_edges, fit_station_text_pnp)


def read(number, x, y, h=16):
    return PillarRead(1.0, x=x, y=y, w=12, h=h, number=number)


def test_highest_same_number_in_a_column_is_top():
    top, lower = classify_top_reads([read("22", 100, 30), read("22", 105, 70)])
    assert top.top
    assert not lower.top


def test_nearby_different_numbers_are_not_one_stack():
    first, second = classify_top_reads([read("22", 100, 30), read("23", 105, 70)])
    assert not first.top
    assert not second.top


def test_tiny_read_is_a_top_panel_at_range():
    (far,) = classify_top_reads([read("22", 100, 30, h=10)])
    assert far.top


def test_wide_separation_does_not_make_a_stack():
    first, second = classify_top_reads([read("22", 100, 30), read("22", 150, 70)])
    assert not first.top
    assert not second.top


class FakeReader:
    def readtext(self, *_args, **_kwargs):
        return [
            ([[10, 20], [20, 20], [20, 40], [10, 40]], "Station22", 0.9),
            ([[30, 50], [40, 50], [40, 70], [30, 70]], "Station?", 0.7),
            ([[1, 1], [2, 1], [2, 2], [1, 2]], "noise", 0.99),
        ]


def test_gpu_reader_rotates_boxes_back_and_keeps_unread_station_bearing():
    reads = GpuPillarReader(reader=FakeReader()).read(np.zeros((100, 160, 3), np.uint8), 3.5)
    assert len(reads) == 2
    assert reads[0].number == "22"
    assert reads[0].t_capture == 3.5
    # x_orig = y_rot, y_orig = H - 1 - x_rot.
    assert (reads[0].x, reads[0].y, reads[0].w, reads[0].h) == (20.0, 79.0, 20.0, 10.0)
    assert np.allclose(reads[0].quad_xy, [[20, 89], [20, 79], [40, 79], [40, 89]])
    assert reads[1].number is None


def test_station_text_pnp_recovers_metric_camera_translation():
    K = np.array([[320.0, 0.0, 320.0], [0.0, 320.0, 180.0], [0.0, 0.0, 1.0]])
    rvec = np.array([0.10, -0.18, 0.04])
    tvec = np.array([0.35, -0.12, 9.0])
    obj = np.array([[-2.503 / 2, -.475 / 2, 0], [2.503 / 2, -.475 / 2, 0],
                    [2.503 / 2, .475 / 2, 0], [-2.503 / 2, .475 / 2, 0]], float)
    quad, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    fit = fit_station_text_pnp(quad.reshape(4, 2), K)
    assert fit is not None
    assert np.allclose(fit.t_cam_text, tvec, atol=1e-5)
    assert fit.reproj_rms_px < 1e-6


def test_pillar_edge_fit_brackets_text_with_vertical_boundaries():
    image = np.zeros((220, 240, 3), np.uint8)
    cv2.line(image, (70, 20), (70, 205), (255, 255, 255), 3)
    cv2.line(image, (150, 20), (150, 205), (255, 255, 255), 3)
    fit = fit_pillar_edges(image, PillarRead(0.0, 92, 80, 35, 65, "22"))
    assert fit is not None
    assert abs(fit.center_u - 110) < 6
    assert abs(fit.width_px - 80) < 10
    assert fit.quality > 0.8
