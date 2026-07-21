import numpy as np

from vq2.pillars import GpuPillarReader, PillarRead, classify_top_reads


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
    assert reads[1].number is None
