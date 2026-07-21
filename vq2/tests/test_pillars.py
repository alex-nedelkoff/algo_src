from vq2.pillars import PillarRead, classify_top_reads


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
