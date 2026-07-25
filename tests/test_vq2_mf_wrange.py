"""Unit tests for the width-as-range staging correction (COR-147)."""
import math

from vq2.live import mf_wrange as W

GATE = (11.37, 5.29)


def test_calibration_anchors_to_believed_range():
    """C is set so vision range EQUALS the believed range at the seed, which
    is what makes the correction independent of the detected feature's true
    width (it is ~0.86 m, not the 1.5 m aperture)."""
    mf_p = [7.27, 0.03]
    c, why = W.calibrate(mf_p, GATE, fg_w=41.0, fg_age=0.05)
    assert why == 'ok'
    r = math.hypot(GATE[0] - mf_p[0], GATE[1] - mf_p[1])
    assert abs(c / 41.0 - r) < 1e-9          # zero residual at the anchor
    dx, dy, d = W.step(mf_p, GATE, 41.0, 0.05, c)
    assert d['why'] == 'ok' and abs(d['resid']) < 1e-9
    assert math.hypot(dx, dy) < 1e-12        # zero to float precision


def test_over_credited_dr_is_pushed_back():
    """The measured failure: DR believes it is closer than the imagery says.
    The correction must move it AWAY from the gate, along the bearing."""
    c, _ = W.calibrate([7.27, 0.03], GATE, fg_w=41.0, fg_age=0.05)
    mf_p = [9.5, 3.5]                         # DR has crept in
    r_dr = math.hypot(GATE[0] - mf_p[0], GATE[1] - mf_p[1])
    dx, dy, d = W.step(mf_p, GATE, 41.0, 0.05, c)   # width unchanged => same range
    assert d['resid'] > 0                     # vision says further than DR
    moved = math.hypot(GATE[0] - (mf_p[0] + dx), GATE[1] - (mf_p[1] + dy))
    assert moved > r_dr                       # pushed away from the gate


def test_correction_is_range_only_never_lateral():
    c, _ = W.calibrate([7.27, 0.03], GATE, fg_w=41.0, fg_age=0.05)
    mf_p = [9.5, 3.5]
    dx, dy, _d = W.step(mf_p, GATE, 60.0, 0.05, c)
    ux = mf_p[0] - GATE[0]
    uy = mf_p[1] - GATE[1]
    n = math.hypot(ux, uy)
    cross = (dx * uy - dy * ux) / n           # component across the bearing
    assert abs(cross) < 1e-12


def test_step_is_bounded():
    """A huge residual must not teleport the DR. (Probe point must sit well
    clear of the gate, else the degenerate-range guard fires first.)"""
    c, _ = W.calibrate([7.27, 0.03], GATE, fg_w=41.0, fg_age=0.05)
    dx, dy, d = W.step([25.0, 5.29], GATE, 40.0, 0.05, c, max_err=99.0)
    assert d['why'] == 'ok'
    assert math.hypot(dx, dy) <= W.MAX_STEP + 1e-12
    assert abs(d['resid']) > 5.0          # residual really was large


def test_fails_closed():
    c, _ = W.calibrate([7.27, 0.03], GATE, fg_w=41.0, fg_age=0.05)
    for kw, why in (
        (dict(fg_w=41.0, fg_age=1.2), 'stale'),        # detection too old
        (dict(fg_w=12.0, fg_age=0.05), 'narrow'),      # width is noise
    ):
        dx, dy, d = W.step([9.5, 3.5], GATE, kw['fg_w'], kw['fg_age'], c)
        assert (dx, dy) == (0.0, 0.0) and d['why'] == why
    dx, dy, d = W.step([9.5, 3.5], GATE, 41.0, 0.05, None)
    assert (dx, dy) == (0.0, 0.0) and d['why'] == 'uncalibrated'
    # wrong object: residual beyond max_err is refused rather than chased
    dx, dy, d = W.step([40.0, 5.29], GATE, 41.0, 0.05, c)
    assert (dx, dy) == (0.0, 0.0) and d['why'] == 'implausible'
    # sitting on the gate: unstable bearing, refused before anything else
    dx, dy, d = W.step([11.3, 5.2], GATE, 41.0, 0.05, c)
    assert (dx, dy) == (0.0, 0.0) and d['why'] == 'degenerate'


def test_calibration_refuses_bad_samples():
    assert W.calibrate([7.27, 0.03], GATE, 41.0, 1.5)[0] is None
    assert W.calibrate([7.27, 0.03], GATE, 5.0, 0.05)[0] is None
    assert W.calibrate([11.37, 5.29], GATE, 41.0, 0.05)[0] is None  # on top of it


def test_shrinking_width_means_receding_gate():
    """mig8's signature: the gate got FURTHER (width fell) while the DR
    believed it was closing. The correction must push the DR back."""
    c, _ = W.calibrate([7.27, 0.03], GATE, fg_w=45.0, fg_age=0.05)
    mf_p = [9.0, 3.0]
    dx, dy, d = W.step(mf_p, GATE, 38.0, 0.05, c)     # width fell => further
    assert d['r_vis'] > d['r_dr']
    assert math.hypot(GATE[0] - (mf_p[0] + dx), GATE[1] - (mf_p[1] + dy)) \
        > math.hypot(GATE[0] - mf_p[0], GATE[1] - mf_p[1])
