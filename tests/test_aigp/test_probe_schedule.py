import numpy as np
from aigp.probe_response import build_schedule, _tilt_deg


def test_tilt_deg_level_and_inverted():
    assert _tilt_deg([1.0, 0.0, 0.0, 0.0]) < 1e-6          # level -> 0 deg
    # 180 deg roll about body-x: quat [0, 1, 0, 0]
    assert abs(_tilt_deg([0.0, 1.0, 0.0, 0.0]) - 180.0) < 1e-6
    # 90 deg roll about body-x: quat [cos45, sin45, 0, 0]
    s = np.sqrt(0.5)
    assert abs(_tilt_deg([s, s, 0.0, 0.0]) - 90.0) < 1e-4


def test_schedule_segments_and_bounds():
    sched = build_schedule(hover_guess=0.5, thrust_band=0.2, rate_amp=1.0,
                           seg_dur=2.0, dt=0.05)
    names = [s["name"] for s in sched]
    assert names == ["thrust_sweep", "rate_roll", "rate_pitch", "rate_yaw"]
    sweep = sched[0]
    # thrust stays within [hover-band, hover+band] and rates zero
    assert all(0.3 - 1e-9 <= c["thrust"] <= 0.7 + 1e-9 for c in sweep["commands"])
    assert all(c["rates"] == [0.0, 0.0, 0.0] for c in sweep["commands"])
    # roll doublet: |rate| <= rate_amp on axis 0 only
    roll = sched[1]
    assert all(abs(c["rates"][0]) <= 1.0 + 1e-9 for c in roll["commands"])
    assert all(c["rates"][1] == 0.0 and c["rates"][2] == 0.0 for c in roll["commands"])
