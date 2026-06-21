import pytest
import numpy as np
import threading
import time
from aigp.drone import Result, Status, FlightConfig, _StatusBox, _check_frame, _check_yaw, _check_positive, Mission
from aigp.navigator import NavGains


def test_flightconfig_safe_defaults_within_envelope():
    c = FlightConfig()
    assert c.amax <= 4.5 and c.tilt_deg <= 20 and c.default_speed <= 6


def test_flightconfig_to_navgains_maps_fields():
    c = FlightConfig(vmax=7.0, amax=4.0, tilt_deg=20.0, zff=-2.4, capture=2.5, kiz=0.8, c_max=18.0)
    g = c.to_navgains()
    assert isinstance(g, NavGains)
    assert g.MAX_SPEED == 7.0 and g.VLAT_MAX == 7.0 and g.FWD_AMAX == 4.0
    assert g.DECEL_MAX >= 4.0 and g.TILT_MAX_DEG == 20.0 and g.Z_FF == -2.4
    assert g.CAPTURE == 2.5 and g.KI_Z == 0.8 and g.C_MAX == 18.0


def test_status_defaults_running():
    s = Status(result=Result.RUNNING, phase="x", pos_ned=np.zeros(3), vel=0.0,
               tilt_deg=0.0, mode="hold", target=None, progress=0.0, error=None)
    assert s.result is Result.RUNNING and s.error is None


def test_statusbox_roundtrip_and_thread_safe_copy():
    box = _StatusBox()
    box.update(phase="leg 1/2", ds_pos=np.array([1.0, 2.0, -3.0]), vel=2.0, tilt=5.0,
               mode="course", target=np.array([4.0, 0.0, -3.0]), progress=0.5)
    s = box.get()
    assert s.phase == "leg 1/2" and s.vel == 2.0 and s.mode == "course"
    assert s.result is Result.RUNNING
    box.set_result(Result.REACHED)
    assert box.get().result is Result.REACHED


def test_validation_raises():
    with pytest.raises(ValueError):
        _check_frame("polar")
    with pytest.raises(ValueError):
        _check_yaw("spin")
    with pytest.raises(ValueError):
        _check_positive("radius", -1.0)
    _check_frame("body"); _check_yaw("lookat"); _check_positive("radius", 2.0)  # no raise


def _fake_run(abort_evt, pause_evt, box, n=50):
    # spins up to n ticks, honoring pause/abort, updating status; returns REACHED if it finishes
    for k in range(n):
        if abort_evt.is_set():
            return Result.ABORT
        while pause_evt.is_set() and not abort_evt.is_set():
            time.sleep(0.005)
        box.update(phase=f"tick {k}", ds_pos=np.zeros(3), vel=1.0, tilt=2.0,
                   mode="hold", target=None, progress=k / n)
        time.sleep(0.005)
    return Result.REACHED


def _mk_mission():
    a, p, box = threading.Event(), threading.Event(), _StatusBox()
    return Mission(lambda ae, pe, b: _fake_run(ae, pe, b), a, p, box)


def test_mission_runs_to_completion():
    m = _mk_mission(); m.start()
    s = m.wait(timeout=5)
    assert m.done and s.result is Result.REACHED


def test_mission_abort_stops_and_reports():
    m = _mk_mission(); m.start()
    time.sleep(0.02); m.abort()
    s = m.wait(timeout=5)
    assert s.result is Result.ABORT


def test_mission_pause_resume_gates_progress():
    m = _mk_mission(); m.start()
    time.sleep(0.02); m.pause()
    time.sleep(0.01)     # let the in-flight tick finish + block inside the pause loop before sampling
    p1 = m.status().progress
    time.sleep(0.05)
    assert abs(m.status().progress - p1) < 1e-9     # frozen while paused
    m.resume()
    s = m.wait(timeout=5)
    assert s.result is Result.REACHED and s.progress > p1
