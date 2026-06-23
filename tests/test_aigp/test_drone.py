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
    c = FlightConfig(vmax=7.0, vlat_max=4.0, amax=4.0, tilt_deg=20.0, zff=-2.4, capture=2.5,
                     kiz=0.8, c_max=18.0)
    g = c.to_navgains()
    assert isinstance(g, NavGains)
    assert g.MAX_SPEED == 7.0 and g.VLAT_MAX == 4.0 and g.FWD_AMAX == 4.0   # vlat NOT slaved to vmax
    assert g.DECEL_MAX >= 4.0 and g.TILT_MAX_DEG == 20.0 and g.Z_FF == -2.4
    assert g.CAPTURE == 2.5 and g.KI_Z == 0.8 and g.C_MAX == 18.0


def test_flightconfig_default_lateral_cap_stays_conservative():
    g = FlightConfig().to_navgains()
    assert g.VLAT_MAX <= 1.5 and g.MAX_SPEED <= 5.0   # default envelope inside the proven stable region


def test_flightconfig_rejects_garbage_envelope():
    for kw in (dict(vmax=-5.0), dict(vmax=float("inf")), dict(vmax=True), dict(tilt_deg=400.0),
               dict(tilt_deg=0.0), dict(vmax=9.0), dict(vlat_max=20.0), dict(kiz=-1.0),
               dict(zff=float("nan")), dict(default_speed=0)):
        with pytest.raises(ValueError):
            FlightConfig(**kw)


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


def test_check_positive_rejects_inf_and_bool():
    from aigp.drone import _check_positive
    for bad in (float("inf"), float("-inf"), float("nan"), 0, -1, True, False, "3"):
        with pytest.raises(ValueError):
            _check_positive("speed", bad)
    _check_positive("speed", 3.0); _check_positive("speed", 1)  # finite >0 ok


def test_check_wp_rejects_nonfinite_and_wrong_shape():
    from aigp.drone import _check_wp
    for bad in [(float("nan"), 0, 0), (float("inf"), 0, 0), (1, 0), (1, 0, 0, 0), (), (1, 2, 3, 4, 5)]:
        with pytest.raises(ValueError):
            _check_wp(bad)
    out = _check_wp((1.0, 2.0, -3.0))
    assert out.shape == (3,) and np.allclose(out, [1, 2, -3])


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


# ---------------------------------------------------------------------------
# Task 6: Drone facade tests
# ---------------------------------------------------------------------------
from aigp.drone import Drone


class _FakeStore2:
    def __init__(self, ds): self._ds = ds
    def get_drone(self): return self._ds
    def get_race_live(self): return True


class _RecCmd:
    def send_attitude_target(self, rate, thr): pass


def _drone():
    from aigp.state import DroneState
    ds = DroneState(np.array([0, 0, -2.0]), np.zeros(3), np.array([0.0, 1, 0, 0]), np.zeros(3), 0)
    return Drone(_FakeStore2(ds), _RecCmd(), (0.5, 13.0, np.array([1.0, 1.0, 1.0])))


def test_goto_validates_inputs():
    d = _drone()
    with pytest.raises(ValueError):
        d.goto((1, 0, 0), frame="polar").wait(timeout=1)
    with pytest.raises(ValueError):
        d.goto((1, 0, 0), yaw="spin").wait(timeout=1)


def test_goto_rejects_bad_waypoint_synchronously():
    d = _drone()
    d.nav.goto = lambda *a, **k: "reached"  # would mask a bad coord if it reached the loop
    for bad in [(float("nan"), 0, 0), (float("inf"), 0, 0), (1, 0), (1, 0, 0, 0)]:
        with pytest.raises(ValueError):
            d.goto(bad, yaw="hold")            # raises in caller, never launches a NaN-streaming mission


def test_follow_empty_raises():
    d = _drone()
    with pytest.raises(ValueError):
        d.follow([], yaw="hold")


def test_inf_scalars_rejected():
    d = _drone()
    with pytest.raises(ValueError):
        d.goto((1, 0, 0), yaw="hold", speed=float("inf"))
    with pytest.raises(ValueError):
        d.orbit((1, 0, 0), radius=float("inf"), seconds=1)
    with pytest.raises(ValueError):
        d.takeoff(float("inf"))


def test_orbit_rejects_bad_center():
    d = _drone()
    with pytest.raises(ValueError):
        d.orbit((float("nan"), 0, 0), radius=2, seconds=1)


def test_goto_returns_mission_and_runs():
    d = _drone()
    # stub the navigator's blocking call so no real loop runs
    d.nav.goto = lambda *a, **k: "reached"
    m = d.goto((3, 0, 0), yaw="hold")
    s = m.wait(timeout=2)
    assert isinstance(m, Mission) and s.result is Result.REACHED


def test_new_mission_auto_aborts_previous():
    d = _drone()
    order = []
    def slow(*a, **k):
        for _ in range(100):
            if d.nav._abort_evt is not None and d.nav._abort_evt.is_set():
                order.append("aborted"); return "abort"
            time.sleep(0.005)
        order.append("finished"); return "reached"
    d.nav.goto = slow
    m1 = d.goto((3, 0, 0), yaw="hold")
    time.sleep(0.02)
    m2 = d.goto((4, 0, 0), yaw="hold")     # should auto-abort m1
    m2.wait(timeout=2)
    assert "aborted" in order and m1.result is Result.ABORT


def test_stop_flag_does_not_leak_between_calls():
    d = _drone()
    d.nav.goto = lambda *a, **k: "reached"
    d.goto((3, 0, 0), yaw="hold", stop=True).wait(timeout=2)
    assert d.nav._stop_each is True
    d.goto((4, 0, 0), yaw="hold", stop=False).wait(timeout=2)
    assert d.nav._stop_each is False     # reset, not leaked


# ---------------------------------------------------------------------------
# Task 7: Motion primitives (orbit, hover, takeoff, descend)
# ---------------------------------------------------------------------------
from aigp.drone import _orbit_ring


def test_orbit_ring_geometry():
    center = np.array([10.0, 0.0, -3.0])
    ring = _orbit_ring(center, radius=5.0, n=24, direction="ccw")
    assert len(ring) == 24
    for p in ring:
        assert abs(np.linalg.norm((p - center)[:2]) - 5.0) < 1e-6   # on the circle
        assert abs(p[2] - center[2]) < 1e-9                          # planar (center altitude)
    # ccw winding: cross of first two spokes is +z (NED down +, so signed area sign is consistent)
    a = (ring[0] - center)[:2]; b = (ring[1] - center)[:2]
    assert (a[0] * b[1] - a[1] * b[0]) > 0


def test_hover_streams_commands_on_still_drone():
    # regression: hover used to route through settle(), which stops sending once vel < SETTLE_V,
    # so a stationary drone received ZERO setpoints. Active hover must keep commanding.
    from aigp.state import DroneState
    sends = {"n": 0}

    class CountCmd:
        def send_attitude_target(self, rate, thr): sends["n"] += 1

    ds = DroneState(np.array([0, 0, -2.0]), np.zeros(3), np.array([0.0, 1, 0, 0]), np.zeros(3), 0)
    d = Drone(_FakeStore2(ds), CountCmd(), (0.5, 13.0, np.array([1.0, 1.0, 1.0])))
    d.set_origin(pos_ned=np.zeros(3), yaw=0.0)
    d.hover(seconds=0.2).wait(timeout=2)
    assert sends["n"] > 0


def test_takeoff_and_descend_targets():
    d = _drone()
    captured = {}
    def capture_nav(targets, **k):
        captured["t"] = np.asarray(targets[0], float)
        return captured["t"]
    d._navigate = capture_nav
    d.takeoff(3.0)
    assert abs(captured["t"][2] - (-2.0 - 3.0)) < 1e-9    # current z (-2) minus 3 = up
    d.descend(2.0)
    assert abs(captured["t"][2] - (-2.0 + 2.0)) < 1e-9    # down
