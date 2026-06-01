from aigp.state import Store
from aigp.race import wait_for_race_live


def test_wait_returns_false_when_never_live():
    s = Store()
    assert wait_for_race_live(s, timeout=0.1, poll=0.01) is False


def test_wait_returns_true_when_live():
    s = Store()
    s.set_race({"race_live": True, "boot_ms": 4000, "race_start_ms": 3298})
    assert wait_for_race_live(s, timeout=0.5, poll=0.01) is True


def test_store_get_race_live_reflects_state():
    s = Store()
    assert s.get_race_live() is False
    s.set_race({"race_live": False})
    assert s.get_race_live() is False
    s.set_race({"race_live": True})
    assert s.get_race_live() is True
