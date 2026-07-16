from __future__ import annotations

from pathlib import Path


def test_gate_scale_batch_is_observe_only_bounded_and_disk_conservative():
    root = Path(__file__).parents[2]
    source = (
        root / "vq2/live/fly_servo_dpvo_gnscale_observe.bat"
    ).read_text()

    assert "set GNSCALE=1" in source
    assert "set GNSCALE_TMAX=12.0" in source
    assert "set GNSCALE_HOLD_TMAX=18.0" in source
    assert "set RECENTER=1" in source
    assert "set VIZ=0" in source
    assert "set DPVO_ROUTE=1" in source
    assert "set DPVO_OBSERVE=1" in source
    assert "set NOFIX=1" in source
    assert "set MF_DR=1" in source
    assert "set GN_UNLOAD=1" in source
    assert "set RRD=" in source
    assert "set RECORD=C:/Users/alexj/vq2_servo_dpvo_gnscale_obs2" in source
    assert "DPVO_OBSERVE=0" not in source
