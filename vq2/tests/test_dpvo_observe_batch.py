from __future__ import annotations

from pathlib import Path


def test_first_live_batch_is_observe_only_and_disk_conservative():
    root = Path(__file__).parents[2]
    source = (root / "vq2/live/fly_servo_dpvo_observe.bat").read_text()
    assert "set DPVO_ROUTE=1" in source
    assert "set DPVO_OBSERVE=1" in source
    assert "set DPVO_PATCHES=32" in source
    assert "set DPVO_CUDA_FRACTION=0.48" in source
    assert "set GN_UNLOAD=1" in source
    assert "set MF_DR=1" in source
    assert "set RRD=" in source
    assert "DPVO_OBSERVE=0" not in source
    assert "set RECORD=C:/Users/alexj/vq2_servo_dpvo_obs2" in source
    assert "DPVO_GPU_SETTLE" not in source
