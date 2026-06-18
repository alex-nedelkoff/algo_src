import numpy as np
from scripts.sysid.ring_loader import load_npz, RunSeries


def _quat_from_tilt(tilt_deg):
    # roll-only quat [w,x,y,z] giving the requested tilt about body-x
    a = np.radians(tilt_deg) / 2.0
    return np.stack([np.cos(a), np.sin(a), np.zeros_like(a), np.zeros_like(a)], axis=1)


def test_load_npz_maps_fields_and_computes_tilt(tmp_path):
    T = 100
    t = np.linspace(0, 1.0, T)
    cmd = np.random.default_rng(0).standard_normal((T, 3))
    omega = cmd * 0.9
    tilt = np.linspace(0, 50, T)
    quat = _quat_from_tilt(tilt)
    f = tmp_path / "run.npz"
    np.savez(f, t=t, wcmd=cmd, omega=omega, quat=quat)
    km = {"t": "t", "cmd": "wcmd", "omega": "omega", "quat": "quat"}
    rs = load_npz(str(f), km)
    assert isinstance(rs, RunSeries)
    assert rs.cmd.shape == (T, 3) and rs.omega.shape == (T, 3)
    assert abs(rs.dt - (t[1] - t[0])) < 1e-9
    assert abs(rs.tilt_deg[-1] - 50.0) < 0.5          # tilt recovered from quat
