import numpy as np
from scripts.sysid.ring_loader import load_npz, RunSeries, bin_coverage


def _quat_from_tilt(tilt_deg):
    # quat [w,x,y,z] giving the requested tilt under the deploy convention (tilt = 1-2(y^2+z^2)):
    # put the half-angle in the y component -> 1-2(sin^2 a) = cos(2a) = cos(tilt).
    a = np.radians(tilt_deg) / 2.0
    return np.stack([np.cos(a), np.zeros_like(a), np.sin(a), np.zeros_like(a)], axis=1)


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


def test_load_real_schema_dir_and_4wide_cmd(tmp_path):
    # real recorder layout: <run_dir>/data.npz with t_wall, cmd (T,4)=[wx,wy,wz,thrust], omega (T,3), quat (T,4)
    T = 80
    t_wall = np.linspace(0.0, 1.1, T)
    cmd4 = np.random.default_rng(1).standard_normal((T, 4))   # 4-wide incl thrust
    omega = cmd4[:, :3] * 0.8
    quat = _quat_from_tilt(np.linspace(0, 40, T))
    run_dir = tmp_path / "20260618T185812_vq_ringid"
    run_dir.mkdir()
    np.savez(run_dir / "data.npz", t_wall=t_wall, cmd=cmd4, omega=omega, quat=quat)
    rs = load_npz(str(run_dir))                                # pass the DIR; default keymap
    assert rs.cmd.shape == (T, 3) and rs.omega.shape == (T, 3)  # cmd sliced 4->3
    assert np.allclose(rs.cmd, cmd4[:, :3])
    assert abs(rs.tilt_deg[-1] - 40.0) < 0.5


def test_bin_coverage_counts_cells():
    T = 50
    rs = RunSeries(dt=1/720,
                   cmd=np.column_stack([np.full(T, 3.0), np.zeros(T), np.zeros(T)]),
                   omega=np.zeros((T, 3)),
                   tilt_deg=np.full(T, 47.0))
    tilt_edges = np.array([0, 45, 60, 90])
    amp_edges = np.array([0, 2, 4, 10])
    cov = bin_coverage([rs], axis=0, tilt_edges=tilt_edges, amp_edges=amp_edges)
    assert cov.shape == (3, 3)
    assert cov[1, 1] == T          # tilt in [45,60), |cmd|=3 in [2,4)
    assert cov.sum() == T
