"""Test for scripts/sysid/ring_id.py — CLI orchestrator for high-tilt rate-loop sysID."""
import json
import numpy as np
from scripts.sysid.ring_model import RateLoopParams, propagate_nl
from scripts.sysid.ring_id import run_id


def _write_run(path, p, dt, T, seed, tilt_hi):
    rng = np.random.default_rng(seed)
    c = np.cumsum(rng.standard_normal(T)) * 0.3
    cmd = np.column_stack([c, c * 0.5, np.zeros(T)])
    omega = np.column_stack([propagate_nl(cmd[:, 0], dt, p),
                             propagate_nl(cmd[:, 1], dt, p), np.zeros(T)]) + rng.standard_normal((T, 3)) * 0.01
    a = np.radians(np.linspace(0, tilt_hi, T)) / 2
    quat = np.column_stack([np.cos(a), np.zeros(T), np.sin(a), np.zeros(T)])  # y-component -> tilt = 1-2(y^2+z^2)
    t = np.arange(T) * dt
    np.savez(path, t=t, wcmd=cmd, omega=omega, quat=quat)


def test_run_id_writes_outputs_and_verdict(tmp_path):
    dt, T = 1/720, 1500
    p = RateLoopParams(k=1.0, wn=28.0, zeta0=0.2, zeta1=-0.03)
    paths = []
    for s in range(4):
        f = tmp_path / f"r{s}.npz"
        _write_run(str(f), p, dt, T, s, 60.0)
        paths.append(str(f))
    km = {"t": "t", "cmd": "wcmd", "omega": "omega", "quat": "quat"}
    rep = run_id(paths, km, str(tmp_path), tilt_edges=[0, 30, 45, 60, 90],
                 amp_edges=[0, 2, 4, 10], holdout_frac=0.25)
    assert (tmp_path / "vq_model_hightilt.json").exists()
    assert (tmp_path / "ring_envelope.json").exists()
    assert (tmp_path / "ring_report.md").exists()
    env = json.loads((tmp_path / "ring_envelope.json").read_text())
    assert env["roll"]["max_tilt_deg"] >= 45.0
    assert rep["verdict"] in ("go", "partial", "no-go")
