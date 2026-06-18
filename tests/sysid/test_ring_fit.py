import numpy as np
from scripts.sysid.ring_model import RateLoopParams, propagate_nl
from scripts.sysid.ring_loader import RunSeries
from scripts.sysid.ring_fit import fit_axis, holdout_r2, envelope

def _make_run(p, dt, T, seed):
    rng = np.random.default_rng(seed)
    cmd1 = np.cumsum(rng.standard_normal(T)) * 0.3          # smooth-ish excitation on roll
    cmd = np.column_stack([cmd1, np.zeros(T), np.zeros(T)])
    om1 = propagate_nl(cmd1, dt, p) + rng.standard_normal(T) * 0.02
    omega = np.column_stack([om1, np.zeros(T), np.zeros(T)])
    return RunSeries(dt=dt, cmd=cmd, omega=omega, tilt_deg=np.zeros(T))

def test_fit_recovers_known_params():
    dt, T = 1/720, 1500
    true = RateLoopParams(k=1.0, wn=28.0, zeta0=0.20, zeta1=-0.03)
    runs = [_make_run(true, dt, T, s) for s in range(3)]
    est = fit_axis(runs, axis=0, dt=dt)
    assert abs(est.wn - true.wn) / true.wn < 0.10
    assert abs(est.zeta0 - true.zeta0) < 0.05
    assert abs(est.zeta1 - true.zeta1) < 0.03

def test_holdout_r2_high_for_matching_model_and_envelope():
    dt, T = 1/720, 1500
    p = RateLoopParams(k=1.0, wn=28.0, zeta0=0.2, zeta1=-0.03)
    rng = np.random.default_rng(7)
    cmd1 = np.cumsum(rng.standard_normal(T)) * 0.3
    om1 = propagate_nl(cmd1, dt, p) + rng.standard_normal(T) * 0.01
    rs = RunSeries(dt=dt,
                   cmd=np.column_stack([cmd1, np.zeros(T), np.zeros(T)]),
                   omega=np.column_stack([om1, np.zeros(T), np.zeros(T)]),
                   tilt_deg=np.linspace(0, 60, T))
    edges = np.array([0, 30, 45, 60, 90])
    r2 = holdout_r2(p, [rs], axis=0, dt=dt, tilt_edges=edges)
    assert np.nanmin(r2[:3]) > 0.9                  # well-modelled where samples exist
    env = envelope(r2, edges, thresh=0.9)
    assert env["max_tilt_deg"] >= 45.0
