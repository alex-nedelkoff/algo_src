import numpy as np
from scripts.sysid.ring_model import RateLoopParams, propagate_nl
from scripts.sysid.ring_loader import RunSeries
from scripts.sysid.ring_fit import fit_axis

def _make_run(p, dt, T, seed):
    rng = np.random.default_rng(seed)
    cmd1 = np.cumsum(rng.standard_normal(T)) * 0.5          # smooth-ish excitation on roll
    cmd = np.column_stack([cmd1, np.zeros(T), np.zeros(T)])
    om1 = propagate_nl(cmd1, dt, p) + rng.standard_normal(T) * 0.01
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
