import numpy as np
from scripts.sysid.ring_model import RateLoopParams, propagate, propagate_nl

def test_underdamped_step_overshoot():
    # Unit-gain 2nd-order, wn=30 rad/s, zeta=0.14 -> analytic overshoot exp(-z*pi/sqrt(1-z^2))
    p = RateLoopParams(k=1.0, wn=30.0, zeta0=0.14)
    dt = 1.0 / 720.0
    n = int(1.0 / dt)
    cmd = np.ones(n)
    y = propagate(cmd, dt, p)
    overshoot = (y.max() - 1.0)
    expected = np.exp(-p.zeta0 * np.pi / np.sqrt(1 - p.zeta0**2))
    assert abs(overshoot - expected) < 0.03           # within 3% of analytic
    assert abs(y[-1] - 1.0) < 0.02                     # settles to unit gain

def test_gain_scales_output():
    p = RateLoopParams(k=2.0, wn=20.0, zeta0=0.7)
    dt = 1.0 / 720.0
    y = propagate(np.ones(int(0.5 / dt)), dt, p)
    assert abs(y[-1] - 2.0) < 0.05                     # steady state = k

def test_negative_zeta1_rings_more_at_high_amplitude():
    dt = 1.0 / 720.0
    n = int(1.0 / dt)
    p = RateLoopParams(k=1.0, wn=30.0, zeta0=0.5, zeta1=-0.08)  # damping drops as |omega| grows
    small = propagate_nl(0.2 * np.ones(n), dt, p)
    large = propagate_nl(5.0 * np.ones(n), dt, p)
    os_small = small.max() / small[-1] - 1.0
    os_large = large.max() / large[-1] - 1.0
    assert os_large > os_small + 0.05                  # bigger relative overshoot at high amplitude

def test_nl_reduces_to_linear_when_zeta1_zero():
    dt = 1.0 / 720.0
    cmd = np.ones(int(0.5 / dt))
    p = RateLoopParams(k=1.0, wn=25.0, zeta0=0.3, zeta1=0.0)
    assert np.allclose(propagate_nl(cmd, dt, p), propagate(cmd, dt, p), atol=1e-9)
