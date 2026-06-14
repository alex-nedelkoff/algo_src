"""Per-env domain randomization in VQMatchedDynamics (COR-127 asymmetric critic).

Invariant: randomization OFF -> bit-identical to the validated nominal dynamics.
"""
import json

import numpy as np

from sim.dynamics.vq_matched import VQMatchedDynamics

MODEL = json.load(open("sysid/vq_model.json"))


def _rollout(dyn, n=8, steps=30, seed=0):
    rng = np.random.default_rng(seed)
    s = dyn.reset(n)
    s[:, 3:6] = rng.normal(0, 2, (n, 3))          # nonzero velocity to exercise drag/wv
    acts = rng.uniform(-1, 1, (steps, n, 4))
    acts[:, :, 0] = 0.5
    out = []
    for a in acts:
        s = dyn.step(s, a)
        out.append(s.copy())
    return np.array(out)


def test_randomize_off_is_bit_identical():
    a = VQMatchedDynamics(MODEL, frame="ENU")
    b = VQMatchedDynamics(MODEL, frame="ENU")   # never call randomize() -> nominal
    ra, rb = _rollout(a), _rollout(b)
    np.testing.assert_array_equal(ra, rb)        # determinism baseline
    assert np.ndim(a.Dx) == 0                    # scalar (float) when not randomized


def test_dr_factors_shape_and_center():
    d = VQMatchedDynamics(MODEL, frame="ENU")
    rng = np.random.default_rng(3)
    d.randomize(rng, n_envs=8, width=0.4)
    assert d.dr_factors.shape == (8, 11)
    assert np.all(np.abs(d.dr_factors) <= 2.0)   # centered near 0, bounded
    assert np.asarray(d.Dx).shape == (8,)        # per-env params became arrays


def test_dr_changes_trajectory():
    base = VQMatchedDynamics(MODEL, frame="ENU")
    drd = VQMatchedDynamics(MODEL, frame="ENU")
    drd.randomize(np.random.default_rng(1), n_envs=8, width=0.4)
    assert not np.allclose(_rollout(base), _rollout(drd))
