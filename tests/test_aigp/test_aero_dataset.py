import numpy as np
import pandas as pd
from aigp.aero_dataset import build_targets, world_to_body_vel, finite_diff_filtered

def test_world_to_body_vel_identity():
    R = np.eye(3); v_world = np.array([1.0, 2.0, 3.0])
    assert np.allclose(world_to_body_vel(v_world, R), v_world)

def test_finite_diff_filtered_constant_rate():
    t = np.linspace(0, 1, 101)
    w = np.stack([2.0 * t, 0 * t, -1.0 * t], axis=1)
    a = finite_diff_filtered(t, w)
    assert np.allclose(np.median(a[5:-5], axis=0), [2.0, 0.0, -1.0], atol=0.1)

def test_build_targets_coast_force_is_specific_force():
    df = pd.DataFrame({
        "t": [0.0, 0.004], "vx": [5.0, 5.0], "vy": [0, 0], "vz": [0, 0],
        "qw": [1, 1], "qx": [0, 0], "qy": [0, 0], "qz": [0, 0],
        "wx": [0, 0], "wy": [0, 0], "wz": [0, 0],
        "fx": [-1.2, -1.2], "fy": [0, 0], "fz": [0, 0],
        "thrust_accel": [0.0, 0.0], "coast": [True, True],
    })
    out = build_targets(df, I_ratio=np.array([3.7, 1.0, 1.0]))
    assert out["a_aero"].shape[1] == 3
    assert np.allclose(out["a_aero"][0], [-1.2, 0.0, 0.0])
    assert out["coast"].all()
