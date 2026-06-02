import json, numpy as np
from aigp.aero_model import force_features, moment_features, FORCE_COLS, MOMENT_COLS
from aigp.aero_validate import single_step_metrics, term_contributions, save_model

def _synth(n=300, seed=5):
    rng = np.random.default_rng(seed)
    thF = np.zeros(len(FORCE_COLS)); thF[FORCE_COLS.index("C_x")] = 0.05
    V = rng.uniform(-15, 15, (n, 3)); W = rng.uniform(-2, 2, (n, 3))
    aF = np.array([force_features(V[i]) @ thF for i in range(n)])
    return V, W, aF, thF

def test_single_step_metrics_perfect_fit():
    V, W, aF, thF = _synth()
    thM = np.zeros(len(MOMENT_COLS))
    m = single_step_metrics(V, W, aF, np.zeros_like(aF), thF, thM, residual=None)
    assert m["rmse_force"] < 1e-9 and m["r2_force"] > 0.999

def test_term_contributions_identifies_dominant():
    V, W, aF, thF = _synth()
    contrib = term_contributions(V, thF, force_features, FORCE_COLS)
    assert max(contrib, key=contrib.get) == "C_x"

def test_save_model_roundtrip(tmp_path):
    p = tmp_path / "sim_aero.json"
    save_model(str(p), theta_F=np.array([1.0, 2.0]), theta_M=np.array([3.0]),
               force_cols=["D_x", "D_y"], moment_cols=["d_x"], meta={"envelope_max_speed": 22.0})
    d = json.load(open(p))
    assert d["theta_F"] == [1.0, 2.0] and d["meta"]["envelope_max_speed"] == 22.0
