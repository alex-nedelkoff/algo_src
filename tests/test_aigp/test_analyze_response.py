import numpy as np
from aigp.analyze_response import finite_diff, fit_thrust_map


def test_finite_diff_linear():
    t = np.linspace(0, 1, 11)
    v = 3.0 * t + 2.0
    assert np.allclose(finite_diff(v, t), 3.0, atol=1e-6)


def test_fit_thrust_map_recovers_constants():
    # thrust_up_accel = k_a*thrust_norm + b ; hover where thrust_up = g
    tn = np.array([0.3, 0.5, 0.7])
    thrust_up = 30.0 * tn - 6.0     # k_a=30, b=-6
    out = fit_thrust_map(tn, thrust_up, g=9.81)
    assert np.isclose(out["k_a"], 30.0, atol=1e-6)
    assert np.isclose(out["hover_thrust"], (9.81 + 6.0) / 30.0, atol=1e-6)


from aigp.analyze_response import fit_rate_gain, summarize


def test_fit_rate_gain():
    cmd = np.array([0.0, 0.5, 1.0, -0.5, -1.0])
    meas = 0.9 * cmd
    assert np.isclose(fit_rate_gain(cmd, meas), 0.9, atol=1e-9)


def test_summarize_from_samples():
    # thrust-sweep samples: vz linear in t so net accel constant per tn level
    samples = []
    t = 0.0
    # three thrust levels, each producing a known constant vertical accel
    # net_up = -dvz/dt ; thrust_up = net_up + g
    for tn, net_up in [(0.3, -3.0), (0.5, 0.0), (0.7, 3.0)]:
        for k in range(5):
            vz = -net_up * (t)        # vz_down; net_up = -dvz/dt
            samples.append({"segment": "thrust_sweep", "t": t, "thrust_norm": tn,
                            "cmd_rates": [0, 0, 0], "vel_ned": [0, 0, vz],
                            "omega": [0, 0, 0]})
            t += 0.1
    # one roll-doublet segment: measured omega = 0.8 * commanded
    for k in range(6):
        c = 1.0 if k < 3 else -1.0
        samples.append({"segment": "rate_roll", "t": t, "thrust_norm": 0.5,
                        "cmd_rates": [c, 0, 0], "vel_ned": [0, 0, 0],
                        "omega": [0.8 * c, 0, 0]})
        t += 0.1
    out = summarize(samples)
    assert 0.0 < out["hover_thrust"] < 1.0
    assert out["k_a"] > 0.0
    assert np.isclose(out["rate_gain"]["roll"], 0.8, atol=0.05)
