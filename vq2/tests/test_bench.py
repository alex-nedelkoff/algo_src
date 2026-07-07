import os

import numpy as np
import pytest

from vq2.bench import POLICIES, run_bench
from vq2.fusion import FusionConfig, run_fusion

ACC = os.path.expanduser("~/Documents/drone-ai-grand-prix/vq2_data/vq2_accept")


def test_policies_registered():
    assert "radius" in POLICIES and "huber" in POLICIES


def test_radius_policy_matches_legacy_binary_gate():
    accept, r_scale = POLICIES["radius"](miss=2.9, rng=10.0)
    assert accept and r_scale == 1.0
    accept, r_scale = POLICIES["radius"](miss=3.1, rng=10.0)
    assert not accept


def test_huber_policy_never_rejects_and_inflates():
    accept, r1 = POLICIES["huber"](miss=0.5, rng=10.0)
    assert accept and r1 == 1.0
    accept, r2 = POLICIES["huber"](miss=6.0, rng=10.0)
    assert accept and r2 > 2.0  # far miss -> soft fix, not discarded


@pytest.mark.skipif(not os.path.isdir(ACC), reason="vq2_accept corpus absent")
def test_fusion_records_obs_waterfall():
    res = run_fusion(ACC, FusionConfig(use_flow=False, use_vision_pos=True))
    assert len(res.obs_events) > 100
    ev = res.obs_events[0]
    for k in ("t_boot_s", "rng", "miss", "gate", "stage", "r_scale"):
        assert k in ev
    stages = {e["stage"] for e in res.obs_events}
    assert "accepted" in stages
    acc = [e for e in res.obs_events if e["stage"] == "accepted"]
    assert all("nis" in e for e in acc)


@pytest.mark.skipif(not os.path.isdir(ACC), reason="vq2_accept corpus absent")
def test_bench_compares_policies_on_corpus():
    out = run_bench(ACC, policies=("radius", "huber"))
    assert set(out) == {"radius", "huber"}
    for name, row in out.items():
        for k in ("accepted", "rejected", "anchor_p50", "anchor_p90",
                  "dist_traveled_m", "nis_p50"):
            assert k in row, (name, k)
    # huber never discards matched obs -> accepts at least as many
    assert out["huber"]["accepted"] >= out["radius"]["accepted"]


def test_range_identity_gate_rejects_infeasible_candidates():
    from vq2.fusion import range_identity_ok
    # measured 11 m, candidate expected 30 m away -> not this gate
    assert not range_identity_ok(rng_meas=11.0, rng_expected=30.0)
    # measured 11 m, candidate expected 13 m -> feasible under drift
    assert range_identity_ok(rng_meas=11.0, rng_expected=13.0)
    # degenerate expected range never crashes
    assert range_identity_ok(rng_meas=11.0, rng_expected=0.0) is False


@pytest.mark.skipif(not os.path.isdir(ACC), reason="vq2_accept corpus absent")
def test_huber_area_filters_junk_but_keeps_soft_accepts():
    out = run_bench(ACC, policies=("huber", "huber_area"))
    ha, hu = out["huber_area"], out["huber"]
    # identity gate kills the hallucinated far gates the plain huber chases
    assert ha["anchor_p90"] < hu["anchor_p90"]
    # but still no hard starvation: accepts more than the binary radius did (642)
    assert ha["accepted"] > 642 * 0.8
    # identity rejections show up as their own waterfall stage
    assert "identity_reject" in ha["reject_stages"]
