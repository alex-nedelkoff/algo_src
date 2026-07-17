"""MAP_JSON wiring (07-17): vq2wp.py cannot be imported (module-level flight),
so these are source-text assertions in the house style, plus functional checks
of the plumbing map through the real loader."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vq2.map_ingest import anchor_warnings, load_course_map

ROOT = Path(__file__).parents[2]
SOURCE = (ROOT / "vq2/live/vq2wp.py").read_text(encoding="utf-8")


def test_map_json_is_flag_gated_and_default_off():
    assert "MAP_JSON = os.environ.get('MAP_JSON', '')" in SOURCE
    assert "if MAP_JSON:" in SOURCE
    # the hardcoded course model must survive as the default
    assert "G1_AP = np.array([5.9, 0.9, -1.1])" in SOURCE


def test_map_json_loads_before_the_spline_is_built():
    loaded = SOURCE.index("load_course_map(MAP_JSON)")
    first_traj = SOURCE.index("TRAJ, S_GATES = build_traj")
    assert loaded < first_traj


def test_map_json_prints_anchor_warnings_instead_of_silently_flying():
    assert "anchor_warnings(_map_gates)" in SOURCE
    assert "MAP_JSON WARNING" in SOURCE


def test_g2_delta_is_map_derived_at_both_sites():
    # both G2TEST delta sites must prefer the map delta over the constant
    assert SOURCE.count(
        "G2_DELTA if G2_DELTA is not None"
    ) == 2, "both the spline and the pad-lock anchor must use the map delta"
    assert "G2_DELTA = _mg_p[1] - _mg_p[0]" in SOURCE


def test_plumbing_map_reproduces_the_flown_constants_exactly():
    gates = load_course_map(ROOT / "vq2/live/course_map_plumbing.json")
    g1 = np.array(gates[0].pos)
    g2 = np.array(gates[1].pos)
    assert np.allclose(g1, [5.9, 0.9, -1.1])
    # the delta is what G2TEST actually flies: must equal the hardcoded one
    assert np.allclose(g2 - g1, [5.14, 5.26, 0.0])


def test_plumbing_map_warns_on_g1_anchor_by_construction():
    gates = load_course_map(ROOT / "vq2/live/course_map_plumbing.json")
    warnings = anchor_warnings(gates)
    assert len(warnings) == 2  # aim points sit ~1m off both aperture anchors


def test_plumbing_map_declares_itself():
    data = json.loads(
        (ROOT / "vq2/live/course_map_plumbing.json").read_text(encoding="utf-8")
    )
    assert "PLUMBING-CHECK" in data["source"]
