"""Pillar-landmark map loader: contract enforcement + the shipped v1 file."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vq2.map_ingest import load_pillar_map, pillar_map_by_number

V1 = Path(__file__).resolve().parents[1] / "pillar_map_v1.json"


def test_shipped_v1_loads_with_twins():
    lms = load_pillar_map(V1)
    assert len(lms) >= 5
    by_num = pillar_map_by_number(lms)
    assert len(by_num["22"]) == 2          # aisle twins are the point
    assert all(len(p) == 3 for ps in by_num.values() for p in ps)
    assert {lm.confidence for lm in lms} <= {"ticked", "observed", "inferred"}


def test_quarantined_never_returned():
    data = json.loads(V1.read_text())
    assert data["quarantined"], "v1 ships with the disputed 06|22 entry"
    nums = {lm.id for lm in load_pillar_map(V1)}
    assert not any("|" in n for n in nums)


@pytest.mark.parametrize("mutate,err", [
    (lambda d: d.update(version="pillar-2"), "version"),
    (lambda d: d["landmarks"][0].update(number="x2"), "digit"),
    (lambda d: d["landmarks"][1].update(id=""), "unique non-empty"),
    (lambda d: d["landmarks"][0].update(confidence="guessed"), "confidence"),
    (lambda d: d.update(landmarks=[]), "non-empty"),
])
def test_contract_violations_fail_closed(tmp_path, mutate, err):
    data = json.loads(V1.read_text())
    mutate(data)
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=err):
        load_pillar_map(p)


def test_duplicate_ids_rejected(tmp_path):
    data = json.loads(V1.read_text())
    data["landmarks"][1]["id"] = data["landmarks"][0]["id"]
    p = tmp_path / "dup.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unique"):
        load_pillar_map(p)
