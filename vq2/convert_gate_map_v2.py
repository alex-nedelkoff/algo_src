"""Convert Janahan's COR-142 `gate_map_v2.json` into the v1 course-map contract
(docs/vq2-map-json-contract.md) so `vq2.map_ingest` can validate it.

The frames are the SAME convention, so this is a field translation, not a
transform: v2 declares `reset-NED (heading-aligned, pinned at drone reset)`,
`x=North(fwd), y=East(right), z=Down(gravity)`, right-handed — which is the
contract's `spawn:x-downcourse,y-right,z-down` (reset == spawn). This asserts
that rather than assuming it: a v2 file declaring any other axes/handedness is
REJECTED, because a silent lateral-sign flip is exactly the bug v2's own
lineage note L-2 records ("the reset-map vs survey-map lateral-sign
discrepancy was a handedness bug").

Tier -> contract confidence (evidence-graded, deliberately conservative:
`ticked` is reserved for judge ticks, which no vision map can claim):
    certified/HIGH -> observed
    MED/LOW/candidate -> inferred

`route_order` is assigned by ascending gate_id index (G0->1, G1->2, ...). That
is an ASSUMPTION: v2 carries no route order, and the judge scores gates in its
own order. Verify against the anchors before flying.

Usage: python -m vq2.convert_gate_map_v2 gate_map_v2.json out_v1.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

EXPECTED_AXES_SUBSTR = ("x=north", "y=east", "z=down")
TIER_TO_CONFIDENCE = {
    "certified": "observed",
    "HIGH": "observed",
    "MED": "inferred",
    "LOW": "inferred",
    "candidate": "inferred",
}


def convert(v2: dict) -> dict:
    if v2.get("schema_version") != "2.0":
        raise ValueError(f"expected schema_version 2.0, got {v2.get('schema_version')!r}")

    frame = v2.get("frame") or {}
    axes = str(frame.get("axes", "")).lower()
    if not all(token in axes for token in EXPECTED_AXES_SUBSTR):
        raise ValueError(
            f"frame axes {axes!r} are not x=North/y=East/z=Down -- refusing to "
            "guess a transform (see lineage L-2: handedness bugs are real here)"
        )
    if frame.get("handedness") != "right":
        raise ValueError(f"handedness must be explicit 'right', got {frame.get('handedness')!r}")

    gates_v2 = v2.get("gates") or {}
    if not gates_v2:
        raise ValueError("v2 map has no gates")

    gates = []
    for order, gate_id in enumerate(sorted(gates_v2), start=1):
        g = gates_v2[gate_id]
        pos = [float(v) for v in g["position_m"]]
        yaw = float(g.get("yaw_rad", 0.0))
        # yaw about z-down rotates +x(fwd) toward +y(right)
        normal = [math.cos(yaw), math.sin(yaw), 0.0]
        tier = g.get("tier", "candidate")
        if tier not in TIER_TO_CONFIDENCE:
            raise ValueError(f"{gate_id}: unknown tier {tier!r}")
        sigma_xyz = (g.get("covariance") or {}).get("sigma_pos_m") or [0.5, 0.5, 0.5]
        sigma = float(max(sigma_xyz))
        gates.append({
            "id": gate_id,
            "route_order": order,
            "pos": pos,
            "normal": normal,
            "aperture_m": 1.5,          # contract: the judge-scored inner square
            "confidence": TIER_TO_CONFIDENCE[tier],
            "pos_sigma_m": sigma,
        })

    datum = v2.get("datum") or {}
    return {
        "version": 1,
        "frame": "spawn:x-downcourse,y-right,z-down",
        "units": "m",
        "source": (
            f"converted from COR-142 gate_map_v2 (datum {datum.get('anchor_gate')}, "
            f"{v2.get('frame', {}).get('name')}); route_order ASSUMED by gate-id order; "
            f"aperture_m ASSUMED 1.5 (v2 carries no aperture size)"
        ),
        "gates": gates,
    }


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("v2_json", type=Path)
    parser.add_argument("out_json", type=Path)
    args = parser.parse_args(argv)
    with args.v2_json.open("r", encoding="utf-8") as fh:
        v2 = json.load(fh)
    v1 = convert(v2)
    with args.out_json.open("w", encoding="utf-8") as fh:
        json.dump(v1, fh, indent=2)
    print(f"wrote {args.out_json} ({len(v1['gates'])} gates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
