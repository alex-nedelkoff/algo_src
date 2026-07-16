"""Validated loader for the VQ2 course-map JSON (docs/vq2-map-json-contract.md).

Turns Janahan's 6-gate map into route-ordered gate records in the spawn frame,
ready to drive the vq2wp spline (GATES_W / build_traj) or map-follow. Fails
closed on any contract violation; warns (but loads) on disagreement with
judge-verified anchors.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np


EXPECTED_FRAME = "spawn:x-downcourse,y-right,z-down"
DEFAULT_SIGMA = {"ticked": 0.1, "observed": 0.5, "inferred": 1.0}
# Judge-verified anchors (aperture centers, spawn frame). Cross-check only.
VERIFIED_ANCHORS = {
    1: np.array([6.3, 0.0, -1.35]),
    2: np.array([11.7, 5.2, -1.35]),
}
ANCHOR_WARN_M = 1.0


@dataclass(frozen=True)
class MapGate:
    id: str
    route_order: int
    pos: tuple[float, float, float]
    normal: tuple[float, float, float]
    aperture_m: float
    confidence: str
    pos_sigma_m: float


def _finite_vec3(value, name: str) -> np.ndarray:
    vec = np.asarray(value, dtype=float)
    if vec.shape != (3,) or not np.all(np.isfinite(vec)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return vec


def _parse_gate(raw: dict, index: int) -> MapGate:
    where = f"gates[{index}]"
    gate_id = raw.get("id")
    if not isinstance(gate_id, str) or not gate_id:
        raise ValueError(f"{where}: id must be a non-empty string")
    order = raw.get("route_order")
    if not isinstance(order, int) or order < 1:
        raise ValueError(f"{where}: route_order must be a positive integer")
    pos = _finite_vec3(raw.get("pos"), f"{where}.pos")
    normal = _finite_vec3(raw.get("normal"), f"{where}.normal")
    norm = float(np.linalg.norm(normal))
    if not 0.95 <= norm <= 1.05:
        raise ValueError(f"{where}: normal norm {norm:.3f} outside 1 +/- 0.05")
    normal = normal / norm
    aperture = float(raw.get("aperture_m", 1.5))
    if not 0.5 < aperture < 5.0:
        raise ValueError(f"{where}: aperture_m {aperture} outside (0.5, 5.0)")
    confidence = raw.get("confidence", "observed")
    if confidence not in DEFAULT_SIGMA:
        raise ValueError(f"{where}: confidence must be one of {sorted(DEFAULT_SIGMA)}")
    sigma = float(raw.get("pos_sigma_m", DEFAULT_SIGMA[confidence]))
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError(f"{where}: pos_sigma_m must be finite and >= 0")
    return MapGate(
        id=gate_id,
        route_order=order,
        pos=tuple(float(v) for v in pos),
        normal=tuple(float(v) for v in normal),
        aperture_m=aperture,
        confidence=confidence,
        pos_sigma_m=sigma,
    )


def load_course_map(path: str | Path) -> list[MapGate]:
    """Load, validate, and route-order the course map. Raises ValueError on
    any contract violation. Returns gates sorted by route_order (1..N)."""
    with Path(path).open("r", encoding="utf-8") as source:
        data = json.load(source)
    if data.get("version") != 1:
        raise ValueError("map version must be 1")
    if data.get("frame") != EXPECTED_FRAME:
        raise ValueError(f"map frame must be {EXPECTED_FRAME!r}")
    if data.get("units") != "m":
        raise ValueError("map units must be 'm'")
    raw_gates = data.get("gates")
    if not isinstance(raw_gates, list) or not raw_gates:
        raise ValueError("gates must be a non-empty list")

    gates = [_parse_gate(raw, i) for i, raw in enumerate(raw_gates)]

    ids = [g.id for g in gates]
    if len(set(ids)) != len(ids):
        raise ValueError("gate ids must be unique")
    orders = sorted(g.route_order for g in gates)
    if orders != list(range(1, len(gates) + 1)):
        raise ValueError(
            f"route_order must be exactly 1..{len(gates)} with no gaps or dups"
        )
    gates.sort(key=lambda g: g.route_order)
    return gates


def anchor_warnings(gates: list[MapGate]) -> list[str]:
    """Cross-check against judge-verified anchors. Returns human-readable
    warnings for disagreements > ANCHOR_WARN_M; empty list = consistent."""
    warnings = []
    for gate in gates:
        anchor = VERIFIED_ANCHORS.get(gate.route_order)
        if anchor is None:
            continue
        distance = float(np.linalg.norm(np.asarray(gate.pos) - anchor))
        if distance > ANCHOR_WARN_M:
            warnings.append(
                f"gate {gate.id} (route {gate.route_order}) is {distance:.2f} m "
                f"from the judge-verified anchor {anchor.tolist()} - judge wins"
            )
    return warnings


def gates_world(gates: list[MapGate]) -> np.ndarray:
    """Route-ordered (N, 3) array of aperture centers - the vq2wp GATES_W shape."""
    return np.array([g.pos for g in gates], dtype=float)


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map_json", type=Path)
    arguments = parser.parse_args(argv)
    gates = load_course_map(arguments.map_json)
    for warning in anchor_warnings(gates):
        print(f"WARNING: {warning}")
    for gate in gates:
        print(
            f"route {gate.route_order}: {gate.id} pos={list(gate.pos)} "
            f"normal={[round(v, 3) for v in gate.normal]} "
            f"aperture={gate.aperture_m} ({gate.confidence}, "
            f"sigma={gate.pos_sigma_m})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
