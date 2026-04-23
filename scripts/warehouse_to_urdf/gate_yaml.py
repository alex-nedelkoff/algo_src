"""UE YAML → NED gate JSON converter.

Reads a UE-frame gate YAML (as authored in UE Editor) and writes a NED-frame
JSON matching the COR-92 gate schema. Downstream code in scripts/warehouse_to_urdf
reads the NED JSON directly; this converter is the only place that knows about
the UE YAML format.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sim.pybullet.coords import ue_to_ned_position, ue_to_ned_quaternion

_INNER_RADIUS_M = 0.75
_OUTER_RADIUS_M = 0.85


def convert_ue_yaml_to_ned_json(
    yaml_path: Path | str,
    out_json_path: Path | str,
) -> dict[str, Any]:
    """Read a UE-frame gate YAML, convert to NED, write JSON matching the COR-92 schema.

    Returns the written dict (for in-process callers / testing).

    Args:
        yaml_path: Path to the UE-frame gate YAML file.
        out_json_path: Destination path for the NED JSON output.

    Returns:
        The gate dict that was written to *out_json_path*.

    Raises:
        ValueError: If required top-level keys or per-gate keys are missing.
    """
    yaml_path = Path(yaml_path)
    out_json_path = Path(out_json_path)

    # Explicit utf-8 — the YAML may contain non-ASCII comments; Path.read_text
    # without an encoding falls back to the platform locale (cp1252 on Windows)
    # and can UnicodeDecodeError on a perfectly valid file.
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    if "playerstart" not in raw:
        raise ValueError(
            f"YAML at {yaml_path} is missing required top-level key 'playerstart'."
        )
    if "gates" not in raw:
        raise ValueError(
            f"YAML at {yaml_path} is missing required top-level key 'gates'."
        )

    ps = raw["playerstart"]
    if "location_cm" not in ps:
        raise ValueError("'playerstart' is missing required key 'location_cm'.")
    if "rotation_deg_rpy" not in ps:
        raise ValueError("'playerstart' is missing required key 'rotation_deg_rpy'.")

    ps_location = np.array(ps["location_cm"], dtype=np.float64)
    ps_rotation = np.array(ps["rotation_deg_rpy"], dtype=np.float64)

    out: dict[str, Any] = {}
    for gate_name, gate_data in raw["gates"].items():
        if "location_cm" not in gate_data:
            raise ValueError(
                f"Gate '{gate_name}' is missing required key 'location_cm'."
            )
        if "rotation_deg_rpy" not in gate_data:
            raise ValueError(
                f"Gate '{gate_name}' is missing required key 'rotation_deg_rpy'."
            )

        pos_ned = ue_to_ned_position(
            np.array(gate_data["location_cm"], dtype=np.float64),
            ps_location,
        )
        q_ned = ue_to_ned_quaternion(
            np.array(gate_data["rotation_deg_rpy"], dtype=np.float64),
            ps_rotation,
        )

        out[gate_name] = {
            "position_ned": pos_ned.tolist(),
            "orientation_wxyz": q_ned.tolist(),
            "inner_radius_m": _INNER_RADIUS_M,
            "outer_radius_m": _OUTER_RADIUS_M,
        }

    out_json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
