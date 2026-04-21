"""Runtime PyBullet loader for the warehouse + gate URDFs.

This module turns the build artifacts under `sim/assets/warehouse_v1/`
into PyBullet bodies. It owns no env logic, no physics step rate, no
control. Whatever wraps this loader (a future env class) decides those.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pybullet as p


@dataclass
class WarehouseHandles:
    warehouse_body_id: int
    gate_body_ids: list[int] = field(default_factory=list)
    gate_names: list[str] = field(default_factory=list)


@dataclass
class WarehouseScene:
    asset_dir: Path

    def load_into(self, client_id: int) -> WarehouseHandles:
        asset_dir = Path(self.asset_dir)
        p.setAdditionalSearchPath(str(asset_dir), physicsClientId=client_id)

        warehouse_id = p.loadURDF(
            str(asset_dir / "warehouse.urdf"),
            basePosition=[0, 0, 0],
            useFixedBase=True,
            physicsClientId=client_id,
        )

        gates = json.loads((asset_dir / "gates_enu.json").read_text())
        gate_ids: list[int] = []
        gate_names: list[str] = []
        for name in sorted(gates.keys()):
            g = gates[name]
            # gates_enu.json stores wxyz; PyBullet expects xyzw.
            qw, qx, qy, qz = g["orientation_enu_wxyz"]
            gate_id = p.loadURDF(
                str(asset_dir / "gate.urdf"),
                basePosition=g["position_enu"],
                baseOrientation=[qx, qy, qz, qw],
                useFixedBase=True,
                physicsClientId=client_id,
            )
            gate_ids.append(gate_id)
            gate_names.append(name)

        return WarehouseHandles(
            warehouse_body_id=warehouse_id,
            gate_body_ids=gate_ids,
            gate_names=gate_names,
        )
