# sim/assets — built warehouse URDF bundles

Each `warehouse_*_v*/` subdirectory holds a built bundle ready for the PyBullet
runtime loader (`sim/pybullet/warehouse_loader.py`). Bundles are **derived
artifacts** — gitignored, reconstructed on demand from the source FBX +
gate YAML.

## Bundle contents

```
warehouse_<name>_v1/
├── warehouse.obj            # Triangulated collision mesh, ENU metres
├── warehouse.urdf           # Fixed-base link wrapping warehouse.obj
├── gate_ring.obj            # Procedural torus, 0.75 m inner / 0.85 m outer
├── gate.urdf                # Shared across all gates in this bundle
├── gates_enu.json           # Per-gate ENU pose + radii
├── gates_ned_intermediate.json  # Debug artifact (NED before ENU flip)
└── manifest.yaml            # Provenance — input hashes, build params, git SHA
```

## Reconstructing `warehouse_fab_v1`

Source assets:
- `data/warehouse_fab/warehouse.fbx` — exported from UE Editor (Industrial
  Warehouse FAB asset, all `Ind_War_*` PLAs unpacked before export)
- `configs/warehouse/warehouse_fab_v1_gates_ue.yaml` — UE-frame gate
  + PlayerStart transforms

Build:

```powershell
$env:PYTHONPATH = "."
conda run -n monorace python -m scripts.warehouse_to_urdf `
    --mesh data/warehouse_fab/warehouse.fbx `
    --gates-ue-yaml configs/warehouse/warehouse_fab_v1_gates_ue.yaml `
    --out sim/assets/warehouse_fab_v1
```

Build time ~30 s on a modern Windows laptop. Output is ~300 MB
(`warehouse.obj` dominates) — too large for git, hence the gitignore.

The `--clip-margin-m` default (15 m) is sized for warehouse-scale scenes.
Smaller scenes can use a tighter margin to drop more far-away geometry.

## Verifying a bundle loads

```powershell
$env:PYTHONPATH = "."
conda run -n monorace python scripts/_validate_warehouse_load.py
```

Should print the warehouse + 5 gate body IDs and the warehouse AABB.
For `warehouse_fab_v1` expect AABB extent ≈ 6.6 × 14.6 × 10.5 m (ENU).

## Source / pipeline

Build pipeline source: `scripts/warehouse_to_urdf/`. Mesh-direct path
documented in `docs/superpowers/plans/2026-04-23-warehouse-to-pybullet-mesh-direct.md`
(COR-95).
