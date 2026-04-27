"""Diagnose what's actually inside warehouse.obj after the pipeline."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import open3d as o3d

OBJ = Path("sim/assets/warehouse_fab_v1/warehouse.obj")


def main() -> int:
    mesh = o3d.io.read_triangle_mesh(str(OBJ))
    print(f"file: {OBJ}  size: {OBJ.stat().st_size / 1e6:.1f} MB")
    print(f"triangles: {len(mesh.triangles):,}")
    print(f"vertices:  {len(mesh.vertices):,}")

    aabb = mesh.get_axis_aligned_bounding_box()
    print(f"AABB min: {aabb.min_bound}")
    print(f"AABB max: {aabb.max_bound}")
    extent = aabb.max_bound - aabb.min_bound
    print(f"extent:   {extent}")

    # Connected components — find isolated chunks of geometry
    labels = np.asarray(mesh.cluster_connected_triangles()[0])
    n_components = labels.max() + 1 if len(labels) > 0 else 0
    print(f"\nconnected components: {n_components}")

    # Size distribution of components
    if n_components > 0:
        sizes = np.bincount(labels)
        sizes_sorted = np.sort(sizes)[::-1]
        print(f"largest 10 components (triangle counts): {sizes_sorted[:10].tolist()}")
        big_threshold = 1000
        n_big = (sizes >= big_threshold).sum()
        n_small = (sizes < big_threshold).sum()
        print(f"components >= {big_threshold} tris: {n_big}")
        print(f"components <  {big_threshold} tris: {n_small} (likely fragments/lines)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
