"""One-off: try all available loaders on the warehouse FBX."""
from __future__ import annotations
import sys
from pathlib import Path

FBX = Path("data/warehouse_fab/Warehouse_Track_Flat.fbx")


def try_open3d():
    print("=== Open3D ===")
    try:
        import open3d as o3d  # type: ignore
        print("open3d version:", o3d.__version__)
        m = o3d.io.read_triangle_mesh(str(FBX))
        print(f"triangles: {len(m.triangles):,}")
        print(f"vertices:  {len(m.vertices):,}")
        if len(m.triangles) > 0:
            aabb = m.get_axis_aligned_bounding_box()
            print(f"bounds min (UE cm): {aabb.min_bound}")
            print(f"bounds max (UE cm): {aabb.max_bound}")
            print(f"extent (m):         {(aabb.max_bound - aabb.min_bound) / 100.0}")
        else:
            print("NO TRIANGLES LOADED — Open3D can't read this FBX")
    except Exception as e:
        print(f"failed: {type(e).__name__}: {e}")


def try_pyassimp():
    print("\n=== pyassimp ===")
    try:
        import pyassimp  # type: ignore
        with pyassimp.load(str(FBX)) as scene:
            print(f"meshes in scene: {len(scene.meshes)}")
            total_faces = sum(len(m.faces) for m in scene.meshes)
            total_verts = sum(len(m.vertices) for m in scene.meshes)
            print(f"total faces:    {total_faces:,}")
            print(f"total vertices: {total_verts:,}")
    except ImportError:
        print("pyassimp not installed")
    except Exception as e:
        print(f"failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    try_open3d()
    try_pyassimp()
    sys.exit(0)
