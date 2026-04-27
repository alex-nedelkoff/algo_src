"""Interactive Open3D viewer for warehouse.obj. Drag = rotate, scroll = zoom, Shift+drag = pan."""
from pathlib import Path
import open3d as o3d

OBJ = Path("sim/assets/warehouse_fab_v1/warehouse.obj")

mesh = o3d.io.read_triangle_mesh(str(OBJ))
mesh.compute_vertex_normals()
print(f"Loaded {len(mesh.triangles):,} triangles. Opening viewer...")
o3d.visualization.draw_geometries([mesh], window_name="warehouse.obj")
