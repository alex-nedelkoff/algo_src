# PyTorch3D Rasterizer PoC — Fast Rendering for Perception-in-the-Loop RL

**Date**: 2026-03-18
**Status**: Draft
**Author**: Claude + Janahan
**Related**: COR-62 (Perception-in-the-Loop RL Training), COR-61 (Gate Detection Training Data Pipeline)

## Problem

COR-62 requires rendering gate scenes at ~1-10ms per frame so DA3 perception inference can run during PPO training. The COR-61 BlenderProc Cycles pipeline (1.3s/frame at 16 samples on RTX 3090) is 100-1000x too slow. Before wiring a fast renderer into the training loop, we need to answer a basic visual fidelity question: **does a ~5ms PyTorch3D rasterized render look good enough for DA3 to detect gates?**

The existing `sim/viz` pipeline already logs 3D scenes + a drone camera view to rerun. The current camera view is a Bresenham wireframe on black (`pinhole.py`'s `render_wireframe`). This PoC replaces that with a PyTorch3D-rasterized RGB image and upgrades the 3D panel from wireframes to solid meshes.

## Solution

Integrate a PyTorch3D rasterizer into the existing `sim/viz` pipeline:

1. **New `SceneRenderer` class** (`sim/viz/rasterizer.py`) — holds gate/drone/floor meshes as PyTorch3D `Meshes` objects, renders RGB images given camera pose and scene layout.
2. **Upgrade `rerun_generator.py`** — log solid `rr.Mesh3D` in the 3D panel (replacing wireframe `rr.LineStrips3D`), use `SceneRenderer` for the camera image (replacing `render_wireframe`).
3. **CLI flag** — `--renderer wireframe|pytorch3d` on `python -m sim.viz` (default: `wireframe` for backward compat).

The `SceneRenderer` class is designed to be reusable: the same interface that renders for rerun viz will later render for perception-in-the-loop training (COR-62 Phase 2).

## Design

### SceneRenderer class (`sim/viz/rasterizer.py`)

```python
class SceneRenderer:
    """PyTorch3D rasterizer for gate racing scenes.

    Holds pre-loaded meshes (gates, drones, floor) and renders RGB images
    from arbitrary camera poses. Designed for both offline viz (rerun) and
    eventual online use in the training loop.
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (240, 320),  # (H, W)
        device: str = "cuda",
    ) -> None: ...

    def set_scene(
        self,
        gate_positions: np.ndarray,      # (G, 3)
        gate_orientations: np.ndarray,   # (G, 4) wxyz
        gate_half_extents: np.ndarray,   # (G, 2)
        gate_colors: list[tuple[int, int, int]] | None = None,
    ) -> None:
        """Configure the static scene geometry (called once per trajectory).

        gate_colors defaults to the _GATE_COLORS cycle from rerun_generator.py
        when None.
        """

    def render(
        self,
        camera_pos: np.ndarray,    # (3,)
        camera_quat: np.ndarray,   # (4,) wxyz
    ) -> np.ndarray:
        """Render the scene from a camera pose.

        Returns:
            (H, W, 3) uint8 RGB image.
        """
```

### Scene composition

Matching the BlenderProc pipeline's elements at rasterization fidelity:

| Element | Mesh source | Rendering |
|---------|------------|-----------|
| Gates (1-3) | `gate_mesh.generate_gate_mesh()` → trimesh → PyTorch3D `Meshes` | Phong-shaded, LED-style solid colors per gate |
| Floor | Procedural quad (2 triangles, 40m x 40m at z=0) | Solid gray color |
| Drone obstacles | `gate_mesh.generate_drone_mesh()` → trimesh → PyTorch3D `Meshes` | Solid dark color (not rendered in v1 — only gates matter for DA3) |
| Lighting | `pytorch3d.renderer.PointLights` | Single point light above scene center |
| Background | Rasterizer background color | Dark gray (not black — avoids trivial segmentation) |

**Explicit non-goals for v1:** No procedural textures, no shadows, no multiple light sources, no drone obstacle meshes. Solid colors only. This is the low end of the fidelity spectrum — if DA3 can detect gates here, it can detect them anywhere.

### PyTorch3D rendering pipeline

```
trimesh.Trimesh (from gate_mesh.py)
  → vertices (V, 3) + faces (F, 3) as torch tensors
  → pytorch3d.structures.Meshes
  → Transform to world pose (per gate position/orientation)
  → pytorch3d.renderer.MeshRasterizer (RasterizationSettings: image_size, bin_size)
  → pytorch3d.renderer.MeshRenderer with:
      - HardPhongShader (ambient + diffuse + specular)
      - PointLights
  → (H, W, 4) float tensor → [:, :, :3] → uint8 RGB
```

Camera setup uses `pytorch3d.renderer.cameras.FoVPerspectiveCameras` with `fov=90` (matching the existing `PinholeCamera` HFOV) and the drone's world-frame pose converted to PyTorch3D's R, T format.

### Coordinate system mapping

Three coordinate systems must align:

**1. Sim body frame** (`sim/viz/pinhole.py`):
- X=forward, Y=left, Z=up (FLU / aerospace convention)
- Quaternions: (w, x, y, z)

**2. Gate mesh local frame** (`gate_mesh.py`):
- Mesh lies in XY plane, centered at origin. Width along X, height along Y, depth along Z.
- The mesh normal is along **Z**, but the sim treats the gate normal as the local **X-axis**.
- **Transform required**: Before placing a gate mesh in the scene, rotate 90 degrees to align the mesh Z-normal with the sim's X-normal convention. Concretely: apply `R_mesh_to_sim = Ry(+90deg)` (rotate around Y so Z maps to X), then apply the per-gate world-frame quaternion.

**3. PyTorch3D camera frame**:
- X=right, Y=up, Z=**out of screen** (camera looks along **-Z**)
- Uses rotation matrices (3x3), not quaternions
- `FoVPerspectiveCameras` is preferred over `PerspectiveCameras` since the codebase parameterizes cameras by HFOV (90 degrees)

**Sim body-to-PyTorch3D camera mapping** (applied in `SceneRenderer.render`):
```
sim body:  X=forward, Y=left,  Z=up
PT3D cam:  X=right,   Y=up,    Z=out-of-screen (looking along -Z)

Mapping: cam_X = -body_Y, cam_Y = body_Z, cam_Z = -body_X
```
This is the same mapping used in `pinhole.py` (lines 98-101) but with Z negated because PyTorch3D looks along -Z instead of +Z.

**NED vs FLU note**: `perception/camera.py` uses NED convention (X-fwd, Y-right, Z-down) while `pinhole.py` uses FLU (X-fwd, Y-left, Z-up). The `SceneRenderer` follows the FLU/viz convention. Training loop integration (future) will need Y and Z sign flips to match NED.

### Mesh packing strategy

All scene geometry (gates + floor) is packed into a single `Meshes` object via `pytorch3d.structures.join_meshes_as_scene()` for efficient single-pass rasterization. Per-gate vertex transforms (mesh-local → sim-world) are applied in numpy before tensor creation:

```
For each gate:
  1. Load base gate trimesh vertices (V, 3)
  2. Apply R_mesh_to_sim (Z-normal → X-normal rotation)
  3. Apply per-gate world rotation (from quaternion)
  4. Translate to per-gate world position
  5. Append to vertex/face lists with face index offsets
Concatenate all → single Meshes object
```

The `SceneRenderer` handles this conversion internally. The public API accepts the same coordinate conventions as the rest of the sim (`pinhole.py`'s body frame conventions).

### Rerun integration

#### 3D panel upgrades

Replace wireframe gate logging with solid meshes:

```python
# Current (wireframe):
rr.log(f"world/gates/gate_{g}", rr.LineStrips3D([loop], colors=[color]), static=True)

# New (solid mesh):
mesh = generate_gate_mesh()
rr.log(f"world/gates/gate_{g}", rr.Mesh3D(
    vertex_positions=transformed_vertices,
    triangle_indices=mesh.faces,
    vertex_colors=np.full((len(mesh.vertices), 3), color, dtype=np.uint8),
), static=True)
```

Similarly for the drone — `rr.Mesh3D` replaces `rr.Boxes3D`. A floor quad mesh is added as a static entity.

Gate markers (center spheres, stalks, ground crosses) are kept alongside meshes for visibility from all angles.

#### Camera panel

```python
# Current:
frame = render_wireframe(cam, all_edges, cam_pos, cam_quat)
rr.log("drone/camera", rr.Image(frame))

# New (pytorch3d mode):
frame = scene_renderer.render(cam_pos, cam_quat)
rr.log("drone/camera", rr.Image(frame))
```

Additionally, log a `rr.Pinhole` entity so the rerun 3D panel can show the camera frustum. The pinhole intrinsics are static but live under the drone transform hierarchy so the frustum tracks the drone:

```python
# Pinhole intrinsics (static — intrinsics don't change per frame)
rr.log("world/drone/camera", rr.Pinhole(
    focal_length=[cam.fx, cam.fy],
    principal_point=[cam.cx, cam.cy],
    resolution=[cam.width, cam.height],
), static=True)

# Camera image logged under the same entity path (linked in rerun viewer)
rr.log("world/drone/camera", rr.Image(frame))
```

Note: this moves the image from `"drone/camera"` to `"world/drone/camera"` so the frustum visualization and camera image share the same entity path under the drone transform hierarchy. The wireframe renderer (`--renderer wireframe`) continues using the existing `"drone/camera"` path.

### CLI interface

```bash
# Default (backward compatible):
python -m sim.viz trajectory.npz

# With PyTorch3D rasterizer:
python -m sim.viz trajectory.npz --renderer pytorch3d

# Both renderers still respect camera decimation:
python -m sim.viz trajectory.npz --renderer pytorch3d --camera-decimation 5
```

### Dependency management

PyTorch3D is an optional dependency (like rerun). Import is guarded:

```python
try:
    import pytorch3d
    _HAS_PYTORCH3D = True
except ImportError:
    _HAS_PYTORCH3D = False
```

Requesting `--renderer pytorch3d` without the package installed gives a clear error message with install instructions.

## Files

| File | Change | Lines (est.) |
|------|--------|-------------|
| `sim/viz/rasterizer.py` | **New** — `SceneRenderer` class | ~200-250 |
| `sim/viz/rerun_generator.py` | Add `renderer` param, mesh logging, `SceneRenderer` integration | ~60 modified |
| `sim/viz/__main__.py` | Add `--renderer` CLI arg | ~5 |

No changes to: `GateRaceEnv`, `numpy_quad.yaml`, `TrajectoryRecorderCallback`, `.npz` format, `pinhole.py`, or any training code.

## Testing

- **Smoke test**: Unit test that `SceneRenderer` instantiates and `render()` returns `(240, 320, 3)` uint8 array. Marked `@pytest.mark.skipif(not torch.cuda.is_available())`.
- **Visual validation**: Run `python -m sim.viz <existing_trajectory.npz> --renderer pytorch3d` and inspect the `.rrd` in rerun viewer. Verify gates are visible as solid shaded meshes from drone camera POV.
- **Regression**: Run with `--renderer wireframe` (default) and verify output is unchanged.

## Dependency note

PyTorch3D is installed inside the project Docker container (not local conda/venv per project policy). The viz pipeline may run locally for interactive use — PyTorch3D import guards allow graceful degradation to wireframe mode.

## Future work (not in this PoC)

- **Training loop integration**: Call `SceneRenderer.render()` from the PPO loop at ~50Hz across 100 envs (batched rendering)
- **DA3 inference validation**: Feed rendered frames to DA3 and compare detections against ground-truth gate poses
- **Timing benchmarks**: Measure actual render latency on RTX 3090 across batch sizes
- **Fidelity improvements**: UV textures, multiple lights, procedural backgrounds — if v1 fidelity proves insufficient
- **Batched rendering**: `render_batch(camera_poses)` for vectorized multi-env rendering
