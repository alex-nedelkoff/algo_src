# PyTorch3D Rasterizer PoC — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the wireframe camera view in `sim/viz` with PyTorch3D-rasterized RGB and upgrade 3D panel gates from wireframes to solid meshes, so we can evaluate visual fidelity for perception-in-the-loop RL training.

**Architecture:** A new `SceneRenderer` class wraps PyTorch3D's `MeshRenderer` to produce RGB images from gate scene meshes. It integrates into the existing `rerun_generator.py` pipeline as an alternative camera renderer. The 3D panel is upgraded to log `rr.Mesh3D` entities alongside existing markers.

**Tech Stack:** Python 3.11, PyTorch3D (optional dep), trimesh, rerun-sdk, NumPy, PyTorch

**Spec:** `docs/superpowers/specs/2026-03-18-pytorch3d-rasterizer-poc-design.md`

**Linear:** COR-62

---

## File Structure

```
sim/viz/
  rasterizer.py            # NEW — SceneRenderer class (PyTorch3D wrapper)
  rerun_generator.py       # MODIFY — add renderer param, mesh logging, SceneRenderer integration
  __main__.py              # MODIFY — add --renderer CLI arg
  pinhole.py               # UNCHANGED
  __init__.py              # UNCHANGED
tests/
  test_sim/
    test_rasterizer.py     # NEW — smoke tests for SceneRenderer
```

---

### Task 1: SceneRenderer — trimesh-to-PyTorch3D conversion and coordinate transforms

**Files:**
- Create: `sim/viz/rasterizer.py`
- Create: `tests/test_sim/test_rasterizer.py`

This task builds the `SceneRenderer` class with `__init__`, `set_scene`, and `render` methods. The hardest part is the coordinate system mapping (gate mesh local frame → sim world frame → PyTorch3D camera frame).

- [ ] **Step 1: Write failing tests for SceneRenderer**

```python
# tests/test_sim/test_rasterizer.py
"""Smoke tests for the PyTorch3D SceneRenderer."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for PyTorch3D rendering", allow_module_level=True)

pytest.importorskip("pytorch3d")

from sim.viz.rasterizer import SceneRenderer, _R_MESH_TO_SIM, _BODY_TO_PT3D


class TestCoordinateTransforms:
    """Verify coordinate system rotation matrices are correct."""

    def test_mesh_to_sim_maps_z_to_x(self):
        """_R_MESH_TO_SIM should map Z-normal (gate_mesh.py) to X-normal (sim)."""
        z_axis = np.array([0.0, 0.0, 1.0])
        result = _R_MESH_TO_SIM @ z_axis
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0], atol=1e-10)

    def test_mesh_to_sim_preserves_y(self):
        """Y axis (height) should be unchanged by Ry rotation."""
        y_axis = np.array([0.0, 1.0, 0.0])
        result = _R_MESH_TO_SIM @ y_axis
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-10)

    def test_body_to_pt3d_forward_maps_to_neg_z(self):
        """Body forward (+X) should map to camera -Z (into screen)."""
        body_forward = np.array([1.0, 0.0, 0.0])
        result = _BODY_TO_PT3D @ body_forward
        np.testing.assert_allclose(result, [0.0, 0.0, -1.0], atol=1e-10)

    def test_body_to_pt3d_right_maps_to_x(self):
        """Body right (-Y) should map to camera +X."""
        body_right = np.array([0.0, -1.0, 0.0])
        result = _BODY_TO_PT3D @ body_right
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0], atol=1e-10)

    def test_body_to_pt3d_up_maps_to_y(self):
        """Body up (+Z) should map to camera +Y."""
        body_up = np.array([0.0, 0.0, 1.0])
        result = _BODY_TO_PT3D @ body_up
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-10)


class TestSceneRendererInit:
    """Test SceneRenderer construction."""

    def test_init_default(self):
        renderer = SceneRenderer()
        assert renderer.image_size == (240, 320)

    def test_init_custom_size(self):
        renderer = SceneRenderer(image_size=(480, 640))
        assert renderer.image_size == (480, 640)


class TestSceneRendererSetScene:
    """Test scene configuration from gate geometry."""

    def test_set_scene_single_gate(self):
        renderer = SceneRenderer()
        renderer.set_scene(
            gate_positions=np.array([[5.0, 0.0, 1.5]]),
            gate_orientations=np.array([[1.0, 0.0, 0.0, 0.0]]),  # identity quat
            gate_half_extents=np.array([[0.75, 0.75]]),
        )
        # Scene mesh should be set (gates + floor)
        assert renderer._scene_mesh is not None

    def test_set_scene_multiple_gates(self):
        renderer = SceneRenderer()
        renderer.set_scene(
            gate_positions=np.array([
                [5.0, 0.0, 1.5],
                [10.0, 3.0, 1.5],
                [7.0, -2.0, 2.0],
            ]),
            gate_orientations=np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.707, 0.0, 0.0, 0.707],  # 90 deg yaw
                [1.0, 0.0, 0.0, 0.0],
            ]),
            gate_half_extents=np.array([
                [0.75, 0.75],
                [0.75, 0.75],
                [0.75, 0.75],
            ]),
        )
        assert renderer._scene_mesh is not None


class TestSceneRendererRender:
    """Test rendering produces correct output shape and dtype."""

    @pytest.fixture()
    def renderer_with_scene(self):
        renderer = SceneRenderer(image_size=(240, 320))
        renderer.set_scene(
            gate_positions=np.array([[5.0, 0.0, 1.5]]),
            gate_orientations=np.array([[1.0, 0.0, 0.0, 0.0]]),
            gate_half_extents=np.array([[0.75, 0.75]]),
        )
        return renderer

    def test_render_output_shape(self, renderer_with_scene):
        frame = renderer_with_scene.render(
            camera_pos=np.array([0.0, 0.0, 1.5]),
            camera_quat=np.array([1.0, 0.0, 0.0, 0.0]),  # facing +X
        )
        assert frame.shape == (240, 320, 3)
        assert frame.dtype == np.uint8

    def test_render_not_all_black(self, renderer_with_scene):
        """Camera facing a gate 5m ahead should produce non-trivial image."""
        frame = renderer_with_scene.render(
            camera_pos=np.array([0.0, 0.0, 1.5]),
            camera_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        # At least some pixels should be non-background
        assert frame.max() > 50, "Image appears all-dark — gate likely not visible"

    def test_render_gate_behind_camera_is_dark(self, renderer_with_scene):
        """Camera facing away from gate should show mostly background."""
        # Rotate 180 degrees around Z (yaw = pi): (w,x,y,z) = (0,0,0,1)
        frame = renderer_with_scene.render(
            camera_pos=np.array([0.0, 0.0, 1.5]),
            camera_quat=np.array([0.0, 0.0, 0.0, 1.0]),  # 180 deg yaw
        )
        # Gate is behind camera — should be mostly background (dark gray)
        # Allow some floor visibility; threshold is empirical, may need tuning
        bright_pixels = np.sum(frame > 100)
        total_pixels = frame.shape[0] * frame.shape[1] * frame.shape[2]
        assert bright_pixels / total_pixels < 0.15, "Too many bright pixels when gate is behind camera"

    def test_render_before_set_scene_raises(self):
        """Calling render() before set_scene() should raise RuntimeError."""
        renderer = SceneRenderer()
        with pytest.raises(RuntimeError, match="set_scene"):
            renderer.render(
                camera_pos=np.array([0.0, 0.0, 1.5]),
                camera_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker exec algo_src pytest tests/test_sim/test_rasterizer.py -v 2>&1 | head -30`
Expected: FAIL with `ModuleNotFoundError: No module named 'sim.viz.rasterizer'`

- [ ] **Step 3: Implement SceneRenderer**

```python
# sim/viz/rasterizer.py
"""PyTorch3D rasterizer for gate racing scenes.

Renders RGB images of gate scenes from arbitrary camera poses using
PyTorch3D's MeshRenderer with Phong shading. Designed for both offline
rerun visualization and eventual online use in the training loop.

Coordinate conventions:
  - Public API uses sim body frame: X=forward, Y=left, Z=up (FLU)
  - Quaternions: (w, x, y, z) convention
  - Gate mesh local frame (gate_mesh.py): XY plane, normal along Z
  - Internally converts to PyTorch3D camera: X=right, Y=up, Z=out-of-screen
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from sim.viz.pinhole import quat_to_rotation_matrix

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)

# Gate color cycle (shared with rerun_generator.py)
GATE_COLORS: list[tuple[int, int, int]] = [
    (255, 80, 80),    # red
    (80, 255, 80),    # green
    (80, 80, 255),    # blue
    (255, 255, 80),   # yellow
    (255, 80, 255),   # magenta
    (80, 255, 255),   # cyan
    (255, 160, 80),   # orange
    (160, 80, 255),   # purple
]

# Background color (dark gray, not black — avoids trivial segmentation)
_BG_COLOR = (0.15, 0.15, 0.15)

# Floor color
_FLOOR_COLOR = (0.3, 0.3, 0.3)

# Floor dimensions (meters)
_FLOOR_HALF_SIZE = 20.0


def _make_floor_mesh() -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Create a floor quad (2 triangles) at z=0.

    Returns:
        Tuple of (vertices (4, 3), faces (2, 3)).
    """
    s = _FLOOR_HALF_SIZE
    verts = np.array([
        [-s, -s, 0.0],
        [+s, -s, 0.0],
        [+s, +s, 0.0],
        [-s, +s, 0.0],
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.int64)
    return verts, faces


# Rotation matrix to convert gate mesh from gate_mesh.py local frame
# (normal along Z) to sim local frame (normal along X).
# This is Ry(+90deg): rotates Z axis toward X axis.
_R_MESH_TO_SIM = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
], dtype=np.float64)

# Body-to-PyTorch3D camera rotation.
# Sim body: X=forward, Y=left, Z=up
# PT3D cam: X=right, Y=up, Z=out-of-screen (looks along -Z)
# Mapping: cam_X = -body_Y, cam_Y = +body_Z, cam_Z = -body_X
_BODY_TO_PT3D = np.array([
    [0.0, -1.0, 0.0],   # cam_X = -body_Y
    [0.0,  0.0, 1.0],   # cam_Y = +body_Z
    [-1.0, 0.0, 0.0],   # cam_Z = -body_X
], dtype=np.float64)


class SceneRenderer:
    """PyTorch3D rasterizer for gate racing scenes.

    Holds pre-loaded meshes (gates, floor) and renders RGB images from
    arbitrary camera poses.

    Args:
        image_size: (height, width) of rendered images.
        device: PyTorch device string ("cuda" or "cpu").
        hfov_deg: Horizontal field of view in degrees.
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (240, 320),
        device: str = "cuda",
        hfov_deg: float = 90.0,
    ) -> None:
        import torch
        from pytorch3d.renderer import (
            FoVPerspectiveCameras,
            HardPhongShader,
            MeshRasterizer,
            MeshRenderer,
            PointLights,
            RasterizationSettings,
        )

        self.image_size = image_size
        self.device = torch.device(device)
        self.hfov_deg = hfov_deg

        # Build renderer components (camera/lights set per-render call)
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        self._rasterizer = MeshRasterizer(raster_settings=raster_settings)
        self._shader_class = HardPhongShader
        self._lights = PointLights(
            device=self.device,
            location=[[0.0, 0.0, 10.0]],  # above scene center
            ambient_color=((0.4, 0.4, 0.4),),
            diffuse_color=((0.6, 0.6, 0.6),),
            specular_color=((0.2, 0.2, 0.2),),
        )
        self._bg_color = _BG_COLOR

        # Scene mesh (set by set_scene)
        self._scene_mesh = None

    def set_scene(
        self,
        gate_positions: NDArray[np.float64],
        gate_orientations: NDArray[np.float64],
        gate_half_extents: NDArray[np.float64],
        gate_colors: list[tuple[int, int, int]] | None = None,
    ) -> None:
        """Configure the static scene geometry.

        Builds a single packed PyTorch3D Meshes object containing all gates
        and a floor plane. Called once per trajectory.

        Args:
            gate_positions: (G, 3) gate center positions in world frame.
            gate_orientations: (G, 4) gate quaternions (w, x, y, z).
            gate_half_extents: (G, 2) [half_width, half_height] per gate.
            gate_colors: Per-gate RGB colors. Defaults to GATE_COLORS cycle.
        """
        import torch
        from pytorch3d.structures import Meshes

        from perception.training.data.gate_detection.gate_mesh import (
            generate_gate_mesh,
        )

        num_gates = gate_positions.shape[0]
        if gate_colors is None:
            gate_colors = [GATE_COLORS[g % len(GATE_COLORS)] for g in range(num_gates)]

        all_verts: list[NDArray[np.float64]] = []
        all_faces: list[NDArray[np.int64]] = []
        all_colors: list[NDArray[np.float64]] = []
        vert_offset = 0

        # Add gate meshes
        for g in range(num_gates):
            hw, hh = gate_half_extents[g]
            mesh = generate_gate_mesh(
                inner_width=float(hw * 2),
                inner_height=float(hh * 2),
            )
            verts = np.array(mesh.vertices, dtype=np.float64)
            faces = np.array(mesh.faces, dtype=np.int64)

            # Transform: mesh local (Z-normal) → sim local (X-normal) → world
            R_gate = quat_to_rotation_matrix(gate_orientations[g])
            verts = (verts @ _R_MESH_TO_SIM.T) @ R_gate.T + gate_positions[g]

            # Per-vertex color (normalized to [0, 1])
            color = np.array(gate_colors[g], dtype=np.float64) / 255.0
            vert_colors = np.tile(color, (len(verts), 1))

            all_verts.append(verts)
            all_faces.append(faces + vert_offset)
            all_colors.append(vert_colors)
            vert_offset += len(verts)

        # Add floor
        floor_verts, floor_faces = _make_floor_mesh()
        floor_colors = np.tile(
            np.array(_FLOOR_COLOR, dtype=np.float64),
            (len(floor_verts), 1),
        )
        all_verts.append(floor_verts)
        all_faces.append(floor_faces + vert_offset)
        all_colors.append(floor_colors)

        # Pack into single Meshes object
        packed_verts = torch.tensor(
            np.concatenate(all_verts, axis=0), dtype=torch.float32, device=self.device,
        )
        packed_faces = torch.tensor(
            np.concatenate(all_faces, axis=0), dtype=torch.int64, device=self.device,
        )
        packed_colors = torch.tensor(
            np.concatenate(all_colors, axis=0), dtype=torch.float32, device=self.device,
        )

        from pytorch3d.renderer import TexturesVertex

        textures = TexturesVertex(verts_features=[packed_colors])
        self._scene_mesh = Meshes(
            verts=[packed_verts], faces=[packed_faces], textures=textures,
        )

        log.debug(
            "Scene built: %d gates, %d total verts, %d total faces",
            num_gates, len(packed_verts), len(packed_faces),
        )

    def render(
        self,
        camera_pos: NDArray[np.float64],
        camera_quat: NDArray[np.float64],
    ) -> NDArray[np.uint8]:
        """Render the scene from a camera pose.

        Args:
            camera_pos: (3,) camera position in world frame.
            camera_quat: (4,) camera orientation quaternion (w, x, y, z).
                Body frame: X=forward, Y=left, Z=up (FLU).

        Returns:
            (H, W, 3) uint8 RGB image.
        """
        if self._scene_mesh is None:
            raise RuntimeError("Call set_scene() before render()")

        import torch
        from pytorch3d.renderer import (
            FoVPerspectiveCameras,
            HardPhongShader,
            MeshRenderer,
        )

        # Build world-to-camera rotation matrix.
        # Column-vector convention: x_cam = R_w2c @ x_world + t
        R_body_to_world = quat_to_rotation_matrix(camera_quat)
        R_world_to_body = R_body_to_world.T
        R_world_to_cam = _BODY_TO_PT3D @ R_world_to_body  # (3, 3)

        # PyTorch3D uses ROW-VECTOR convention: x_cam = x_world @ R + T
        # So R_pt3d = R_world_to_cam.T and T_pt3d = -cam_pos @ R_pt3d
        R_pt3d = R_world_to_cam.T  # (3, 3) — transposed for row-vector convention
        T_pt3d = -camera_pos @ R_pt3d  # (3,)

        R_tensor = torch.tensor(
            R_pt3d[np.newaxis], dtype=torch.float32, device=self.device,
        )
        T_tensor = torch.tensor(
            T_pt3d[np.newaxis], dtype=torch.float32, device=self.device,
        )

        cameras = FoVPerspectiveCameras(
            R=R_tensor,
            T=T_tensor,
            fov=self.hfov_deg,
            device=self.device,
        )

        # Build renderer with current camera
        renderer = MeshRenderer(
            rasterizer=self._rasterizer,
            shader=HardPhongShader(
                device=self.device,
                cameras=cameras,
                lights=self._lights,
            ),
        )

        # Render
        with torch.no_grad():
            images = renderer(self._scene_mesh, cameras=cameras)  # (1, H, W, 4)

        # Extract RGB, clamp, convert to uint8
        rgb = images[0, :, :, :3].cpu().numpy()
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        return rgb
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker exec algo_src pytest tests/test_sim/test_rasterizer.py -v`
Expected: All 12 tests PASS (5 coordinate transform + 2 init + 2 set_scene + 3 render)

- [ ] **Step 5: Commit**

```bash
git add sim/viz/rasterizer.py tests/test_sim/test_rasterizer.py
git commit -m "feat(viz): add PyTorch3D SceneRenderer for rasterized gate scenes (COR-62)"
```

---

### Task 2: Upgrade rerun 3D panel — solid gate meshes + floor + drone mesh

**Files:**
- Modify: `sim/viz/rerun_generator.py:219-231` (gate logging block)
- Modify: `sim/viz/rerun_generator.py:299-303` (drone box logging)

This task upgrades the rerun 3D panel from wireframes to solid meshes. The camera panel is unchanged in this task (still wireframe). This can be tested by running `python -m sim.viz` on an existing trajectory and visually inspecting.

- [ ] **Step 1: Add mesh imports and a helper to transform gate mesh vertices**

At the top of `rerun_generator.py`, add:

```python
from perception.training.data.gate_detection.gate_mesh import (
    generate_gate_mesh,
    generate_drone_mesh,
)
```

Add a helper function after `_gate_wireframe_edges`:

```python
def _transform_gate_mesh_verts(
    mesh_verts: NDArray[np.float64],
    position: NDArray[np.float64],
    orientation: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Transform gate mesh vertices from mesh-local to world frame.

    gate_mesh.py produces meshes in XY plane with normal along Z.
    The sim uses gate normal along local X-axis. We apply Ry(+90deg)
    to align Z→X, then the per-gate world rotation and translation.

    Args:
        mesh_verts: (V, 3) vertices in gate mesh local frame.
        position: (3,) gate center in world frame.
        orientation: (4,) gate quaternion (w, x, y, z).

    Returns:
        (V, 3) vertices in world frame.
    """
    # Ry(+90deg): rotates Z-normal to X-normal
    R_mesh_to_sim = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ], dtype=np.float64)
    R_gate = quat_to_rotation_matrix(orientation)
    return (mesh_verts @ R_mesh_to_sim.T) @ R_gate.T + position[np.newaxis, :]
```

- [ ] **Step 2: Replace gate wireframe logging with rr.Mesh3D**

Replace the gate logging block (lines 219-231 of `rerun_generator.py`) — the section that starts with `# --- 3a. Static scene: gates ---`. The new code logs both a solid mesh and keeps the wireframe for dual visibility:

```python
    # --- 3a. Static scene: gates ---
    _base_gate_mesh = generate_gate_mesh()
    _base_gate_verts = np.array(_base_gate_mesh.vertices, dtype=np.float64)
    _base_gate_faces = np.array(_base_gate_mesh.faces, dtype=np.int32)

    for g in range(gate_positions.shape[0]):
        color = _GATE_COLORS[g % len(_GATE_COLORS)]

        # Solid mesh
        world_verts = _transform_gate_mesh_verts(
            _base_gate_verts, gate_positions[g], gate_orientations[g],
        )
        vert_colors = np.tile(
            np.array(color, dtype=np.uint8), (len(world_verts), 1),
        )
        rr.log(
            f"world/gates/gate_{g}",
            rr.Mesh3D(
                vertex_positions=world_verts.astype(np.float32),
                triangle_indices=_base_gate_faces,
                vertex_colors=vert_colors,
            ),
            static=True,
        )
```

- [ ] **Step 3: Add floor plane mesh**

After the gate markers block (after line 281), add:

```python
    # --- 3a-ter. Static scene: floor plane ---
    _floor_half = 20.0
    floor_verts = np.array([
        [-_floor_half, -_floor_half, 0.0],
        [+_floor_half, -_floor_half, 0.0],
        [+_floor_half, +_floor_half, 0.0],
        [-_floor_half, +_floor_half, 0.0],
    ], dtype=np.float32)
    floor_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    floor_colors = np.full((4, 3), 80, dtype=np.uint8)  # gray
    rr.log(
        "world/floor",
        rr.Mesh3D(
            vertex_positions=floor_verts,
            triangle_indices=floor_faces,
            vertex_colors=floor_colors,
        ),
        static=True,
    )
```

- [ ] **Step 4: Replace drone box with drone mesh**

Replace the drone box logging (lines 299-303):

```python
        # Drone bounding box
        rr.log(
            "world/drone/box",
            rr.Boxes3D(half_sizes=[[0.1, 0.1, 0.03]]),
        )
```

With drone mesh logging. Generate the mesh once before the loop and log per-timestep (mesh is in body frame, inherits the drone transform):

Before the per-timestep loop (before line 284), add:

```python
    _drone_mesh = generate_drone_mesh()
    _drone_verts = np.array(_drone_mesh.vertices, dtype=np.float32)
    _drone_faces = np.array(_drone_mesh.faces, dtype=np.int32)
    _drone_colors = np.full((len(_drone_verts), 3), 200, dtype=np.uint8)  # light gray
```

Then replace the `Boxes3D` block with:

```python
        # Drone mesh (body frame, inherits world/drone transform)
        rr.log(
            "world/drone/mesh",
            rr.Mesh3D(
                vertex_positions=_drone_verts,
                triangle_indices=_drone_faces,
                vertex_colors=_drone_colors,
            ),
        )
```

- [ ] **Step 5: Visual validation**

Run on an existing trajectory:
```bash
docker exec algo_src python -m sim.viz /path/to/existing/trajectory.npz --output-dir /tmp/rrd_test
```
Open the `.rrd` in the rerun viewer. Verify:
- Gates appear as solid colored meshes (not wireframe lines)
- Floor plane is visible at z=0
- Drone is a cross-shaped mesh (not a box)
- Gate markers (spheres, stalks, ground crosses) are still visible
- Flight trail still renders correctly

- [ ] **Step 6: Commit**

```bash
git add sim/viz/rerun_generator.py
git commit -m "feat(viz): upgrade rerun 3D panel to solid meshes (COR-62)"
```

---

### Task 3: PyTorch3D camera rendering in rerun pipeline

**Files:**
- Modify: `sim/viz/rerun_generator.py:124-128` (generate_rrd signature)
- Modify: `sim/viz/rerun_generator.py:345-362` (camera rendering block)
- Modify: `sim/viz/rerun_generator.py:372-376` (batch_generate signature)

This task wires `SceneRenderer.render()` into the camera panel, replacing `render_wireframe()` when `renderer="pytorch3d"`.

- [ ] **Step 1: Add renderer parameter to generate_rrd**

Update the `generate_rrd` signature (line 124-128):

```python
def generate_rrd(
    npz_path: str | Path,
    output_path: str | Path | None = None,
    camera_decimation: int = 10,
    renderer: str = "wireframe",
) -> Path:
```

Update the docstring to document the `renderer` parameter:
```
        renderer: Camera renderer backend. "wireframe" (default) uses
                  Bresenham line rasterization. "pytorch3d" uses PyTorch3D
                  Phong-shaded mesh rendering.
```

- [ ] **Step 2: Add PyTorch3D renderer initialization in generate_rrd**

After the `cam = PinholeCamera.from_hfov(90.0, 320, 240)` line (line 213), add:

```python
    # --- Optional PyTorch3D renderer ---
    scene_renderer = None
    if renderer == "pytorch3d":
        try:
            from sim.viz.rasterizer import SceneRenderer
        except ImportError:
            raise ImportError(
                "PyTorch3D is required for --renderer pytorch3d but is not installed.\n"
                "Install it with: pip install pytorch3d\n"
                "Or use --renderer wireframe (default)."
            )
        scene_renderer = SceneRenderer(
            image_size=(cam.height, cam.width),
            hfov_deg=90.0,
        )
        scene_renderer.set_scene(
            gate_positions=gate_positions,
            gate_orientations=gate_orientations,
            gate_half_extents=gate_half_extents,
        )

        # Log pinhole intrinsics for camera frustum visualization
        rr.log("world/drone/camera", rr.Pinhole(
            focal_length=[cam.fx, cam.fy],
            principal_point=[cam.cx, cam.cy],
            resolution=[cam.width, cam.height],
        ), static=True)
```

- [ ] **Step 3: Update camera rendering block to use SceneRenderer**

Replace the camera rendering block (lines 345-362):

```python
        # Pinhole camera view (decimated)
        if t % camera_decimation == 0:
            # Camera is body-mounted, same pose as drone
            cam_pos = positions[t]
            cam_quat = quaternions[t]  # (w, x, y, z)

            if scene_renderer is not None:
                # PyTorch3D rasterized render
                frame = scene_renderer.render(cam_pos, cam_quat)
                rr.log("world/drone/camera", rr.Image(frame))
            else:
                # Wireframe render (original path)
                all_edges: list[tuple[NDArray[np.float64], NDArray[np.float64], tuple[int, int, int]]] = []
                for gate_edges in all_gate_edges:
                    all_edges.extend(gate_edges)
                horizon_edges = render_horizon(cam, cam_pos, cam_quat)
                all_edges.extend(horizon_edges)
                frame = render_wireframe(cam, all_edges, cam_pos, cam_quat)
                rr.log("drone/camera", rr.Image(frame))
```

- [ ] **Step 4: Update batch_generate to pass renderer**

Update `batch_generate` signature (line 372-376):

```python
def batch_generate(
    trajectory_dir: str | Path,
    output_dir: str | Path | None = None,
    camera_decimation: int = 10,
    renderer: str = "wireframe",
) -> list[Path]:
```

Update the `generate_rrd` call inside `batch_generate` (line 409-413):

```python
        rrd_path = generate_rrd(
            npz_file,
            output_path=out_path,
            camera_decimation=camera_decimation,
            renderer=renderer,
        )
```

- [ ] **Step 5: Commit**

```bash
git add sim/viz/rerun_generator.py
git commit -m "feat(viz): wire PyTorch3D renderer into camera panel (COR-62)"
```

---

### Task 4: CLI integration — `--renderer` flag

**Files:**
- Modify: `sim/viz/__main__.py:11-31` (argument parsing)
- Modify: `sim/viz/__main__.py:52-68` (function calls)

- [ ] **Step 1: Add --renderer argument to CLI parser**

After the `--camera-decimation` argument (line 31), add:

```python
    parser.add_argument(
        "--renderer",
        choices=["wireframe", "pytorch3d"],
        default="wireframe",
        help="Camera renderer backend (default: wireframe)",
    )
```

- [ ] **Step 2: Pass renderer to generate_rrd and batch_generate**

Update the `generate_rrd` call (lines 55-61):

```python
        rrd_path = generate_rrd(
            path,
            output_path=args.output_dir / path.with_suffix(".rrd").name
            if args.output_dir
            else None,
            camera_decimation=args.camera_decimation,
            renderer=args.renderer,
        )
```

Update the `batch_generate` call (lines 63-68):

```python
        batch_generate(
            path,
            output_dir=args.output_dir,
            camera_decimation=args.camera_decimation,
            renderer=args.renderer,
        )
```

- [ ] **Step 3: Test CLI help output**

Run: `docker exec algo_src python -m sim.viz --help`
Expected: shows `--renderer {wireframe,pytorch3d}` in the output.

- [ ] **Step 4: Test wireframe mode still works (regression)**

Run: `docker exec algo_src python -m sim.viz /path/to/trajectory.npz --renderer wireframe`
Expected: generates `.rrd` without errors, identical to default behavior.

- [ ] **Step 5: Test pytorch3d mode end-to-end**

Run: `docker exec algo_src python -m sim.viz /path/to/trajectory.npz --renderer pytorch3d --output-dir /tmp/rrd_pt3d`
Expected: generates `.rrd` with PyTorch3D-rendered camera frames. Open in rerun viewer and verify:
- 3D panel shows solid gate meshes, floor, drone mesh
- Camera panel shows Phong-shaded gate rendering from drone POV
- Camera frustum is visible in the 3D panel
- Scrubbing through timesteps updates both panels

- [ ] **Step 6: Commit**

```bash
git add sim/viz/__main__.py
git commit -m "feat(viz): add --renderer CLI flag for pytorch3d rendering (COR-62)"
```
