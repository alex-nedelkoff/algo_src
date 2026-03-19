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

            # Transform: mesh local (Z-normal) -> sim local (X-normal) -> world
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
        R_pt3d = R_world_to_cam.T  # (3, 3) -- transposed for row-vector convention
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
