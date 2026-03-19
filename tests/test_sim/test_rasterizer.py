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
            gate_orientations=np.array([[1.0, 0.0, 0.0, 0.0]]),
            gate_half_extents=np.array([[0.75, 0.75]]),
        )
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
                [0.707, 0.0, 0.0, 0.707],
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
            camera_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        assert frame.shape == (240, 320, 3)
        assert frame.dtype == np.uint8

    def test_render_not_all_black(self, renderer_with_scene):
        """Camera facing a gate 5m ahead should produce non-trivial image."""
        frame = renderer_with_scene.render(
            camera_pos=np.array([0.0, 0.0, 1.5]),
            camera_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        assert frame.max() > 50, "Image appears all-dark — gate likely not visible"

    def test_render_gate_behind_camera_is_dark(self, renderer_with_scene):
        """Camera facing away from gate should show mostly background."""
        frame = renderer_with_scene.render(
            camera_pos=np.array([0.0, 0.0, 1.5]),
            camera_quat=np.array([0.0, 0.0, 0.0, 1.0]),
        )
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
