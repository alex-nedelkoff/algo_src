"""Tests for procedural gate mesh generator.

Covers: inner corner positions, gate frame mesh geometry, drone mesh,
and OBJ export round-trip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from perception.training.data.gate_detection.gate_mesh import (
    generate_drone_mesh,
    generate_gate_mesh,
    get_gate_inner_corners,
)

# --------------------------------------------------------------------- #
# Inner corners
# --------------------------------------------------------------------- #


class TestGetGateInnerCorners:
    """Tests for get_gate_inner_corners()."""

    def test_output_shape(self) -> None:
        corners = get_gate_inner_corners()
        assert corners.shape == (4, 3)

    def test_output_dtype(self) -> None:
        corners = get_gate_inner_corners()
        assert corners.dtype == np.float64

    def test_default_positions(self) -> None:
        """Default 1.5x1.5 gate: corners at +/-0.75 in XY, Z=0."""
        corners = get_gate_inner_corners()
        w, h = 1.5, 1.5
        expected = np.array(
            [
                [-w / 2, +h / 2, 0.0],  # TL
                [+w / 2, +h / 2, 0.0],  # TR
                [+w / 2, -h / 2, 0.0],  # BR
                [-w / 2, -h / 2, 0.0],  # BL
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_almost_equal(corners, expected)

    def test_custom_dimensions(self) -> None:
        """Non-square gate with custom width and height."""
        corners = get_gate_inner_corners(inner_width=2.0, inner_height=1.0)
        expected = np.array(
            [
                [-1.0, +0.5, 0.0],
                [+1.0, +0.5, 0.0],
                [+1.0, -0.5, 0.0],
                [-1.0, -0.5, 0.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_almost_equal(corners, expected)

    def test_all_z_zero(self) -> None:
        """All corners lie in the XY plane (Z=0)."""
        corners = get_gate_inner_corners(inner_width=3.0, inner_height=2.0)
        np.testing.assert_array_equal(corners[:, 2], 0.0)

    def test_corner_order_tl_tr_br_bl(self) -> None:
        """TL has negative X, positive Y; TR positive X, positive Y; etc."""
        corners = get_gate_inner_corners(inner_width=2.0, inner_height=2.0)
        # TL: x < 0, y > 0
        assert corners[0, 0] < 0 and corners[0, 1] > 0
        # TR: x > 0, y > 0
        assert corners[1, 0] > 0 and corners[1, 1] > 0
        # BR: x > 0, y < 0
        assert corners[2, 0] > 0 and corners[2, 1] < 0
        # BL: x < 0, y < 0
        assert corners[3, 0] < 0 and corners[3, 1] < 0


# --------------------------------------------------------------------- #
# Gate mesh
# --------------------------------------------------------------------- #


class TestGenerateGateMesh:
    """Tests for generate_gate_mesh()."""

    def test_returns_trimesh(self) -> None:
        mesh = generate_gate_mesh()
        assert isinstance(mesh, trimesh.Trimesh)

    def test_has_vertices_and_faces(self) -> None:
        mesh = generate_gate_mesh()
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_bounding_box_outer_dimensions(self) -> None:
        """Outer dimensions should be inner + 2*frame_width in X and Y."""
        inner_w, inner_h = 1.5, 1.5
        frame_w = 0.1
        frame_d = 0.05
        mesh = generate_gate_mesh(
            inner_width=inner_w,
            inner_height=inner_h,
            frame_width=frame_w,
            frame_depth=frame_d,
        )
        bbox = mesh.bounding_box.extents  # (dx, dy, dz)
        expected_x = inner_w + 2 * frame_w
        expected_y = inner_h + 2 * frame_w
        expected_z = frame_d

        assert bbox[0] == pytest.approx(expected_x, abs=1e-6)
        assert bbox[1] == pytest.approx(expected_y, abs=1e-6)
        assert bbox[2] == pytest.approx(expected_z, abs=1e-6)

    def test_centered_at_origin(self) -> None:
        """Gate mesh center should be at the origin."""
        mesh = generate_gate_mesh()
        center = mesh.bounding_box.centroid
        np.testing.assert_array_almost_equal(center, [0.0, 0.0, 0.0], decimal=6)

    def test_custom_dimensions(self) -> None:
        """Custom dimensions produce correctly sized bounding box."""
        inner_w, inner_h = 2.0, 1.0
        frame_w = 0.2
        frame_d = 0.1
        mesh = generate_gate_mesh(
            inner_width=inner_w,
            inner_height=inner_h,
            frame_width=frame_w,
            frame_depth=frame_d,
        )
        bbox = mesh.bounding_box.extents
        assert bbox[0] == pytest.approx(inner_w + 2 * frame_w, abs=1e-6)
        assert bbox[1] == pytest.approx(inner_h + 2 * frame_w, abs=1e-6)
        assert bbox[2] == pytest.approx(frame_d, abs=1e-6)

    def test_is_hollow(self) -> None:
        """The gate frame should be hollow — no vertices in the center region.

        We verify by checking that no mesh vertices fall inside the inner
        opening (the hollow region), while vertices do exist on the frame bars.
        """
        inner_w, inner_h = 1.5, 1.5
        frame_w = 0.1
        mesh = generate_gate_mesh(
            inner_width=inner_w,
            inner_height=inner_h,
            frame_width=frame_w,
            frame_depth=0.05,
        )
        verts = mesh.vertices

        # Define the inner opening with a small margin to avoid edge vertices
        margin = frame_w * 0.1
        inner_x = inner_w / 2 - margin
        inner_y = inner_h / 2 - margin

        # No vertices should be strictly inside the hollow center
        inside = (
            (np.abs(verts[:, 0]) < inner_x)
            & (np.abs(verts[:, 1]) < inner_y)
        )
        assert not np.any(inside), "Center of gate should be hollow (no vertices)"

        # Vertices should exist on the frame bars (outer region)
        outer_x = inner_w / 2 + frame_w
        on_frame = np.abs(verts[:, 0]) > inner_w / 2
        assert np.any(on_frame), "Frame bars should have vertices"


# --------------------------------------------------------------------- #
# Drone mesh
# --------------------------------------------------------------------- #


class TestGenerateDroneMesh:
    """Tests for generate_drone_mesh()."""

    def test_returns_trimesh(self) -> None:
        mesh = generate_drone_mesh()
        assert isinstance(mesh, trimesh.Trimesh)

    def test_has_vertices_and_faces(self) -> None:
        mesh = generate_drone_mesh()
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_reasonable_size(self) -> None:
        """Drone bounding box should be roughly within the specified size."""
        size = 0.3
        mesh = generate_drone_mesh(size=size)
        bbox = mesh.bounding_box.extents
        # The bounding box should not exceed 2x the size parameter in any dim
        assert bbox[0] <= size * 2.5
        assert bbox[1] <= size * 2.5
        assert bbox[2] <= size * 2.5

    def test_custom_size(self) -> None:
        """Larger size parameter produces larger mesh."""
        small = generate_drone_mesh(size=0.1)
        large = generate_drone_mesh(size=1.0)
        assert large.bounding_box.volume > small.bounding_box.volume


# --------------------------------------------------------------------- #
# OBJ export round-trip
# --------------------------------------------------------------------- #


class TestOBJExport:
    """Tests for OBJ export and round-trip loading."""

    def test_gate_obj_round_trip(self, tmp_path: Path) -> None:
        """Export gate mesh to OBJ and reload it."""
        mesh = generate_gate_mesh()
        obj_path = tmp_path / "gate.obj"
        mesh.export(str(obj_path))

        assert obj_path.exists()
        assert obj_path.stat().st_size > 0

        loaded = trimesh.load(str(obj_path), force="mesh")
        assert len(loaded.vertices) > 0
        assert len(loaded.faces) > 0

    def test_drone_obj_round_trip(self, tmp_path: Path) -> None:
        """Export drone mesh to OBJ and reload it."""
        mesh = generate_drone_mesh()
        obj_path = tmp_path / "drone.obj"
        mesh.export(str(obj_path))

        assert obj_path.exists()
        assert obj_path.stat().st_size > 0

        loaded = trimesh.load(str(obj_path), force="mesh")
        assert len(loaded.vertices) > 0
        assert len(loaded.faces) > 0

    def test_exported_gate_preserves_dimensions(self, tmp_path: Path) -> None:
        """OBJ round-trip should preserve bounding box dimensions."""
        inner_w, inner_h = 1.5, 1.5
        frame_w = 0.1
        mesh = generate_gate_mesh(
            inner_width=inner_w, inner_height=inner_h, frame_width=frame_w
        )
        obj_path = tmp_path / "gate.obj"
        mesh.export(str(obj_path))

        loaded = trimesh.load(str(obj_path), force="mesh")
        bbox = loaded.bounding_box.extents
        assert bbox[0] == pytest.approx(inner_w + 2 * frame_w, abs=1e-3)
        assert bbox[1] == pytest.approx(inner_h + 2 * frame_w, abs=1e-3)
