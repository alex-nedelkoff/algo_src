"""Tests for perception.planning.occupancy_grid."""
from __future__ import annotations

import numpy as np
import pytest

from perception.planning.occupancy_grid import GridExtent, OccupancyGrid2D5


def _empty_grid() -> OccupancyGrid2D5:
    return OccupancyGrid2D5(
        GridExtent(x_min=-5.0, x_max=5.0, y_min=-5.0, y_max=5.0, cell_size_m=0.5),
        altitude_m=1.5,
    )


class TestCoords:
    def test_world_to_grid_at_origin(self) -> None:
        g = _empty_grid()
        i, j = g.world_to_grid(np.array([0.0, 0.0]))
        # Origin is at the grid centre; with x range [-5, 5] and cell_size=0.5,
        # n_cells = 20, so origin sits at cell index 10, 10.
        assert (i, j) == (10, 10)

    def test_grid_to_world_centre(self) -> None:
        g = _empty_grid()
        xy = g.grid_to_world((10, 10))
        np.testing.assert_allclose(xy, [0.25, 0.25])  # cell centre offset by 0.5/2

    def test_round_trip(self) -> None:
        g = _empty_grid()
        for xy in [(1.2, 3.8), (-2.5, 0.0), (-4.9, 4.9)]:
            ij = g.world_to_grid(np.array(xy))
            assert g.in_bounds(ij)


class TestAddCylinder:
    def test_single_cylinder(self) -> None:
        g = _empty_grid()
        g.add_cylinder(np.array([0.0, 0.0]), radius_m=1.0)
        # Cells within 1 m of origin should be occupied. Total occupied
        # cells ≈ π · r² / cell² = π / 0.25 ≈ 12.6 → about 12-14 cells.
        n_occupied = int(g.grid.sum())
        assert 8 <= n_occupied <= 18

    def test_outside_grid_no_op(self) -> None:
        g = _empty_grid()
        g.add_cylinder(np.array([100.0, 100.0]), radius_m=1.0)
        assert g.grid.sum() == 0

    def test_known_cell_occupied(self) -> None:
        g = _empty_grid()
        g.add_cylinder(np.array([2.0, 2.0]), radius_m=0.6)
        ij = g.world_to_grid(np.array([2.0, 2.0]))
        assert g.grid[ij[0], ij[1]] == 1


class TestDilate:
    def test_dilation_expands_obstacle(self) -> None:
        g = _empty_grid()
        g.add_cylinder(np.array([0.0, 0.0]), radius_m=0.4)
        before = int(g.grid.sum())
        d = g.dilate(radius_m=0.6)
        after = int(d.grid.sum())
        assert after > before
        # Original grid unchanged.
        assert int(g.grid.sum()) == before

    def test_dilation_zero_radius_noop(self) -> None:
        g = _empty_grid()
        g.add_cylinder(np.array([0.0, 0.0]), radius_m=0.4)
        d = g.dilate(radius_m=0.0)
        np.testing.assert_array_equal(d.grid, g.grid)


class TestImage:
    def test_image_shape(self) -> None:
        g = _empty_grid()
        g.add_cylinder(np.array([1.0, 1.0]), radius_m=0.5)
        img = g.to_image()
        assert img.shape == (g.extent.n_cells_y, g.extent.n_cells_x, 3)
        assert img.dtype == np.uint8


class TestIsFreeWorld:
    def test_free_outside_obstacles(self) -> None:
        g = _empty_grid()
        g.add_cylinder(np.array([0.0, 0.0]), radius_m=0.5)
        assert g.is_free_world(np.array([4.0, 4.0]))

    def test_blocked_inside_obstacle(self) -> None:
        g = _empty_grid()
        g.add_cylinder(np.array([0.0, 0.0]), radius_m=0.5)
        assert not g.is_free_world(np.array([0.0, 0.0]))

    def test_outside_grid_blocked(self) -> None:
        g = _empty_grid()
        # Anything outside the grid is treated as blocked (planner must
        # stay in bounds).
        assert not g.is_free_world(np.array([100.0, 100.0]))
