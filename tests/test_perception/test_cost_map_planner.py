"""Tests for perception.planning.cost_map_planner."""
from __future__ import annotations

import numpy as np
import pytest

from perception.planning.cost_map_planner import (
    astar_grid, line_of_sight_free, plan, smooth_path_los,
)
from perception.planning.occupancy_grid import GridExtent, OccupancyGrid2D5


def _empty_grid() -> OccupancyGrid2D5:
    return OccupancyGrid2D5(
        GridExtent(x_min=-5.0, x_max=5.0, y_min=-5.0, y_max=5.0, cell_size_m=0.5),
        altitude_m=1.5,
    )


class TestAstar:
    def test_open_grid_diagonal(self) -> None:
        grid = np.zeros((10, 10), dtype=np.uint8)
        path = astar_grid(grid, (0, 0), (9, 9))
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (9, 9)
        # On an open grid, A* should produce ≤ 10 cells (one diagonal step per cell).
        assert len(path) <= 10

    def test_blocked_path_routes_around(self) -> None:
        grid = np.zeros((10, 10), dtype=np.uint8)
        # Wall across the middle, opening on the left.
        grid[1:9, 5] = 1
        path = astar_grid(grid, (0, 0), (9, 9))
        assert path is not None
        # Path should route through (0, 5) area, not through the wall.
        for c in path:
            assert grid[c[0], c[1]] == 0

    def test_unreachable(self) -> None:
        grid = np.zeros((10, 10), dtype=np.uint8)
        # Full vertical wall — no path.
        grid[:, 5] = 1
        path = astar_grid(grid, (0, 0), (9, 9))
        assert path is None

    def test_blocked_goal_uses_nearest_free(self) -> None:
        grid = np.zeros((10, 10), dtype=np.uint8)
        # Goal is inside a small obstacle.
        grid[5, 5] = 1
        path = astar_grid(grid, (0, 0), (5, 5))
        assert path is not None
        # The path's last cell should be a free neighbour of (5, 5).
        last = path[-1]
        assert grid[last[0], last[1]] == 0


class TestLineOfSight:
    def test_open_los(self) -> None:
        grid = np.zeros((10, 10), dtype=np.uint8)
        assert line_of_sight_free(grid, (0, 0), (9, 9))

    def test_blocked_los(self) -> None:
        grid = np.zeros((10, 10), dtype=np.uint8)
        grid[5, 5] = 1
        assert not line_of_sight_free(grid, (0, 0), (9, 9))


class TestSmoothing:
    def test_smoothes_open_corridor(self) -> None:
        grid = np.zeros((20, 20), dtype=np.uint8)
        # 20-cell zig-zag path, all in a clear corridor.
        path = [(i, i) for i in range(15)]
        smoothed = smooth_path_los(grid, path)
        # On an open grid the smoother should collapse to start + end.
        assert smoothed[0] == path[0]
        assert smoothed[-1] == path[-1]
        assert len(smoothed) <= 3

    def test_preserves_corner(self) -> None:
        grid = np.zeros((10, 10), dtype=np.uint8)
        grid[3:7, 5] = 1   # vertical wall
        path = astar_grid(grid, (0, 0), (9, 9))
        smoothed = smooth_path_los(grid, path)
        # Smoothed path must avoid the wall, so it can't be a straight line
        # from start to end — must keep at least one waypoint near the corner.
        assert len(smoothed) >= 3
        for c in smoothed:
            assert grid[c[0], c[1]] == 0


class TestPlanIntegration:
    def test_plan_open_world(self) -> None:
        g = _empty_grid()
        wps = plan(
            g,
            start_xy=np.array([-3.0, -3.0]), goal_xy=np.array([3.0, 3.0]),
            target_spacing_m=None,
        )
        assert wps is not None
        np.testing.assert_allclose(wps[0, 0:2], [-3.0, -3.0], atol=1e-9)
        np.testing.assert_allclose(wps[-1, 0:2], [3.0, 3.0], atol=1e-9)
        # All at altitude.
        assert np.allclose(wps[:, 2], g.altitude_m)

    def test_plan_around_obstacle(self) -> None:
        g = _empty_grid()
        g.add_cylinder(np.array([0.0, 0.0]), radius_m=1.0)
        # Dilate so the planner respects drone radius.
        dilated = g.dilate(0.3)
        wps = plan(
            dilated,
            start_xy=np.array([-3.0, 0.0]), goal_xy=np.array([3.0, 0.0]),
            target_spacing_m=None,
        )
        assert wps is not None
        # No waypoint should land inside the inflated obstacle.
        for wp in wps:
            assert dilated.is_free_world(wp[0:2])

    def test_unreachable_returns_none(self) -> None:
        g = _empty_grid()
        # Block the entire grid.
        g.grid[:] = 1
        wps = plan(g, np.array([-3.0, 0.0]), np.array([3.0, 0.0]))
        assert wps is None

    def test_target_spacing_resamples(self) -> None:
        g = _empty_grid()
        wps = plan(
            g,
            start_xy=np.array([-4.0, 0.0]), goal_xy=np.array([4.0, 0.0]),
            target_spacing_m=2.0,
        )
        assert wps is not None
        # Total path = 8 m, target spacing 2 m → ~4-5 waypoints.
        assert 4 <= wps.shape[0] <= 6
