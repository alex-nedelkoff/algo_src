"""Tests for gate detection shared format module.

Covers: heatmap generation, gate mask rendering, save/load round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from perception.training.data.gate_detection.format import (
    generate_corner_heatmaps,
    load_sample,
    render_gate_mask,
    save_sample,
)

# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #

H, W = 480, 640
STRIDE = 4
SIGMA = 2.0


@pytest.fixture
def single_gate_coords() -> list[dict]:
    """A single gate roughly centered in a 640x480 image."""
    return [
        {
            "gate_id": 1,
            "corners": [
                [200.0, 150.0],  # TL
                [440.0, 150.0],  # TR
                [440.0, 330.0],  # BR
                [200.0, 330.0],  # BL
            ],
            "confidence": 1.0,
        }
    ]


@pytest.fixture
def multi_gate_coords() -> list[dict]:
    """Two gates in a 640x480 image, non-overlapping."""
    return [
        {
            "gate_id": 1,
            "corners": [
                [50.0, 50.0],
                [200.0, 50.0],
                [200.0, 200.0],
                [50.0, 200.0],
            ],
            "confidence": 1.0,
        },
        {
            "gate_id": 2,
            "corners": [
                [400.0, 250.0],
                [600.0, 250.0],
                [600.0, 430.0],
                [400.0, 430.0],
            ],
            "confidence": 0.8,
        },
    ]


# --------------------------------------------------------------------- #
# Heatmap tests
# --------------------------------------------------------------------- #


class TestGenerateCornerHeatmaps:
    """Tests for CenterNet-style corner heatmap generation."""

    def test_output_shape_single_gate(self, single_gate_coords: list[dict]) -> None:
        hm = generate_corner_heatmaps(single_gate_coords, H, W, stride=STRIDE, sigma=SIGMA)
        assert hm.shape == (4, H // STRIDE, W // STRIDE)

    def test_output_dtype(self, single_gate_coords: list[dict]) -> None:
        hm = generate_corner_heatmaps(single_gate_coords, H, W, stride=STRIDE, sigma=SIGMA)
        assert hm.dtype == np.float32

    def test_value_range(self, single_gate_coords: list[dict]) -> None:
        hm = generate_corner_heatmaps(single_gate_coords, H, W, stride=STRIDE, sigma=SIGMA)
        assert hm.min() >= 0.0
        assert hm.max() <= 1.0

    def test_peak_at_correct_location_single_gate(
        self, single_gate_coords: list[dict]
    ) -> None:
        """Each heatmap channel should peak near the corresponding corner."""
        hm = generate_corner_heatmaps(single_gate_coords, H, W, stride=STRIDE, sigma=SIGMA)
        corners = single_gate_coords[0]["corners"]

        for ch, (cx, cy) in enumerate(corners):
            peak_y, peak_x = np.unravel_index(np.argmax(hm[ch]), hm[ch].shape)
            expected_x = cx / STRIDE
            expected_y = cy / STRIDE
            # Peak should be within 1 grid cell of expected location
            assert abs(peak_x - expected_x) <= 1.0, (
                f"Channel {ch}: peak_x={peak_x}, expected~{expected_x}"
            )
            assert abs(peak_y - expected_y) <= 1.0, (
                f"Channel {ch}: peak_y={peak_y}, expected~{expected_y}"
            )

    def test_peak_value_is_one(self, single_gate_coords: list[dict]) -> None:
        """The Gaussian kernel peak should be exactly 1.0."""
        hm = generate_corner_heatmaps(single_gate_coords, H, W, stride=STRIDE, sigma=SIGMA)
        for ch in range(4):
            assert hm[ch].max() == pytest.approx(1.0, abs=1e-5)

    def test_multiple_gates_produce_multiple_peaks(
        self, multi_gate_coords: list[dict]
    ) -> None:
        """With 2 gates, each channel should have 2 distinct peaks."""
        hm = generate_corner_heatmaps(multi_gate_coords, H, W, stride=STRIDE, sigma=SIGMA)

        for ch in range(4):
            channel = hm[ch]
            # Check that there are at least 2 local maxima above a threshold
            # by verifying both gate corner locations have high values
            for gate in multi_gate_coords:
                cx, cy = gate["corners"][ch]
                gx, gy = int(round(cx / STRIDE)), int(round(cy / STRIDE))
                # Clamp to valid range
                gx = min(gx, channel.shape[1] - 1)
                gy = min(gy, channel.shape[0] - 1)
                assert channel[gy, gx] > 0.5, (
                    f"Channel {ch}, gate {gate['gate_id']}: "
                    f"value at ({gx},{gy})={channel[gy, gx]:.3f}, expected > 0.5"
                )

    def test_empty_coords_returns_zeros(self) -> None:
        """No gates -> all-zero heatmaps."""
        hm = generate_corner_heatmaps([], H, W, stride=STRIDE, sigma=SIGMA)
        assert hm.shape == (4, H // STRIDE, W // STRIDE)
        assert np.all(hm == 0.0)

    def test_custom_stride(self, single_gate_coords: list[dict]) -> None:
        """Different stride produces correctly sized output."""
        stride = 8
        hm = generate_corner_heatmaps(single_gate_coords, H, W, stride=stride, sigma=SIGMA)
        assert hm.shape == (4, H // stride, W // stride)

    def test_gaussians_are_symmetric(self, single_gate_coords: list[dict]) -> None:
        """Heatmap values should decay symmetrically around the peak."""
        hm = generate_corner_heatmaps(single_gate_coords, H, W, stride=STRIDE, sigma=SIGMA)
        ch = 0  # Test TL corner
        peak_y, peak_x = np.unravel_index(np.argmax(hm[ch]), hm[ch].shape)

        # Values at equal distance from peak should be approximately equal
        if peak_y > 0 and peak_y < hm.shape[1] - 1:
            val_above = hm[ch, peak_y - 1, peak_x]
            val_below = hm[ch, peak_y + 1, peak_x]
            assert val_above == pytest.approx(val_below, abs=0.05)


# --------------------------------------------------------------------- #
# Mask tests
# --------------------------------------------------------------------- #


class TestRenderGateMask:
    """Tests for gate instance segmentation mask rendering."""

    def test_output_shape(self, single_gate_coords: list[dict]) -> None:
        mask = render_gate_mask(single_gate_coords, H, W)
        assert mask.shape == (H, W)

    def test_output_dtype(self, single_gate_coords: list[dict]) -> None:
        mask = render_gate_mask(single_gate_coords, H, W)
        assert mask.dtype == np.uint8

    def test_single_gate_has_correct_ids(self, single_gate_coords: list[dict]) -> None:
        """Single gate mask should have bg (0) and gate instance (1)."""
        mask = render_gate_mask(single_gate_coords, H, W)
        unique_vals = set(np.unique(mask))
        assert 0 in unique_vals, "Background (0) should be present"
        assert 1 in unique_vals, "Gate instance 1 should be present"
        assert len(unique_vals) == 2

    def test_single_gate_interior_is_filled(self, single_gate_coords: list[dict]) -> None:
        """Center of the gate quadrilateral should be filled."""
        mask = render_gate_mask(single_gate_coords, H, W)
        # Center of gate corners
        corners = np.array(single_gate_coords[0]["corners"])
        cx, cy = corners.mean(axis=0).astype(int)
        assert mask[cy, cx] == 1

    def test_multi_gate_has_distinct_ids(self, multi_gate_coords: list[dict]) -> None:
        """Two gates should produce instance IDs 1 and 2."""
        mask = render_gate_mask(multi_gate_coords, H, W)
        unique_vals = set(np.unique(mask))
        assert 0 in unique_vals
        assert 1 in unique_vals
        assert 2 in unique_vals

    def test_multi_gate_each_filled(self, multi_gate_coords: list[dict]) -> None:
        """Each gate's center should have its respective instance ID."""
        mask = render_gate_mask(multi_gate_coords, H, W)

        for i, gate in enumerate(multi_gate_coords, start=1):
            corners = np.array(gate["corners"])
            cx, cy = corners.mean(axis=0).astype(int)
            assert mask[cy, cx] == i, (
                f"Gate {i} center ({cx},{cy}) has value {mask[cy, cx]}, expected {i}"
            )

    def test_empty_coords_returns_zeros(self) -> None:
        """No gates -> all-zero mask."""
        mask = render_gate_mask([], H, W)
        assert mask.shape == (H, W)
        assert np.all(mask == 0)

    def test_background_outside_gate(self, single_gate_coords: list[dict]) -> None:
        """Pixels far from the gate should be background."""
        mask = render_gate_mask(single_gate_coords, H, W)
        # Top-left corner of image is far from the gate
        assert mask[0, 0] == 0
        assert mask[0, W - 1] == 0


# --------------------------------------------------------------------- #
# Save / Load round-trip tests
# --------------------------------------------------------------------- #


class TestSaveLoadRoundTrip:
    """Tests for save_sample / load_sample round-trip fidelity."""

    @pytest.fixture
    def sample_data(self, single_gate_coords: list[dict]) -> dict:
        """Create a complete sample dataset for testing."""
        rng = np.random.default_rng(42)
        rgb = rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)
        gate_mask = render_gate_mask(single_gate_coords, H, W)
        obstacle_mask = np.zeros((H, W), dtype=np.uint8)
        corner_heatmaps = generate_corner_heatmaps(
            single_gate_coords, H, W, stride=STRIDE, sigma=SIGMA
        )
        metadata = {
            "source": "test",
            "frame_id": 42,
            "camera_intrinsics": [500.0, 500.0, 320.0, 240.0],
        }
        return {
            "rgb": rgb,
            "gate_mask": gate_mask,
            "obstacle_mask": obstacle_mask,
            "corner_coords": single_gate_coords,
            "corner_heatmaps": corner_heatmaps,
            "metadata": metadata,
        }

    def test_save_creates_expected_files(
        self, tmp_path: Path, sample_data: dict
    ) -> None:
        sample_dir = tmp_path / "sample_000"
        save_sample(sample_dir, **sample_data)

        assert (sample_dir / "rgb.png").exists()
        assert (sample_dir / "gate_mask.png").exists()
        assert (sample_dir / "obstacle_mask.png").exists()
        assert (sample_dir / "corner_heatmaps.npy").exists()
        assert (sample_dir / "corner_coords.json").exists()
        assert (sample_dir / "metadata.json").exists()

    def test_round_trip_rgb(self, tmp_path: Path, sample_data: dict) -> None:
        sample_dir = tmp_path / "sample_000"
        save_sample(sample_dir, **sample_data)
        loaded = load_sample(sample_dir)

        np.testing.assert_array_equal(loaded["rgb"], sample_data["rgb"])

    def test_round_trip_gate_mask(self, tmp_path: Path, sample_data: dict) -> None:
        sample_dir = tmp_path / "sample_000"
        save_sample(sample_dir, **sample_data)
        loaded = load_sample(sample_dir)

        np.testing.assert_array_equal(loaded["gate_mask"], sample_data["gate_mask"])

    def test_round_trip_obstacle_mask(self, tmp_path: Path, sample_data: dict) -> None:
        sample_dir = tmp_path / "sample_000"
        save_sample(sample_dir, **sample_data)
        loaded = load_sample(sample_dir)

        np.testing.assert_array_equal(
            loaded["obstacle_mask"], sample_data["obstacle_mask"]
        )

    def test_round_trip_corner_heatmaps(
        self, tmp_path: Path, sample_data: dict
    ) -> None:
        sample_dir = tmp_path / "sample_000"
        save_sample(sample_dir, **sample_data)
        loaded = load_sample(sample_dir)

        np.testing.assert_array_almost_equal(
            loaded["corner_heatmaps"], sample_data["corner_heatmaps"]
        )

    def test_round_trip_corner_coords(
        self, tmp_path: Path, sample_data: dict
    ) -> None:
        sample_dir = tmp_path / "sample_000"
        save_sample(sample_dir, **sample_data)
        loaded = load_sample(sample_dir)

        assert loaded["corner_coords"] == sample_data["corner_coords"]

    def test_round_trip_metadata(self, tmp_path: Path, sample_data: dict) -> None:
        sample_dir = tmp_path / "sample_000"
        save_sample(sample_dir, **sample_data)
        loaded = load_sample(sample_dir)

        assert loaded["metadata"] == sample_data["metadata"]

    def test_save_creates_directory(self, tmp_path: Path, sample_data: dict) -> None:
        """save_sample should create the sample directory if it doesn't exist."""
        sample_dir = tmp_path / "nested" / "deep" / "sample_000"
        save_sample(sample_dir, **sample_data)
        assert sample_dir.exists()

    def test_load_returns_correct_dtypes(
        self, tmp_path: Path, sample_data: dict
    ) -> None:
        sample_dir = tmp_path / "sample_000"
        save_sample(sample_dir, **sample_data)
        loaded = load_sample(sample_dir)

        assert loaded["rgb"].dtype == np.uint8
        assert loaded["gate_mask"].dtype == np.uint8
        assert loaded["obstacle_mask"].dtype == np.uint8
        assert loaded["corner_heatmaps"].dtype == np.float32

    def test_save_with_string_path(self, tmp_path: Path, sample_data: dict) -> None:
        """save_sample should accept str paths, not just Path objects."""
        sample_dir = str(tmp_path / "sample_str")
        save_sample(sample_dir, **sample_data)
        loaded = load_sample(sample_dir)
        assert loaded["rgb"].shape == sample_data["rgb"].shape

    def test_corner_coords_json_is_valid(
        self, tmp_path: Path, sample_data: dict
    ) -> None:
        """The saved corner_coords.json should be valid JSON."""
        sample_dir = tmp_path / "sample_000"
        save_sample(sample_dir, **sample_data)

        with open(sample_dir / "corner_coords.json") as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["gate_id"] == 1
        assert len(data[0]["corners"]) == 4
