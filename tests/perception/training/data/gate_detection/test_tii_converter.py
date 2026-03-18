"""Tests for TII dataset converter.

Covers: parse_tii_label, convert_tii_flight round-trip.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from perception.training.data.gate_detection.tii_converter import (
    convert_tii_flight,
    parse_tii_label,
)

# --------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------- #

IMG_W, IMG_H = 640, 480


def _make_all_visible_label_line(
    cx: float = 0.5,
    cy: float = 0.5,
    w: float = 0.3,
    h: float = 0.3,
) -> str:
    """Build a TII label line with all 4 corners visible (vis=2).

    Returns a string with 17 space-separated tokens:
      class cx cy w h tlx tly tlv trx try trv brx bry brv blx bly blv
    Corner positions are derived from the bbox.
    """
    # Derive corner positions from bbox
    tl_x = cx - w / 2
    tl_y = cy - h / 2
    tr_x = cx + w / 2
    tr_y = cy - h / 2
    br_x = cx + w / 2
    br_y = cy + h / 2
    bl_x = cx - w / 2
    bl_y = cy + h / 2

    tokens = [
        "0",  # class
        f"{cx:.6f}",
        f"{cy:.6f}",
        f"{w:.6f}",
        f"{h:.6f}",
        # TL
        f"{tl_x:.6f}",
        f"{tl_y:.6f}",
        "2",
        # TR
        f"{tr_x:.6f}",
        f"{tr_y:.6f}",
        "2",
        # BR
        f"{br_x:.6f}",
        f"{br_y:.6f}",
        "2",
        # BL
        f"{bl_x:.6f}",
        f"{bl_y:.6f}",
        "2",
    ]
    return " ".join(tokens)


def _make_partially_visible_label_line() -> str:
    """Build a TII label line where one corner has visibility 0 (outside image)."""
    tokens = [
        "0",
        "0.5",
        "0.5",
        "0.3",
        "0.3",
        # TL — visible
        "0.35",
        "0.35",
        "2",
        # TR — NOT visible
        "0.65",
        "0.35",
        "0",
        # BR — visible
        "0.65",
        "0.65",
        "2",
        # BL — visible
        "0.35",
        "0.65",
        "2",
    ]
    return " ".join(tokens)


def _make_fake_tii_sample(
    tmpdir: Path,
    frame_name: str = "000001",
    flight_name: str = "flight-test",
    label_lines: list[str] | None = None,
) -> Path:
    """Create a minimal fake TII flight directory with one image + label.

    Returns the flight_dir (e.g. tmpdir / "autonomous" / "flight-test").
    """
    flight_dir = tmpdir / "autonomous" / flight_name
    img_dir = flight_dir / f"camera_{flight_name}"
    lbl_dir = flight_dir / f"labels_{flight_name}"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # Write a dummy 640x480 BGR JPEG
    dummy_img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    dummy_img[100:380, 100:540] = (128, 64, 32)  # some non-zero region
    cv2.imwrite(str(img_dir / f"{frame_name}.jpg"), dummy_img)

    # Write label file
    if label_lines is None:
        label_lines = [_make_all_visible_label_line()]
    lbl_dir.mkdir(parents=True, exist_ok=True)
    (lbl_dir / f"{frame_name}.txt").write_text("\n".join(label_lines) + "\n")

    return flight_dir


# --------------------------------------------------------------------- #
# parse_tii_label tests
# --------------------------------------------------------------------- #


class TestParseTiiLabel:
    """Tests for parse_tii_label."""

    def test_all_visible_returns_one_gate(self) -> None:
        line = _make_all_visible_label_line()
        gates = parse_tii_label(line, IMG_W, IMG_H)
        assert len(gates) == 1

    def test_gate_dict_structure(self) -> None:
        line = _make_all_visible_label_line()
        gates = parse_tii_label(line, IMG_W, IMG_H)
        gate = gates[0]

        assert "gate_id" in gate
        assert "corners" in gate
        assert "confidence" in gate
        assert isinstance(gate["gate_id"], int)
        assert gate["confidence"] == 1.0
        assert len(gate["corners"]) == 4

    def test_corners_denormalized_to_pixel_space(self) -> None:
        """Corners should be in pixel coordinates, not normalized."""
        line = _make_all_visible_label_line(cx=0.5, cy=0.5, w=0.3, h=0.3)
        gates = parse_tii_label(line, IMG_W, IMG_H)
        corners = gates[0]["corners"]

        # TL corner: normalized (0.35, 0.35) -> pixel (224, 168)
        expected_tl_x = 0.35 * IMG_W  # 224.0
        expected_tl_y = 0.35 * IMG_H  # 168.0
        assert corners[0][0] == pytest.approx(expected_tl_x, abs=0.1)
        assert corners[0][1] == pytest.approx(expected_tl_y, abs=0.1)

        # BR corner: normalized (0.65, 0.65) -> pixel (416, 312)
        expected_br_x = 0.65 * IMG_W  # 416.0
        expected_br_y = 0.65 * IMG_H  # 312.0
        assert corners[2][0] == pytest.approx(expected_br_x, abs=0.1)
        assert corners[2][1] == pytest.approx(expected_br_y, abs=0.1)

    def test_invisible_corner_drops_gate(self) -> None:
        """If any corner has visibility != 2, the gate should be dropped."""
        line = _make_partially_visible_label_line()
        gates = parse_tii_label(line, IMG_W, IMG_H)
        assert len(gates) == 0

    def test_corner_order_is_tl_tr_br_bl(self) -> None:
        """Corners should follow TL, TR, BR, BL convention."""
        line = _make_all_visible_label_line(cx=0.5, cy=0.5, w=0.4, h=0.4)
        gates = parse_tii_label(line, IMG_W, IMG_H)
        corners = gates[0]["corners"]

        tl, tr, br, bl = corners
        # TL.x < TR.x,  TL.y < BL.y
        assert tl[0] < tr[0], "TL.x should be less than TR.x"
        assert tl[1] < bl[1], "TL.y should be less than BL.y"
        # BR is bottom-right
        assert br[0] > bl[0], "BR.x should be greater than BL.x"
        assert br[1] > tr[1], "BR.y should be greater than TR.y"

    def test_gate_id_is_zero(self) -> None:
        """gate_id should be the class index from the label (always 0 for TII)."""
        line = _make_all_visible_label_line()
        gates = parse_tii_label(line, IMG_W, IMG_H)
        assert gates[0]["gate_id"] == 0


# --------------------------------------------------------------------- #
# convert_tii_flight tests
# --------------------------------------------------------------------- #


class TestConvertTiiFlight:
    """Tests for convert_tii_flight round-trip."""

    def test_single_frame_produces_one_sample(self, tmp_path: Path) -> None:
        flight_dir = _make_fake_tii_sample(tmp_path)
        output_dir = tmp_path / "output"

        count = convert_tii_flight(flight_dir, output_dir)
        assert count == 1

    def test_output_files_exist(self, tmp_path: Path) -> None:
        flight_dir = _make_fake_tii_sample(tmp_path)
        output_dir = tmp_path / "output"

        convert_tii_flight(flight_dir, output_dir)

        # There should be exactly one sample directory
        sample_dirs = sorted(output_dir.iterdir())
        assert len(sample_dirs) == 1

        sample_dir = sample_dirs[0]
        assert (sample_dir / "rgb.png").exists()
        assert (sample_dir / "gate_mask.png").exists()
        assert (sample_dir / "obstacle_mask.png").exists()
        assert (sample_dir / "corner_heatmaps.npy").exists()
        assert (sample_dir / "corner_coords.json").exists()
        assert (sample_dir / "metadata.json").exists()

    def test_metadata_fields(self, tmp_path: Path) -> None:
        flight_dir = _make_fake_tii_sample(tmp_path)
        output_dir = tmp_path / "output"

        convert_tii_flight(flight_dir, output_dir)

        import json

        sample_dirs = sorted(output_dir.iterdir())
        with open(sample_dirs[0] / "metadata.json") as f:
            meta = json.load(f)

        assert meta["source"] == "tii"
        assert meta["gate_dims_m"] == [1.52, 1.52]

    def test_invisible_gate_frame_skipped(self, tmp_path: Path) -> None:
        """A frame where the only gate has an invisible corner should be skipped."""
        flight_dir = _make_fake_tii_sample(
            tmp_path,
            label_lines=[_make_partially_visible_label_line()],
        )
        output_dir = tmp_path / "output"

        count = convert_tii_flight(flight_dir, output_dir)
        assert count == 0

    def test_max_samples_respected(self, tmp_path: Path) -> None:
        """convert_tii_flight should stop after max_samples."""
        flight_dir = tmp_path / "autonomous" / "flight-multi"
        img_dir = flight_dir / "camera_flight-multi"
        lbl_dir = flight_dir / "labels_flight-multi"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        # Create 5 frames
        for i in range(5):
            name = f"{i:06d}"
            dummy = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / f"{name}.jpg"), dummy)
            (lbl_dir / f"{name}.txt").write_text(
                _make_all_visible_label_line() + "\n"
            )

        output_dir = tmp_path / "output"
        count = convert_tii_flight(flight_dir, output_dir, max_samples=3)
        assert count == 3

    def test_multi_gate_label(self, tmp_path: Path) -> None:
        """A label file with 2 gates should produce a sample with 2 gates."""
        lines = [
            _make_all_visible_label_line(cx=0.3, cy=0.3, w=0.2, h=0.2),
            _make_all_visible_label_line(cx=0.7, cy=0.7, w=0.2, h=0.2),
        ]
        flight_dir = _make_fake_tii_sample(tmp_path, label_lines=lines)
        output_dir = tmp_path / "output"

        count = convert_tii_flight(flight_dir, output_dir)
        assert count == 1

        import json

        sample_dirs = sorted(output_dir.iterdir())
        with open(sample_dirs[0] / "corner_coords.json") as f:
            coords = json.load(f)

        assert len(coords) == 2

    def test_obstacle_mask_all_zeros(self, tmp_path: Path) -> None:
        """TII has no obstacle annotations, so obstacle_mask should be all zeros."""
        flight_dir = _make_fake_tii_sample(tmp_path)
        output_dir = tmp_path / "output"

        convert_tii_flight(flight_dir, output_dir)

        sample_dirs = sorted(output_dir.iterdir())
        obs_mask = cv2.imread(
            str(sample_dirs[0] / "obstacle_mask.png"), cv2.IMREAD_GRAYSCALE
        )
        assert np.all(obs_mask == 0)
