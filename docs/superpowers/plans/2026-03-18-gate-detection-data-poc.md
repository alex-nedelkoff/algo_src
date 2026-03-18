# Gate Detection Training Data — POC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate the gate detection data pipeline end-to-end with 30 images from each source (TII real + BlenderProc synthetic) before scaling to 60k.

**Architecture:** Two independent data sources produce samples in a unified format. TII converter downloads real drone racing footage and converts annotations. BlenderProc pipeline renders DCL-style gates in randomized 3D scenes. A merge script combines both into a single manifest with train/val split. A validation script visually confirms format correctness.

**Tech Stack:** Python 3.11, OpenCV, NumPy, PyArrow, BlenderProc (bundles Blender 4.2), trimesh (gate mesh generation), Docker (nvidia/cuda base)

**Spec:** `docs/superpowers/specs/2026-03-17-gate-detection-training-data-design.md`

**Linear:** COR-61

---

## File Structure

```
perception/training/
  data/
    gate_detection/
      __init__.py               # Empty
      format.py                 # Shared format constants, I/O helpers, heatmap generation
      gate_mesh.py              # Procedural DCL gate OBJ generator (trimesh)
      tii_converter.py          # Download + subsample + convert TII dataset
      blenderproc_pipeline.py   # BlenderProc scene generation script (run via `blenderproc run`)
      merge.py                  # Merge sources → manifest.parquet + splits
      validate.py               # Visual sanity checks + distribution stats
  docker/
    Dockerfile.gate-data        # Extends existing Docker pattern with BlenderProc + trimesh
tests/
  perception/
    training/
      data/
        gate_detection/
          test_format.py        # Tests for format helpers + heatmap generation
          test_gate_mesh.py     # Tests for procedural gate mesh
          test_tii_converter.py # Tests for TII annotation conversion
          test_merge.py         # Tests for merge + manifest
```

---

### Task 1: Shared Format Module

**Files:**
- Create: `perception/training/data/gate_detection/__init__.py`
- Create: `perception/training/data/gate_detection/format.py`
- Test: `tests/perception/training/data/gate_detection/test_format.py`

- [ ] **Step 1: Write failing tests for format helpers**

```python
# tests/perception/training/data/gate_detection/test_format.py
"""Tests for gate detection dataset format helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


def test_generate_corner_heatmaps_single_gate():
    """Single gate corners produce 4 heatmaps with peaks at correct locations."""
    from perception.training.data.gate_detection.format import generate_corner_heatmaps

    corners = [{"gate_id": 0, "corners": [[100, 50], [200, 50], [200, 150], [100, 150]], "confidence": 1.0}]
    heatmaps = generate_corner_heatmaps(corners, height=480, width=640, stride=4, sigma=2.0)

    assert heatmaps.shape == (4, 120, 160)
    assert heatmaps.dtype == np.float32
    # Each channel should have a peak near the corresponding corner
    for ch in range(4):
        peak_y, peak_x = np.unravel_index(heatmaps[ch].argmax(), heatmaps[ch].shape)
        expected_x = int(corners[0]["corners"][ch][0] / 4)
        expected_y = int(corners[0]["corners"][ch][1] / 4)
        assert abs(peak_x - expected_x) <= 1
        assert abs(peak_y - expected_y) <= 1


def test_generate_corner_heatmaps_multiple_gates():
    """Two gates produce 4 heatmaps each with 2 peaks."""
    from perception.training.data.gate_detection.format import generate_corner_heatmaps

    corners = [
        {"gate_id": 0, "corners": [[100, 50], [200, 50], [200, 150], [100, 150]], "confidence": 1.0},
        {"gate_id": 1, "corners": [[300, 200], [400, 200], [400, 300], [300, 300]], "confidence": 1.0},
    ]
    heatmaps = generate_corner_heatmaps(corners, height=480, width=640, stride=4, sigma=2.0)

    assert heatmaps.shape == (4, 120, 160)
    # Each channel should have 2 distinct peaks (above some threshold)
    for ch in range(4):
        peaks = heatmaps[ch] > 0.5
        # At least 2 connected regions of high activation
        assert peaks.sum() >= 2


def test_render_gate_mask_single_gate():
    """Filled quadrilateral from 4 corners produces a binary mask."""
    from perception.training.data.gate_detection.format import render_gate_mask

    corners = [{"gate_id": 0, "corners": [[100, 50], [200, 50], [200, 150], [100, 150]], "confidence": 1.0}]
    mask = render_gate_mask(corners, height=480, width=640)

    assert mask.shape == (480, 640)
    assert mask.dtype == np.uint8
    assert mask.max() == 1  # instance ID 1
    assert mask[100, 150] == 1  # center of gate should be filled
    assert mask[0, 0] == 0  # corner of image should be bg


def test_render_gate_mask_multiple_gates():
    """Multiple gates get distinct instance IDs."""
    from perception.training.data.gate_detection.format import render_gate_mask

    corners = [
        {"gate_id": 0, "corners": [[100, 50], [200, 50], [200, 150], [100, 150]], "confidence": 1.0},
        {"gate_id": 1, "corners": [[300, 200], [400, 200], [400, 300], [300, 300]], "confidence": 1.0},
    ]
    mask = render_gate_mask(corners, height=480, width=640)

    assert 1 in mask  # gate 0 → instance 1
    assert 2 in mask  # gate 1 → instance 2


def test_save_and_load_sample(tmp_path):
    """Round-trip save/load of a sample preserves all fields."""
    from perception.training.data.gate_detection.format import save_sample, load_sample

    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    gate_mask = np.zeros((480, 640), dtype=np.uint8)
    gate_mask[50:150, 100:200] = 1
    obstacle_mask = np.zeros((480, 640), dtype=np.uint8)
    corner_coords = [{"gate_id": 0, "corners": [[100, 50], [200, 50], [200, 150], [100, 150]], "confidence": 1.0}]
    heatmaps = np.random.rand(4, 120, 160).astype(np.float32)
    metadata = {"source": "test", "resolution": [480, 640], "intrinsics": None, "distortion_model": None, "distortion_coeffs": None, "gate_dims_m": [1.5, 1.5]}

    sample_dir = tmp_path / "test_sample"
    save_sample(sample_dir, rgb=rgb, gate_mask=gate_mask, obstacle_mask=obstacle_mask, corner_coords=corner_coords, corner_heatmaps=heatmaps, metadata=metadata)

    loaded = load_sample(sample_dir)
    np.testing.assert_array_equal(loaded["rgb"], rgb)
    np.testing.assert_array_equal(loaded["gate_mask"], gate_mask)
    np.testing.assert_array_equal(loaded["obstacle_mask"], obstacle_mask)
    np.testing.assert_array_almost_equal(loaded["corner_heatmaps"], heatmaps)
    assert loaded["corner_coords"] == corner_coords
    assert loaded["metadata"]["source"] == "test"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/perception/training/data/gate_detection/test_format.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement format.py**

```python
# perception/training/data/gate_detection/format.py
"""Shared format constants and I/O helpers for gate detection dataset."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Corner order convention: TL=0, TR=1, BR=2, BL=3
CORNER_NAMES = ["TL", "TR", "BR", "BL"]


def generate_corner_heatmaps(
    corner_coords: list[dict],
    height: int,
    width: int,
    stride: int = 4,
    sigma: float = 2.0,
) -> np.ndarray:
    """Generate CenterNet-style corner heatmaps.

    Args:
        corner_coords: List of dicts with 'gate_id', 'corners' ([[x,y]x4]), 'confidence'.
        height: Image height in pixels.
        width: Image width in pixels.
        stride: Output stride (heatmap is height//stride x width//stride).
        sigma: Gaussian kernel sigma in heatmap pixels.

    Returns:
        (4, H//stride, W//stride) float32 heatmaps, one channel per corner type.
    """
    h_out = height // stride
    w_out = width // stride
    heatmaps = np.zeros((4, h_out, w_out), dtype=np.float32)

    radius = int(3 * sigma)
    for gate in corner_coords:
        for ch, (cx, cy) in enumerate(gate["corners"]):
            # Map to heatmap coordinates
            hx = int(cx / stride)
            hy = int(cy / stride)
            # Gaussian kernel
            y_min = max(0, hy - radius)
            y_max = min(h_out, hy + radius + 1)
            x_min = max(0, hx - radius)
            x_max = min(w_out, hx + radius + 1)
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    d2 = (x - hx) ** 2 + (y - hy) ** 2
                    val = np.exp(-d2 / (2 * sigma**2))
                    heatmaps[ch, y, x] = max(heatmaps[ch, y, x], val)

    return heatmaps


def render_gate_mask(
    corner_coords: list[dict],
    height: int,
    width: int,
) -> np.ndarray:
    """Render filled gate quadrilaterals as an instance mask.

    Args:
        corner_coords: List of dicts with 'gate_id', 'corners' ([[x,y]x4]), 'confidence'.
        height: Image height.
        width: Image width.

    Returns:
        (H, W) uint8 mask. 0=background, 1+=gate instance IDs.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for i, gate in enumerate(corner_coords):
        pts = np.array(gate["corners"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], color=int(i + 1))
    return mask


def save_sample(
    sample_dir: str | Path,
    *,
    rgb: np.ndarray,
    gate_mask: np.ndarray,
    obstacle_mask: np.ndarray,
    corner_coords: list[dict],
    corner_heatmaps: np.ndarray,
    metadata: dict,
) -> None:
    """Save a single dataset sample to disk."""
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(sample_dir / "rgb.png"), rgb)
    cv2.imwrite(str(sample_dir / "gate_mask.png"), gate_mask)
    cv2.imwrite(str(sample_dir / "obstacle_mask.png"), obstacle_mask)
    np.save(str(sample_dir / "corner_heatmaps.npy"), corner_heatmaps)

    with open(sample_dir / "corner_coords.json", "w") as f:
        json.dump(corner_coords, f)
    with open(sample_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)


def load_sample(sample_dir: str | Path) -> dict:
    """Load a single dataset sample from disk."""
    sample_dir = Path(sample_dir)

    rgb = cv2.imread(str(sample_dir / "rgb.png"), cv2.IMREAD_COLOR)
    gate_mask = cv2.imread(str(sample_dir / "gate_mask.png"), cv2.IMREAD_GRAYSCALE)
    obstacle_mask = cv2.imread(str(sample_dir / "obstacle_mask.png"), cv2.IMREAD_GRAYSCALE)
    corner_heatmaps = np.load(str(sample_dir / "corner_heatmaps.npy"))

    with open(sample_dir / "corner_coords.json") as f:
        corner_coords = json.load(f)
    with open(sample_dir / "metadata.json") as f:
        metadata = json.load(f)

    return {
        "rgb": rgb,
        "gate_mask": gate_mask,
        "obstacle_mask": obstacle_mask,
        "corner_heatmaps": corner_heatmaps,
        "corner_coords": corner_coords,
        "metadata": metadata,
    }
```

Also create the empty `__init__.py`:

```python
# perception/training/data/gate_detection/__init__.py
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/perception/training/data/gate_detection/test_format.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add perception/training/data/gate_detection/__init__.py \
       perception/training/data/gate_detection/format.py \
       tests/perception/training/data/gate_detection/test_format.py
git commit -m "feat(gate-data): add shared format module with heatmap + mask generation"
```

---

### Task 2: Procedural Gate Mesh Generator

**Files:**
- Create: `perception/training/data/gate_detection/gate_mesh.py`
- Test: `tests/perception/training/data/gate_detection/test_gate_mesh.py`

- [ ] **Step 1: Write failing tests for gate mesh**

```python
# tests/perception/training/data/gate_detection/test_gate_mesh.py
"""Tests for procedural DCL gate mesh generator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")


def test_generate_gate_mesh_dimensions():
    """Gate mesh has correct inner opening dimensions."""
    from perception.training.data.gate_detection.gate_mesh import generate_gate_mesh

    mesh = generate_gate_mesh(inner_width=1.5, inner_height=1.5, frame_width=0.1)

    # Bounding box should be slightly larger than inner opening
    bounds = mesh.bounds  # (2, 3) array: [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    outer_width = bounds[1][0] - bounds[0][0]
    outer_height = bounds[1][1] - bounds[0][1]
    assert abs(outer_width - 1.7) < 0.01  # 1.5 + 2*0.1
    assert abs(outer_height - 1.7) < 0.01


def test_generate_gate_mesh_is_watertight():
    """Gate mesh is a valid closed mesh."""
    from perception.training.data.gate_detection.gate_mesh import generate_gate_mesh

    mesh = generate_gate_mesh()
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0


def test_generate_gate_mesh_corners():
    """Gate inner corners are at expected positions."""
    from perception.training.data.gate_detection.gate_mesh import get_gate_inner_corners

    corners = get_gate_inner_corners(inner_width=1.5, inner_height=1.5)
    assert corners.shape == (4, 3)
    # TL, TR, BR, BL in gate-local frame (gate in XY plane, centered at origin)
    expected = np.array([
        [-0.75,  0.75, 0.0],  # TL
        [ 0.75,  0.75, 0.0],  # TR
        [ 0.75, -0.75, 0.0],  # BR
        [-0.75, -0.75, 0.0],  # BL
    ])
    np.testing.assert_array_almost_equal(corners, expected)


def test_export_gate_obj(tmp_path):
    """Gate mesh exports to valid OBJ file."""
    from perception.training.data.gate_detection.gate_mesh import generate_gate_mesh

    mesh = generate_gate_mesh()
    obj_path = tmp_path / "gate.obj"
    mesh.export(str(obj_path))
    assert obj_path.exists()
    assert obj_path.stat().st_size > 0
    # Reload and verify
    reloaded = trimesh.load(str(obj_path))
    assert len(reloaded.vertices) == len(mesh.vertices)


def test_generate_drone_mesh():
    """Drone mesh is a simple placeholder with reasonable size."""
    from perception.training.data.gate_detection.gate_mesh import generate_drone_mesh

    mesh = generate_drone_mesh(size=0.3)
    assert len(mesh.vertices) > 0
    bounds = mesh.bounds
    max_dim = (bounds[1] - bounds[0]).max()
    assert 0.2 < max_dim < 0.5  # roughly drone-sized
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/perception/training/data/gate_detection/test_gate_mesh.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement gate_mesh.py**

```python
# perception/training/data/gate_detection/gate_mesh.py
"""Procedural DCL-style gate and drone mesh generation using trimesh."""

from __future__ import annotations

import numpy as np
import trimesh


def get_gate_inner_corners(
    inner_width: float = 1.5,
    inner_height: float = 1.5,
) -> np.ndarray:
    """Return 4 inner corner positions in gate-local frame (XY plane, centered at origin).

    Order: TL, TR, BR, BL.

    Returns:
        (4, 3) float64 array of corner positions in meters.
    """
    hw = inner_width / 2
    hh = inner_height / 2
    return np.array([
        [-hw,  hh, 0.0],  # TL
        [ hw,  hh, 0.0],  # TR
        [ hw, -hh, 0.0],  # BR
        [-hw, -hh, 0.0],  # BL
    ], dtype=np.float64)


def generate_gate_mesh(
    inner_width: float = 1.5,
    inner_height: float = 1.5,
    frame_width: float = 0.1,
    frame_depth: float = 0.05,
) -> trimesh.Trimesh:
    """Generate a DCL-style square gate frame mesh.

    The gate is a hollow rectangular frame in the XY plane, centered at origin.
    Inner opening is inner_width x inner_height. Frame bars are frame_width wide
    and frame_depth thick (in Z).

    Args:
        inner_width: Inner opening width in meters.
        inner_height: Inner opening height in meters.
        frame_width: Width of each frame bar in meters.
        frame_depth: Depth (thickness in Z) of the frame in meters.

    Returns:
        trimesh.Trimesh of the gate frame.
    """
    ow = inner_width / 2 + frame_width
    oh = inner_height / 2 + frame_width
    iw = inner_width / 2
    ih = inner_height / 2
    d = frame_depth / 2

    # Outer box minus inner box = frame
    outer = trimesh.creation.box(extents=[ow * 2, oh * 2, frame_depth])
    inner = trimesh.creation.box(extents=[iw * 2, ih * 2, frame_depth * 2])  # slightly thicker to ensure clean cut
    frame = trimesh.boolean.difference([outer, inner], engine="blender")

    # If boolean fails (no blender engine), fall back to 4 bars
    if frame is None or len(frame.vertices) == 0:
        bars = []
        # Top bar
        bars.append(trimesh.creation.box(
            extents=[ow * 2, frame_width, frame_depth],
            transform=trimesh.transformations.translation_matrix([0, ih + frame_width / 2, 0]),
        ))
        # Bottom bar
        bars.append(trimesh.creation.box(
            extents=[ow * 2, frame_width, frame_depth],
            transform=trimesh.transformations.translation_matrix([0, -(ih + frame_width / 2), 0]),
        ))
        # Left bar
        bars.append(trimesh.creation.box(
            extents=[frame_width, inner_height, frame_depth],
            transform=trimesh.transformations.translation_matrix([-(iw + frame_width / 2), 0, 0]),
        ))
        # Right bar
        bars.append(trimesh.creation.box(
            extents=[frame_width, inner_height, frame_depth],
            transform=trimesh.transformations.translation_matrix([(iw + frame_width / 2), 0, 0]),
        ))
        frame = trimesh.util.concatenate(bars)

    return frame


def generate_drone_mesh(size: float = 0.3) -> trimesh.Trimesh:
    """Generate a simple placeholder drone mesh (cross-shaped body + 4 rotors).

    Args:
        size: Overall diameter of drone in meters.

    Returns:
        trimesh.Trimesh of the drone.
    """
    arm_length = size / 2
    arm_width = size * 0.08
    arm_height = size * 0.04
    rotor_radius = size * 0.15
    rotor_height = size * 0.02

    parts = []
    # Central body
    parts.append(trimesh.creation.box(extents=[arm_width * 2, arm_width * 2, arm_height * 2]))

    # 4 arms at 45-degree offsets
    for angle_deg in [45, 135, 225, 315]:
        angle = np.radians(angle_deg)
        dx = np.cos(angle) * arm_length / 2
        dy = np.sin(angle) * arm_length / 2
        arm = trimesh.creation.box(
            extents=[arm_length, arm_width, arm_height],
            transform=trimesh.transformations.compose_matrix(
                translate=[dx, dy, 0],
                angles=[0, 0, angle],
            ),
        )
        parts.append(arm)
        # Rotor disc at arm tip
        rotor = trimesh.creation.cylinder(
            radius=rotor_radius, height=rotor_height,
            transform=trimesh.transformations.translation_matrix([
                np.cos(angle) * arm_length, np.sin(angle) * arm_length, arm_height,
            ]),
        )
        parts.append(rotor)

    return trimesh.util.concatenate(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/perception/training/data/gate_detection/test_gate_mesh.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Export OBJ assets for later use**

```bash
python -c "
from perception.training.data.gate_detection.gate_mesh import generate_gate_mesh, generate_drone_mesh
generate_gate_mesh().export('perception/training/data/gate_detection/assets/gate.obj')
generate_drone_mesh().export('perception/training/data/gate_detection/assets/drone.obj')
print('Exported gate.obj and drone.obj')
"
```

- [ ] **Step 6: Commit**

```bash
git add perception/training/data/gate_detection/gate_mesh.py \
       perception/training/data/gate_detection/assets/ \
       tests/perception/training/data/gate_detection/test_gate_mesh.py
git commit -m "feat(gate-data): procedural DCL gate + drone mesh generator"
```

---

### Task 3: TII Dataset Converter (30 images POC)

**Files:**
- Create: `perception/training/data/gate_detection/tii_converter.py`
- Test: `tests/perception/training/data/gate_detection/test_tii_converter.py`

**Prereq:** Download TII autonomous data (~4.95 GB) to `~/corvidx/data/raw/tii/`:
```bash
mkdir -p ~/corvidx/data/raw/tii && cd ~/corvidx/data/raw/tii
wget https://github.com/Drone-Racing/drone-racing-dataset/releases/download/v3.0.0/autonomous_zipchunk01
wget https://github.com/Drone-Racing/drone-racing-dataset/releases/download/v3.0.0/autonomous_zipchunk02
wget https://github.com/Drone-Racing/drone-racing-dataset/releases/download/v3.0.0/autonomous_zipchunk03
cat autonomous_zipchunk* > autonomous.zip && rm autonomous_zipchunk*
unzip autonomous.zip
find autonomous -name '*.zip' -execdir unzip {} \;
```

- [ ] **Step 1: Write failing tests for TII parsing**

```python
# tests/perception/training/data/gate_detection/test_tii_converter.py
"""Tests for TII dataset annotation conversion."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest


def _make_fake_tii_sample(tmpdir: Path, frame_name: str = "000001") -> Path:
    """Create a minimal fake TII sample (image + label) for testing."""
    flight_dir = tmpdir / "autonomous" / "flight-test"
    img_dir = flight_dir / "camera_flight-test"
    lbl_dir = flight_dir / "labels_flight-test"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    # Create a dummy 640x480 image
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir / f"{frame_name}.jpg"), img)

    # Create a TII-format label: 1 gate with 4 corners (normalized coords)
    # Format: class cx cy w h tlx tly tlv trx try trv brx bry brv blx bly blv
    label = "0 0.234 0.312 0.156 0.208 0.156 0.208 2 0.312 0.208 2 0.312 0.416 2 0.156 0.416 2\n"
    (lbl_dir / f"{frame_name}.txt").write_text(label)

    return tmpdir


def test_parse_tii_label():
    """TII label line parses to corner coords in pixel space."""
    from perception.training.data.gate_detection.tii_converter import parse_tii_label

    line = "0 0.234 0.312 0.156 0.208 0.156 0.208 2 0.312 0.208 2 0.312 0.416 2 0.156 0.416 2"
    gates = parse_tii_label(line, img_width=640, img_height=480)

    assert len(gates) == 1
    assert gates[0]["gate_id"] == 0
    corners = gates[0]["corners"]
    assert len(corners) == 4
    # Corners should be in pixel space (denormalized)
    for cx, cy in corners:
        assert 0 <= cx <= 640
        assert 0 <= cy <= 480


def test_parse_tii_label_invisible_corner():
    """Corners with visibility=0 are excluded; gate dropped if <4 visible."""
    from perception.training.data.gate_detection.tii_converter import parse_tii_label

    # visibility 0 means outside image
    line = "0 0.5 0.5 0.2 0.2 0.1 0.1 0 0.9 0.1 2 0.9 0.9 2 0.1 0.9 2"
    gates = parse_tii_label(line, img_width=640, img_height=480)
    # Gate has only 3 visible corners → should be dropped
    assert len(gates) == 0


def test_convert_tii_sample(tmp_path):
    """Full conversion of a fake TII sample produces valid format output."""
    from perception.training.data.gate_detection.tii_converter import convert_tii_flight
    from perception.training.data.gate_detection.format import load_sample

    src = _make_fake_tii_sample(tmp_path)
    out_dir = tmp_path / "output"

    n = convert_tii_flight(
        flight_dir=src / "autonomous" / "flight-test",
        output_dir=out_dir,
        max_samples=5,
    )

    assert n > 0
    # Check that output samples exist and are loadable
    samples = list(out_dir.iterdir())
    assert len(samples) == n
    sample = load_sample(samples[0])
    assert sample["rgb"].shape[0] == 480
    assert sample["rgb"].shape[1] == 640
    assert sample["gate_mask"].shape == (480, 640)
    assert sample["metadata"]["source"] == "tii"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/perception/training/data/gate_detection/test_tii_converter.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement tii_converter.py**

```python
# perception/training/data/gate_detection/tii_converter.py
"""Convert TII Race Against the Machine dataset to gate detection format."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from perception.training.data.gate_detection.format import (
    generate_corner_heatmaps,
    render_gate_mask,
    save_sample,
)

logger = logging.getLogger(__name__)


def parse_tii_label(line: str, img_width: int, img_height: int) -> list[dict]:
    """Parse a single TII label line into gate corner coordinates.

    TII format per gate: class cx cy w h tlx tly tlv trx try trv brx bry brv blx bly blv
    All values normalized [0,1]. Visibility: 0=outside, 2=inside image.

    Returns list of dicts with keys: gate_id, corners ([[x,y]x4]), confidence.
    Only returns gates with all 4 corners visible (visibility == 2).
    """
    tokens = line.strip().split()
    if len(tokens) < 17:
        return []

    gates = []
    # Each gate uses 17 tokens (1 class + 4 bbox + 4*(x,y,vis))
    idx = 0
    gate_id = 0
    while idx + 17 <= len(tokens):
        # Skip class + bbox (5 tokens)
        idx += 5
        corners = []
        all_visible = True
        for _ in range(4):
            cx = float(tokens[idx]) * img_width
            cy = float(tokens[idx + 1]) * img_height
            vis = int(tokens[idx + 2])
            if vis == 0:
                all_visible = False
            corners.append([cx, cy])
            idx += 3

        if all_visible:
            gates.append({
                "gate_id": gate_id,
                "corners": corners,
                "confidence": 1.0,
            })
        gate_id += 1

    return gates


def convert_tii_flight(
    flight_dir: str | Path,
    output_dir: str | Path,
    max_samples: int | None = None,
) -> int:
    """Convert a single TII flight directory to gate detection format.

    Args:
        flight_dir: Path to flight directory (contains camera_flight-*/  and labels_flight-*/).
        output_dir: Where to write converted samples.
        max_samples: Maximum samples to convert (None = all).

    Returns:
        Number of samples written.
    """
    flight_dir = Path(flight_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find image and label directories
    img_dirs = list(flight_dir.glob("camera_flight-*"))
    lbl_dirs = list(flight_dir.glob("labels_flight-*"))
    if not img_dirs or not lbl_dirs:
        logger.warning("No camera/labels dirs in %s", flight_dir)
        return 0

    img_dir = img_dirs[0]
    lbl_dir = lbl_dirs[0]

    # Get sorted frame list (matching image + label pairs)
    img_files = sorted(img_dir.glob("*.jpg"))
    written = 0

    for img_path in img_files:
        if max_samples is not None and written >= max_samples:
            break

        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        # Read image
        rgb = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if rgb is None:
            continue
        h, w = rgb.shape[:2]

        # Parse labels
        label_text = lbl_path.read_text().strip()
        if not label_text:
            continue

        gates = []
        for line in label_text.split("\n"):
            gates.extend(parse_tii_label(line, img_width=w, img_height=h))

        if not gates:
            continue

        # Generate annotations
        gate_mask = render_gate_mask(gates, height=h, width=w)
        heatmaps = generate_corner_heatmaps(gates, height=h, width=w)
        obstacle_mask = np.zeros((h, w), dtype=np.uint8)

        flight_name = flight_dir.name
        sample_id = f"{flight_name}_{img_path.stem}"
        metadata = {
            "source": "tii",
            "resolution": [h, w],
            "intrinsics": None,  # TODO: load from TII calibration files
            "distortion_model": None,
            "distortion_coeffs": None,
            "gate_dims_m": [1.52, 1.52],  # TII gate: 5ft = 152cm
        }

        save_sample(
            output_dir / sample_id,
            rgb=rgb,
            gate_mask=gate_mask,
            obstacle_mask=obstacle_mask,
            corner_coords=gates,
            corner_heatmaps=heatmaps,
            metadata=metadata,
        )
        written += 1

    logger.info("Wrote %d samples from %s", written, flight_dir.name)
    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Convert TII dataset to gate detection format")
    parser.add_argument("--data-dir", required=True, type=Path, help="Path to raw TII data (contains autonomous/)")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--n-samples", type=int, default=30, help="Max samples to convert")
    parser.add_argument("--flights", nargs="*", default=None, help="Flight names to process (default: first available)")
    args = parser.parse_args()

    auto_dir = args.data_dir / "autonomous"
    if not auto_dir.exists():
        logger.error("No autonomous/ directory in %s", args.data_dir)
        return

    flights = sorted(auto_dir.iterdir())
    if args.flights:
        flights = [auto_dir / f for f in args.flights]

    total = 0
    for flight_dir in flights:
        if not flight_dir.is_dir():
            continue
        remaining = args.n_samples - total if args.n_samples else None
        if remaining is not None and remaining <= 0:
            break
        n = convert_tii_flight(flight_dir, args.output_dir, max_samples=remaining)
        total += n

    logger.info("Total: %d samples converted", total)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/perception/training/data/gate_detection/test_tii_converter.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add perception/training/data/gate_detection/tii_converter.py \
       tests/perception/training/data/gate_detection/test_tii_converter.py
git commit -m "feat(gate-data): TII dataset converter with annotation parsing"
```

---

### Task 4: BlenderProc Pipeline (30 images POC)

**Files:**
- Create: `perception/training/data/gate_detection/blenderproc_pipeline.py`

**Note:** This script runs via `blenderproc run`, not `python`. It cannot be unit-tested with pytest directly since it requires Blender's Python environment. We test it by running it and inspecting output with the validate script (Task 6).

- [ ] **Step 1: Implement blenderproc_pipeline.py**

```python
# perception/training/data/gate_detection/blenderproc_pipeline.py
"""BlenderProc synthetic gate detection data generator.

Run with: blenderproc run blenderproc_pipeline.py --gate-obj assets/gate.obj --output-dir /output --n-samples 30

This script runs inside Blender's Python environment via BlenderProc.
It cannot be run with standard Python or tested with pytest.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import blenderproc as bproc
import bpy
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def random_color_material(obj, rng):
    """Assign a random emissive color to simulate LED gate lighting."""
    # DCL-style LED colors: blue, purple, green, white, cyan
    led_colors = [
        [0.0, 0.3, 1.0, 1.0],   # blue
        [0.6, 0.0, 1.0, 1.0],   # purple
        [0.0, 1.0, 0.3, 1.0],   # green
        [1.0, 1.0, 1.0, 1.0],   # white
        [0.0, 1.0, 1.0, 1.0],   # cyan
    ]
    color = led_colors[rng.integers(len(led_colors))]
    mat = obj.get_materials()[0] if obj.get_materials() else bproc.material.create("gate_mat")
    mat.set_principled_shader_value("Base Color", color)
    mat.set_principled_shader_value("Emission Color", color)
    mat.set_principled_shader_value("Emission Strength", rng.uniform(2.0, 10.0))
    mat.set_principled_shader_value("Metallic", rng.uniform(0.5, 0.9))
    obj.set_material(0, mat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-obj", required=True, type=str, help="Path to gate OBJ mesh")
    parser.add_argument("--drone-obj", default=None, type=str, help="Path to drone OBJ mesh")
    parser.add_argument("--output-dir", required=True, type=str, help="Output directory")
    parser.add_argument("--n-samples", type=int, default=30, help="Number of images to render")
    parser.add_argument("--seed", type=int, default=42)
    # Parse only known args (blenderproc adds its own)
    args, _ = parser.parse_known_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    bproc.init()

    # Renderer settings — use Cycles for headless compatibility
    bproc.renderer.set_max_amount_of_samples(64)  # low for POC speed
    bproc.renderer.set_noise_threshold(0.05)
    bproc.camera.set_resolution(640, 480)

    # Enable outputs
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_segmentation_output(
        map_by=["category_id", "instance"],
        default_values={"category_id": 0},
    )

    for sample_idx in range(args.n_samples):
        bproc.utility.reset_keyframes()

        # Clean scene (do NOT re-call bproc.init() — it should only be called once)
        bproc.clean_up()

        # Random background color
        bg_color = rng.uniform(0.05, 0.5, size=3).tolist() + [1.0]
        bproc.world.set_world_background_hdr_img(None)
        bpy.context.scene.world.use_nodes = True
        bg_node = bpy.context.scene.world.node_tree.nodes.get("Background")
        if bg_node:
            bg_node.inputs["Color"].default_value = bg_color

        # Load and place 1-3 gates
        n_gates = rng.integers(1, 4)
        gate_objects = []
        gate_corners_3d = []

        for gate_idx in range(n_gates):
            objs = bproc.loader.load_obj(args.gate_obj)
            gate = objs[0]
            gate.set_cp("category_id", 1)  # class 1 = gate

            # Random position: 2-12m in front, +/-3m lateral, +/-2m vertical
            x = rng.uniform(2.0, 12.0)
            y = rng.uniform(-3.0, 3.0)
            z = rng.uniform(-2.0, 2.0)
            gate.set_location([x, y, z])

            # Random rotation
            gate.set_rotation_euler([
                rng.uniform(-0.3, 0.3),
                rng.uniform(-0.3, 0.3),
                rng.uniform(-np.pi, np.pi),
            ])

            random_color_material(gate, rng)
            gate_objects.append(gate)

        # Load and place 0-2 drones
        n_drones = 0
        if args.drone_obj and Path(args.drone_obj).exists():
            n_drones = rng.integers(0, 3)
            for _ in range(n_drones):
                drone_objs = bproc.loader.load_obj(args.drone_obj)
                drone = drone_objs[0]
                drone.set_cp("category_id", 2)  # class 2 = drone
                drone.set_location([
                    rng.uniform(2.0, 15.0),
                    rng.uniform(-5.0, 5.0),
                    rng.uniform(-3.0, 3.0),
                ])
                drone.set_rotation_euler(rng.uniform(-np.pi, np.pi, 3).tolist())

        # Add random lighting
        light = bproc.types.Light()
        light.set_type("POINT")
        light.set_location(rng.uniform(-5, 5, 3).tolist())
        light.set_energy(rng.uniform(100, 500))

        # Camera at origin looking forward (+X)
        cam_pose = np.eye(4)
        bproc.camera.add_camera_pose(cam_pose)

        # Render
        data = bproc.renderer.render()

        # Extract RGB
        rgb = data["colors"][0]  # (H, W, 4) RGBA uint8
        rgb_bgr = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)

        # Extract instance segmentation
        inst_seg = data["instance_segmaps"][0]  # (H, W) int
        cat_seg = data["category_id_segmaps"][0]  # (H, W) int

        # Build gate mask (category_id == 1)
        gate_mask = np.zeros_like(cat_seg, dtype=np.uint8)
        gate_instance_id = 1
        for inst_val in np.unique(inst_seg):
            if inst_val == 0:
                continue
            region = inst_seg == inst_val
            cat_val = cat_seg[region][0] if region.any() else 0
            if cat_val == 1:  # gate
                gate_mask[region] = gate_instance_id
                gate_instance_id += 1

        # Build obstacle mask (category_id == 2 → drone)
        obstacle_mask = np.zeros_like(cat_seg, dtype=np.uint8)
        obstacle_mask[cat_seg == 2] = 2  # drone class

        # Project gate corners to 2D
        K = bproc.camera.get_intrinsics_as_K_matrix()
        h, w = rgb.shape[:2]
        corner_coords = []
        # Gate inner corners in local frame (from gate_mesh.py convention)
        hw_gate = 0.75  # 1.5m / 2
        local_corners = np.array([
            [-hw_gate,  hw_gate, 0.0],  # TL
            [ hw_gate,  hw_gate, 0.0],  # TR
            [ hw_gate, -hw_gate, 0.0],  # BR
            [-hw_gate, -hw_gate, 0.0],  # BL
        ])

        for gate_idx, gate in enumerate(gate_objects):
            world_corners = gate.blender_obj.matrix_world @ \
                np.hstack([local_corners, np.ones((4, 1))]).T
            world_corners = world_corners[:3].T  # (4, 3)
            pixels = bproc.camera.project_points(world_corners, frame=0)
            # Check if corners are in frame
            in_frame = np.all((pixels >= 0) & (pixels < [w, h]), axis=1)
            if in_frame.all():
                corner_coords.append({
                    "gate_id": gate_idx,
                    "corners": pixels.tolist(),
                    "confidence": 1.0,
                })

        # Generate heatmaps
        from perception.training.data.gate_detection.format import generate_corner_heatmaps
        heatmaps = generate_corner_heatmaps(corner_coords, height=h, width=w)

        # Save sample
        import cv2
        sample_id = f"blenderproc_{sample_idx:06d}"
        sample_dir = output_dir / sample_id

        from perception.training.data.gate_detection.format import save_sample
        metadata = {
            "source": "blenderproc",
            "resolution": [h, w],
            "intrinsics": K.tolist(),
            "distortion_model": "pinhole",
            "distortion_coeffs": None,
            "gate_dims_m": [1.5, 1.5],
            "n_gates": len(gate_objects),
            "n_drones": n_drones,
        }

        save_sample(
            sample_dir,
            rgb=rgb_bgr,
            gate_mask=gate_mask,
            obstacle_mask=obstacle_mask,
            corner_coords=corner_coords,
            corner_heatmaps=heatmaps,
            metadata=metadata,
        )

        print(f"[{sample_idx+1}/{args.n_samples}] Saved {sample_id} ({len(corner_coords)} gates in frame)")

    print(f"Done. {args.n_samples} samples written to {output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test by running the pipeline with 5 images**

```bash
blenderproc run perception/training/data/gate_detection/blenderproc_pipeline.py \
  --gate-obj perception/training/data/gate_detection/assets/gate.obj \
  --drone-obj perception/training/data/gate_detection/assets/drone.obj \
  --output-dir ~/corvidx/data/gate_detection/blenderproc \
  --n-samples 5
```

Expected: 5 sample directories created, each with rgb.png, gate_mask.png, etc.

- [ ] **Step 3: Visually inspect a few samples**

```bash
python -c "
from perception.training.data.gate_detection.format import load_sample
from pathlib import Path
import cv2, numpy as np
d = Path.home() / 'corvidx/data/gate_detection/blenderproc'
s = load_sample(sorted(d.iterdir())[0])
print('RGB shape:', s['rgb'].shape)
print('Gate mask unique:', np.unique(s['gate_mask']))
print('Corners:', s['corner_coords'])
print('Heatmaps shape:', s['corner_heatmaps'].shape)
# Save overlay for visual check
overlay = s['rgb'].copy()
overlay[s['gate_mask'] > 0] = [0, 255, 0]
cv2.imwrite('/tmp/blenderproc_check.png', cv2.addWeighted(s['rgb'], 0.7, overlay, 0.3, 0))
print('Overlay saved to /tmp/blenderproc_check.png')
"
```

- [ ] **Step 4: Commit**

```bash
git add perception/training/data/gate_detection/blenderproc_pipeline.py
git commit -m "feat(gate-data): BlenderProc synthetic gate rendering pipeline"
```

---

### Task 5: Merge + Manifest

**Files:**
- Create: `perception/training/data/gate_detection/merge.py`
- Test: `tests/perception/training/data/gate_detection/test_merge.py`

- [ ] **Step 1: Write failing tests for merge**

```python
# tests/perception/training/data/gate_detection/test_merge.py
"""Tests for dataset merge and manifest generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest


def _make_fake_samples(base_dir: Path, source: str, n: int = 3):
    """Create n minimal fake samples in source subdirectory."""
    from perception.training.data.gate_detection.format import save_sample, generate_corner_heatmaps, render_gate_mask

    src_dir = base_dir / source
    corners = [{"gate_id": 0, "corners": [[100, 50], [200, 50], [200, 150], [100, 150]], "confidence": 1.0}]

    for i in range(n):
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        save_sample(
            src_dir / f"{source}_{i:04d}",
            rgb=rgb,
            gate_mask=render_gate_mask(corners, 480, 640),
            obstacle_mask=np.zeros((480, 640), dtype=np.uint8),
            corner_coords=corners,
            corner_heatmaps=generate_corner_heatmaps(corners, 480, 640),
            metadata={"source": source, "resolution": [480, 640], "intrinsics": None,
                       "distortion_model": None, "distortion_coeffs": None, "gate_dims_m": [1.5, 1.5]},
        )


def test_build_manifest(tmp_path):
    """Manifest parquet has correct schema and row count."""
    from perception.training.data.gate_detection.merge import build_manifest

    _make_fake_samples(tmp_path, "tii", 3)
    _make_fake_samples(tmp_path, "blenderproc", 5)

    manifest_path = tmp_path / "manifest.parquet"
    build_manifest(tmp_path, manifest_path)

    table = pq.read_table(manifest_path)
    assert len(table) == 8
    assert "source" in table.column_names
    assert "sample_id" in table.column_names
    assert set(table.column("source").to_pylist()) == {"tii", "blenderproc"}


def test_create_splits(tmp_path):
    """Train/val split produces correct file counts."""
    from perception.training.data.gate_detection.merge import build_manifest, create_splits

    _make_fake_samples(tmp_path, "tii", 5)
    _make_fake_samples(tmp_path, "blenderproc", 5)

    manifest_path = tmp_path / "manifest.parquet"
    build_manifest(tmp_path, manifest_path)

    splits_dir = tmp_path / "splits"
    create_splits(manifest_path, splits_dir, val_ratio=0.2, seed=42)

    train_ids = (splits_dir / "train.txt").read_text().strip().split("\n")
    val_ids = (splits_dir / "val.txt").read_text().strip().split("\n")
    assert len(train_ids) + len(val_ids) == 10
    assert len(val_ids) == 2  # 20% of 10
    # No overlap
    assert set(train_ids) & set(val_ids) == set()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/perception/training/data/gate_detection/test_merge.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement merge.py**

```python
# perception/training/data/gate_detection/merge.py
"""Merge gate detection sources into unified manifest + train/val splits."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

SOURCES = ["tii", "blenderproc"]

SCHEMA = pa.schema([
    ("sample_id", pa.string()),
    ("source", pa.string()),
    ("n_gates", pa.int32()),
    ("n_drones", pa.int32()),
    ("resolution_h", pa.int32()),
    ("resolution_w", pa.int32()),
    ("has_distortion", pa.bool_()),
])


def build_manifest(data_dir: str | Path, output_path: str | Path) -> None:
    """Scan all source subdirectories and build manifest.parquet."""
    data_dir = Path(data_dir)
    output_path = Path(output_path)

    rows = []
    for source in SOURCES:
        src_dir = data_dir / source
        if not src_dir.exists():
            continue
        for sample_dir in sorted(src_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            meta_path = sample_dir / "metadata.json"
            if not meta_path.exists():
                continue
            with open(meta_path) as f:
                meta = json.load(f)

            corner_path = sample_dir / "corner_coords.json"
            n_gates = 0
            if corner_path.exists():
                with open(corner_path) as f:
                    n_gates = len(json.load(f))

            res = meta.get("resolution", [0, 0])
            rows.append({
                "sample_id": sample_dir.name,
                "source": source,
                "n_gates": n_gates,
                "n_drones": meta.get("n_drones", 0),
                "resolution_h": res[0],
                "resolution_w": res[1],
                "has_distortion": meta.get("distortion_model") not in (None, "pinhole"),
            })

    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    pq.write_table(table, output_path)
    logger.info("Manifest: %d samples written to %s", len(rows), output_path)


def create_splits(
    manifest_path: str | Path,
    splits_dir: str | Path,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """Create stratified train/val split from manifest."""
    manifest_path = Path(manifest_path)
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(manifest_path)
    sample_ids = table.column("sample_id").to_pylist()
    sources = table.column("source").to_pylist()

    rng = np.random.default_rng(seed)
    train_ids = []
    val_ids = []

    # Stratify by source
    source_groups = {}
    for sid, src in zip(sample_ids, sources):
        source_groups.setdefault(src, []).append(sid)

    for src, ids in source_groups.items():
        ids = list(ids)
        rng.shuffle(ids)
        n_val = max(1, int(len(ids) * val_ratio))
        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])

    (splits_dir / "train.txt").write_text("\n".join(train_ids) + "\n")
    (splits_dir / "val.txt").write_text("\n".join(val_ids) + "\n")
    logger.info("Split: %d train, %d val", len(train_ids), len(val_ids))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Merge gate detection datasets")
    parser.add_argument("--data-dir", required=True, type=Path, help="Root data directory")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest_path = args.data_dir / "manifest.parquet"
    build_manifest(args.data_dir, manifest_path)
    create_splits(manifest_path, args.data_dir / "splits", val_ratio=args.val_ratio, seed=args.seed)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/perception/training/data/gate_detection/test_merge.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add perception/training/data/gate_detection/merge.py \
       tests/perception/training/data/gate_detection/test_merge.py
git commit -m "feat(gate-data): merge script with manifest + stratified splits"
```

---

### Task 6: Validation Script

**Files:**
- Create: `perception/training/data/gate_detection/validate.py`

- [ ] **Step 1: Implement validate.py**

```python
# perception/training/data/gate_detection/validate.py
"""Visual validation and distribution stats for gate detection dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from perception.training.data.gate_detection.format import load_sample

logger = logging.getLogger(__name__)


def overlay_annotations(rgb: np.ndarray, gate_mask: np.ndarray, corner_coords: list[dict]) -> np.ndarray:
    """Draw gate mask overlay and corner markers on RGB image."""
    vis = rgb.copy()
    # Green overlay for gate mask
    green_overlay = vis.copy()
    green_overlay[gate_mask > 0] = [0, 255, 0]
    vis = cv2.addWeighted(vis, 0.7, green_overlay, 0.3, 0)

    # Draw corners as colored circles
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # TL, TR, BR, BL
    for gate in corner_coords:
        for i, (cx, cy) in enumerate(gate["corners"]):
            cv2.circle(vis, (int(cx), int(cy)), 5, colors[i], -1)
            cv2.putText(vis, f"G{gate['gate_id']}", (int(cx) + 8, int(cy) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return vis


def validate_dataset(data_dir: str | Path, n_vis: int = 10, output_dir: str | Path | None = None) -> None:
    """Run validation checks and save visualizations."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir) if output_dir else data_dir / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = data_dir / "manifest.parquet"
    if not manifest_path.exists():
        logger.error("No manifest.parquet in %s — run merge first", data_dir)
        return

    table = pq.read_table(manifest_path)
    sources = table.column("source").to_pylist()
    sample_ids = table.column("sample_id").to_pylist()
    n_gates_list = table.column("n_gates").to_pylist()

    # Distribution stats
    print(f"\n=== Dataset Stats ({len(table)} total samples) ===")
    for src in sorted(set(sources)):
        count = sources.count(src)
        print(f"  {src}: {count} samples")

    gate_counts = np.array(n_gates_list)
    print(f"\n  Gate count distribution: min={gate_counts.min()}, max={gate_counts.max()}, "
          f"mean={gate_counts.mean():.1f}")
    for n in sorted(set(gate_counts)):
        print(f"    {n} gates: {(gate_counts == n).sum()} samples")

    # Visual samples
    rng = np.random.default_rng(42)
    for src in sorted(set(sources)):
        src_ids = [sid for sid, s in zip(sample_ids, sources) if s == src]
        chosen = rng.choice(src_ids, size=min(n_vis, len(src_ids)), replace=False)
        for sid in chosen:
            sample_dir = data_dir / src / sid
            if not sample_dir.exists():
                continue
            sample = load_sample(sample_dir)
            vis = overlay_annotations(sample["rgb"], sample["gate_mask"], sample["corner_coords"])
            out_path = output_dir / f"{src}_{sid}.png"
            cv2.imwrite(str(out_path), vis)

    print(f"\nVisualizations saved to {output_dir}/")

    # Cross-check: corners on mask edges
    errors = 0
    for sid, src in zip(sample_ids, sources):
        sample_dir = data_dir / src / sid
        if not sample_dir.exists():
            continue
        sample = load_sample(sample_dir)
        mask = sample["gate_mask"]
        for gate in sample["corner_coords"]:
            for cx, cy in gate["corners"]:
                x, y = int(round(cx)), int(round(cy))
                if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                    # Corner should be near mask edge (within 3px of a mask pixel)
                    region = mask[max(0, y-3):y+4, max(0, x-3):x+4]
                    if region.max() == 0:
                        errors += 1

    if errors > 0:
        print(f"\nWARNING: {errors} corners not near mask edges (>3px tolerance)")
    else:
        print("\nAll corners align with mask edges. ✓")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Validate gate detection dataset")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--n-vis", type=int, default=10, help="Visualizations per source")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    validate_dataset(args.data_dir, n_vis=args.n_vis, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add perception/training/data/gate_detection/validate.py
git commit -m "feat(gate-data): validation script with visual overlays + distribution stats"
```

---

### Task 7: Docker Image

**Files:**
- Create: `perception/training/docker/Dockerfile.gate-data`

- [ ] **Step 1: Write Dockerfile**

```dockerfile
# perception/training/docker/Dockerfile.gate-data
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip wget unzip \
    libgl1-mesa-glx libglib2.0-0 libegl1-mesa libxi6 libxxf86vm1 \
    libxkbcommon0 libsm6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

RUN pip install --no-cache-dir \
    numpy opencv-python-headless pyarrow>=14.0 trimesh blenderproc tqdm

# Pre-download Blender so it's baked into the image
RUN python -c "import blenderproc; blenderproc.init()" 2>/dev/null || true

WORKDIR /app
COPY . /app/perception/training/
ENV PYTHONPATH=/app
VOLUME /data

CMD ["bash"]
```

- [ ] **Step 2: Build and test**

```bash
cd ~/corvidx/algo_src
docker build -f perception/training/docker/Dockerfile.gate-data -t gate-data .
docker run --rm --gpus all -v ~/corvidx/data:/data gate-data \
    python -c "from perception.training.data.gate_detection.format import load_sample; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add perception/training/docker/Dockerfile.gate-data
git commit -m "feat(gate-data): Docker image with BlenderProc + trimesh"
```

---

### Task 8: End-to-End POC Run

This task runs the full pipeline to produce 30 TII + 30 BlenderProc samples and validates them.

- [ ] **Step 1: Download TII autonomous data (if not already done)**

```bash
mkdir -p ~/corvidx/data/raw/tii && cd ~/corvidx/data/raw/tii
wget https://github.com/Drone-Racing/drone-racing-dataset/releases/download/v3.0.0/autonomous_zipchunk01
wget https://github.com/Drone-Racing/drone-racing-dataset/releases/download/v3.0.0/autonomous_zipchunk02
wget https://github.com/Drone-Racing/drone-racing-dataset/releases/download/v3.0.0/autonomous_zipchunk03
cat autonomous_zipchunk* > autonomous.zip && rm autonomous_zipchunk*
unzip autonomous.zip
find autonomous -name '*.zip' -execdir unzip {} \;
```

- [ ] **Step 2: Run TII converter (30 samples)**

```bash
python -m perception.training.data.gate_detection.tii_converter \
  --data-dir ~/corvidx/data/raw/tii \
  --output-dir ~/corvidx/data/gate_detection/tii \
  --n-samples 30
```

Expected: 30 sample directories in `~/corvidx/data/gate_detection/tii/`

- [ ] **Step 3: Export gate + drone meshes**

```bash
mkdir -p perception/training/data/gate_detection/assets
python -c "
from perception.training.data.gate_detection.gate_mesh import generate_gate_mesh, generate_drone_mesh
generate_gate_mesh().export('perception/training/data/gate_detection/assets/gate.obj')
generate_drone_mesh().export('perception/training/data/gate_detection/assets/drone.obj')
print('Done')
"
```

- [ ] **Step 4: Run BlenderProc pipeline (30 samples)**

```bash
blenderproc run perception/training/data/gate_detection/blenderproc_pipeline.py \
  --gate-obj perception/training/data/gate_detection/assets/gate.obj \
  --drone-obj perception/training/data/gate_detection/assets/drone.obj \
  --output-dir ~/corvidx/data/gate_detection/blenderproc \
  --n-samples 30
```

Expected: 30 sample directories in `~/corvidx/data/gate_detection/blenderproc/`

- [ ] **Step 5: Merge + validate**

```bash
python -m perception.training.data.gate_detection.merge \
  --data-dir ~/corvidx/data/gate_detection \
  --val-ratio 0.1

python -m perception.training.data.gate_detection.validate \
  --data-dir ~/corvidx/data/gate_detection \
  --n-vis 10
```

Expected output:
```
=== Dataset Stats (60 total samples) ===
  tii: 30 samples
  blenderproc: 30 samples

  Gate count distribution: min=1, max=3, mean=~1.5
    ...

Visualizations saved to ~/corvidx/data/gate_detection/validation/
All corners align with mask edges. ✓
```

- [ ] **Step 6: Visually inspect validation images**

Open `~/corvidx/data/gate_detection/validation/` and check:
- TII samples: green gate overlays match actual gates, corner circles on gate corners
- BlenderProc samples: gates look reasonable, masks align, corners match

- [ ] **Step 7: Commit POC results summary**

```bash
git add -A
git commit -m "feat(gate-data): end-to-end POC pipeline — 30 TII + 30 BlenderProc samples validated"
```
