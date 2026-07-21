"""Pillar-read utilities shared by offline and live perception.

The visual detector/OCR supplies a bounding box and, when readable, a station
number.  This module deliberately does not prescribe a detector: the smoother
only needs an image bearing, while the OCR result selects map candidates.

One physical pillar can carry the same number at several heights.  A lower
marking is safe for the azimuth-only PnL factor but must never be interpreted
as the surveyed top panel for elevation/range.  ``classify_top_reads`` applies
the production rule from the pillar-estimator handoff.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PillarRead:
    """One detector/OCR observation of a station-number panel.

    Coordinates use the existing OpenCV image convention.  ``number`` is
    ``None`` when the panel was detected but OCR could not identify it.
    """

    t_capture: float
    x: float
    y: float
    w: float
    h: float
    number: str | None = None
    confidence: float = 1.0
    top: bool = False
    quad_xy: np.ndarray | None = None  # detector quad, text reading order

    @property
    def center_u(self) -> float:
        return self.x + self.w / 2.0


def classify_top_reads(
    reads: Iterable[PillarRead],
    column_px: float = 30.0,
    far_panel_h_px: float = 10.0,
) -> list[PillarRead]:
    """Return reads with conservative top-panel flags.

    A readable panel is top when it is the highest member of a stack of at
    least two same-number reads in one image column.  Tiny readable panels are
    also top: at long range the lit top panel is the only reliably readable
    marking.  All other observations remain azimuth-only.
    """
    result = list(reads)
    for i, read in enumerate(result):
        is_top = bool(read.number is not None and read.h <= far_panel_h_px)
        if read.number is not None:
            stack = [other for other in result
                     if other.number == read.number
                     and abs(other.center_u - read.center_u) <= column_px]
            if len(stack) >= 2:
                is_top = is_top or read.y == min(other.y for other in stack)
        result[i] = replace(read, top=is_top)
    return result


_STATION_RE = re.compile(r"^station\s*([0-9]{1,2})$", re.IGNORECASE)

# Measured station-text decal geometry from the VQ2 renderer's text-box GT.
# This is the wide ``Station NN`` decal, not the smaller numeric placard.
STATION_TEXT_WIDTH_M = 2.503
STATION_TEXT_HEIGHT_M = 0.475


@dataclass(frozen=True)
class TextPnPFit:
    """Camera-relative pose of the centre of a fitted station-text decal."""

    rvec: np.ndarray             # text-plane -> camera Rodrigues vector
    t_cam_text: np.ndarray       # text-panel centre in camera coordinates, m
    reproj_rms_px: float


def fit_station_text_pnp(
    quad_xy,
    camera_matrix,
    dist_coeffs=None,
    *,
    width_m: float = STATION_TEXT_WIDTH_M,
    height_m: float = STATION_TEXT_HEIGHT_M,
) -> TextPnPFit | None:
    """Fit a detected station-text quad to its known metric rectangle.

    ``quad_xy`` must be in text-reading order (top-left, top-right,
    bottom-right, bottom-left) after the clockwise image rotation used by the
    GPU OCR reader has been mapped back to the original image.  Planar IPPE
    returns the physically valid, lowest-reprojection solution.  The result is
    deliberately camera-relative: converting it to a mapped pillar landmark
    requires a surveyed text-panel offset, which the current map contract does
    not yet carry.
    """
    import cv2

    image = np.asarray(quad_xy, dtype=np.float64)
    K = np.asarray(camera_matrix, dtype=np.float64)
    if image.shape != (4, 2) or K.shape != (3, 3):
        raise ValueError("quad_xy must be (4,2) and camera_matrix must be (3,3)")
    if width_m <= 0.0 or height_m <= 0.0:
        raise ValueError("text dimensions must be positive")
    object_points = np.array([
        [-width_m / 2.0, -height_m / 2.0, 0.0],
        [ width_m / 2.0, -height_m / 2.0, 0.0],
        [ width_m / 2.0,  height_m / 2.0, 0.0],
        [-width_m / 2.0,  height_m / 2.0, 0.0],
    ], dtype=np.float64)
    ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        object_points, image, K, dist_coeffs, flags=cv2.SOLVEPNP_IPPE,
    )
    if not ok:
        return None
    candidates = []
    for rvec, tvec in zip(rvecs, tvecs):
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
        if tvec[2] <= 0.0:
            continue
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist_coeffs)
        rms = float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - image) ** 2, axis=1))))
        candidates.append((rms, np.asarray(rvec, dtype=np.float64).reshape(3), tvec))
    if not candidates:
        return None
    rms, rvec, tvec = min(candidates, key=lambda row: row[0])
    return TextPnPFit(rvec=rvec, t_cam_text=tvec, reproj_rms_px=rms)


class GpuPillarReader:
    """GPU text detector/OCR adapter for the simulator's vertical station text.

    The camera sees ``Station NN`` lettering vertically.  The image is rotated
    clockwise before EasyOCR runs, then each text box is transformed back to
    original OpenCV coordinates.  A recognition such as ``Station24`` is a
    *candidate* identity; callers must perform temporal voting and map
    association before allowing it to influence the smoother.

    EasyOCR is intentionally imported only when this optional GPU producer is
    instantiated, so offline geometry and unit tests do not require it.
    """

    def __init__(self, reader=None, *, gpu: bool = True,
                 text_threshold: float = 0.35, low_text: float = 0.20):
        if reader is None:
            try:
                import easyocr
            except ImportError as error:
                raise RuntimeError(
                    "GpuPillarReader requires easyocr; install it in the "
                    "flight environment"
                ) from error
            reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
        self.reader = reader
        self.text_threshold = float(text_threshold)
        self.low_text = float(low_text)

    @staticmethod
    def _cw_quad_to_original(box, original_height: int) -> np.ndarray:
        """Map a clockwise-rotated EasyOCR quad into original x/y points."""
        quad = np.asarray(box, dtype=float)
        if quad.shape != (4, 2):
            raise ValueError("OCR box must be four 2D points")
        # OpenCV clockwise rotation: x_rot = H - 1 - y, y_rot = x.
        return np.column_stack((quad[:, 1], original_height - 1.0 - quad[:, 0]))

    def read(self, image_bgr, t_capture: float) -> list[PillarRead]:
        """Detect station text and return PnL-ready reads for one BGR frame."""
        import cv2

        if image_bgr is None or len(image_bgr.shape) != 3:
            raise ValueError("image_bgr must be an HxWxC image")
        height = int(image_bgr.shape[0])
        rotated = cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
        rows = self.reader.readtext(
            rotated, detail=1, paragraph=False,
            text_threshold=self.text_threshold, low_text=self.low_text,
        )
        reads = []
        for box, text, confidence in rows:
            match = _STATION_RE.fullmatch(str(text).strip())
            # Keep a recognized station-text region even if its digits are
            # uncertain: it is a valid bearing-only PnL observation.
            if not str(text).strip().lower().startswith("station"):
                continue
            quad = self._cw_quad_to_original(box, height)
            x, y = quad.min(axis=0)
            w, h = quad.max(axis=0) - (x, y)
            reads.append(PillarRead(
                t_capture=float(t_capture), x=x, y=y, w=w, h=h,
                number=None if match is None else match.group(1),
                confidence=float(confidence), quad_xy=quad,
            ))
        return classify_top_reads(reads)
