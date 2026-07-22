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


@dataclass(frozen=True)
class PillarEdgeFit:
    """Image-space support for a physical pillar around a text observation."""
    left_line: tuple[float, float, float, float]
    right_line: tuple[float, float, float, float]
    center_u: float
    width_px: float
    quality: float


def fit_pillar_edges(image_bgr, read: PillarRead, *,
                     horizontal_margin_px: float = 45.0,
                     vertical_margin_px: float = 90.0,
                     min_span_px: float = 32.0) -> PillarEdgeFit | None:
    """Fit a Manhattan-vertical pillar silhouette around an OCR text box.

    OCR supplies association; Canny/Hough only accepts a left/right pair that
    brackets it.  The caller should de-roll the image when attitude is large.
    """
    import cv2
    if image_bgr is None or image_bgr.ndim < 2 or read.w <= 0 or read.h <= 0:
        return None
    h_img, w_img = image_bgr.shape[:2]
    x0 = max(0, int(np.floor(read.x - horizontal_margin_px)))
    x1 = min(w_img, int(np.ceil(read.x + read.w + horizontal_margin_px)))
    y0 = max(0, int(np.floor(read.y - vertical_margin_px)))
    y1 = min(h_img, int(np.ceil(read.y + read.h + vertical_margin_px)))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    gray = cv2.cvtColor(image_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    lines = cv2.HoughLinesP(cv2.Canny(gray, 45, 135), 1, np.pi / 180.0,
                            threshold=24, minLineLength=max(20, int(min_span_px)),
                            maxLineGap=10)
    if lines is None:
        return None
    candidates = []
    ym = read.y + read.h / 2.0
    for ax1, ay1, ax2, ay2 in lines.reshape(-1, 4):
        gx1, gy1, gx2, gy2 = map(float, (ax1 + x0, ay1 + y0, ax2 + x0, ay2 + y0))
        dx, dy = gx2 - gx1, gy2 - gy1
        span = float(np.hypot(dx, dy))
        if span < min_span_px or abs(dy) < 2.0 * abs(dx):
            continue
        xx = gx1 + (ym - gy1) * dx / dy
        candidates.append((xx, span, (gx1, gy1, gx2, gy2)))
    left = [row for row in candidates if row[0] <= read.x + 2.0]
    right = [row for row in candidates if row[0] >= read.x + read.w - 2.0]
    if not left or not right:
        return None
    l = max(left, key=lambda q: q[1] - .15 * abs(q[0] - read.x))
    r = max(right, key=lambda q: q[1] - .15 * abs(q[0] - (read.x + read.w)))
    width = r[0] - l[0]
    if width < max(read.w, 6.0) or width > read.w + 2.0 * horizontal_margin_px:
        return None
    quality = min(1.0, (l[1] + r[1]) / max(2.0 * min_span_px, 2.0 * read.h))
    return PillarEdgeFit(l[2], r[2], (l[0] + r[0]) / 2.0, width, quality)


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
