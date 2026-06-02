"""Classical gate detector: find the largest visible gate (area is a proxy for range/nearness).
Pure (no sim deps). Gates are the only saturated objects in the desaturated env."""
from __future__ import annotations
import json
import os
from dataclasses import dataclass
import cv2
import numpy as np

DEFAULT_PARAMS = {"h_lo1": 0, "h_hi1": 12, "h_lo2": 165, "h_hi2": 180,
                  "s_min": 110, "v_min": 70, "min_area_px": 80,
                  "square_tol": 0.5, "max_v_frac": 0.72}


def load_params(path: str = "detect_params.json") -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return {**DEFAULT_PARAMS, **json.load(f)}
    return dict(DEFAULT_PARAMS)


@dataclass
class GateDetection:
    u: float
    v: float
    w_px: float
    h_px: float
    area: float
    bbox: tuple[int, int, int, int]  # (x, y, w, h)


def red_mask(bgr: np.ndarray, p: dict) -> np.ndarray:
    """Return a binary mask of red pixels; bgr must be an H×W×3 BGR image."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo1 = np.array([p["h_lo1"], p["s_min"], p["v_min"]]); hi1 = np.array([p["h_hi1"], 255, 255])
    lo2 = np.array([p["h_lo2"], p["s_min"], p["v_min"]]); hi2 = np.array([p["h_hi2"], 255, 255])
    m = cv2.inRange(hsv, lo1, hi1) | cv2.inRange(hsv, lo2, hi2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)


def detect_gate(bgr, params: dict | None = None) -> "GateDetection | None":
    p = params or load_params()
    H, W = bgr.shape[:2]
    mask = red_mask(bgr, p)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = None
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < p["min_area_px"]:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue
        if abs(w / float(h) - 1.0) > p["square_tol"]:
            continue
        if (y + h / 2.0) > p["max_v_frac"] * H:
            continue
        if largest is None or area > largest[0]:  # largest area ≈ nearest gate (area is a range proxy)
            largest = (area, x, y, w, h)
    if largest is None:
        return None
    area, x, y, w, h = largest
    return GateDetection(x + w / 2.0, y + h / 2.0, float(w), float(h), area, (x, y, w, h))


def draw_overlay(bgr, det) -> np.ndarray:
    out = bgr.copy()
    if det is not None:
        x, y, w, h = det.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(out, (int(det.u), int(det.v)), 3, (0, 255, 0), -1)
    return out
