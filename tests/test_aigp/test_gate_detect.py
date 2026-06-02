import os
import cv2
import numpy as np
from aigp.gate_detect import detect_gate, red_mask, load_params, GateDetection

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "detect_params.json")
P = load_params(PARAMS_PATH)


def _frame(n):
    return cv2.imread(os.path.join(FIX, f"gate_frame_{n}.jpg"))


def test_mask_has_gate_pixels():
    m = red_mask(_frame(0), P)
    assert m.dtype == np.uint8 and m.shape == (360, 640)
    assert np.count_nonzero(m) > 50


def test_detect_nearest_gate_center():
    det = detect_gate(_frame(0), P)
    assert isinstance(det, GateDetection)
    assert abs(det.u - 326) < 30 and abs(det.v - 179) < 30  # measured from gate_frame_0.jpg
    assert det.area >= P["min_area_px"]


def test_detect_none_on_desaturated():
    gray = np.full((360, 640, 3), 60, np.uint8)
    assert detect_gate(gray, P) is None


def test_rejects_low_in_frame():
    img = np.zeros((360, 640, 3), np.uint8)
    cv2.rectangle(img, (300, 330), (340, 358), (0, 0, 255), -1)  # red square low in frame
    assert detect_gate(img, P) is None


def test_rejects_nonsquare():
    img = np.zeros((360, 640, 3), np.uint8)
    cv2.rectangle(img, (310, 80), (330, 160), (0, 0, 255), -1)  # 20x80 red, upper-center
    assert detect_gate(img, P) is None


def test_draw_overlay_shape():
    from aigp.gate_detect import draw_overlay
    f = _frame(0)
    assert draw_overlay(f, None).shape == f.shape           # None path = copy
    det = detect_gate(f, P)
    assert draw_overlay(f, det).shape == f.shape            # real detection
