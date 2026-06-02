import os
import cv2
import numpy as np
from aigp.gate_detect import detect_gate, red_mask, load_params, GateDetection

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
P = load_params()  # pytest runs from worktree root; default path finds detect_params.json


def _frame(n):
    return cv2.imread(os.path.join(FIX, f"gate_frame_{n}.jpg"))


def test_mask_has_gate_pixels():
    m = red_mask(_frame(0), P)
    assert m.dtype == np.uint8 and m.shape == (360, 640)
    assert m.sum() > 0


def test_detect_nearest_gate_center():
    det = detect_gate(_frame(0), P)
    assert isinstance(det, GateDetection)
    assert abs(det.u - 315) < 80 and abs(det.v - 180) < 80
    assert det.area >= P["min_area_px"]


def test_detect_none_on_desaturated():
    gray = np.full((360, 640, 3), 60, np.uint8)
    assert detect_gate(gray, P) is None


def test_rejects_low_reflection():
    img = np.zeros((360, 640, 3), np.uint8)
    cv2.rectangle(img, (300, 330), (340, 358), (0, 0, 255), -1)  # red square low in frame
    assert detect_gate(img, P) is None
