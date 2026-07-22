"""In-VM monocular VO (COR-147): KLT + essential-matrix, keyframe-by-parallax.

DPVO's lietorch CUDA kernel access-violates on Vagon (see HANDOFF_vo_loop.md);
this path uses only OpenCV — no compiled extension — and runs clean in-VM.

Core discipline (measured 2026-07-22): frame-to-frame baseline is often
sub-pixel (drone slow / 38% duplicate re-sends), which makes the essential
matrix degenerate. So we KEYFRAME BY PARALLAX: accumulate frames until median
optical flow from the current keyframe crosses KF_FLOW_PX (~8-12px, cf DPVO
KEYFRAME_THRESH=15px), then solve the relative pose and emit ONE OdomDelta.

Monocular => translation DIRECTION only (unit); OdomDelta.scale_locked=False.
The association/smoother layer applies scale (climb-cal / GNSCALE) and rotates
the body-frame delta into the state's yaw-relative frame using attitude at t0.

Frames follow vq2.camera: cam x-right/y-down/z-fwd (OpenCV), body x-fwd/y-right/
z-down, M_BODY_CAM maps cam->body.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import cv2

from ..camera import FX, FY, CX, CY, M_BODY_CAM
from ..relpose import OdomDelta

K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])

_FEAT = dict(maxCorners=600, qualityLevel=0.01, minDistance=7, blockSize=7)
_LK = dict(winSize=(21, 21), maxLevel=3,
           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


@dataclass
class VoStep:
    """One keyframe-to-keyframe relative-pose solve. Translation is UNIT
    (monocular). Rotation/translation expressed in the BODY frame at t0."""
    t0: float
    t1: float
    R_body: np.ndarray          # (3,3) relative rotation, body frame (cam0->cam1)
    t_body_unit: np.ndarray     # (3,) unit translation direction, body frame @ t0
    dyaw: float                 # yaw component of R_body (rad, wrapped)
    n_inliers: int              # recoverPose cheirality inliers
    n_tracked: int
    median_flow_px: float


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class MonoVO:
    """Streaming monocular VO. Feed grayscale frames in time order via step();
    get a VoStep back on each keyframe boundary (else None)."""

    def __init__(self, kf_flow_px: float = 9.0, min_inliers: int = 30,
                 min_tracked: int = 40, ransac_thresh_px: float = 1.0):
        self.kf_flow_px = kf_flow_px
        self.min_inliers = min_inliers
        self.min_tracked = min_tracked
        self.ransac_thresh_px = ransac_thresh_px
        self._kf_img = None         # keyframe image
        self._kf_pts = None         # (N,1,2) features detected in the keyframe
        self._kf_t = None
        self._cur_pts = None        # keyframe features tracked into the latest frame
        self._prev_img = None       # latest frame (for incremental KLT)

    def _set_keyframe(self, t, img):
        self._kf_img = img
        self._kf_t = t
        self._kf_pts = cv2.goodFeaturesToTrack(img, mask=None, **_FEAT)
        self._cur_pts = self._kf_pts
        self._prev_img = img

    def step(self, t: float, gray: np.ndarray) -> VoStep | None:
        """gray: (H,W) uint8. Returns a VoStep when a keyframe closes, else None."""
        if self._kf_img is None or self._kf_pts is None or len(self._kf_pts) < self.min_tracked:
            self._set_keyframe(t, gray)
            return None

        # incremental KLT from the previous frame keeps tracks stable across
        # the whole keyframe interval; keyframe correspondence is (kf_pts, cur)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self._prev_img, gray, self._cur_pts, None, **_LK)
        if p1 is None:
            self._set_keyframe(t, gray)
            return None
        st = st.reshape(-1).astype(bool)
        kf_pts = self._kf_pts[st]
        cur_pts = p1[st]
        self._kf_pts = kf_pts
        self._cur_pts = cur_pts
        self._prev_img = gray

        n_tracked = len(cur_pts)
        if n_tracked < self.min_tracked:
            step = self._solve(t, kf_pts, cur_pts, n_tracked)   # last-chance solve
            self._set_keyframe(t, gray)
            return step

        q0 = kf_pts.reshape(-1, 2)
        q1 = cur_pts.reshape(-1, 2)
        median_flow = float(np.median(np.linalg.norm(q1 - q0, axis=1)))
        if median_flow < self.kf_flow_px:
            return None                                          # accumulate baseline

        step = self._solve(t, kf_pts, cur_pts, n_tracked, median_flow)
        self._set_keyframe(t, gray)                              # close the keyframe
        return step

    def _solve(self, t, kf_pts, cur_pts, n_tracked, median_flow=None):
        q0 = kf_pts.reshape(-1, 2).astype(np.float32)
        q1 = cur_pts.reshape(-1, 2).astype(np.float32)
        if median_flow is None:
            median_flow = float(np.median(np.linalg.norm(q1 - q0, axis=1)))
        if len(q0) < 8:
            return None
        E, mask = cv2.findEssentialMat(q1, q0, K, method=cv2.RANSAC,
                                       prob=0.999, threshold=self.ransac_thresh_px)
        if E is None or E.shape != (3, 3):
            return None
        n_inl, R, t_cam, _ = cv2.recoverPose(E, q1, q0, K, mask=mask.copy())
        if n_inl < self.min_inliers or not np.all(np.isfinite(R)):
            return None

        # cam (kf->cur) -> body frame. recoverPose returns R,t with
        # X_cur = R @ X_kf + t; t is the cur-cam origin in the kf-cam frame
        # (unit). Motion of the body in body-frame @ t0:
        R_body = M_BODY_CAM @ R @ M_BODY_CAM.T
        t_body = M_BODY_CAM @ t_cam.reshape(3)
        n = np.linalg.norm(t_body)
        if n < 1e-9:
            return None
        t_body_unit = t_body / n
        dyaw = _wrap(math.atan2(R_body[1, 0], R_body[0, 0]))
        return VoStep(t0=self._kf_t, t1=t, R_body=R_body, t_body_unit=t_body_unit,
                      dyaw=dyaw, n_inliers=int(n_inl), n_tracked=int(n_tracked),
                      median_flow_px=float(median_flow))


def step_to_odom_delta(step: VoStep, yaw0: float = 0.0, sigma_p: float = 0.5,
                       sigma_yaw: float = 0.05) -> OdomDelta:
    """Lower a VoStep to an OdomDelta. dp_local is the UNIT body-frame
    direction rotated into the yaw-relative frame of state t0 (yaw0 = heading
    at t0; default 0 => body==yaw-relative when heading is datum-aligned).
    scale_locked=False: magnitude is 1, the smoother/association applies scale."""
    c, s = math.cos(-yaw0), math.sin(-yaw0)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    dp = Rz @ step.t_body_unit
    return OdomDelta(t0=step.t0, t1=step.t1, dp_local=dp, dyaw=step.dyaw,
                     source="vo_generic", sigma_p=sigma_p, sigma_yaw=sigma_yaw,
                     scale_locked=False)
