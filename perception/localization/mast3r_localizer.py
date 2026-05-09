"""MASt3R-SLAM-based pose estimation against monocular RGB.

The other side of the localizer interface from ``IcpLocalizer``: instead
of fusing depth into a known map, this wraps the MASt3R-SLAM frame
tracker and emits a refined world-frame body pose every step.

Why this exists alongside ICP: ICP needs a pre-built map and a depth
sensor (or depth model). MASt3R-SLAM needs only RGB and the foundation
model's learned 3D priors — closer to the real on-board sensor stack
for the AI Grand Prix.

## Coordinate frames

MASt3R-SLAM lives in its own *SLAM frame* whose origin is wherever
frame 0 happened to be (``T_S_C0 = identity`` by convention). To turn
those poses into our env frame we apply a one-shot rigid alignment
``T_env_S`` computed from the *known* initial drone pose at reset time::

    T_env_S = T_env_body0 @ T_body_cam     # since T_S_C0 = I

then per frame::

    T_S_Cf = (whatever SLAM returns)
    T_env_Cf = T_env_S @ T_S_Cf
    T_env_bf = T_env_Cf @ T_cam_body

The math doesn't care which body-rate sensor or how the env frame was
defined — only that the caller hands us ``T_env_body0`` at reset.

## Why Sim(3), and why we ignore scale here

MASt3R-SLAM's optimizer is parameterised on ``lietorch.Sim3`` (7-DoF
similarity), so the output transform has a free scale. With
``use_calib=True`` and a known intrinsic matrix K, the projection
constraint pins scale near 1 *iff the underlying point cloud is
metric-correct*, and the resulting 4×4 ``T_S_C.matrix()`` is
effectively SE(3). We extract it directly. If the scale were free
(no calib) we'd need to track it as a one-shot calibration constant.

## Caveat: foundation-model depth must be in-distribution

The "scale pins near 1" property only holds when MASt3R's predicted
metric depth is correct. **Any synthetic rendering pipeline tested so
far is out-of-distribution for the metric depth head**, including:

  - PyBullet's procedurally-textured room (range 2.4–10 m → predicted
    1.9–3.6 m, ~25–50 % translation under-scale on short trajectories)
  - 3D Gaussian Splatting renders of MipNeRF 360 'room' (range ~9 m →
    predicted ~3 m, ~57 % translation under-scale on a 30-frame
    interpolation of training cameras)

Both MASt3R and Depth-Anything-V2 Metric Indoor produce the same
biased prediction on 3DGS renders (median 2.764 vs 2.767 m), confirming
the issue is the rendered-image domain shift, not the model. **Real-
camera input works**: on ETH3D's `sofa_1` (laser-GT-pose monocular
benchmark), this same wrapper achieves 11.6 cm mean / 32.4 cm max
pose error over 100 frames with no global BA — comparable to
published mono-SLAM-no-loop-closure literature.

Diagnostic: ``mast3r_inference_mono``'s output Z range should span
metres-to-tens-of-metres on a typical indoor scene. If it's compressed
to < ~5 m of variation, you're OOD and should either:
  - switch depth source (anchored GT depth, real camera, or RGB-D
    sensor — see ``debug_mast3r_depth_anchor.py``)
  - use real-camera input directly (ETH3D / TUM RGB-D / your own
    walkthrough video — see ``_phase2_eth3d_slam.py``)
  - accept that the SLAM stack can't be validated on synthetic data
    and use VIO + ICP-against-TSDF instead

See ``scripts/perception/debug_mast3r_*.py`` and ``_phase2_*.py`` for
the regression harnesses; COR-106 Phase 1 + Phase 2 for the full
investigation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import threading
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray


class _SingleProcessManager:
    """Drop-in replacement for ``multiprocessing.Manager`` for a single
    process. Avoids the manager-spawn hang we hit on Windows + CUDA, where
    ``mp.Manager()`` and ``manager.RLock()`` can take minutes or never
    return.

    MASt3R-SLAM's SharedKeyframes/SharedStates use the manager only for
    locks and a couple of int counters. If we never fork, plain threading
    primitives + a tiny .value wrapper give the same surface API at zero
    cost.
    """

    class _Value:
        def __init__(self, _typecode: str, initial: Any = 0) -> None:
            self.value = initial

    def RLock(self) -> threading.RLock:  # noqa: N802 (mp.Manager API name)
        return threading.RLock()

    def Value(self, typecode: str, initial: Any = 0) -> "_SingleProcessManager._Value":  # noqa: N802
        return self._Value(typecode, initial)

    def list(self) -> list:
        return []

# MASt3R-SLAM imports. These pull in lietorch + the foundation model
# weights — heavy. Keep at module top so failures surface at import.
import lietorch  # noqa: F401  (used implicitly for Sim3.Identity)
from mast3r_slam.config import config as mast3r_config, load_config
from mast3r_slam.frame import (  # noqa: E402
    Mode, SharedKeyframes, SharedStates, create_frame,
)
from mast3r_slam.mast3r_utils import load_mast3r, mast3r_inference_mono
from mast3r_slam.tracker import FrameTracker


def _find_repo_root() -> Path:
    """Walk up from this file to find the algo_src project root.

    A directory is the root iff it contains ``external_packages/MASt3R-SLAM``.
    Works from worktrees, subprojects, or installed locations as long as
    that submodule is in the expected place.
    """
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        if (ancestor / "external_packages" / "MASt3R-SLAM").exists():
            return ancestor
    raise RuntimeError(
        "Could not locate algo_src root from "
        f"{here} — no parent contains external_packages/MASt3R-SLAM. "
        "Set MAST3R_SLAM_ROOT env var to override."
    )


# Allow override via env var; fall back to project-root resolution. This
# lets the cloud bootstrap script point at e.g. /workspace/algo_src
# without code changes, and keeps the worktree-relative path on dev.
import os as _os
_MAST3R_REPO = (
    Path(_os.environ["MAST3R_SLAM_ROOT"]).resolve()
    if "MAST3R_SLAM_ROOT" in _os.environ
    else _find_repo_root() / "external_packages" / "MASt3R-SLAM"
)
_DEFAULT_CONFIG = _MAST3R_REPO / "config" / "calib.yaml"
_DEFAULT_CKPT = (_MAST3R_REPO / "checkpoints"
                 / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")


@dataclass
class LocalizerResult:
    """Common return shape for ICP and MASt3R-SLAM localizers."""
    success: bool
    refined_T_world_body: NDArray[np.float64]   # 4×4
    mode: str                                    # "init", "tracking", "reloc", ...
    fitness: float                               # match fraction (MASt3R) or
                                                  # ICP fitness — meaning differs
                                                  # but ranks the same way
    extra: dict                                  # diagnostic — varies by impl


def _sim3_to_matrix(T_sim3) -> NDArray[np.float64]:
    """Extract a 4×4 numpy matrix from a lietorch.Sim3.

    .matrix() returns (B, 4, 4) with the scale baked into the upper-left
    3×3. With use_calib=True the scale is constrained to ~1 by the
    projection model, so we treat the matrix as SE(3) directly. If you
    ever go calibration-free, normalise the rotation block:
        R = T[:3, :3]; s = (det R)**(1/3); R /= s; t = T[:3, 3] / s
    """
    T = T_sim3.matrix().detach().cpu().numpy()
    if T.ndim == 3:
        T = T[0]
    return T.astype(np.float64)


def _matrix_to_sim3(T: NDArray[np.float64], device: str = "cuda") -> "lietorch.Sim3":
    """Wrap a 4×4 SE(3) matrix as a lietorch.Sim3 (scale=1).

    lietorch.Sim3 stores 7 parameters (qw, qx, qy, qz, tx, ty, tz) plus
    log-scale. We construct via Sim3 from a homogeneous transform.
    """
    import lietorch as _l
    T_t = torch.as_tensor(T, dtype=torch.float32, device=device)[None]
    # lietorch's "from matrix" path: Sim3.exp(log(SE3.fromMatrix)) is
    # the public-facing way; but the simplest is to decompose explicitly.
    R = T[:3, :3]
    t = T[:3, 3]
    qw = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if qw < 1e-8:
        # Fallback: identity quaternion if pose is near-degenerate.
        qx, qy, qz = 0.0, 0.0, 0.0
        qw = 1.0
    else:
        qx = (R[2, 1] - R[1, 2]) / (4.0 * qw)
        qy = (R[0, 2] - R[2, 0]) / (4.0 * qw)
        qz = (R[1, 0] - R[0, 1]) / (4.0 * qw)
    log_s = 0.0
    data = torch.tensor(
        [[t[0], t[1], t[2], qx, qy, qz, qw, log_s]],
        dtype=torch.float32, device=device,
    )
    return _l.Sim3.InitFromVec(data)


def _cam_to_body(R_body_to_cam: NDArray[np.float64]) -> NDArray[np.float64]:
    """4×4 transform mapping camera-frame points → body-frame.

    Same convention as ICP localizer: ``R_body_to_cam @ [0,0,1]_cam =
    [1,0,0]_body`` (cam look → drone forward). No translation since the
    camera origin is the body origin in our setup.
    """
    T = np.eye(4)
    T[:3, :3] = R_body_to_cam
    return T


class Mast3rLocalizer:
    """Live monocular pose estimation via MASt3R-SLAM.

    Usage::

        loc = Mast3rLocalizer(K=k_3x3, R_body_to_cam=R)
        loc.reset(T_env_body0)            # one-shot env←slam alignment
        for t in range(N):
            rgb = render(...)             # (H, W, 3) uint8 or float
            res = loc.localize(rgb, t)
            T_env_body = res.refined_T_world_body
    """

    def __init__(
        self,
        K: NDArray[np.float64],              # 3×3 intrinsic matrix in pixels
        R_body_to_cam: NDArray[np.float64],
        *,
        img_size_wh: tuple[int, int] = (512, 384),
        config_path: Path | str = _DEFAULT_CONFIG,
        checkpoint_path: Path | str = _DEFAULT_CKPT,
        keyframe_buffer: int = 64,
        device: str = "cuda",
        model_dtype: torch.dtype = torch.float32,
        cuda_low_vram: bool = False,
    ) -> None:
        # MASt3R-SLAM resolves config "inherit" paths relative to its own
        # working directory, so the load_config call needs the path to be
        # interpretable from there. We re-set CWD to its repo root.
        import os
        prev_cwd = os.getcwd()
        os.chdir(_MAST3R_REPO)
        try:
            load_config(str(config_path))
        finally:
            os.chdir(prev_cwd)

        # We always run with calibration so the Sim3 scale stays near 1
        # and the depth/translation come out metric.
        mast3r_config["use_calib"] = True

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_grad_enabled(False)

        # In single-process mode use a threading-based shim — mp.Manager
        # spawns a child process via spawn() on Windows + CUDA, and that
        # path has been observed to hang here.
        self._manager = _SingleProcessManager()

        self.device = device
        self.img_w, self.img_h = img_size_wh
        self.T_body_cam = _cam_to_body(R_body_to_cam)
        # T_cam_body: inverse of T_body_cam (rotation only, no translation)
        self.T_cam_body = self.T_body_cam.T.copy()

        # CUDA-allocator warm-up. On Windows (no expandable_segments
        # support), the very first model.to(device) call on a 4 GB GPU
        # OOMs at ~60 % load due to fragmentation. Pre-allocating a
        # large block forces the allocator to claim a contiguous segment
        # from the driver, then freeing it leaves a single big segment
        # in the allocator's free pool that the subsequent model load
        # can subdivide cleanly. Empirically robust on RTX 3050 4 GB.
        # Load the model. On Linux with expandable_segments and a 24 GB+
        # GPU the standard .to(device) is fine. On 4 GB Windows GPUs we
        # had to use an in-place parameter walk to avoid fragmentation
        # OOM — that path is preserved as _model_to_device_inplace and
        # opt-in via cuda_low_vram.
        print(f"[Mast3rLocalizer] loading model "
              f"(dtype={model_dtype}): {checkpoint_path}")
        from mast3r.model import AsymmetricMASt3R
        self._model = AsymmetricMASt3R.from_pretrained(str(checkpoint_path))
        if model_dtype != torch.float32:
            self._model = self._model.to(model_dtype)
        if cuda_low_vram and device.startswith("cuda"):
            self._model_to_device_inplace(self._model, device)
        else:
            self._model = self._model.to(device)
        self._model.eval()
        self._model.share_memory()
        self._model_dtype = model_dtype

        # Now SharedKeyframes / SharedStates. Default buffer (512) costs
        # ~1.5 GB of VRAM just for feature tensors at 512×384; tighten
        # to keep us under 4 GB total.
        self._keyframes = SharedKeyframes(
            self._manager, self.img_h, self.img_w,
            buffer=keyframe_buffer, device=device,
        )
        self._states = SharedStates(
            self._manager, self.img_h, self.img_w, device=device,
        )

        # Intrinsics: MASt3R-SLAM expects a 3×3 K tensor on device. Pass
        # the same K we used to render, after MASt3R's internal centre-
        # principal-point shift (config["dataset"]["center_principle_point"]
        # is True by default — it doesn't actually shift K, just confirms
        # the assumption that cx=W/2, cy=H/2).
        K_t = torch.as_tensor(K, dtype=torch.float32, device=device)
        self._keyframes.set_intrinsics(K_t)
        self._K_np = np.asarray(K, dtype=np.float64).copy()

        self._tracker = FrameTracker(self._model, self._keyframes, device)

        # Alignment from SLAM → env frame, set in reset().
        self._T_env_slam: Optional[NDArray[np.float64]] = None

        # Frame counter; SLAM frame indices need to be monotonically
        # increasing.
        self._frame_idx = 0

    @staticmethod
    def _model_to_device_inplace(model: torch.nn.Module, device: str) -> None:
        """Move parameters and buffers to ``device`` one at a time, in-
        place, with periodic empty_cache(). Mitigates Windows-CUDA
        allocator fragmentation that breaks ``model.to(device)`` on a
        4 GB GPU around 60 % loaded.
        """
        n_params = 0
        n_bytes = 0
        for name, param in list(model.named_parameters()):
            new_data = param.data.to(device, non_blocking=False)
            param.data = new_data
            n_params += 1
            n_bytes += new_data.numel() * new_data.element_size()
            # Light-touch periodic flush so old CPU tensors don't accumulate.
            if n_params % 50 == 0:
                torch.cuda.empty_cache()
        for name, buf in list(model.named_buffers()):
            buf.data = buf.data.to(device, non_blocking=False)
        torch.cuda.empty_cache()
        print(f"  moved {n_params} params ({n_bytes / 1e9:.2f} GB) to {device}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, T_env_body0: NDArray[np.float64]) -> None:
        """Set the env←slam alignment using the known initial pose.

        Call this once before the first ``localize()``. The alignment
        anchors the SLAM frame (which is identity at frame 0) to the env
        frame: ``T_env_S = T_env_C0 = T_env_body0 @ T_body_cam``.
        """
        T_env_C0 = T_env_body0 @ self.T_body_cam
        self._T_env_slam = T_env_C0.astype(np.float64)
        # Reset SLAM internal state. set mode → INIT so the first
        # localize() call runs mono inference and seeds keyframe 0.
        self._states.set_mode(Mode.INIT)
        self._tracker.reset_idx_f2k()
        self._frame_idx = 0

    @torch.inference_mode()
    def localize(self, rgb: NDArray[np.uint8], t: int) -> LocalizerResult:
        """Track one RGB frame, return env-frame body pose."""
        if self._T_env_slam is None:
            raise RuntimeError("call reset(T_env_body0) before localize()")

        # MASt3R's resize_img wants float [0,1] HWC.
        if rgb.dtype == np.uint8:
            rgb_f = rgb.astype(np.float32) / 255.0
        else:
            rgb_f = rgb.astype(np.float32)
            if rgb_f.max() > 1.0 + 1e-3:
                rgb_f = rgb_f / 255.0

        # Initial guess: previous SLAM pose, or identity at frame 0.
        if self._frame_idx == 0:
            T_S_C_init = lietorch.Sim3.Identity(1, device=self.device)
        else:
            T_S_C_init = self._states.get_frame().T_WC

        frame = create_frame(
            self._frame_idx, rgb_f, T_S_C_init,
            img_size=512, device=self.device,
        )
        # Match the model's dtype on the encoder input (model is loaded
        # in fp16 to fit a 4 GB GPU). Other frame tensors stay fp32 — the
        # GN optimizer / lietorch math needs the precision.
        if self._model_dtype != torch.float32:
            frame.img = frame.img.to(self._model_dtype)

        mode = self._states.get_mode()
        success = True
        diag: dict = {}

        if mode == Mode.INIT:
            # Seed keyframe 0 from monocular inference.
            X_init, C_init = mast3r_inference_mono(self._model, frame)
            frame.update_pointmap(X_init, C_init)
            self._keyframes.append(frame)
            self._states.set_mode(Mode.TRACKING)
            self._states.set_frame(frame)
            mode_name = "init"
            T_S_C = _sim3_to_matrix(frame.T_WC)
        elif mode == Mode.TRACKING:
            add_new_kf, match_info, try_reloc = self._tracker.track(frame)
            if try_reloc:
                self._states.set_mode(Mode.RELOC)
                mode_name = "tracking_lost"
                success = False
                # Use last known pose so the caller can fall back smoothly.
                T_S_C = _sim3_to_matrix(self._states.get_frame().T_WC)
                diag["match_info"] = match_info
            else:
                self._states.set_frame(frame)
                if add_new_kf:
                    self._keyframes.append(frame)
                    diag["new_keyframe"] = True
                mode_name = "tracking"
                T_S_C = _sim3_to_matrix(frame.T_WC)
        else:
            # RELOC mode — minimal handling for the smoke test: fall back
            # to last-known pose. Real reloc needs the retrieval database
            # + factor graph (see main.py:relocalization).
            mode_name = "reloc_pending"
            success = False
            T_S_C = _sim3_to_matrix(self._states.get_frame().T_WC)

        # Convert SLAM-frame camera pose → env-frame body pose.
        T_env_C = self._T_env_slam @ T_S_C
        T_env_body = T_env_C @ self.T_cam_body

        self._frame_idx += 1

        diag["T_S_C"] = T_S_C
        diag["frame_idx"] = self._frame_idx
        diag["n_keyframes"] = len(self._keyframes)
        return LocalizerResult(
            success=success,
            refined_T_world_body=T_env_body,
            mode=mode_name,
            fitness=1.0 if success else 0.0,
            extra=diag,
        )
