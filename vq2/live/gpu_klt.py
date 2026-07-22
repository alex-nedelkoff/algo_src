"""Torch-native GPU pyramidal Lucas-Kanade sparse optical flow (COR-147).

Drop-in replacement for cv2.calcOpticalFlowPyrLK's role in the VO front-end,
run on the GPU. The point on a 2-core VM: torch RELEASES THE GIL during CUDA
ops, so GPU tracking and the CPU essential-matrix solve run concurrently
instead of contending — which the cv2 (CPU) path could not do.

Method: classic pyramidal LK. Per level, coarse->fine, per feature point we
solve the 2x2 normal equations G d = b over a WxW window, iterating with
bilinear re-warping (grid_sample). All points and windows are vectorized.

Interface mirrors what MonoVO needs:
    klt = GpuKLT(device='cuda')
    klt.set_prev(prev_gray)                 # (H,W) uint8
    nxt_pts, status = klt.track(prev_pts, cur_gray)   # prev_pts (N,1,2) float32
returns (N,1,2) float32 and (N,) uint8 status, matching cv2's shapes.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
_SOBEL_Y = _SOBEL_X.t().contiguous()
# 3x3 binomial blur for pyramid downsampling (approx cv2 pyrDown)
_BINOM = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32) / 16.0


class GpuKLT:
    def __init__(self, device: str = "cuda", win: int = 21, levels: int = 4,
                 iters: int = 5, min_eig: float = 1e-3):
        self.device = torch.device(device)
        self.win = win
        self.levels = levels
        self.iters = iters
        self.min_eig = min_eig
        self.hw = win // 2
        # window offset grid (win*win, 2) as (dx, dy)
        r = torch.arange(-self.hw, self.hw + 1, dtype=torch.float32)
        oy, ox = torch.meshgrid(r, r, indexing="ij")
        self._off = torch.stack([ox.reshape(-1), oy.reshape(-1)], dim=-1).to(self.device)  # (w*w,2)
        self._sx = _SOBEL_X.view(1, 1, 3, 3).to(self.device)
        self._sy = _SOBEL_Y.view(1, 1, 3, 3).to(self.device)
        self._blur = _BINOM.view(1, 1, 3, 3).to(self.device)
        self._prev_pyr = None

    # -------------------------------------------------------- pyramids --
    def _to_tensor(self, gray: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(gray).to(self.device, torch.float32)
        return t.view(1, 1, *gray.shape)

    def _pyramid(self, img: torch.Tensor):
        """List of (img, Ix, Iy) per level, coarse index high."""
        pyr = []
        cur = img
        for _ in range(self.levels):
            ix = F.conv2d(cur, self._sx, padding=1)
            iy = F.conv2d(cur, self._sy, padding=1)
            pyr.append((cur, ix, iy))
            blurred = F.conv2d(cur, self._blur, padding=1)
            cur = blurred[:, :, ::2, ::2]
        return pyr

    def set_prev(self, gray: np.ndarray) -> None:
        self._prev_pyr = self._pyramid(self._to_tensor(gray))

    # ---------------------------------------------------------- sample --
    @staticmethod
    def _sample(img: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """img (1,1,H,W); xy (K,2) pixel coords -> (K,) bilinear values."""
        H, W = img.shape[-2:]
        gx = 2.0 * xy[:, 0] / (W - 1) - 1.0
        gy = 2.0 * xy[:, 1] / (H - 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)
        out = F.grid_sample(img, grid, align_corners=True,
                            padding_mode="border", mode="bilinear")
        return out.view(-1)

    def _win_sample(self, img, pts):
        """pts (N,2) -> (N, w*w) window samples."""
        coords = pts[:, None, :] + self._off[None, :, :]      # (N, w*w, 2)
        flat = coords.reshape(-1, 2)
        return self._sample(img, flat).view(pts.shape[0], -1)

    # ----------------------------------------------------------- track --
    def track(self, prev_pts: np.ndarray, cur_gray: np.ndarray):
        """prev_pts (N,1,2) float32 in prev image; returns (N,1,2), status (N,)."""
        assert self._prev_pyr is not None, "call set_prev() first"
        cur_pyr = self._pyramid(self._to_tensor(cur_gray))
        N = prev_pts.shape[0]
        p0 = torch.from_numpy(prev_pts.reshape(N, 2)).to(self.device, torch.float32)
        flow = torch.zeros_like(p0)

        for lvl in range(self.levels - 1, -1, -1):
            scale = float(2 ** lvl)
            pl = p0 / scale
            I0 = self._win_sample(self._prev_pyr[lvl][0], pl)          # (N,w*w)
            gx = self._win_sample(self._prev_pyr[lvl][1], pl)
            gy = self._win_sample(self._prev_pyr[lvl][2], pl)
            Gxx = (gx * gx).sum(1); Gyy = (gy * gy).sum(1); Gxy = (gx * gy).sum(1)
            det = Gxx * Gyy - Gxy * Gxy
            # min eigenvalue of the 2x2 structure tensor (corner quality)
            tr = Gxx + Gyy
            eig = 0.5 * (tr - torch.sqrt(torch.clamp(tr * tr - 4 * det, min=0.0)))
            good = det > 1e-6
            cur_img = cur_pyr[lvl][0]
            f = flow / scale
            for _ in range(self.iters):
                I1 = self._win_sample(cur_img, pl + f)
                It = I1 - I0
                bx = (gx * It).sum(1); by = (gy * It).sum(1)
                # d = -G^{-1} b
                dx = -(Gyy * bx - Gxy * by) / (det + 1e-12)
                dy = -(-Gxy * bx + Gxx * by) / (det + 1e-12)
                f = f + torch.stack([dx, dy], dim=-1)
            flow = f * scale

        nxt = (p0 + flow)
        H, W = cur_gray.shape
        inb = ((nxt[:, 0] >= 0) & (nxt[:, 0] < W - 1) &
               (nxt[:, 1] >= 0) & (nxt[:, 1] < H - 1))
        status = (good & inb & (eig > self.min_eig * (self.win ** 2)))
        # promote current pyramid to prev for the next incremental call
        self._prev_pyr = cur_pyr
        return (nxt.detach().cpu().numpy().reshape(N, 1, 2).astype(np.float32),
                status.detach().cpu().numpy().astype(np.uint8))


class GpuKLTGraph:
    """CUDA-graph GPU KLT: the whole fixed-shape LK is captured once and
    replayed with a SINGLE launch, so the CPU pays ~one launch instead of the
    ~50 per-kernel WDDM submissions that keep naive torch CPU-bound on Windows.

    Fixed-N: tracks a padded MAX_N point buffer every frame (invalid points
    masked by status), because CUDA graphs require static shapes. Pure-tensor
    forward writes in-place into static output tensors.
    """

    def __init__(self, H: int, W: int, max_n: int = 600, device: str = "cuda",
                 win: int = 21, levels: int = 3, iters: int = 3,
                 min_eig: float = 1e-3):
        self.device = torch.device(device)
        self.H, self.W, self.max_n = H, W, max_n
        self.win, self.levels, self.iters, self.min_eig = win, levels, iters, min_eig
        self.hw = win // 2
        r = torch.arange(-self.hw, self.hw + 1, dtype=torch.float32)
        oy, ox = torch.meshgrid(r, r, indexing="ij")
        self._off = torch.stack([ox.reshape(-1), oy.reshape(-1)], dim=-1).to(self.device)
        self._sx = _SOBEL_X.view(1, 1, 3, 3).to(self.device)
        self._sy = _SOBEL_Y.view(1, 1, 3, 3).to(self.device)
        self._blur = _BINOM.view(1, 1, 3, 3).to(self.device)
        # static graph I/O
        self.s_prev = torch.zeros(1, 1, H, W, device=self.device)
        self.s_cur = torch.zeros(1, 1, H, W, device=self.device)
        self.s_pts = torch.zeros(max_n, 2, device=self.device)
        self.o_nxt = torch.zeros(max_n, 2, device=self.device)
        self.o_status = torch.zeros(max_n, device=self.device)
        self._graph = None
        self._capture()

    def _pyr(self, img):
        pyr = []
        cur = img
        for _ in range(self.levels):
            ix = F.conv2d(cur, self._sx, padding=1)
            iy = F.conv2d(cur, self._sy, padding=1)
            pyr.append((cur, ix, iy))
            cur = F.conv2d(cur, self._blur, padding=1)[:, :, ::2, ::2]
        return pyr

    def _sample(self, img, xy):
        H, W = img.shape[-2:]
        gx = 2.0 * xy[:, 0] / (W - 1) - 1.0
        gy = 2.0 * xy[:, 1] / (H - 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)
        return F.grid_sample(img, grid, align_corners=True,
                             padding_mode="border").view(-1)

    def _win(self, img, pts):
        coords = pts[:, None, :] + self._off[None, :, :]
        return self._sample(img, coords.reshape(-1, 2)).view(pts.shape[0], -1)

    def _forward(self):
        """Pure tensor; writes o_nxt / o_status in place. Graph-capturable."""
        prev_pyr = self._pyr(self.s_prev)
        cur_pyr = self._pyr(self.s_cur)
        p0 = self.s_pts
        flow = torch.zeros_like(p0)
        good = torch.zeros(self.max_n, device=self.device)
        eig = torch.zeros(self.max_n, device=self.device)
        for lvl in range(self.levels - 1, -1, -1):
            scale = float(2 ** lvl)
            pl = p0 / scale
            I0 = self._win(prev_pyr[lvl][0], pl)
            gx = self._win(prev_pyr[lvl][1], pl)
            gy = self._win(prev_pyr[lvl][2], pl)
            Gxx = (gx * gx).sum(1); Gyy = (gy * gy).sum(1); Gxy = (gx * gy).sum(1)
            det = Gxx * Gyy - Gxy * Gxy
            tr = Gxx + Gyy
            eig = 0.5 * (tr - torch.sqrt(torch.clamp(tr * tr - 4 * det, min=0.0)))
            good = (det > 1e-6).float()
            cur_img = cur_pyr[lvl][0]
            f = flow / scale
            for _ in range(self.iters):
                It = self._win(cur_img, pl + f) - I0
                bx = (gx * It).sum(1); by = (gy * It).sum(1)
                dx = -(Gyy * bx - Gxy * by) / (det + 1e-12)
                dy = -(-Gxy * bx + Gxx * by) / (det + 1e-12)
                f = f + torch.stack([dx, dy], dim=-1)
            flow = f * scale
        nxt = p0 + flow
        inb = ((nxt[:, 0] >= 0) & (nxt[:, 0] < self.W - 1) &
               (nxt[:, 1] >= 0) & (nxt[:, 1] < self.H - 1)).float()
        status = good * inb * (eig > self.min_eig * (self.win ** 2)).float()
        self.o_nxt.copy_(nxt)
        self.o_status.copy_(status)

    def _capture(self):
        # warm on a side stream, then capture
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._forward()
        torch.cuda.current_stream().wait_stream(s)
        self._graph = torch.cuda.CUDAGraph()
        # thread_local capture: the DEFAULT 'global' mode errors on ANY CUDA
        # op in ANY thread during capture, which crashes a co-resident GateNet
        # detector thread (and can poison the shared context). thread_local
        # only guards this thread, letting GateNet keep inferring during capture.
        with torch.cuda.graph(self._graph, capture_error_mode="thread_local"):
            self._forward()

    def set_prev(self, gray: np.ndarray) -> None:
        self.s_prev[0, 0].copy_(torch.from_numpy(gray).to(self.device, torch.float32))

    def track(self, prev_pts: np.ndarray, cur_gray: np.ndarray):
        """prev_pts (N,1,2); N<=max_n. Returns (N,1,2), status (N,)."""
        N = prev_pts.shape[0]
        self.s_pts.zero_()
        self.s_pts[:N].copy_(torch.from_numpy(prev_pts.reshape(N, 2)).to(self.device))
        self.s_cur[0, 0].copy_(torch.from_numpy(cur_gray).to(self.device, torch.float32))
        self._graph.replay()
        nxt = self.o_nxt[:N].cpu().numpy().reshape(N, 1, 2).astype(np.float32)
        st = self.o_status[:N].cpu().numpy().astype(np.uint8)
        self.s_prev.copy_(self.s_cur)          # promote cur -> prev
        return nxt, st
