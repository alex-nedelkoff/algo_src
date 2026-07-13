"""Live DPVO odometry thread for vq2wp (task #24, v1).

Feeds the camera stream into DPVO and fuses camera positions into the
flight KF. Replaces GateNet in the pre-tick control loop: the estimator
finally SEES the real motion (the un-modelled push that capped blind
gate-1 at ~12%), so the carrot corrects truth, not just the estimate.

Frame anchoring: DPVO's world = its first fed camera frame. We feed from
'airborne', so the anchor is the KF pose captured at that moment (at-rest
attitude exact, position ~0.1 m). Monocular scale is calibrated during
the climb/recenter phase against KF displacement (noiseless-IMU DR,
still independent because no DPVO updates have been applied yet) and
FROZEN at the first KF update to avoid circularity.

Convention notes:
- DPVO pg.poses_ stores world-to-camera; .inv() gives camera-to-world
  (translation = camera position in the cam0 frame).
- M_BODY_CAM columns are the camera axes in body frame (cam->body).
"""
import math
import os
import threading
import time

import numpy as np

NETWORK = r'C:\Users\alexj\DPVO\dpvo.pth'
DPVO_CFG = r'C:\Users\alexj\DPVO\config\default.yaml'
INTRINSICS = (320.0, 320.0, 320.0, 180.0)


def _euler_R(roll, pitch, yaw):
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


class DpvoOdom(threading.Thread):
    def __init__(self, state, kf, kf_lock, m_body_cam, jlog=None):
        super().__init__(daemon=True)
        self.state = state
        self.kf = kf
        self.kf_lock = kf_lock
        self.m_body_cam = m_body_cam
        self.jlog = jlog or (lambda *a, **k: None)
        self.r_scale = float(os.environ.get('DPVO_R', '1.0'))
        self.stride = int(os.environ.get('DPVO_STRIDE', '2'))

    def run(self):
        try:
            self._run()
        except Exception as e:  # never take down the flight
            import traceback
            print(f'DPVO thread died: {e!r}', flush=True)
            traceback.print_exc()
            self.jlog('dpvo_died', err=repr(e))

    def _run(self):
        import sys
        # the pip wheel omits repo-only submodules (dpvo.loop_closure);
        # import from the repo checkout instead
        if r'C:\Users\alexj\DPVO' not in sys.path:
            sys.path.insert(0, r'C:\Users\alexj\DPVO')
        for _m in [m for m in list(sys.modules) if m.startswith('dpvo')]:
            del sys.modules[_m]
        import torch
        from lietorch import SE3
        from dpvo.dpvo import DPVO
        from dpvo.config import cfg as dcfg

        dcfg.merge_from_file(DPVO_CFG)
        dcfg.PATCHES_PER_FRAME = int(os.environ.get('DPVO_PATCHES', '48'))
        dcfg.MIXED_PRECISION = True
        dcfg.LOOP_CLOSURE = False
        dcfg.CLASSIC_LOOP_CLOSURE = False

        intr = torch.as_tensor(INTRINSICS, dtype=torch.float, device='cuda')
        st = self.state
        slam = None
        anchor_p = None
        R0 = None                 # cam0 -> spawn rotation
        scale = None
        scale_locked = False
        cal = []                  # (p_cam0, p_kf) pairs pre-update
        last_ns = 0
        skip = 0
        t_idx = 0
        last_upd = 0.0

        print('DPVO odom thread up, waiting for airborne', flush=True)
        while not st.get('stop'):
            if not st.get('airborne') or st.get('frame') is None:
                time.sleep(0.02)
                continue
            ns = st.get('frame_ns', 0)
            if ns == last_ns:
                time.sleep(0.004)
                continue
            last_ns = ns
            skip += 1
            if skip % self.stride:
                continue
            img = st['frame']
            if slam is None:
                time.sleep(1.0)   # let GateNet's VRAM release settle (4 GB GPU)
                slam = DPVO(dcfg, NETWORK, ht=img.shape[0], wd=img.shape[1],
                            viz=False)
                with self.kf_lock:
                    anchor_p = self.kf.p.copy()
                R_wb = _euler_R(st['roll'], st['pitch'], st['yaw'])
                R0 = R_wb @ self.m_body_cam       # cam0 -> spawn
                print(f'DPVO init: anchor {np.round(anchor_p, 2)}', flush=True)
                self.jlog('dpvo_init', anchor=anchor_p.round(3).tolist())
            t = torch.from_numpy(np.ascontiguousarray(img[..., ::-1]))
            t = t.permute(2, 0, 1).cuda()
            slam(t_idx, t, intr)
            t_idx += 1
            if slam.n < 9:        # DPVO still initializing
                continue
            pose = SE3(slam.pg.poses_[slam.n - 1]).inv()
            p_cam0 = pose.data.detach().cpu().numpy().reshape(-1)[:3]
            with self.kf_lock:
                p_kf = self.kf.p.copy()

            if not scale_locked:
                cal.append((p_cam0.copy(), p_kf.copy(), time.time()))
                d_dp = np.linalg.norm(cal[-1][0] - cal[0][0])
                d_kf = np.linalg.norm(cal[-1][1] - cal[0][1])
                if d_dp > 1e-3 and d_kf > 0.3:
                    scale = d_kf / d_dp
                if len(cal) % 10 == 0:
                    self.jlog('dpvo_cal', n=len(cal), d_dp=round(float(d_dp), 4),
                              d_kf=round(float(d_kf), 3),
                              s=None if scale is None else round(scale, 3))
                if scale is not None and len(cal) >= 12:
                    d_dp2 = np.linalg.norm(cal[-1][0] - cal[len(cal)//2][0])
                    d_kf2 = np.linalg.norm(cal[-1][1] - cal[len(cal)//2][1])
                    if d_dp2 > 1e-3 and d_kf2 > 0.4:
                        s2 = d_kf2 / d_dp2
                        if abs(s2 - scale) / scale < 0.4:
                            scale = 0.5 * (scale + s2)
                            scale_locked = True
                # bounded fallback: after 12 s of calibration with SOME
                # estimate, lock it -- an approximate scale that lets the
                # estimator see the disturbance beats flying blind
                if (not scale_locked and scale is not None
                        and time.time() - cal[0][2] > 12.0):
                    scale_locked = True
                if scale_locked:
                    print(f'DPVO scale locked: {scale:.2f} m/unit', flush=True)
                    self.jlog('dpvo_scale', s=round(scale, 3))
                else:
                    continue

            p_spawn = anchor_p + R0 @ (scale * p_cam0)
            st['dpvo_p'] = p_spawn
            st['dpvo_wall'] = time.time()

            now = time.time()
            if now - last_upd > 0.1:
                last_upd = now
                with self.kf_lock:
                    ok = self.kf.update_position(p_spawn, rng=5.0,
                                                 r_scale=self.r_scale)
                self.jlog('dpvo_upd', ok=bool(ok),
                          p=np.round(p_spawn, 3).tolist())
        if slam is not None:
            print('DPVO odom thread exiting', flush=True)
