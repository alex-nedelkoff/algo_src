"""Rerun telemetry for gate fly-through (rerun 0.33). Best-effort: never raise into the
control loop. Run `rerun` on the Mac first; this connects to it over Tailscale."""
from __future__ import annotations
import numpy as np

MAC_VIEWER = "rerun+http://100.101.13.126:9876/proxy"
_ok = False


def init(rrd_path=None):
    """Connect to the Rerun viewer on the Mac. On failure, telemetry is silently disabled."""
    global _ok
    try:
        import rerun as rr
        rr.init("aigp-gate")
        if rrd_path:
            rr.save(rrd_path)          # write a shareable .rrd recording
        else:
            rr.connect_grpc(MAC_VIEWER)
        try:
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)  # NED: z down
        except Exception:
            pass
        _ok = True
    except Exception as e:
        print(f"[viz] disabled: {e}", flush=True)
        _ok = False


def log_step(t_s, drone_pos=None, drone_vel=None, frame_bgr=None, mask=None,
             overlay=None, det=None, cmd=None, trail=None):
    if not _ok:
        return
    try:
        import rerun as rr
        rr.set_time("t", duration=float(t_s))
        if drone_pos is not None:
            rr.log("world/drone", rr.Points3D([np.asarray(drone_pos, float)], radii=0.4))
        if trail is not None and len(trail) >= 2:
            rr.log("world/trail", rr.LineStrips3D([np.asarray(trail, float)]))
        if drone_pos is not None and drone_vel is not None:
            rr.log("world/vel", rr.Arrows3D(origins=[np.asarray(drone_pos, float)],
                                            vectors=[np.asarray(drone_vel, float)]))
        if frame_bgr is not None:
            rr.log("cam/frame", rr.Image(frame_bgr[:, :, ::-1]))
        if mask is not None:
            rr.log("cam/mask", rr.Image(mask))
        if overlay is not None:
            rr.log("cam/overlay", rr.Image(overlay[:, :, ::-1]))
        if det is not None:
            rr.log("det/u", rr.Scalars(float(det.u)))
            rr.log("det/v", rr.Scalars(float(det.v)))
            rr.log("det/size", rr.Scalars(float(det.w_px)))
        if cmd is not None:
            rr.log("ctrl/yaw_sp", rr.Scalars(float(cmd.yaw_sp)))
            rr.log("ctrl/fwd", rr.Scalars(float(cmd.fwd_speed)))
            rr.log("ctrl/have_gate", rr.Scalars(1.0 if cmd.have_gate else 0.0))
        if drone_vel is not None:
            rr.log("ctrl/speed", rr.Scalars(float(np.linalg.norm(np.asarray(drone_vel, float)[:2]))))
    except Exception:
        pass
