"""Live flight-control debug telemetry over Rerun (0.33), built around the turn-tumble diagnosis.

Two layers:
  * PURE helpers (tilt_deg, euler_rpy_deg, sideslip_deg, along_cross) -- unit-tested, no Rerun dep.
  * FlightLog -- best-effort Rerun sink: decimated, async, NEVER raises into the control loop, and
    ALWAYS called AFTER the command is sent so telemetry can't delay a command. Uses a separate
    gRPC connection to the Mac viewer (not the MAVLink command link).

Workflow: run `rerun` on the Mac, then run an instrumented controller on Windows with --viz.
"""
from __future__ import annotations
import numpy as np
from aigp.geometry import quat_to_R

MAC_VIEWER = "rerun+http://100.101.13.126:9876/proxy"   # Mac Tailscale IP (matches aigp/viz.py)


# ---------------------------------------------------------------------------
# Pure, testable signal math (NED world / FRD body; camera-forward = -body_x).
# ---------------------------------------------------------------------------
def tilt_deg(quat_wxyz) -> float:
    """Angle of the body-z axis from world-down, i.e. how far the drone is tilted from level."""
    R = quat_to_R(np.asarray(quat_wxyz, float))
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))


def euler_rpy_deg(quat_wxyz):
    """(roll, pitch, yaw) in degrees, ZYX NED convention."""
    R = quat_to_R(np.asarray(quat_wxyz, float))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2]))
    roll = np.arctan2(R[2, 1], R[2, 2])
    return float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw))


def sideslip_deg(quat_wxyz, vel_ned) -> float:
    """Signed angle (deg) between the nose (camera-forward = -body_x, horizontal) and the horizontal
    velocity. 0 = flying nose-first (weathervane-stable); +/-90 = pure strafe; 180 = tail-first.
    Returns 0 when nearly stationary (no meaningful airflow direction)."""
    v = np.asarray(vel_ned, float)[:2]
    if float(np.linalg.norm(v)) < 1e-3:
        return 0.0
    R = quat_to_R(np.asarray(quat_wxyz, float))
    cam = -R[:2, 0]                       # nose direction in the horizontal plane
    nc = np.linalg.norm(cam)
    if nc < 1e-9:
        return 0.0
    cam = cam / nc
    vd = v / np.linalg.norm(v)
    dot = float(cam @ vd)
    cross = float(cam[0] * vd[1] - cam[1] * vd[0])
    return float(np.degrees(np.arctan2(cross, dot)))


def along_cross(vel_ned, tangent_h):
    """Decompose horizontal velocity into (along-track, cross-track) about a horizontal unit tangent."""
    v = np.asarray(vel_ned, float)[:2]
    t = np.asarray(tangent_h, float)[:2]
    lat = np.array([-t[1], t[0]])
    return float(v @ t), float(v @ lat)


# ---------------------------------------------------------------------------
# Rerun sink
# ---------------------------------------------------------------------------
class FlightLog:
    """Best-effort Rerun telemetry. Construct once; call step() each control iteration AFTER sending
    the command. Logs at ~1/decimate of the call rate. Disabled (no-op) if Rerun can't connect."""

    def __init__(self, rate_gain=None, decimate=5, rrd_path=None, run_name="aigp-flight"):
        self.ok = False
        self.rg = np.asarray(rate_gain, float) if rate_gain is not None else None
        self.decimate = max(1, int(decimate))
        self._i = 0
        self._trail = []
        try:
            import rerun as rr
            self._rr = rr
            rr.init(run_name)
            if rrd_path:
                rr.save(rrd_path)
            else:
                rr.connect_grpc(MAC_VIEWER)
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)  # NED: z down
            self._send_blueprint()
            self.ok = True
        except Exception as e:
            print(f"[flightlog] disabled: {e}", flush=True)

    def _send_blueprint(self):
        try:
            import rerun.blueprint as rrb
            ts = lambda origin, name: rrb.TimeSeriesView(origin=origin, name=name)
            charts = rrb.Grid(
                ts("rates/roll", "roll rate cmd/act (deg/s)"),
                ts("rates/pitch", "pitch rate cmd/act (deg/s)"),
                ts("rates/yaw", "yaw rate cmd/act (deg/s)"),
                ts("aero/sideslip", "sideslip (deg)  <- weathervane"),
                ts("att/tilt", "tilt cmd/act (deg)"),
                ts("att/yaw", "yaw sp/act (deg)"),
                ts("track", "speed / cruise / cross / along (m/s)"),
                ts("ctrl", "thrust cmd / state"),
                grid_columns=2,
            )
            bp = rrb.Blueprint(
                rrb.Horizontal(rrb.Spatial3DView(origin="world", name="3D actual vs commanded"),
                               charts, column_shares=[1.1, 1.0]),
                collapse_panels=True,
            )
            self._rr.send_blueprint(bp)
        except Exception as e:
            print(f"[flightlog] blueprint skipped: {e}", flush=True)

    def set_path(self, points, name="cmd_path"):
        """Log the commanded/target path (waypoints or spline samples) once, as a 3D line + markers."""
        if not self.ok:
            return
        try:
            p = np.asarray(points, float)
            self._rr.log(f"world/{name}", self._rr.LineStrips3D([p], colors=[120, 180, 255]), static=True)
            self._rr.log(f"world/{name}_pts", self._rr.Points3D(p, radii=0.25, colors=[120, 180, 255]),
                         static=True)
        except Exception:
            pass

    def _scal(self, path, v):
        self._rr.log(path, self._rr.Scalars(float(v)))

    def step(self, t_s, ds, dbg=None, nearest=None, tangent=None, cruise=None,
             running=None, armed=None):
        """t_s: seconds. ds: DroneState. dbg: {a(3), w_des(3 rad/s), q_des(4), thr} from cmd().
        nearest/tangent/cruise/running/armed: optional context. Decimated + best-effort."""
        if not self.ok:
            return
        pos = np.asarray(ds.pos_ned, float)
        self._trail.append(pos.copy())
        if len(self._trail) > 4000:
            self._trail = self._trail[-4000:]
        self._i += 1
        if self._i % self.decimate:
            return
        try:
            rr = self._rr
            rr.set_time("t", duration=float(t_s))
            vel = np.asarray(ds.vel_ned, float)
            R = quat_to_R(np.asarray(ds.quat_wxyz, float))
            cam = -R[:, 0]                                  # nose (camera-forward) in world
            # --- 3D ---
            rr.log("world/drone", rr.Points3D([pos], radii=0.4, colors=[255, 200, 0]))
            if len(self._trail) >= 2:
                rr.log("world/trail", rr.LineStrips3D([np.array(self._trail)], colors=[255, 200, 0]))
            rr.log("world/vel", rr.Arrows3D(origins=[pos], vectors=[vel], colors=[0, 220, 120]))
            rr.log("world/nose", rr.Arrows3D(origins=[pos], vectors=[cam * 1.5], colors=[255, 80, 80]))
            if dbg is not None and dbg.get("a") is not None:
                rr.log("world/accel_cmd", rr.Arrows3D(origins=[pos], vectors=[np.asarray(dbg["a"], float)],
                                                      colors=[180, 120, 255]))
            # --- attitude + aero ---
            roll, pitch, yaw = euler_rpy_deg(ds.quat_wxyz)
            self._scal("att/tilt/act", tilt_deg(ds.quat_wxyz))
            self._scal("att/yaw/act", yaw)
            self._scal("aero/sideslip/beta", sideslip_deg(ds.quat_wxyz, vel))
            # --- body rates: commanded (w_des, rad/s) vs actual (omega) ---  the COUPLING read
            om = np.degrees(np.asarray(ds.omega, float))
            for ax, name in zip(range(3), ("roll", "pitch", "yaw")):
                self._scal(f"rates/{name}/act", om[ax])
            if dbg is not None and dbg.get("w_des") is not None:
                wd = np.degrees(np.asarray(dbg["w_des"], float))
                for ax, name in zip(range(3), ("roll", "pitch", "yaw")):
                    self._scal(f"rates/{name}/cmd", wd[ax])
            if dbg is not None and dbg.get("q_des") is not None:
                self._scal("att/tilt/cmd", tilt_deg(dbg["q_des"]))
                self._scal("att/yaw/sp", euler_rpy_deg(dbg["q_des"])[2])
            if dbg is not None and dbg.get("thr") is not None:
                self._scal("ctrl/thrust", dbg["thr"])
            # --- tracking ---
            self._scal("track/speed", float(np.linalg.norm(vel[:2])))
            if cruise is not None:
                self._scal("track/cruise", float(cruise))
            if tangent is not None:
                al, ct = along_cross(vel, tangent)
                self._scal("track/along", al)
                self._scal("track/cross", ct)
            if nearest is not None:
                rr.log("world/nearest", rr.Points3D([np.asarray(nearest, float)], radii=0.2,
                                                    colors=[120, 180, 255]))
            # --- state ---
            if running is not None:
                self._scal("ctrl/running", 1.0 if running else 0.0)
            if armed is not None:
                self._scal("ctrl/armed", 1.0 if armed else 0.0)
        except Exception:
            pass
