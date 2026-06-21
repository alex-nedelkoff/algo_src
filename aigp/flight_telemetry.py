"""Live flight-control debug telemetry over Rerun (0.33), built around the turn-tumble diagnosis.

Two layers:
  * PURE helpers (tilt_deg, euler_rpy_deg, sideslip_deg, along_cross) -- unit-tested, no Rerun dep.
  * FlightLog -- best-effort Rerun sink: decimated, async, NEVER raises into the control loop, and
    ALWAYS called AFTER the command is sent so telemetry can't delay a command. Uses a separate
    gRPC connection to the Mac viewer (not the MAVLink command link).

Workflow: run `rerun` on the Mac, then run an instrumented controller on Windows with --viz.
"""
from __future__ import annotations
import threading
import time as _time
import numpy as np
from aigp.geometry import quat_to_R

MAC_VIEWER = "rerun+http://100.101.13.126:9876/proxy"   # Mac Tailscale IP (matches aigp/viz.py)


# ---------------------------------------------------------------------------
# Pure, testable signal math (NED world / FRD body; camera-forward = -body_x).
# ---------------------------------------------------------------------------
_T20 = np.radians(20.0)


def _qfix(q):
    """live shuffled quat (wxyz) -> TRUE attitude quat (canonical: q_true = q_live[[1,2,3,0]])."""
    q = np.asarray(q, float)
    return q[[1, 2, 3, 0]]


def _r_opt_body_true(s_cam):
    """optical (x right, y down, z view) -> TRUE body, camera = s_cam*body_x pitched 20 deg up.
    s_cam=-1 reproduces fly_gate3's live-frame matrix exactly."""
    S, C = np.sin(_T20), np.cos(_T20)
    return np.array([[0.0, s_cam * S, s_cam * C],
                     [s_cam, 0.0, 0.0],
                     [0.0, C, -S]])


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
def strip_viz_args(argv):
    """Remove viz flags (--viz, --no-viz, --rrd <path>) so a controller can parse its positionals."""
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a in ("--viz", "--no-viz"):
            i += 1
        elif a == "--rrd":
            i += 2
        else:
            out.append(a); i += 1
    return out


def from_args(argv, rate_gain=None, run_name="aigp-flight", store=None):
    """Build a FlightLog from CLI args. ON BY DEFAULT (every run feeds the dashboard); --no-viz
    disables it, --rrd <path> records to a file instead of streaming live. Returns None if disabled.
    Pass `store` so the COLLISION flag is streamed too (hard rule: every sim run feeds the dashboard)."""
    if "--no-viz" in argv:
        return None
    rrd = None
    if "--rrd" in argv:
        try:
            rrd = argv[argv.index("--rrd") + 1]
        except IndexError:
            rrd = None
    return FlightLog(rate_gain=rate_gain, hz=40, rrd_path=rrd, run_name=run_name, store=store)


class FlightLog:
    """Best-effort Rerun telemetry, fully OFF the control thread. The control loop calls push()
    each iteration AFTER sending its command -- push() only stows the latest snapshot under a lock
    (microseconds), so it can never jitter the loop. A daemon thread does ALL the Rerun I/O at `hz`,
    naturally decimating from the (faster) control rate. Disabled (no-op) if Rerun can't connect.

    Why threaded: ~15 rr.log calls/step cost ~3 ms on the control thread (measured) -- too much for a
    250 Hz+ loop. Off-thread, push() is ~microseconds and the logging cost is borne by a background
    thread that just drops frames if it can't keep up."""

    def __init__(self, rate_gain=None, hz=40, rrd_path=None, run_name="aigp-flight", store=None):
        self.ok = False
        self.rg = np.asarray(rate_gain, float) if rate_gain is not None else None
        self._store = store        # if set, the COLLISION flag + FPV camera are streamed from it
        self._last_seq = 0
        self._last_frame_seq = -1
        self._last_frame_t = 0.0
        self._trail = []
        self._latest = None
        self._lock = threading.Lock()
        self._stop = False
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
            self._th = threading.Thread(target=self._run, args=(max(1.0, float(hz)),), daemon=True)
            self._th.start()
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
                ts("events/collision_count", "COLLISION count  <- crashes (step per hit)"),
                grid_columns=2,
            )
            bp = rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Vertical(rrb.Spatial3DView(origin="world", name="3D actual vs commanded"),
                                 rrb.Spatial2DView(origin="camera", name="FPV camera"),
                                 row_shares=[1.0, 0.8]),
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

    def clear_trail(self):
        """Wipe the actual-flight trail + stale commanded path so the NEXT mission renders clean
        (actual-vs-commanded for one maneuver instead of the whole session)."""
        if not self.ok:
            return
        with self._lock:
            self._trail = []
        try:
            for ent in ("world/trail", "world/cmd_path", "world/cmd_path_pts", "world/nearest"):
                self._rr.log(ent, self._rr.Clear(recursive=False))
        except Exception:
            pass

    def log_gates(self, points, name="gates", color=(0, 255, 0), radii=2.0):
        """Log gate centers as labeled spheres + a connecting line (static). points in world NED.
        radii defaults to the 2 m geometric-pass detection radius so the trail-vs-sphere overlap is visible."""
        if not self.ok:
            return
        try:
            p = np.asarray(points, float)
            labels = [f"{name}{i}" for i in range(len(p))]
            self._rr.log(f"world/{name}", self._rr.Points3D(p, radii=float(radii), colors=list(color), labels=labels),
                         static=True)
            self._rr.log(f"world/{name}_line", self._rr.LineStrips3D([p], colors=list(color)), static=True)
        except Exception:
            pass

    def _scal(self, path, v):
        self._rr.log(path, self._rr.Scalars(float(v)))

    def push(self, t_s, ds, dbg=None, nearest=None, tangent=None, cruise=None,
             running=None, armed=None, los=None):
        """Stow the latest snapshot for the logging thread. ~microseconds -- safe every control loop.
        dbg: {a(3), w_des(3 rad/s), q_des(4), thr} from cmd(). los: world-NED point the camera is
        EXPECTED to point at (commanded line-of-sight) -- drawn as a ray to compare vs the actual
        frustum. Other args are optional context."""
        if not self.ok:
            return
        with self._lock:
            self._latest = (float(t_s), ds, dbg, nearest, tangent, cruise, running, armed, los)

    def close(self):
        self._stop = True

    def _run(self, hz):
        period = 1.0 / hz
        last_t = None
        while not self._stop:
            with self._lock:
                snap = self._latest
            if snap is not None and snap[0] != last_t:
                last_t = snap[0]
                try:
                    self._log(*snap)
                except Exception:
                    pass
            _time.sleep(period)

    def _log(self, t_s, ds, dbg, nearest, tangent, cruise, running, armed, los=None):
        """All Rerun I/O -- runs ONLY on the daemon thread, never the control loop."""
        rr = self._rr
        pos = np.asarray(ds.pos_ned, float)
        self._trail.append(pos.copy())
        if len(self._trail) > 4000:
            self._trail = self._trail[-4000:]
        rr.set_time("t", duration=float(t_s))
        vel = np.asarray(ds.vel_ned, float)
        if True:
            R = quat_to_R(np.asarray(ds.quat_wxyz, float))
            cam = -R[:, 0]                                  # nose (camera-forward) in world
            # --- 3D ---
            rr.log("world/drone", rr.Points3D([pos], radii=0.4, colors=[255, 200, 0]))
            if len(self._trail) >= 2:
                rr.log("world/trail", rr.LineStrips3D([np.array(self._trail)], colors=[255, 200, 0]))
            rr.log("world/vel", rr.Arrows3D(origins=[pos], vectors=[vel], colors=[0, 220, 120]))
            rr.log("world/nose", rr.Arrows3D(origins=[pos], vectors=[cam * 1.5], colors=[255, 80, 80]))
            if los is not None:                              # EXPECTED line-of-sight (cyan) vs actual frustum
                losp = np.asarray(los, float)
                rr.log("world/expected_los", rr.LineStrips3D([np.array([pos, losp])], colors=[0, 255, 255]))
                rr.log("world/los_target", rr.Points3D([losp], radii=0.35, colors=[0, 255, 255]))
            # --- camera FOV frustum (spec intrinsics: 640x360, fx=fy=320 -> 90 deg horizontal;
            # camera tilted 20 deg UP off the nose). Pose from the TRUE attitude (qfix) -- the raw
            # live quat is a shuffled chart, NOT a rotation; using it made the frustum drift in
            # motion (the canonical fake-tilt warp). Camera sign s_cam calibrated once at the first
            # sample against the proven live camera heading (fly_gate3 recipe).
            R_t = quat_to_R(_qfix(ds.quat_wxyz))
            if not getattr(self, "_pinhole_sent", False):
                self._pinhole_sent = True
                yaw_live = float(np.arctan2(R[1, 0], R[0, 0]))
                cam_live = -np.array([np.cos(yaw_live), np.sin(yaw_live)])
                self._s_cam = 1.0 if float(R_t[:2, 0] @ cam_live) > 0 else -1.0
                self._r_opt = _r_opt_body_true(self._s_cam)
                print(f"[viz] frustum s_cam={self._s_cam:+.0f}", flush=True)
                rr.log("world/cam_fov", rr.Pinhole(resolution=[640, 360], focal_length=[320.0, 320.0],
                                                   principal_point=[320.0, 180.0],
                                                   image_plane_distance=3.0), static=True)
            rr.log("world/cam_fov", rr.Transform3D(translation=pos, mat3x3=R_t @ self._r_opt))
            if los is not None:
                # ACTUAL camera view axis in world (same basis as the frustum above) vs the DESIRED
                # line-of-sight (drone->los). aim_err_deg = signed horizontal angle between them:
                # ~0 = camera on target; ~180 = pointing opposite (nose/camera flip); else a sign bug.
                cam_axis = R_t @ self._r_opt[:, 2]
                na = float(np.linalg.norm(cam_axis))
                if na > 1e-9:
                    cam_axis = cam_axis / na
                    rr.log("world/cam_axis", rr.Arrows3D(origins=[pos], vectors=[cam_axis * 4.0],
                                                         colors=[255, 0, 255]))   # magenta = ACTUAL camera
                    dh = (np.asarray(los, float) - pos)[:2]; ah = cam_axis[:2]
                    if np.linalg.norm(dh) > 1e-6 and np.linalg.norm(ah) > 1e-6:
                        self._scal("cam/aim_err_deg", float(np.degrees(
                            np.arctan2(dh[0] * ah[1] - dh[1] * ah[0], float(dh @ ah)))))
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
            # --- COLLISION flag (hard rule: every sim run streams this so crashes are visible) ---
            if self._store is not None:
                ev, seq = self._store.get_collision()
                self._scal("events/collision_count", float(seq))   # monotonic: steps up on each hit
                if seq != self._last_seq:
                    self._last_seq = seq
                    rr.log("world/collision", rr.Points3D([pos], radii=0.7, colors=[255, 0, 0]))
                    if ev is not None:
                        rr.log("events/collision_log", rr.TextLog(
                            f"COLLISION id={ev['id']} (1001=gate,1002=env) impulse={ev['impulse']:.2f}",
                            level="ERROR"))
            # --- FPV camera feed (hard rule: stream it so the view is observable on the dashboard) ---
            if self._store is not None:
                fr, fseq = self._store.get_frame()
                if fr is not None and fseq != self._last_frame_seq and (_time.time() - self._last_frame_t) > 0.1:
                    self._last_frame_seq = fseq
                    self._last_frame_t = _time.time()
                    try:
                        rr.log("camera", rr.Image(fr[0][:, :, ::-1]))   # cv2 BGR -> RGB
                    except Exception:
                        pass
