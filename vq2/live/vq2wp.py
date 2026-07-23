"""VQ2 waypoint racer: GateNet vision servo + map-prior spline route between gates.

vq2gate.py's flight-proven stack (wfix IMU attitude, PosVelKF, GateNet detector,
velocity-PD level_cmd, in-flight sign probes, approach/punch phases) with two fixes
for the post-G1 failure mode (2026-07-05 run: G1 ticked, then approach-timeout on a
ghost lock while every real obs was rejected):

1. ROUTE phase: after a tick (or a lost lock) fly the course-map spline toward the
   next anchored gate on KF dead reckoning, nose along the carrot so the camera
   sweeps the gate corridor. GateTrajectory (VQ1-proven, drone-locked s_ref carrot)
   supplies the path; the map (course_map_v5) supplies G1/G2. Vision takes over the
   moment a validated lock appears.
2. Map-validated landmark pinning: a new landmark is only pinned when the obs lands
   within VAL_R of the expected next map gate (kills the garbage-solve ghosts:
   z = -17..-30 m "gates" from the failed run), plus a hard |dz| sanity gate.

Success = RACE_STATUS ticks (judge signal). Mission: tick gate 2, then land.
"""
import sys, os, socket, struct, time, json, math, threading
from collections import deque

sys.path.insert(0, r'C:\Users\alexj\Documents\algo_src_main')
sys.path.insert(0, r'C:\Users\alexj')
import numpy as np
import cv2
import torch
# transformers 5.x evaluates torch.float8_e8m0fnu at import (finegrained_fp8.py);
# torch 2.5.1 predates that dtype. GateNet never touches the fp8-quantization
# path, so a placeholder dtype satisfies the module-level reference.
for _fp8 in ('float8_e8m0fnu', 'float8_e4m3fn', 'float8_e5m2'):
    if not hasattr(torch, _fp8):
        setattr(torch, _fp8, torch.uint8)
from pymavlink import mavutil
from eskf import PosVelKF, accel_level
from gate_traj import GateTrajectory
from scripts.dcl import vq1_detect_overlay as OV
from perception.training import train_gatenet as TG
from perception.decode import associate as AS

OUT = 'C:/Users/alexj/vq2_wp'
os.makedirs(OUT + '/frames', exist_ok=True)

# corpus-format recording for offline VIO work (RECORD=<dir> to enable):
# raw mavlink + every decoded frame at camera rate. det_loop's frame writes
# only run at inference rate (~3 fps) -- too sparse for optical flow.
REC_DIR = os.environ.get('RECORD', '')
if REC_DIR:
    os.makedirs(REC_DIR + '/frames', exist_ok=True)
    rec_mav = open(REC_DIR + '/mavlink.jsonl', 'w')
    rec_frames = open(REC_DIR + '/frames_dedup.jsonl', 'w')
    rec_cmd = open(REC_DIR + '/cmds.jsonl', 'w')
CKPT = r'C:\Users\alexj\gatenet_b2_cov.pt'
CFG = r'C:\Users\alexj\Documents\algo_src_main\configs\perception\gatenet_b2_multi_pb_cov.yaml'

HOVER = 0.2675  # 07-10: the vehicle NEVER changed. THRPROBE's 'hover 0.12'
                # was measured GROUNDED (98-100%% floor contact in every probe
                # corpus); the flat |f|~g 'governor band' was the floor's
                # normal force. Tick-era value restored.
CMD_HZ = 50.0
TILT_ABORT = math.radians(55)
RATE_GAIN = 1.93
# SIGN_R flipped -1 -> +1 with the roll-integration wfix (07-15): the old -1
# made the rate loop converge the MIRRORED est roll (true roll went the
# opposite way, unobservable at rest). With est roll now truth-signed, +1
# drives TRUE roll to the reference.
SIGN_R, SIGN_P = +1.0, +1.0
KP = 1.8
K_V = 0.07  # outer-loop limit cycle (Rerun cmd-vs-act, 07-10): vision-velocity jitter -> roll_ref oscillation; cut gain
CAM_TILT = math.radians(20.0)
MISSION_S = 240.0
TARGET_TICKS = int(os.environ.get('TICKS', '1'))  # prove ONE tick first; TICKS=2 chains to the next gate

# ---- course map (frame = KF world: spawn origin, x downcourse, z down) ----
# COURSE GEOMETRY (flight #21 frames + #5 pad detections): the judge counts
# gates in ORDER. Gate 0 is a HIGH gate ~9 m out and ~6 m UP (nine consistent
# GateNet pad detections at [9.0, 0.0, -6.3] were real, not garbage); the
# course ribbon drops vertically from it into the red gate (gate 1) at x=11,
# then runs to G2. The pad-z "artifact" was the detector alternating between
# the stacked high + red gates.
# THE 2-METRE BUG (07-07, run-13 frames): PnP range measures to the gate-
# frame ORIGIN, which sits ~2 m BEHIND the visible structure (recovered
# 8-kp model: corners at z_gate -1.8..-2.2). G1_W [11,0] is therefore a
# point in EMPTY SPACE behind the gate -- every "crossing" transited the
# real aperture plane (x ~ 9.1, triangulated corner map) un-aimed.
# Fly at G1_AP; keep G1_W for obs-derived map matching.
ORIGIN_OFFSET = 0.0   # affine range correction (RANGE_AFF_*) already
                      # references ranges to the APERTURE plane
# QUALIFIER COURSE RE-MAP (07-07, VQ2-COURSE-01): the ribbon's gate 1 is at
# [26.9, 8.6] (191 georeferenced obs, dominant cluster; confirmed visually --
# the ribbon threads it). The [11, 0] object is a DECOY the ribbon bypasses;
# every prior 'crossing' threaded it perfectly for zero points.
# CORNER-TRIANGULATED (tick corpus, 79 obs, 8.2 px): the ribbon gate's
# PHYSICAL structure is at [11.7, 5.2] -- the [26.9,8.6] cluster was the
# PnP origin's far-projection (origin-offset illusion at range). Aperture
# centroid [11.68, 5.17, -1.06], face along y => through-normal ~ +x.
# COURSE MAP v2 (07-08, VQ2-SCALE-01): the old map ([11.68,5.17] gate,
# [26.9,8.6] cluster, [7.25,-0.17] arch) was built from UNCORRECTED PnP
# ranges (4/3 scale + ~2.7 m origin offset baked in) -- inflated ~2x and
# bent. Raw-IMU truth, replicated on arch33 + accept19: gate 1 sits
# straight down the start chute, ~5.9 m from the pad. Frames: glowing
# start corridor, green lights, gate dead ahead.
N1 = np.array([1.0, 0.0, 0.0])
G1_AP = np.array([5.9, 0.9, -1.1])    # official inner aperture 1.5 m sq,
                                      # center ~1.35 m up; fly z -1.1
G1_W = G1_AP.copy()                   # corrected ranges are aperture-referenced
RIB_W = G1_AP.copy()
HIGH_W = G1_AP.copy()                 # route/punch aim point
G2_W = np.array([31.6, 1.4, -1.5])    # old [44.8,1.9] descaled (provisional)
DECOY_W = np.array([100.0, 100.0, -1.3])  # RETIRED: old arch anchor, scale-stale;
                                          # parked far away so nothing matches it
N2 = N1.copy()
GATES_W = [HIGH_W, G2_W, G2_W]
THRU = [N1, N2, N2]

# MAP_JSON (07-17): course model from a validated map file (contract:
# docs/vq2-map-json-contract.md; loader: map_ingest.py, fails closed on
# contract violations and prints judge-anchor disagreements). Overrides the
# constants above; everything downstream inherits: the pad-lock anchor block
# treats G1 as a prior exactly as before, and the G2TEST gate-2 delta becomes
# map-derived (route-2 minus route-1) instead of the hardcoded [5.14,5.26,0].
# Unset (default) = the constants above, byte-identical behavior.
G2_DELTA = None
MAP_JSON = os.environ.get('MAP_JSON', '')
if MAP_JSON:
    try:
        from map_ingest import load_course_map, anchor_warnings
    except ImportError:                    # running from the repo checkout
        from vq2.map_ingest import load_course_map, anchor_warnings
    _map_gates = load_course_map(MAP_JSON)
    for _mw in anchor_warnings(_map_gates):
        print(f'MAP_JSON WARNING: {_mw}', flush=True)
    _mg_p = [np.array(g.pos, dtype=float) for g in _map_gates]
    _mg_n = [np.array(g.normal, dtype=float) for g in _map_gates]
    N1 = _mg_n[0]
    G1_AP = _mg_p[0].copy()
    G1_W = G1_AP.copy()
    RIB_W = G1_AP.copy()
    HIGH_W = G1_AP.copy()
    if len(_mg_p) > 1:
        G2_W = _mg_p[1].copy()
        N2 = _mg_n[1]
        G2_DELTA = _mg_p[1] - _mg_p[0]
    GATES_W = [HIGH_W, G2_W, G2_W]
    THRU = [N1, N2, N2]
    print(f'MAP_JSON: {len(_mg_p)} gates from {MAP_JSON} | '
          f'G1 {G1_AP.round(2).tolist()} G2 {G2_W.round(2).tolist()} '
          f'delta {None if G2_DELTA is None else G2_DELTA.round(2).tolist()}',
          flush=True)

def build_traj(gh, g1, g2):
    """Spline: climb to the high gate, controlled-sink dive to the red gate,
    then level run to G2. vz_max caps the descent rate (VQ1 controlled-sink)."""
    pts = np.array([
        [0.0, 0.0, -1.3],
        [4.0, 0.8, -1.3],           # bear gently right off the pad
        [9.4, 1.7, -1.3],           # JUDGE GATE 1 = START ARCH (replay pose
        [10.4, 2.9, -1.2],          #   at inside-arch frame); tight right turn
        gh,                         # gate-2 aperture (corner-measured, 3.5 m on)
        gh + 2.0 * N1,              # carry through
        g2 - 3.0 * N2,
        g2,
    ])
    tr = GateTrajectory(pts, v_cruise=1.6, phi_max_deg=15.0, tilt_budget_deg=12.0,
                        vz_max=0.55)
    return tr, [tr.nearest_s(gh), tr.nearest_s(g2), tr.s_max]

ARCHTEST = os.environ.get('ARCHTEST', '0') == '1'
if ARCHTEST:
    # verification mission: straight through the arch center, carry 3 m,
    # land. No turns, no retreat. Thread verified by recorded RACE_STATUS
    # + mid-crossing frames.
    def build_traj(gh, g1, g2):
        # COURSE MAP v2: straight down the start chute through gate 1's
        # aperture (raw-IMU truth [5.9, 0.9], official 1.5 m sq inner,
        # crossing z -1.1), carry 2 m, land. The chute frames (arch33
        # -10 s) show exactly this line: green start lights both sides,
        # gate dead ahead.
        gate = G1_AP
        # TICK-GEOMETRY REPLICA (07-10): every scored tick transited the
        # start-arch zone before the gate; today's straight chute (0/6,
        # frame-verified centered crossings) never did. Yesterday's
        # ticking route, descaled to true units: S-curve through the
        # arch zone (~[4.3, -0.1], cross low z -0.85 like arch17's
        # raw-DR) then into the gate.
        # DIRECTION-CONVENTION TEST (07-11, Alex protocol): fly a line
        # deliberately OFFSET from the gate -- 'left' = 2 m at -y,
        # 'up' = 2 m at -z -- and verify by eye in sim + viewer.
        off = {'left': np.array([0.0, -2.0, 0.0]),
               'up': np.array([0.0, 0.0, -2.0])}.get(
                   os.environ.get('DIRTEST', ''), np.zeros(3))
        tgt = gate + off
        pts = [
            [0.0, 0.0, -1.3],
            [2.0, off[1] * 0.5, -1.2 + off[2] * 0.5],
            tgt - 2.0 * N1,
            tgt,
            tgt + 2.0 * N1,
        ]
        if os.environ.get('G2TEST') == '1':
            # GATE-2 SEGMENT MAP (07-11): gate-2 center from the ribbon
            # triangulation artifact (g2rib_corners_world), expressed as a
            # DELTA from the same pipeline's gate-1 center so the shared
            # detection-chain bias cancels: [5.14, 5.26, 0]. Applied to
            # the judge-calibrated ticking aperture (tgt). Frame-verified:
            # after gate 1 the ribbon turns right ~24 deg to gate 2,
            # crossed along +x. Same aperture z (official gates identical).
            # MAP_JSON supplies the delta (route2 - route1) when set.
            g2 = tgt + (G2_DELTA if G2_DELTA is not None
                        else np.array([5.14, 5.26, 0.0]))
            pts += [
                (tgt + g2) / 2 + np.array([-0.5, 0.0, 0.0]),  # swing wide into the right turn
                g2 - 2.5 * np.array([1.0, 0.0, 0.0]),
                g2,
                g2 + 2.5 * np.array([1.0, 0.0, 0.0]),
            ]
        pts = np.array(pts)
        tr = GateTrajectory(pts, v_cruise=0.6, phi_max_deg=15.0,
                            tilt_budget_deg=12.0, vz_max=0.55)
        if os.environ.get('G2TEST') == '1':
            return tr, [tr.nearest_s(gate), tr.nearest_s(pts[-2]), tr.s_max]
        return tr, [tr.nearest_s(gate), tr.s_max, tr.s_max]

TRAJ, S_GATES = build_traj(HIGH_W, G1_W, G2_W)
# ARCHTEST perception-aware zone: whole chute (straight course, gate in
# view from early on with corrected ranges)
ARCH_SWEEP_S = TRAJ.nearest_s(np.array([1.5, 0.2, -1.2])) if ARCHTEST else 1e9

# ---- Rerun live dashboard (best-effort: never raises into the control loop) ----
MAC_VIEWER = 'rerun+http://100.101.13.126:9876/proxy'   # Mac over Tailscale (VQ1 convention)
VIZ = os.environ.get('NOVIZ') != '1'
_rr = None
if VIZ:
    try:
        import rerun as rr
        rr.init('vq2wp')
        # RRD=path -> record to a local .rrd file (robust: no tailnet, replayable
        # in `rerun <file>`). Otherwise stream to the Mac viewer (VQ1 convention).
        _rrd = os.environ.get('RRD')
        if _rrd:
            rr.save(_rrd)
            print(f'rerun recording -> {_rrd}', flush=True)
        else:
            rr.connect_grpc(MAC_VIEWER)
        # declare the world frame handedness: spawn frame is FRD
        # (x forward/downcourse, y right, z down). Without this, rerun
        # renders z-down coords in its default z-up right-handed space:
        # left/right appears MIRRORED and roll spikes render in the
        # pitch plane (Alex spotted both on the live stream, 07-10).
        rr.log('world', rr.ViewCoordinates.FRD, static=True)
        # handedness beacons: labeled points at pad-right and pad-left so
        # the viewer's left/right is decidable at a glance from any eye
        rr.log('world/beacon_right', _rr_pts_right := None or rr.Points3D(
            [[1.0, 2.0, -1.0]], radii=0.15, colors=[255, 0, 0],
            labels=['RIGHT of pad (+y)']), static=True)
        rr.log('world/beacon_left', rr.Points3D(
            [[1.0, -2.0, -1.0]], radii=0.15, colors=[0, 100, 255],
            labels=['LEFT of pad (-y)']), static=True)
        _rr = rr
    except Exception as e:
        print(f'rerun viz unavailable: {e}', flush=True)

def _viz_static():
    """Log the static scene: NED axes, map gates, theoretical spline."""
    if _rr is None:
        return
    try:
        rr = _rr
        rr.log('world', rr.ViewCoordinates.FRD, static=True)
        # handedness beacons: labeled points at pad-right and pad-left so
        # the viewer's left/right is decidable at a glance from any eye
        rr.log('world/beacon_right', _rr_pts_right := None or rr.Points3D(
            [[1.0, 2.0, -1.0]], radii=0.15, colors=[255, 0, 0],
            labels=['RIGHT of pad (+y)']), static=True)
        rr.log('world/beacon_left', rr.Points3D(
            [[1.0, -2.0, -1.0]], radii=0.15, colors=[0, 100, 255],
            labels=['LEFT of pad (-y)']), static=True)
        centers, sizes, labels = [], [], []
        for i, g in enumerate(GATES_W[:2]):
            centers.append(g.tolist())
            sizes.append([0.15, 1.0, 1.0])
            labels.append(f'gate{i}')
        rr.log('world/gates', rr.Boxes3D(centers=centers, half_sizes=sizes,
                                         labels=labels, colors=[255, 60, 30]), static=True)
        pts = [TRAJ.sample(s)['pos'].tolist() for s in np.linspace(0, TRAJ.s_max, 120)]
        rr.log('world/ref_path', rr.LineStrips3D([pts], colors=[0, 200, 255]), static=True)
    except Exception:
        pass

_trail = []
_fixpts = deque(maxlen=200)   # accepted vision-fix positions (viewer markers)
_viz_last = [0.0, 0.0]   # [pose_wall, cam_wall]

def viz_tick(p, ref_pos=None):
    """~10 Hz pose/trail/carrot + ~5 Hz FPV. Call AFTER commands are sent."""
    if _rr is None:
        return
    now = time.time()
    try:
        rr = _rr
        if now - _viz_last[0] > 0.1:
            _viz_last[0] = now
            rr.set_time('t', duration=now)
            r_, p_, y_ = state['roll'], state['pitch'], state['yaw']
            cr, sr = math.cos(r_), math.sin(r_); cp, sp = math.cos(p_), math.sin(p_)
            cy, sy = math.cos(y_), math.sin(y_)
            R = np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                          [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                          [-sp,   cp*sr,            cp*cr]])
            rr.log('world/drone', rr.Transform3D(translation=p.tolist(), mat3x3=R.tolist()))
            rr.log('world/drone/body', rr.Boxes3D(half_sizes=[[0.18, 0.18, 0.05]],
                                                  colors=[80, 255, 80]))
            _trail.append(p.tolist())
            if len(_trail) % 5 == 0:
                rr.log('world/actual_path', rr.LineStrips3D([_trail], colors=[80, 255, 80]))
            if ref_pos is not None:
                rr.log('world/carrot', rr.Points3D([list(ref_pos)], radii=0.12,
                                                   colors=[255, 255, 0]))
            rr.log('plots/alt_m', rr.Scalars(-float(p[2])))
            rr.log('plots/gate_idx', rr.Scalars(float(state['gate_idx'])))
            dp = state.get('dpvo_p')
            if dp is not None and now - state.get('dpvo_wall', 0) < 1.0:
                rr.log('world/dpvo', rr.Points3D([list(dp)], radii=0.1,
                                                 colors=[60, 120, 255]))
            # vision-fix markers: red dots where accepted fixes snapped the
            # KF -- attributes the trail's jerks to fixes, not IMU noise
            lf = state.get('_last_fix')
            if lf is not None and now - lf[2] < 0.5:
                _fixpts.append(lf[0])
                rr.log('world/vision_fixes',
                       rr.Points3D(list(_fixpts), radii=0.07,
                                   colors=[255, 60, 60]))
            # CMD vs ACTUAL (07-10, Alex: 'sway -- disconnect between
            # command and actual'): commanded body rates + thrust against
            # measured gyro and estimated velocity, same timeline
            lc = state.get('last_cmd')
            if lc is not None:
                rr.log('cmd/roll_rate', rr.Scalars(float(lc[0])))
                rr.log('cmd/pitch_rate', rr.Scalars(float(lc[1])))
                rr.log('cmd/yaw_rate', rr.Scalars(float(lc[2])))
                rr.log('cmd/thrust', rr.Scalars(float(lc[3])))
            g_ = state['gyr']
            rr.log('act/roll_rate', rr.Scalars(float(g_[0])))
            rr.log('act/pitch_rate', rr.Scalars(-float(g_[1])))
            rr.log('act/yaw_rate', rr.Scalars(-float(g_[2])))
            rr.log('act/vx_b', rr.Scalars(float(state['vx_b'])))
            rr.log('act/vy_b', rr.Scalars(float(state['vy_b'])))
            rr.log('act/vz_up', rr.Scalars(float(state['vz_up'])))
            rr.log('act/tilt_deg', rr.Scalars(math.degrees(tilt()) if state.get('airborne') else 0.0))
        jp = state.get('jpeg')
        if jp is not None and now - _viz_last[1] > 0.2:
            _viz_last[1] = now
            rr.set_time('t', duration=now)
            rr.log('camera/fpv', rr.EncodedImage(contents=jp, media_type='image/jpeg'))
    except Exception:
        pass
LEAD = 2.5                      # carrot lead (m); drone-locked s_ref cannot run away
CARROT_DS = 0.12                # max s advance per 20 Hz iter (~2.4 m/s along-track)
V_ROUTE_MAX = float(os.environ.get('VMAX', '1.0'))  # slow everywhere: keep the detector locked to punch range
ACCEPT_R = 3.0                  # legacy radius gate (POLICY=radius rollback + hysteresis bound)
OBS_POLICY = os.environ.get('POLICY', 'huber_area')  # 'huber_area' | 'radius'
def _load_npy(p):
    try:
        return np.load(p)
    except Exception:
        return None
# accept19 multi-view triangulated map (the old g1_corners map was pad-ray
# bearing-only: consistent from the pad, metres wrong in depth)
DECOY_C = _load_npy(r'C:\Users\alexj\decoy_corners_world.npy')
G2RIB_C = _load_npy(r'C:\Users\alexj\g2rib_corners_world.npy')
IDENT_ON = OBS_POLICY in ('huber_area', 'huber') and \
    os.environ.get('NOIDENT', '0') != '1'
HUBER_DELTA = 1.5               # m: miss below this = full-weight fix
RANGE_RATIO_MIN = 0.55          # identity gate (see det_loop comment)
# AFFINE RANGE CORRECTION (07-08, VQ2-SCALE-01): the 07-05 "ranges are
# TRUE / KF under-integrates" call was backwards. Official qualifier gate
# inner aperture = 1.5 m; the PnP solver assumes the VQ1 2.0 m aperture,
# so ranges read 4/3 too long, plus the solver's gate-frame origin sits
# ~2.7 m (assumed-scale) behind the aperture. Regressed on arch33
# (straight chute, raw-IMU truth: pnp = 1.30*true + 2.9) and accept19
# (129 tracked pairs: 1.40*true + 3.41), k pinned to the exact 4/3:
#     true_aperture_range = (pnp_range - 2.7) * 0.75
RANGE_AFF_B = 2.7
RANGE_AFF_K = 0.75

# Single-sourced pinhole intrinsic (see vq2/camera.py). Deploy layout ships
# flat modules (from eskf import ...), repo/test layout is the vq2 package;
# support both. NOTE: deploy must now copy camera.py flat alongside vq2wp.py.
try:
    from camera import FX as CAM_FX, FY as CAM_FY, CX as CAM_CX, CY as CAM_CY
except ImportError:  # package/repo context
    from vq2.camera import FX as CAM_FX, FY as CAM_FY, CX as CAM_CX, CY as CAM_CY

# camera axes in body frame (nose camera, 20 deg up -- flight-validated 07-06)
ct, st = math.cos(CAM_TILT), math.sin(CAM_TILT)
C_Z = np.array([ct, 0.0, -st])
C_Y = np.array([st, 0.0, ct])
C_X = np.cross(C_Y, C_Z)
M_BODY_CAM = np.stack([C_X, C_Y, C_Z], axis=1)

ATT_BUF = deque(maxlen=600)
TRIM_WIN = []
KF = PosVelKF()
KF_LOCK = threading.Lock()
state = {'acc': (0, 0, -9.81), 'gyr': (0, 0, 0), 't_us': 0,
         'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'vx_b': 0.0, 'vy_b': 0.0, 'vz_up': 0.0,
         'collision': None, 'gate_idx': 0, 'race_finish_ns': -1, 'stop': False,
         'obs': None, 'obs_wall': 0.0, 'frame': None, 'frame_ns': 0,
         'next_gate_w': HIGH_W.copy(), 'nav_ready': False,
         'gatenet_unloaded': False, 'gatenet_unloaded_wall': 0.0}
log_f = open(OUT + '/log.jsonl', 'w')
llock = threading.Lock()

def jlog(kind, **kw):
    kw['kind'] = kind
    kw['t'] = time.time()
    with llock:
        log_f.write(json.dumps(kw, default=str) + '\n')

m = mavutil.mavlink_connection('udpin:127.0.0.1:14550')
m.wait_heartbeat(timeout=8)
print('hb ok', flush=True)

def rx_loop():
    last_us = None
    while not state['stop']:
        msg = m.recv_match(blocking=True, timeout=0.5)
        if not msg: continue
        t = msg.get_type()
        if REC_DIR and t != 'BAD_DATA':
            d = msg.to_dict()
            d['_rx_wall'] = time.time()
            if t == 'ENCAPSULATED_DATA':
                d['data'] = bytes(msg.data)[:64].hex()
            rec_mav.write(json.dumps(d, default=str) + '\n')
        if t == 'HIGHRES_IMU':
            acc = (msg.xacc, msg.yacc, msg.zacc)
            gyr = (msg.xgyro, msg.ygyro, msg.zgyro)
            us = msg.time_usec
            # 38% of HIGHRES_IMU messages are exact re-sends of the previous
            # sample ~0.2 ms later (07-12 corpus audit, vq2_night1: 1541 of
            # 1549 sub-ms pairs bit-identical). Skip them: no information,
            # and they double-integrate the same sample over the sub-ms dt.
            if last_us is not None and 0 <= us - last_us < 1000:
                continue
            if last_us is not None and us > last_us:
                dt = (us - last_us) / 1e6
                if os.environ.get('EULERFIX') == '1':
                    # PROPER BODY-RATE -> EULER-RATE KINEMATICS (07-14): the
                    # raw integrator treats body rates as direct angle rates,
                    # dropping the coupling terms. Yawing at 0.35 rad/s with
                    # ~10 deg pitch loses ~3.6 deg/s of roll rate -> gravity
                    # leaks laterally -> phantom velocity (servo_nt3 runaway:
                    # est 8 m/s at a real 0.8). This was the error the roll
                    # trim mopped up in cruise (and why the blind chute needed
                    # heading-lock); integrate correctly instead.
                    _p, _q, _r = gyr[0], -gyr[1], -gyr[2]   # wfix signs
                    _sf, _cf = math.sin(state['roll']), math.cos(state['roll'])
                    _tt = max(-3.0, min(3.0, math.tan(state['pitch'])))
                    _ct = max(0.3, math.cos(state['pitch']))
                    state['roll'] += (_p + _q * _sf * _tt + _r * _cf * _tt) * dt
                    state['pitch'] += (_q * _cf - _r * _sf) * dt
                    state['yaw'] += ((_q * _sf + _r * _cf) / _ct) * dt
                else:
                    # wfix ALL THREE axes (07-15, fg44 frames): roll was the
                    # only axis integrating the raw gyro sign -- unlike pitch/
                    # yaw it has no at-rest truth (spawn roll=0) so the sign
                    # was never calibrated. Frames at est roll -0.10/-0.18
                    # show the horizon banked the OPPOSITE way: est roll was
                    # a MIRROR of true roll (rate loop stayed self-consistent
                    # via SIGN_R=-1, hiding it). This mirror is the root of
                    # the "IMU-invisible rightward push" family: every est-
                    # frame roll correction physically banked the wrong way.
                    state['roll'] += (-gyr[0]) * dt       # wfix (07-15)
                    state['pitch'] += (-gyr[1]) * dt      # wfix
                    state['yaw'] += (-gyr[2]) * dt        # wfix
                # windowed force-balance ROLL TRIM (VQ2-DRIFT-01 root cause:
                # per-flight roll bias -> phantom lateral force -> every gate
                # miss, invisible to DR). Over a 1 s contiguous-quiet window,
                # mean a_y = -g*sin(roll_true); pull est roll to it. Benched
                # on 4 corpora: bias to within +-0.4 deg (was up to 3.7).
                # trim ONLY in steady route cruise (run-16 lesson: the
                # approach maneuver's REAL lateral accel satisfied the quiet
                # gates and was read as +5 deg of roll error -> crash). Route
                # loop sets trim_ok; approach/punch/retreat clear it. Per-
                # window correction clamped to 2 deg (bias converges over
                # 2-3 windows; a false window can no longer wreck attitude).
                if (state.get('airborne') and state.get('trim_ok')
                        and os.environ.get('NOTRIM') != '1'):
                    gm = max(abs(gyr[0]), abs(gyr[1]), abs(gyr[2]))
                    an = math.sqrt(acc[0]**2 + acc[1]**2 + acc[2]**2)
                    if gm < 0.4 and abs(an - 9.81) < 1.0 and abs(gyr[2]) < 0.03:
                        # |gz| gate: on an ARC the bank is real lateral accel
                        # (run-20: trims clamped +-2 deg oscillating on the
                        # curved route) -- only trim on straight segments
                        TRIM_WIN.append((us / 1e6, acc[1], an, acc[0]))
                    else:
                        TRIM_WIN.clear()
                    if TRIM_WIN and TRIM_WIN[-1][0] - TRIM_WIN[0][0] >= 1.0:
                        ay = sum(w[1] for w in TRIM_WIN) / len(TRIM_WIN)
                        roll_true = math.asin(max(-1.0, min(1.0, -ay / 9.81)))
                        err = roll_true - state['roll']
                        err = max(-0.035, min(0.035, err))
                        state['roll'] += err
                        # PITCH TRIM (07-09 replay): est v_x sat at -0.8 m/s
                        # for 52 s while the route commanded +0.6 -- a ~5 deg
                        # untrimmed pitch bias leaks gravity backward; roll
                        # had a trim, pitch never did. Same window, same
                        # clamp. At rest the spawn's known -17.8 deg pitch
                        # validates the sign: pitch_true = asin(a_x / g).
                        axm = sum(w[3] for w in TRIM_WIN) / len(TRIM_WIN)
                        pitch_true = math.asin(max(-1.0, min(1.0, axm / 9.81)))
                        perr = pitch_true - state['pitch']
                        perr = max(-0.035, min(0.035, perr))
                        state['pitch'] += perr
                        jlog('roll_trim', err_deg=round(math.degrees(err), 2),
                             pitch_err_deg=round(math.degrees(perr), 2))
                        TRIM_WIN.clear()
                in_contact = (state.get('contact') and
                              time.time() - state.get('contact_wall', 0) < 0.3)
                a_lvl = accel_level(acc, state['roll'], state['pitch'])
                cyw, syw = math.cos(state['yaw']), math.sin(state['yaw'])
                a_w = np.array([cyw * a_lvl[0] - syw * a_lvl[1],
                                syw * a_lvl[0] + cyw * a_lvl[1], a_lvl[2]])
                with KF_LOCK:
                    if in_contact:
                        # freeze velocity propagation under contact forces
                        KF.predict(np.zeros(3), dt)
                    else:
                        KF.predict(a_w, dt)
                    vw = KF.v
                state['vx_b'] = cyw * vw[0] + syw * vw[1]
                state['vy_b'] = -syw * vw[0] + cyw * vw[1]
                state['vz_up'] = -vw[2]
            elif last_us is None:
                state['roll'] = math.atan2(acc[1], -acc[2])
                state['pitch'] = math.atan2(acc[0], math.sqrt(acc[1]**2 + acc[2]**2))
            last_us = us
            state['acc'], state['gyr'], state['t_us'] = acc, gyr, us
            ATT_BUF.append((time.time(), state['roll'], state['pitch']))
        elif t == 'COLLISION':
            state['collision'] = msg.to_dict()
            # CONTACT FLAG (07-10, Alex's insight): horizontal_minimum_delta
            # < 5 cm = touching the world. Contact forces are not motion:
            # the KF must not integrate them and guards must not fire on
            # the resulting estimate garbage. 98-100%% of tonight's probe
            # corpora were grounded -- every 'vehicle anomaly' was floor
            # artifacts.
            state['contact'] = msg.horizontal_minimum_delta < 0.05
            state['contact_wall'] = time.time()
            jlog('collision', **msg.to_dict())
        elif t == 'ENCAPSULATED_DATA':
            d = bytes(msg.data)
            if d and d[0] == 1:
                try:
                    rs = struct.unpack('<BQqqIq', d[:37])
                    if rs[4] != state['gate_idx']:
                        jlog('gate_tick', idx=rs[4])
                        if rs[4] > state['gate_idx']:
                            # Camera frame stamps share the receive-wall epoch,
                            # unlike the judge's internal tickstamp.
                            state['gate_tick_ns'] = time.time_ns()
                            state['judge_tickstamp'] = rs[5]
                    state['gate_idx'] = rs[4]
                    state['race_finish_ns'] = rs[3]
                    state['race_ms'] = rs[1]   # race clock (resets on hard reset)
                except Exception:
                    pass

def cam_loop():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('127.0.0.1', 5600)); s.settimeout(0.5)
    H = struct.Struct('<IHHIIQ')
    frames, seen = {}, set()
    while not state['stop']:
        try: pkt = s.recv(65536)
        except socket.timeout: continue
        fid, cid, total, jsize, psize, ns = H.unpack_from(pkt)
        if ns in seen: continue
        frames.setdefault(fid, {})[cid] = pkt[H.size:H.size+psize]
        f = frames[fid]
        if len(f) == total:
            data = b''.join(f[i] for i in range(total))
            if len(data) == jsize:
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    state['dpvo_frame'] = (ns, data)
                    state['frame'], state['frame_ns'] = img, ns
                    state['frame_wall'] = time.time()
                    state['jpeg'] = data          # raw jpeg for the Rerun stream
                    if REC_DIR:
                        # raw bytes straight to disk (no re-encode); single
                        # cam thread owns rec_frames, no lock needed
                        with open(f'{REC_DIR}/frames/{ns}.jpg', 'wb') as rf:
                            rf.write(data)
                        rec_frames.write(json.dumps(
                            {'sim_ns': ns, 'fid': fid, 'rx_wall': time.time()}) + '\n')
                        # kf_pose @ frame cadence (COR-147 g1 DR-bridge):
                        # kf_upd rows stop at the pad under NOFIX, so banked
                        # corpora carry NO metric position through the blind
                        # leg. One row per recorded frame, keyed by the same
                        # sim_ns as the jpg, joins offline without a t->ns
                        # clock fit.
                        with KF_LOCK:
                            _kp, _kv = KF.p.copy(), KF.v.copy()
                        jlog('kf_pose', ns=ns, p=_kp.round(3).tolist(),
                             v=_kv.round(3).tolist())
                seen.add(ns)
            del frames[fid]

def det_loop():
    try:
        _det_loop()
    except Exception as e:
        import traceback
        jlog('det_crash', err=traceback.format_exc())
        print('DET THREAD CRASHED:', e, flush=True)

def _det_loop():
    dev = torch.device('cuda')
    lm = OV.load_model(CKPT, CFG, dev)
    dk, sk = OV._multi_decode_knobs(lm.cfg, stride=lm.stride)
    # warm-up inference: post-reboot the first CUDA forward pass takes
    # 20-30 s (cuDNN autotune); fly3 arch40-42 all aborted their 15 s pad
    # window while it compiled. Pay the cost here, before 'ready'.
    dummy = np.zeros((360, 640, 3), np.uint8)
    padded = OV.pad_bottom(dummy, lm.pad_to_h)
    with torch.no_grad():
        x = TG.frames_to_input(torch.from_numpy(padded[None]), dev)
        TG._float_output(lm.model(x))
    print('detector ready', flush=True)
    last_ns = 0
    while not state['stop']:
        img, ns = state['frame'], state['frame_ns']
        f_wall = state.get('frame_wall', 0.0)
        if img is None or ns == last_ns:
            time.sleep(0.01); continue
        last_ns = ns
        if (state.get('go_passed')
            # GN_UNLOAD (07-15): fastgate is CPU-only and owns the whole
            # flight now -- GateNet's only job is the at-rest pad lock.
            # Release the GPU right after GO regardless of DPVO so a DPVO
            # thread (or anything else) gets the full 4 GB in flight.
            # (go_passed FIRST: 'ticks' is a main-thread global that does
            # not exist until the mission loop starts -- fg58 det crash.)
                and (os.environ.get('GN_UNLOAD') == '1'
                     or (os.environ.get('DPVO') == '1'
                         and os.environ.get('NOFIX') == '1'
                         and state.get('gate_idx', 0) == 0))):
            # DPVO owns the pre-tick estimate; UNLOAD GateNet entirely --
            # sim (~1.8G) + GateNet (~1G) + DPVO (~1.5G) exceed the RTX
            # 3050's 4 GB and the process died silently at DPVO init
            # (pair-5 autopsy: 64 rows, no traceback = VRAM kill).
            # GNSCALE (07-13): keep GateNet loaded a few more seconds so it can
            # publish gate-referenced metric drone positions (state['gatenet_p']
            # = G1_W - g_w) for the bridge to anchor DPVO scale on a drift-free
            # reference. Unload once the scale locks (bridge sets
            # dpvo_scale_locked) or a VRAM-safety timeout, so DPVO gets the GPU
            # for the crossing. NOTE: brief GateNet+WSL2-DPVO overlap -- if this
            # OOM-kills the process, drop GNSCALE_TMAX or DPVO patches.
            _hold = (os.environ.get('GNSCALE') == '1'
                     and os.environ.get('DPVO_OBSERVE') == '1'
                     and not state.get('dpvo_scale_locked')
                     and time.time() - state.get('go_wall', 0.0)
                         < float(os.environ.get('GNSCALE_TMAX', '12.0')))
            if not _hold:
                if lm is not None:
                    del lm, x, out
                    lm = None
                    torch.cuda.empty_cache()
                    _handoff_wall = time.time()
                    state['gatenet_unloaded'] = True
                    state['gatenet_unloaded_wall'] = _handoff_wall
                    jlog('gatenet_unloaded',
                         go_delay_ms=round(
                             (_handoff_wall - state.get('go_wall', _handoff_wall))
                             * 1000.0, 1))
                    _why = ('scale locked' if state.get('dpvo_scale_locked')
                            else 'timeout' if os.environ.get('GNSCALE') == '1'
                            else 'GPU handoff')
                    print(f'GateNet unloaded (DPVO owns the GPU pre-tick) [{_why}]',
                          flush=True)
                time.sleep(0.05)
                continue
            # else (GNSCALE hold): fall through and keep detecting so the NOFIX
            # block below can publish state['gatenet_p'] for scale cal
        if lm is None:
            # post-tick: reload for the gate-2 servo leg (warm-up cost
            # accepted; the 8 s tick grace covers part of it)
            print('GateNet reloading for gate-2 leg', flush=True)
            lm = OV.load_model(CKPT, CFG, dev)
            dk, sk = OV._multi_decode_knobs(lm.cfg, stride=lm.stride)
            print('GateNet reloaded', flush=True)
        padded = OV.pad_bottom(img, lm.pad_to_h)
        with torch.no_grad():
            x = TG.frames_to_input(torch.from_numpy(padded[None]), dev)
            out = TG._float_output(lm.model(x))
            insts = AS.decode_frame_multi(out, 0, **dk)
        best = None
        for dec in insts:
            ip = AS.solve_instance(dec, image_size=(int(x.shape[-1]), int(x.shape[-2])),
                                   direct_pose_vec=None, **sk)
            if not ip.solved or ip.t is None or ip.low_confidence:
                continue
            z = float(ip.t[2])
            if not (1.5 < z < 45.0):
                continue
            tx = float(ip.t[0])
            key = ((z + 2.0 * abs(tx)) if z < 22.0 else 100.0 + z, -float(dec.center_score))
            if best is None or key < best[0]:
                best = (key, np.asarray(ip.t, float),
                        np.asarray(dec.corner_xy, float),
                        np.asarray(dec.visibility, float)
                        if dec.visibility is not None else None)
        if best is not None:
            rng_pnp = float(np.linalg.norm(best[1]))
            rng_true = max(0.3, (rng_pnp - RANGE_AFF_B) * RANGE_AFF_K)
            best = (best[0], best[1] * (rng_true / rng_pnp), best[2], best[3])
        if best is not None and state['nav_ready']:
            # nav_ready gates the whole chain: frames rendered DURING the sim
            # reset produced a false pin (flight #8, [7.95,-1.77]) that blew up
            # the KF with garbage fixes before takeoff
            g_b = M_BODY_CAM @ best[1]
            r, p = state['roll'], state['pitch']
            best_dt = 1e9
            ATT_ITER = ATT_BUF if f_wall > 0 else []
            for (tw, rb, pb) in reversed(ATT_ITER):
                d = abs(tw - f_wall)
                if d < best_dt:
                    best_dt, r, p = d, rb, pb
                elif tw < f_wall - 0.5:
                    break
            sr, cr = math.sin(r), math.cos(r)
            sp, cp = math.sin(p), math.cos(p)
            Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
            g_lvl = Ry @ (Rx @ g_b)
            state['det_wall'] = time.time()
            # phase-dependent z sanity: ON THE PAD (at rest, attitude exact,
            # obs-z bias absent) apertures read z ~ -0.6, so 2.0 rejects the
            # STACKED HIGH gate (z -4.95 pad-locked attempt 2 and poisoned
            # every fix after). IN FLIGHT the per-run obs-z bias (-0.6..-4.9)
            # puts LEGIT gates past any tight bound (a 2.5 guard starved
            # run 4 to 6 fixes/flight) -> keep the loose 8.0 there.
            _rng0 = float(np.linalg.norm(g_lvl))
            # pad guard scales with range: the course gate at ~28 m carries a
            # range-scaled obs-z bias the flat 2.0 (tuned on the 11 m decoy)
            # would reject
            z_lim = 8.0 if state.get('airborne') else max(2.0, 0.12 * _rng0)
            if abs(g_lvl[2]) > z_lim:
                jlog('obs_insane', ns=ns, g_lvl=g_lvl.round(3).tolist())
                cv2.imwrite(f'{OUT}/frames/{ns}.jpg', img)
                continue
            # MAP-anchored acceptance, huber_area policy (VQ2-POLICY-01,
            # bench-proven on both corpora): (1) range-ratio identity gate --
            # PnP range is size-derived and TRUE, so a candidate whose
            # expected range disagrees >~2x is a different gate or a
            # hallucination, robust to ~10 m of drift; (2) huber soft
            # acceptance -- far misses inflate R instead of being discarded,
            # so the filter can ALWAYS be walked back (run-5 starvation mode
            # structurally gone). POLICY=radius reverts to the legacy gate.
            cyw, syw = math.cos(state['yaw']), math.sin(state['yaw'])
            g_w = np.array([cyw * g_lvl[0] - syw * g_lvl[1],
                            syw * g_lvl[0] + cyw * g_lvl[1], g_lvl[2]])
            with KF_LOCK:
                p_kf = KF.p.copy()
            rng_meas = float(np.linalg.norm(g_lvl))
            if (os.environ.get('NOFIX') == '1' and state.get('airborne')
                    and ticks == 0):
                # PURE-DR CHUTE (07-12 error budget): sim IMU is NOISELESS
                # (at-rest gyro/accel sigma = 0 to 5 decimals) while in-
                # flight detection lateral noise is sigma 0.3-1.4 m per obs
                # -- position fixes at 1.7 Hz were injecting the entire
                # +-1.5 m crossing spread into a near-perfect DR chain.
                # Pre-tick: pad anchor only, no in-flight position fixes.
                # SERVO PASS-THROUGH (07-12, gate-1 1/8 ceiling): every
                # engineered tick in campaign history came from the
                # relative-obs servo -- KF updates stay off, but a
                # plausible dead-ahead gate-1 sighting still feeds the
                # terminal handoff (relative steering is immune to the
                # est-truth gap that blind DR cannot see).
                # GATE-1 IDENTITY (07-14, servo_nt6 frames): the hangar is FULL
                # of station-bay gates; with the servo enabled (no DIRTEST) a
                # drifted drone locks a WRONG bay gate at <8 m and punches at it
                # (nt6 punched in the right-side stalls, parked jet dead ahead).
                # DR is clean now (NOTRIM+EULERFIX) -- only pass through obs
                # whose DR-implied world position matches the pad-locked G1.
                _gseen = p_kf + g_w
                _g1miss = float(np.linalg.norm((_gseen - G1_W)[:2]))
                if (rng_meas < 8.0 and abs(float(g_lvl[1])) < 2.5
                        and _g1miss < float(os.environ.get('G1_IDENT_R', '2.5'))):
                    state['obs'] = g_lvl
                    state['obs_wall'] = time.time()
                    state['tgt'] = g_lvl.copy()
                    with KF_LOCK:
                        # arm the obs-DR punch (07-14): _last_fresh was only set
                        # in the non-NOFIX fix path, so under NOFIX the terminal
                        # punch NEVER fired -- every approach coasted blind past
                        # the plane after the ~4 m detector dropout.
                        state['_last_fresh'] = (float(g_lvl[0]), float(g_lvl[1]),
                                                float(g_lvl[2]), KF.p.copy(),
                                                time.time())
                elif rng_meas < 8.0:
                    jlog('obs_wronggate', ns=ns, rng=round(rng_meas, 1),
                         miss=round(_g1miss, 1))
                if os.environ.get('GNSCALE') == '1':
                    # GATE-REFERENCED METRIC DRONE POSITION (07-13/14): a gate's
                    # world position is fixed (pad-locked G1_W, and G2_W = HIGH_W
                    # + fixed offset, same frame); a detection gives it relative
                    # to the drone (g_w), so drone_world = gate_W - g_w -- a
                    # drift-free metric position the bridge anchors DPVO scale on
                    # (does NOT touch the control KF). In level flight the 20-up
                    # cam mostly sees the FARTHER gate 2, so accept ANY range and
                    # assign to the nearest KNOWN gate by RANGE-RATIO identity
                    # (identical gates -> range disambiguates g1/g2; ratio <
                    # RANGE_RATIO_MIN rejects junk/mis-ID). Tag the gate id so the
                    # bridge never mixes G1 and a coarsely-mapped G2 in one
                    # displacement. A constant gate-map error cancels in the
                    # scale's displacement ratio.
                    _gid, _gwm, _gbest = None, None, RANGE_RATIO_MIN
                    for _gi, _gwc in ((0, G1_W), (1, G2_W)):
                        _re = float(np.linalg.norm((_gwc - p_kf)[:2]))
                        if _re < 0.5:
                            continue
                        _ra = min(rng_meas / _re, _re / rng_meas)
                        if _ra > _gbest:
                            _gbest, _gid, _gwm = _ra, _gi, _gwc
                    if _gwm is not None:
                        _gp = (_gwm - g_w).astype(float)
                        state['gatenet_sample'] = (
                            int(ns), int(_gid),
                            tuple(float(value) for value in _gp))
                        state['gatenet_p'] = _gp
                        state['gatenet_gate'] = _gid
                        state['gatenet_p_wall'] = time.time()
                        # tag the SOURCE frame ns: this detection lags ~1 s, so
                        # the bridge must pair it with the DPVO pose from THIS
                        # frame, not the current one (else ~1 s of motion leaks
                        # into the scale). f_wall is the frame's rx wall time.
                        state['gatenet_ns'] = ns
                        state['gatenet_fwall'] = f_wall
                        jlog('gnscale_anchor', ns=ns, gate=_gid,
                             rng=round(rng_meas, 1), ratio=round(_gbest, 2))
                jlog('obs_nofix', ns=ns, rng=round(rng_meas, 1))
                cv2.imwrite(f'{OUT}/frames/{ns}.jpg', img)
                continue
            _rng_cap = 14.0
            if os.environ.get('G2TEST') == '1':
                # G2TEST (07-12, run 5): the course gates are IDENTICAL, so
                # constellation identity cannot tell gate 2 from gate 1.
                # Gate-2 sightings at rng 10-13 m identity-passed as G1 and
                # huber-dragged the KF backward (miss 3.4-4.9 accepted).
                # Every legitimate fix on this route is < 8 m: gate 1 from
                # the pad is 6.9, gate 2 from the gate-1 crossing is 7.4.
                _rng_cap = 8.0
            if ARCHTEST and rng_meas > _rng_cap:
                # the verification course is <= 14 m end to end; far solves
                # are hangar junk, and huber's inflate-don't-discard walked
                # the KF into a pillar on them (arch19: miss 8-21 accepted
                # at ranges 22-39 m)
                jlog('obs_far_archtest', ns=ns, rng=round(rng_meas, 1))
                cv2.imwrite(f'{OUT}/frames/{ns}.jpg', img)
                continue
            if (os.environ.get('G2TEST') == '1' and ticks == 0
                    and os.environ.get('DIRTEST')):
                # BLIND-MODE ONLY (DIRTEST set): pre-tick blackout past the
                # crossing zone -- gate 2 is also < 8 m from the late chute
                # and indistinguishable from gate 1, and near-range fixes
                # ride a huge identity tolerance. LATCHED (run 7: a soft
                # boundary chattered). In servo mode (no DIRTEST) the
                # approach handoff needs these obs: 07-12 frames prove the
                # detection bearing is unbiased now, so the servo -- the
                # machinery behind every historical tick -- steers on them.
                if p_kf[0] > 5.5:
                    state['_g2_blackout'] = True
                if state.get('_g2_blackout'):
                    jlog('obs_blackout_g2test', ns=ns, rng=round(rng_meas, 1))
                    continue
            ident_full = False
            miss, match_g = 1e9, None
            # pre-tick under G2TEST the moved-in G2_W anchor (12 m, was
            # parked at 31.6) becomes range-plausible from the mid-chute and
            # steals matches from G1 (runs 5-6: est dragged +y/backward off
            # the ticking line). Gate 2 only becomes a legal anchor after
            # tick 1.
            _cands = ((G1_W, RIB_W) if (os.environ.get('G2TEST') == '1'
                                        and ticks == 0)
                      else (G1_W, RIB_W, G2_W, DECOY_W))
            for gw_map in _cands:
                if OBS_POLICY in ('huber_area', 'huber'):
                    rng_exp = float(np.linalg.norm((gw_map - p_kf)[:2]))
                    ratio = (min(rng_meas / max(rng_exp, 0.1),
                                 max(rng_exp, 0.1) / rng_meas)
                             if rng_meas > 0.1 else 0.0)
                    if ratio < RANGE_RATIO_MIN:
                        continue
                m_ = float(np.linalg.norm((g_w - (gw_map - p_kf))[:2]))
                if m_ < miss:
                    miss, match_g = m_, gw_map
            # match HYSTERESIS: if the previous match is still plausible, keep
            # it -- alternating matches whipsawed the position fixes between
            # two anchors and exploded the velocity (flights #38/#39)
            prev_m = state.get('_match_prev')
            if (match_g is not None and prev_m is not None
                    and time.time() - prev_m[1] < 2.0 and match_g is not prev_m[0]):
                m_prev = float(np.linalg.norm((g_w - (prev_m[0] - p_kf))[:2]))
                if m_prev < ACCEPT_R + 1.5:
                    miss, match_g = m_prev, prev_m[0]
            # constellation identity (VQ2-POLICY: 5c0a230 bench): obs
            # matched to G1 must positively fit the G1 corner map; in
            # strict course-context, non-G1-identified obs are junk unless
            # a genuinely far G2 sighting. Kills truss/fixture solves the
            # range gate can't (run-6 poison).
            _cmap = (G2RIB_C if (match_g is G1_W or match_g is RIB_W) else
                     DECOY_C if match_g is DECOY_W else None)
            if IDENT_ON and _cmap is not None and best[3] is not None:
                r_, p_, y_ = state['roll'], state['pitch'], state['yaw']
                sr_, cr_ = math.sin(r_), math.cos(r_)
                sp_, cp_ = math.sin(p_), math.cos(p_)
                cy_, sy_ = math.cos(y_), math.sin(y_)
                Ry_ = np.array([[cp_, 0, sp_], [0, 1, 0], [-sp_, 0, cp_]])
                Rx_ = np.array([[1, 0, 0], [0, cr_, -sr_], [0, sr_, cr_]])
                Rz_ = np.array([[cy_, -sy_, 0], [sy_, cy_, 0], [0, 0, 1]])
                R_wc_ = (Rz_ @ Ry_ @ Rx_) @ M_BODY_CAM
                rel_ = (_cmap - p_kf) @ R_wc_
                n_match = 0
                for k_ in range(min(len(best[2]), len(_cmap))):
                    if best[3][k_] < 0.5 or rel_[k_, 2] <= 0.2:
                        continue
                    uv_ = np.array([rel_[k_, 0] / rel_[k_, 2] * CAM_FX + CAM_CX,
                                    rel_[k_, 1] / rel_[k_, 2] * CAM_FY + CAM_CY])
                    tol_ = 60.0 * 11.0 / max(rng_meas, 3.0)  # angular-constant
                    if np.linalg.norm(uv_ - best[2][k_]) < tol_:
                        n_match += 1
                is_g1 = n_match >= 2
                if is_g1:
                    ident_full = True   # content-confirmed: full-weight fix below
                # pre-first-tick: identity-or-nothing. Run-11 wreck: junk
                # obs at rng 15-28 sailed through the far_ok>18 loophole and
                # repeated soft pulls (rs to 19) dragged y to 21 m. Nothing
                # legitimate exists on this leg except identified G1; the
                # blind dropout zone rides DR (holdout: 0.3-0.5 m / 3-8 s).
                if ticks == 0:
                    bad = not is_g1
                else:
                    bad = ((match_g is G1_W and not is_g1) or
                           (match_g is G2_W and not is_g1 and rng_meas < 18.0))
                if bad and ARCHTEST and rng_meas < 7.5 and abs(float(g_lvl[1])) < 2.5:
                    # post-arch corridor: the only detectable gate inside
                    # 7.5 m forward IS the target (gate 2 sits ~8.5 m out).
                    # arch22/24 missed because identity kept failing here
                    # and the handoff starved; take the obs soft-weighted
                    bad = False
                if bad:
                    dists_ = []
                    for k_ in range(min(len(best[2]), len(_cmap))):
                        if rel_[k_, 2] <= 0.2:
                            dists_.append(None); continue
                        uv_ = np.array([rel_[k_, 0] / rel_[k_, 2] * CAM_FX + CAM_CX,
                                        rel_[k_, 1] / rel_[k_, 2] * CAM_FY + CAM_CY])
                        dists_.append(round(float(np.linalg.norm(uv_ - best[2][k_])), 1))
                    jlog('obs_ident_fail', ns=ns, n_match=n_match,
                         miss=round(miss, 2), rng=round(rng_meas, 1),
                         p=p_kf.round(2).tolist(),
                         rpy=[round(math.degrees(r_), 1), round(math.degrees(p_), 1),
                              round(math.degrees(y_), 1)],
                         vis=[round(float(v), 2) for v in best[3]],
                         dists=dists_)
                    cv2.imwrite(f'{OUT}/frames/{ns}.jpg', img)
                    continue
            if OBS_POLICY in ('huber_area', 'huber'):
                accepted = match_g is not None
                r_scale = 1.0 if miss <= HUBER_DELTA else miss / HUBER_DELTA
                if ident_full:
                    # identity-confirmed G1: huber's miss-based distrust is
                    # estimate-referenced and soft-pedals the very fixes that
                    # rescue a drifted filter (run-6/8 lateral miss). Content
                    # beats miss: full weight.
                    r_scale = 1.0
            else:  # legacy radius gate
                accepted = miss <= ACCEPT_R
                r_scale = 1.0
            if accepted:
                state['_match_prev'] = (match_g, time.time())
            if not accepted:
                jlog('obs_ident' if match_g is None else 'obs_offmap',
                     ns=ns, g_w=g_w.round(3).tolist(), miss=round(miss, 2))
            else:
                g_w_upd = g_w.copy()
                with KF_LOCK:
                    if os.environ.get('OBSZ', '1') != '1':
                        # legacy xy-only quarantine (OLD sim's -0.6..-4.9
                        # per-run obs-z bias): zero the z innovation
                        g_w_upd[2] = float(match_g[2]) - KF.p[2]
                    else:
                        # SOFT z: full-weight z fixes sawtooth against the
                        # z DR drift; damp the innovation instead -- bounds
                        # the drift without the fight
                        z_full = float(match_g[2]) - KF.p[2]
                        g_w_upd[2] = z_full + 0.4 * (g_w_upd[2] - z_full)
                    # updated sim (07-09): obs-z reads sane (canary implied
                    # pad-hover height 0.3 m) AND its accelerometer
                    # under-reads specific force under thrust ~4%, so z
                    # MUST be vision-bounded or it runs away (est z +78 m
                    # while hovering at the pad)
                    ok = KF.update_position(match_g - g_w_upd,
                                            rng=rng_meas, r_scale=r_scale)
                    nis = getattr(KF, 'last_nis', None)
                    p_now = KF.p.round(2).tolist(); v_now = KF.v.round(2).tolist()
                state['fix_count'] = state.get('fix_count', 0) + 1
                # viewer attribution: mark each accepted fix so the trail's
                # snaps read as "vision fix here", not IMU noise
                state['_last_fix'] = (list(p_now), round(miss, 2), time.time())
                jlog('kf_upd', ok=ok, miss=round(miss, 2),
                     r_scale=round(r_scale, 2),
                     nis=round(nis, 2) if nis is not None else None,
                     p=p_now, v=v_now)
                # VISION VELOCITY: the gate is static, so drone velocity =
                # -d(gate offset)/dt between consecutive detections. This is
                # the only absolute velocity source (no ODOMETRY, no ZUPT) --
                # without it the KF over-integrates metres ahead at speed and
                # every blind final approach misses the aperture (flights
                # #31-#34, confirmed visually by Alex).
                prev = state.get('_vv_prev')
                t_now = time.time()
                if (prev is not None and 0.05 < t_now - prev[1] < 1.2 and prev[2] is match_g
                        and np.linalg.norm((g_w - prev[0])[:2]) < 0.7):
                    # consistency gate: consecutive obs must be the SAME physical
                    # gate (two neighbors matched to one map slot injected an
                    # 8 m/s phantom velocity, flight #38 runaway)
                    v_meas = -(g_w - prev[0]) / (t_now - prev[1])
                    v_meas[2] = 0.0     # z channel corrupt; xy only
                    if np.linalg.norm(v_meas[:2]) < 4.0:
                        with KF_LOCK:
                            KF.update_velocity(np.array([v_meas[0], v_meas[1], KF.v[2]]),
                                               sigma=1.5)
                state['_vv_prev'] = (g_w.copy(), t_now, match_g)
                if match_g is DECOY_W:
                    # landmark fix only: never target the decoy
                    cv2.imwrite(f'{OUT}/frames/{ns}.jpg', img)
                    continue
                state['obs'] = g_lvl
                state['obs_wall'] = time.time()
                state['tgt'] = g_lvl.copy()
                with KF_LOCK:
                    state['_last_fresh'] = (float(g_lvl[0]), float(g_lvl[1]),
                                            float(g_lvl[2]), KF.p.copy(), time.time())
                jlog('obs', ns=ns, score=best[0], t_cam=best[1].round(3).tolist(),
                     g_lvl=g_lvl.round(3).tolist())
        cv2.imwrite(f'{OUT}/frames/{ns}.jpg', img)

def fastgate_loop():
    """FASTGATE (07-14): millisecond classical aperture detector (fastgate.py,
    orange frame + dark hole -> HOLE center bearing, 1.8 ms/frame, holds the
    gate through the terminal band where GateNet drops out at ~4 m). Publishes
    state['fg_bear'] = (lat/fwd, dwn/fwd) level-frame bearing at ~30 Hz for the
    approach servo + punch to steer closed-loop through the crossing. Range is
    glow-corrupted at close range -- bearing only."""
    try:
        import fastgate as FG
    except Exception as e:
        print(f'fastgate unavailable: {e}', flush=True)
        return
    _FLOW = None
    if os.environ.get('FLOWOBS') == '1':
        # OBSERVE-ONLY ground-plane flow velocity (07-17): vq2/flow_vel.py
        # fed at frame rate; publishes state['flow_v'] + jlog('flow') next to
        # the KF velocity for divergence analysis. NO control authority and
        # NO KF fusion in this mode -- wire update_velocity only after the
        # observe flights certify it. (Piggybacks this loop: FLOWOBS needs
        # FASTGATE=1.)
        try:
            from flow_vel import FlowVelocity
            _FLOW = FlowVelocity()
            print('flow observe up (FLOWOBS=1, observe-only)', flush=True)
        except Exception as e:
            print(f'flow_vel unavailable: {e}', flush=True)
    last_ns = 0
    print('fastgate loop up', flush=True)
    while not state['stop']:
        img, ns = state['frame'], state['frame_ns']
        if img is None or ns == last_ns:
            time.sleep(0.005)
            continue
        last_ns = ns
        if os.environ.get('LINEFOLLOW') == '1':
            # E3 cyan-line detector (07-15 handoff, re-applied 07-17 after
            # the DPVO rewrite wiped it): the sim paints the racing line
            # through every gate; its image ROW is a direct height-over-
            # course observation (est-z under-reads climbs ~40% -- THE
            # gate-2 ceiling-drift killer). Lower 45% of frame = floor.
            try:
                _h2, _w2 = img.shape[:2]
                _roi = img[int(_h2 * 0.55):, :]
                _b_ = _roi[:, :, 0].astype('int16')
                _g_ = _roi[:, :, 1].astype('int16')
                _r_ = _roi[:, :, 2].astype('int16')
                _cy = ((_b_ > 120) & (_g_ > 100) & (_b_ - _r_ > 40)
                       & (_g_ - _r_ > 20))
                _n_ = int(_cy.sum())
                if _n_ > int(os.environ.get('LINE_MIN_PX', '150')):
                    _ys, _xs = np.nonzero(_cy)
                    state['line_off'] = (float(_xs.mean()) - _w2 / 2.0) / _w2
                    state['line_row'] = (int(_h2 * 0.55)
                                         + float(_ys.mean())) / _h2
                    state['line_n'] = _n_
                    # LINE HEAD (07-19, test15 postmortem): centroid-only
                    # guidance has no lookahead -- the drone parked ON the
                    # line (row 0.95+, count 10k->170) with nothing driving
                    # progress along it. The head = mean x of the TOPMOST
                    # band of line pixels (the farthest visible piece) =
                    # where the line GOES; yaw steers at the head, roll
                    # centers the centroid.
                    _yt = float(_ys.min())
                    _band = _ys <= _yt + 0.3 * max(1.0,
                                                   float(_ys.max()) - _yt)
                    if int(_band.sum()) >= 40:
                        state['line_head_off'] = \
                            (float(_xs[_band].mean()) - _w2 / 2.0) / _w2
                    else:
                        state['line_head_off'] = state['line_off']
                    state['line_wall'] = time.time()
            except Exception:
                pass
        if _FLOW is not None:
            try:
                _fl_res = _FLOW.process(
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), ns * 1e-9,
                    (state['roll'], state['pitch'], state['yaw']),
                    float(0.15 - KF.p[2]))
                if _fl_res is not None:
                    _fv, _fsig, _finl, _ftr = _fl_res
                    state['flow_v'] = _fv
                    state['flow_wall'] = time.time()
                    if time.time() - state.get('_flow_log', 0) > 0.3:
                        state['_flow_log'] = time.time()
                        jlog('flow', v=[round(float(x), 3) for x in _fv],
                             sig=round(float(_fsig), 3), n=int(_finl),
                             tr=int(_ftr),
                             kfv=[round(float(x), 3) for x in KF.v],
                             z=round(float(KF.p[2]), 2))
            except Exception:
                pass
        try:
            dets = FG.detect(img)
        except Exception:
            continue
        if not dets:
            continue
        r_, p_ = state['roll'], state['pitch']
        sr, cr = math.sin(r_), math.cos(r_)
        sp, cp = math.sin(p_), math.cos(p_)
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        # associate: prefer the det nearest in bearing to the current GateNet
        # obs (identity-confirmed); else the most-centered forward det
        ref = state.get('obs')
        ref_ok = ref is not None and time.time() - state.get('obs_wall', 0) < 3.0 \
            and float(ref[0]) > 0.3
        # TEMPORAL CONTINUITY (07-14, fg3 frames): when the chute gate's hole
        # clipped out of frame at 5.2 m, the biggest-hole rule silently
        # re-locked onto the stacked gate-2 pair and the pursuit flew a perfect
        # approach to the WRONG gate. Once locked, follow the SAME hole
        # (nearest to the last bearing); size/centeredness only for the
        # initial acquisition.
        last_b = state.get('fg_bear')
        last_ok = last_b is not None and time.time() - state.get('fg_wall', 0) < 0.6
        best = None
        for d in dets:
            t_cam, hole_w = d[0], d[1]
            # CLOSE-GATES-ONLY (07-14, Alex): only holes >= FG_MIN_W px may
            # influence trajectory -- far bay/course gates (<40 px) can never
            # steal the track. With gate-to-gate pursuit this makes the course
            # fly itself in proximity order.
            if hole_w < float(os.environ.get('FG_MIN_W', '40')):
                continue
            if len(d) > 3 and d[3]:
                # CLIPPED HOLE (fg4): a half-visible hole's remnant center is
                # biased toward the visible side -- at 3.5 m the jump (0.28 rad)
                # slid under the continuity gate and the pursuit chased the
                # phantom hard-right. Never steer on clipped detections; the
                # fallback coasts straight on the already-nulled line.
                continue
            g = Ry @ (Rx @ (M_BODY_CAM @ t_cam))
            if g[0] < 0.3:
                continue
            bear = (float(g[1] / g[0]), float(g[2] / g[0]))
            if last_ok:
                cost = abs(bear[0] - last_b[0]) + abs(bear[1] - last_b[1])
                if cost > 0.15:          # continuity gate: same hole only
                    continue
                # THROUGH-HOLE ALIAS (07-15, fg22 frames): a far gate visible
                # THROUGH the tracked aperture sits at the SAME bearing, so
                # when the near hole clips out, bearing continuity hands the
                # track to it (w 94 -> 44 in ONE frame) and the pursuit sails
                # past the near gate aiming at the far one. A real hole cannot
                # halve between frames at 30 Hz -- reject size discontinuities
                # and let the coast/punch logic own the crossing.
                if hole_w < 0.55 * state.get('fg_w', 0.0):
                    continue
            else:
                # ACQUISITION BY SIZE (fg6: 'most-centered' acquired a 9 px FAR
                # bay gate and continuity faithfully tracked the wrong target).
                # At start range (~6 m) the chute hole is ~80 px; bay gates are
                # <=20 px. Only a big hole can start a track.
                if hole_w < float(os.environ.get('FG_ACQ_W', '40')):
                    continue
                cost = -hole_w                       # largest qualifying hole
            if best is None or cost < best[0]:
                best = (cost, bear, hole_w)
        if best is not None:
            state['fg_bear'] = best[1]
            state['fg_wall'] = time.time()
            state['fg_w'] = best[2]
            if time.time() - state.get('_fg_log', 0) > 0.4:
                state['_fg_log'] = time.time()
                jlog('fg', bear=[round(best[1][0], 3), round(best[1][1], 3)],
                     w=round(best[2], 0))

def send_rate(rr, pr, yr, thr):
    state['last_thr'] = thr
    state['last_cmd'] = (rr, pr, yr, thr)
    m.mav.set_attitude_target_send(
        int(time.time()*1000) & 0xFFFFFFFF, m.target_system, m.target_component,
        mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE,
        [1.0, 0, 0, 0], rr, pr, yr, thr)
    if REC_DIR:
        # command log for flight_report control metrics (saturation duty,
        # hover balance, cmd-vs-achieved rate transfer). ~40 B x 50 Hz.
        rec_cmd.write(json.dumps({'t': time.time(), 'rr': round(rr, 4),
                                  'pr': round(pr, 4), 'yr': round(yr, 4),
                                  'thr': round(thr, 4)}) + '\n')

def level_cmd(vx_ref=0.0, vy_ref=0.0, vz_ref=0.0, thr_base=HOVER, pitch_bias=0.0, yr=0.0,
              roll_bias=0.0, att=False):
    # att=True: PURE-ATTITUDE MODE (07-15) -- references come straight from
    # vision, NO est-velocity feedback (the est banks ~1 m/s offsets per
    # maneuver and the velocity loop converts them into real wrong motion:
    # fg33 flew backward-up while est read "vx +0.9, satisfied"). In att
    # mode vz_ref is a THROTTLE DELTA (same +/-0.06 clamp); roll_bias/
    # pitch_bias ARE the attitude references. Same hard clamps either way.
    if att:
        roll_ref = max(-0.25, min(0.25, roll_bias))
        pitch_ref = max(-0.35, min(0.35, pitch_bias))
    else:
        roll_ref = max(-0.25, min(0.25, -K_V * (state['vy_b'] - vy_ref) + roll_bias))
        pitch_ref = max(-0.35, min(0.35, K_V * (state['vx_b'] - vx_ref) + pitch_bias))
    rr = SIGN_R * (KP * (roll_ref - state['roll'])) / RATE_GAIN
    pr = SIGN_P * (KP * (pitch_ref - state['pitch'])) / RATE_GAIN
    # RATE DISCIPLINE (07-06 collapse root cause): commanded transients hit
    # 356 deg/s measured; HIGHRES_IMU cadence is bursty (unique samples at 7/14/28 ms), and
    # gyro integration across gaps at those rates accrues 1-3 deg PERMANENT
    # attitude error -> gravity leak -> estimate runaway. The attitude chain
    # is proven clean below ~1 rad/s actual; RATE_GAIN 1.93 means +-0.6
    # commanded ~ +-1.2 actual worst case. Do not raise without re-deriving
    # the gap-error budget.
    RATE_MAX = float(os.environ.get('RATE_MAX', '0.6'))
    rr = max(-RATE_MAX, min(RATE_MAX, rr)); pr = max(-RATE_MAX, min(RATE_MAX, pr))
    if att:
        dthr = max(-0.06, min(0.06, vz_ref))
    else:
        dthr = max(-0.06, min(0.06, 0.10 * (vz_ref - state['vz_up'])))
    send_rate(rr, pr, yr, max(0.05, min(0.6, thr_base + dthr)))

def tilt(): return math.sqrt(state['roll']**2 + state['pitch']**2)

def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi

def land(reason):
    print('LAND:', reason, flush=True); jlog('land', reason=reason)
    t0 = time.time()
    while time.time() - t0 < 5.0:
        level_cmd(0, 0, -0.7); time.sleep(1/CMD_HZ)
    while time.time() - t0 < 7.0:
        send_rate(0, 0, 0, 0.10); time.sleep(1/CMD_HZ)
    m.mav.command_long_send(m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0, 0, 0, 0, 0, 0, 0)
    if REC_DIR:
        # OUT/log.jsonl is shared across runs and each launch truncates it;
        # snapshot into the per-run corpus (arch17's tick-run route log was
        # lost to the next launch)
        try:
            log_f.flush()
            import shutil as _sh
            _sh.copy(OUT + '/log.jsonl', REC_DIR + '/livelog.jsonl')
        except Exception as e:
            print('livelog snapshot failed:', e, flush=True)

_threads = [rx_loop, cam_loop, det_loop]
if os.environ.get('FASTGATE') == '1':
    _threads.append(fastgate_loop)
for th in _threads:
    threading.Thread(target=th, daemon=True).start()
time.sleep(2.0)

print('sim HARD reset (param1=1: restarts the RACE + countdown)', flush=True)
m.mav.command_long_send(m.target_system, m.target_component, 31000, 0, 1, 0, 0, 0, 0, 0, 0)

select_dpvo_route_position = None
_dpvo_route_started = False
if (os.environ.get('DPVO') == '1'
        and os.environ.get('DPVO_ROUTE') == '1'):
    if os.environ.get('DPVO_BRIDGE') != '1':
        raise RuntimeError('DPVO_ROUTE requires the low-memory WSL2 bridge')
    from dpvo_odom_bridge import DpvoOdom
    from dpvo_route import select_route_position
    select_dpvo_route_position = select_route_position
    print('DPVO: prewarming independent WSL2 route bridge', flush=True)
    DpvoOdom(state, KF, KF_LOCK, M_BODY_CAM, jlog=jlog).start()
    _dpvo_route_started = True
    print('DPVO route prewarm thread started', flush=True)

time.sleep(6.0)
state['collision'] = None
time.sleep(1.0)
acc = state['acc']
state['roll'] = math.atan2(acc[1], -acc[2])
state['pitch'] = math.atan2(acc[0], math.sqrt(acc[1]**2 + acc[2]**2))
state['vx_b'] = state['vy_b'] = state['vz_up'] = 0.0
with KF_LOCK:
    KF.reset_at_rest()
state['obs'] = None
state['tgt'] = None
state['landmark_w'] = None
state['obs_wall'] = 0.0
state['next_gate_w'] = HIGH_W.copy()
state['nav_ready'] = True          # reset done -- detections may pin/update from here
gate0_idx = state['gate_idx']
ticks = 0
print(f'post-reset pitch {math.degrees(state["pitch"]):.1f} gate_idx {gate0_idx}', flush=True)

# PAD ACQUISITION: lock gate 1 from the spawn tilt before takeoff
pad_t0 = time.time()
while (state['obs'] is None and state.get('fix_count', 0) < 3
       and time.time() - pad_t0 < 15.0):
    time.sleep(0.1)
if state['obs'] is None and state.get('fix_count', 0) < 3:
    print('NO PAD ACQUISITION -- aborting before takeoff', flush=True)
    sys.exit(1)
if state['obs'] is not None:
    print(f'pad lock: g_lvl {state["obs"].round(2)} range {np.linalg.norm(state["obs"]):.1f} m', flush=True)
else:
    print(f'pad vision alive via landmark fixes (n={state.get("fix_count",0)}) -- course gate not visible from pad', flush=True)

# PER-FLIGHT ANCHOR (07-07): the pad measurement is bias-free (at rest,
# attitude exact) -- anchor the spline aim AND the obs-matching anchor to
# it instead of canned constants. Aperture = measured origin minus the
# ORIGIN_OFFSET along the course axis; z stays map-verified -1.3.
_pad = state['obs'].copy() if state['obs'] is not None else None
_pad_w = (np.array([float(_pad[0]), float(_pad[1]), -1.3])
          if _pad is not None else None)
if _pad_w is not None and np.linalg.norm((_pad_w - G1_W)[:2]) < 4.0:
    # pad measurement confirms the mapped course gate: anchor to it
    G1_W = _pad_w
    G1_AP = G1_W - ORIGIN_OFFSET * N1
    # JUDGE-CALIBRATED AIM BIAS (07-11): the accidental DIRTEST=left tick
    # proved the true aperture sits ~2 m left (-y) of the pad-locked
    # bearing (PnP origin displaced along the gate's rotated normal).
    # Applied to the anchor so route, servo target, and coast all inherit.
    G1_AP[1] += float(os.environ.get('AIMBIAS_Y', '0.0'))
    # AIMBIAS_Z (07-12 batch6): under NOFIX the z channel is pure DR too,
    # and it drifts ~+0.5 m up through the takeoff climb -- three of six
    # crossings clipped the TOP bar (frames: yellow warning stripe).
    # Positive value = aim lower (FRD z down).
    G1_AP[2] += float(os.environ.get('AIMBIAS_Z', '0.0'))
    HIGH_W = G1_AP.copy()
    GATES_W[0] = HIGH_W
    if os.environ.get('G2TEST') == '1':
        # keep the tick-handler aim/z targets on the SAME gate-2 the
        # G2TEST spline flies (delta from the ticking aperture;
        # MAP_JSON supplies the delta when set)
        G2_W = HIGH_W + (G2_DELTA if G2_DELTA is not None
                         else np.array([5.14, 5.26, 0.0]))
        GATES_W[1] = GATES_W[2] = G2_W.copy()
    TRAJ, S_GATES = build_traj(HIGH_W, G1_W, G2_W)
    state['next_gate_w'] = HIGH_W.copy()
    print(f'spline anchored to pad lock: aperture {G1_AP.round(2)} origin {G1_W.round(2)}', flush=True)
    if _rr is not None:
        try:
            _rr.log('world/gate', _rr.Points3D([G1_AP.tolist()], radii=0.25,
                                               colors=[255, 140, 0],
                                               labels=['GATE (pad lock)']), static=True)
        except Exception:
            pass
else:
    print('no course-gate pad obs -- flying the map', flush=True)

# ACTUATION IS FROZEN UNTIL RACE GO (~8 s race clock): commands sent
# pre-GO are clamped and release violently at the green. The 4-tick-era
# runs cleared GO by lucky timing; post-07-08 relaunch timing shifted
# and every takeoff hit the clamp (misdiagnosed as a vehicle update).
_go_t0 = time.time()
while state.get('race_ms', 0) < 8500 and time.time() - _go_t0 < 40.0:
    time.sleep(0.05)
if state.get('race_ms', 0) < 8500:
    print('NO RACE CLOCK after 40 s (rx dead or race not armed) -- aborting', flush=True)
    sys.exit(1)
print('race GO (clock %.1f s)' % (state.get('race_ms', 0) / 1e3), flush=True)
state['go_wall'] = time.time()   # authoritative GPU-handoff clock
state['go_passed'] = True   # pad lock done; det thread may release the GPU
m.mav.command_long_send(m.target_system, m.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
time.sleep(0.5)
print('armed', flush=True)

if os.environ.get('DPVO') == '1' and not _dpvo_route_started:
    # Legacy DPVO modes keep their established post-arm startup. Route mode
    # was started during the hard-reset settle so it can be ready before GO.
    if os.environ.get('DPVO_BRIDGE') == '1':
        from dpvo_odom_bridge import DpvoOdom
        print('DPVO: using WSL2 bridge', flush=True)
    else:
        from dpvo_odom import DpvoOdom
    DpvoOdom(state, KF, KF_LOCK, M_BODY_CAM, jlog=jlog).start()
    print('DPVO odometry thread started', flush=True)
if os.environ.get('THRPROBE') == '1':
    # Thrust-curve probe (post sim-update the old HOVER=0.2675 produces
    # ~2-4x the expected climb): step throttle at/near ground, log the
    # accel response per step via jlog + the RECORD stream, then disarm.
    for thr_step in (0.08, 0.12, 0.16, 0.20, 0.24):
        t0 = time.time()
        while time.time() - t0 < 2.0:
            send_rate(0, 0, 0, thr_step)
            time.sleep(1/CMD_HZ)
        jlog('thrprobe', thr=thr_step,
             az=round(float(state['acc'][2]), 3),
             vz=round(float(state['vz_up']), 3))
        print(f'THRPROBE thr={thr_step:.2f} az={state["acc"][2]:.2f} vz={state["vz_up"]:.2f}',
              flush=True)
    for _ in range(int(2.0 * CMD_HZ)):   # settle down before disarm
        send_rate(0, 0, 0, 0.05); time.sleep(1/CMD_HZ)
    land('thrust probe complete')
    sys.exit(0)

VEL_MASK = (mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE)

def send_vel(vx, vy, vz):
    # sim-internal velocity controller (PyAIPilotExample-v2 documented mode)
    m.mav.set_position_target_local_ned_send(
        int(time.time() * 1000) & 0xFFFFFFFF, m.target_system,
        m.target_component, mavutil.mavlink.MAV_FRAME_LOCAL_NED, VEL_MASK,
        0, 0, 0, vx, vy, vz, 0, 0, 0, 0, 0)

if os.environ.get('VELPROBE') == '1':
    # velocity-mode probe: does the sim's internal controller fly clean
    # velocity commands, and how does its NED map to the spawn frame?
    def _vhold(dur, vx, vy, vz):
        t0 = time.time()
        while time.time() - t0 < dur:
            send_vel(vx, vy, vz)
            time.sleep(1/CMD_HZ)
    # actuation is FROZEN until race GO (~8 s race clock): both v1/v2
    # probes sat inert through their first segments, then executed the
    # in-flight command violently at GO. Wait for the green.
    while state.get('race_ms', 0) < 8500:
        time.sleep(0.05)
    jlog('velprobe', seg='go', race_ms=state.get('race_ms'))
    print('race GO (clock %.1f s) -- probing' % (state.get('race_ms', 0) / 1e3), flush=True)
    # spool window: the controller engages ~6 s after the first setpoint
    # (constant across three probes); prime with zero-velocity setpoints
    jlog('velprobe', seg='prime'); _vhold(7.0, 0.0, 0.0, 0.0)
    jlog('velprobe', seg='up');    _vhold(2.0, 0.0, 0.0, -0.4)
    jlog('velprobe', seg='hold');  _vhold(1.0, 0.0, 0.0, 0.0)
    jlog('velprobe', seg='px');    _vhold(4.0, 0.5, 0.0, 0.0)
    jlog('velprobe', seg='hold2'); _vhold(1.0, 0.0, 0.0, 0.0)
    land('velocity probe complete')
    sys.exit(0)

if os.environ.get('RATEPROBE') == '1':
    # rate-command deadband probe: hover-scale corrections (~0.03 rad/s)
    # produce no motion while EXCITE pulses (>=0.25) all worked. Step the
    # pitch-rate command upward and find where the gyro responds.
    while state.get('race_ms', 0) < 8500:
        time.sleep(0.05)
    _t0 = time.time()
    while time.time() - _t0 < 1.2:
        send_rate(0, 0, 0, 0.16); time.sleep(1/CMD_HZ)   # lift
    _t0 = time.time()
    while time.time() - _t0 < 2.0:
        level_cmd(0, 0, 0); time.sleep(1/CMD_HZ)          # settle
    for amp in (0.02, 0.04, 0.08, 0.12, 0.16, 0.25):
        jlog('rateprobe', amp=amp)
        _t0 = time.time()
        while time.time() - _t0 < 0.8:
            send_rate(0, amp, 0, HOVER); time.sleep(1/CMD_HZ)
        _t0 = time.time()
        while time.time() - _t0 < 1.5:
            level_cmd(0, 0, 0); time.sleep(1/CMD_HZ)
    land('rate probe complete')
    sys.exit(0)

if os.environ.get('EXCITE') == '1':
    # SysID excitation flight for the post-update vehicle (VQ1 playbook:
    # excite from steady state; fit offline, never in-air). Fine throttle
    # ladder around hover, then per-axis rate pulses with level
    # recoveries; everything lands in the RECORD stream for the fit.
    def _hold(dur, thr, rr=0.0, pr=0.0, yr=0.0):
        t0 = time.time()
        while time.time() - t0 < dur:
            send_rate(rr, pr, yr, thr)
            time.sleep(1/CMD_HZ)

    def _recover(dur=1.2):
        t0 = time.time()
        while time.time() - t0 < dur:
            level_cmd(0, 0, 0)
            time.sleep(1/CMD_HZ)

    _hold(1.2, 0.16)          # lift to ~1 m
    _recover(2.0)
    for thr in (0.11, 0.13, 0.15, 0.12, 0.14, 0.10):
        jlog('excite', what='thr', val=thr)
        _hold(1.2, thr)
        _recover(1.0)
    for kw in ('rr', 'pr', 'yr'):
        for amp in (0.25, -0.25, 0.45, -0.45):
            if tilt() > 0.5:
                break
            jlog('excite', what=kw, val=amp)
            _hold(0.4, HOVER, **{kw: amp})
            _recover(1.4)
    land('excitation complete')
    sys.exit(0)

def guards(phase):
    tilt_lim = TILT_ABORT * (1.6 if phase == 'punch' else 1.0)
    if tilt() > tilt_lim: return f'tilt abort ({phase})'
    c = state['collision']
    imp_lim = 4.0 if phase == 'punch' else 2.5
    if c and c.get('threat_level', 0) >= 2 and c.get('horizontal_minimum_delta', 0) > imp_lim:
        return f'collision ({phase}): imp {c.get("horizontal_minimum_delta"):.1f}'
    # ATTMODE flies on vision (range-rate governor bounds REAL speed); the
    # est velocity it ignores drifts to fiction in ~10-15 s and was falsely
    # killing healthy pursuits (fg38 mid-punch, fg39 mid-approach). Keep a
    # loose sanity ceiling there; full strictness everywhere else.
    _vlim = 20.0 if (os.environ.get('ATTMODE') == '1'
                     and phase in ('fgpursuit', 'punch')) else 8.0
    if (os.environ.get('MF_DR') == '1'
            and phase in ('fgpursuit', 'punch')):
        _vlim = 1e9   # kinematic DR: est velocity is unused fantasy (fg61:
                      # the 20 m/s check killed a healthy staged approach)
    if abs(state['vx_b']) > _vlim or abs(state['vy_b']) > _vlim:
        return f'velocity runaway ({phase})'
    return None

_viz_static()
aborted = None
# rotate level at LOW thrust, then climb gently
t0 = time.time()
while time.time() - t0 < 0.7:
    level_cmd(0, 0, 0, thr_base=0.10); time.sleep(1/CMD_HZ)
t0 = time.time()
# CLIMB_S (07-15, fg15/17/19 frames): 1.6 s @ 1 m/s + the 0.55 m takeoff
# displacement parks the drone at 2.5-3.5 m TRUTH (est-z under-reads it as
# ~1.3) -- far above the 1.35 m aperture line, and the throttle channel
# (dthr +/-0.06 on a drifting est-vz) cannot descend out of it. Start AT
# gate height instead of descending to it.
_climb_s = float(os.environ.get('CLIMB_S', '1.6'))
while time.time() - t0 < _climb_s and not aborted:
    # perception-aware climb: keep the nose on the pad-locked gate so
    # fixes keep flowing from t0 (map v2 retired the near-pad arch
    # anchor; if the gate leaves view, DR owns the flight and wanders)
    yr_c = 0.0
    if (state['obs'] is not None and time.time() - state['obs_wall'] < 1.0
            and os.environ.get('STRAIGHTTEST') != '1'):
        brg = math.atan2(float(state['obs'][1]), float(state['obs'][0]))
        yr_c = max(-0.3, min(0.3, 1.0 * brg))   # SZ=+1 (flight-measured default; defined later)
    level_cmd(0, 0, 1.0, yr=yr_c); aborted = guards('climb'); time.sleep(1/CMD_HZ)
state['airborne'] = True   # loosens the det z-guard (in-flight obs-z bias)
print(f'airborne tilt {math.degrees(tilt()):.1f} vz {state["vz_up"]:.2f}', flush=True)

if os.environ.get('RELEVEL') == '1' and not aborted:
    # TRANSIENT WIPE (07-12): the spawn sits at -17.8 deg pitch and the
    # takeoff level-off rotates ~0.5 rad/s across the stream's 28 ms
    # sample gaps -- unsampled rotation banks 2-4 deg of permanent
    # attitude error (the entire -2.0/+0.5 aim-bias budget; sensor
    # itself is noiseless). Hover 1.2 s, then re-zero attitude from the
    # accel (exact at hover) and zero the KF velocity once (its error
    # comes from the same wrong attitude; the controller would chase it).
    _acc_win = []
    t0 = time.time()
    while time.time() - t0 < 1.2 and not aborted:
        level_cmd(0, 0, 0)
        if time.time() - t0 > 0.4:
            _acc_win.append(state['acc'])
        aborted = guards('climb'); time.sleep(1/CMD_HZ)
    if _acc_win and not aborted:
        axm = sum(a[0] for a in _acc_win) / len(_acc_win)
        aym = sum(a[1] for a in _acc_win) / len(_acc_win)
        azm = sum(a[2] for a in _acc_win) / len(_acc_win)
        an = math.sqrt(axm*axm + aym*aym + azm*azm)
        if abs(an - 9.81) < 0.8:      # genuine near-hover: level formula valid
            r_old, p_old = state['roll'], state['pitch']
            state['roll'] = math.atan2(aym, -azm)
            state['pitch'] = math.atan2(axm, math.sqrt(aym*aym + azm*azm))
            with KF_LOCK:
                KF.x[3:6] = 0.0
            print(f'RELEVEL: roll {math.degrees(r_old):+.1f} -> '
                  f'{math.degrees(state["roll"]):+.1f} deg, pitch '
                  f'{math.degrees(p_old):+.1f} -> {math.degrees(state["pitch"]):+.1f} deg, v zeroed',
                  flush=True)
            jlog('relevel', dr=round(math.degrees(state['roll'] - r_old), 2),
                 dp=round(math.degrees(state['pitch'] - p_old), 2))
        else:
            print(f'RELEVEL skipped: |f| {an:.2f} not hover-quiet', flush=True)

# AXIS PROBES (frame tripwires -- physical defaults, flip only on strong evidence).
# Flight 2026-07-05 #3 measured SX/SY/SZ all +1 AND the probe shove (2 m off-axis,
# +16 deg yaw) misaligned the G1 approach into a gate-post clip. Probes now
# opt-in via PROBES=1; the measured physical signs are the default.
SX, SY = 1.0, 1.0

if os.environ.get('RECENTER') == '1' and not aborted:
    # FLY OUT THE CLIMB DRIFT (07-12 straight probe): takeoff physically
    # displaces the drone ~+1.0 m y / -0.55 m z and the estimator TRACKS
    # it (real motion, noiseless DR). Don't compensate in the aim --
    # converge back to the spawn line on the estimate, then fly the
    # chute. Kills the +-0.75 run-to-run crossing lottery at its source.
    # RECENTER_Y (07-14, nt10 frames): takeoff displaces the drone ~+2 m right
    # of the est frame (accel under-reads under thrust -- the est-truth gap that
    # the historical AIMBIAS_Y=-2 was really compensating). Converge est-y to
    # the biased line HERE, in hover, so the blind chute needs zero lateral
    # motion at the crossing (nt10 crossed at est -0.48: the carrot never
    # finished the diagonal shift).
    # RECENTER_Z (07-15, fg19-21 frames): est-z under-reads the climb by
    # ~1.2-1.7 m, so the old -1.3 target parks the drone at TRUTH 2.5-3 m --
    # above the 1.35 m aperture line, seeing near gates oblique-from-above
    # (hole width shrinks while closing -> pillar/pole hits at ~23 s in
    # fg19/20/21). -0.4 est ~= 1.3-1.6 m truth ~= the aperture line; err low:
    # from below, the 20-up camera keeps the hole in view (below-height
    # approach, memory note).
    _tgt = np.array([0.0, float(os.environ.get('RECENTER_Y', '0.0')),
                     float(os.environ.get('RECENTER_Z', '-1.3'))])
    t0 = time.time()
    # GNSCALE SCALE-CAL HOLD (07-14): stay near the pad (~6 m from gate 1, SAFE)
    # while DPVO calibrates metric scale off the gate anchors, then commit to the
    # spline. Without this the drone dives into the gate before scale locks
    # (scored7: banked 5 of 12 samples, crashed at ~2 m). A slow FORWARD creep to
    # a safe hold-x gives a monotonic DPVO baseline (symmetric bobs -> ~0 net
    # displacement -> ill-posed scale) and keeps the gate in view.
    _gn = os.environ.get('GNSCALE') == '1'
    _tmax = float(os.environ.get('GNSCALE_HOLD_TMAX', '18.0')) if _gn else 8.0
    _holdx = float(os.environ.get('GNSCALE_HOLD_X', '1.5'))
    while time.time() - t0 < _tmax and not aborted:
        with KF_LOCK:
            _p = KF.p.copy()
        _d = _tgt - _p
        _conv = float(np.hypot(_d[1], _d[2])) < 0.15
        _wait_scale = _gn and not state.get('dpvo_scale_locked')
        if _conv and not _wait_scale:
            break
        _vy = max(-0.5, min(0.5, 0.8 * float(_d[1])))
        _vz = max(-0.5, min(0.5, -0.8 * float(_d[2])))   # vz_ref is up-positive
        _vx = 0.0
        if _conv and _wait_scale:
            _vx = 0.3 if _p[0] < _holdx else 0.0         # creep to safe hold-x, then hover
        level_cmd(_vx, _vy, _vz)
        aborted = guards('climb')
        time.sleep(1/CMD_HZ)
    with KF_LOCK:
        _p = KF.p.copy()
    _sl = 'scale LOCKED' if state.get('dpvo_scale_locked') else 'scale NOT locked'
    print(f'RECENTER done at [{_p[0]:.2f} {_p[1]:.2f} {_p[2]:.2f}] '
          f'({time.time()-t0:.1f} s, {_sl})', flush=True)
    jlog('recenter', p=[round(float(x), 2) for x in _p], t=round(time.time()-t0, 1))

if os.environ.get('YAWPROBE') == '1' and not aborted:
    # YAW-SIGN PROBE (07-14): hover, spin ~180 deg at 0.4 rad/s, stop, and log
    # integrated roll/pitch vs gravity-derived truth throughout. If the gyro
    # reports FRD body rates, EULERFIX keeps roll/pitch true through the spin;
    # if it reports Euler-angle rates, EULERFIX's coupling terms inject phantom
    # roll (fg14 runaway). Run with EULERFIX=1 and =0, compare drift.
    print(f'YAWPROBE start (EULERFIX={os.environ.get("EULERFIX","0")})', flush=True)
    _t0 = time.time()
    while time.time() - _t0 < 14.0 and not aborted:
        _el = time.time() - _t0
        _yr = 0.4 if 2.0 < _el < 10.0 else 0.0
        level_cmd(0.0, 0.0, 0.0, yr=_yr)
        acc_ = state['acc']
        _r_acc = math.atan2(acc_[1], -acc_[2])
        _p_acc = math.atan2(acc_[0], math.sqrt(acc_[1]**2 + acc_[2]**2))
        jlog('yawprobe', el=round(_el, 2), yr=_yr,
             r_est=round(math.degrees(state['roll']), 2),
             p_est=round(math.degrees(state['pitch']), 2),
             y_est=round(math.degrees(state['yaw']), 1),
             r_acc=round(math.degrees(_r_acc), 2),
             p_acc=round(math.degrees(_p_acc), 2),
             vx=round(state['vx_b'], 2), vy=round(state['vy_b'], 2))
        time.sleep(1 / CMD_HZ)
    land(aborted or 'yawprobe complete')
    sys.exit(0)

if os.environ.get('HOVERPROBE') == '1' and not aborted:
    # HOVER-DRIFT PROBE (07-15): is the rightward terminal drift a REAL
    # environmental push or flight-coupled (attitude banking during
    # maneuvers)? Hover 12 s with ZERO commands right after RELEVEL, with
    # the pad-locked chute hole in view: fastgate's 30 Hz bearing stream is
    # a truth-referenced lateral velocity measurement (drift_rad/s x range).
    # If the hole bearing walks left at hover -> the drone drifts right with
    # zeroed attitude and no commands -> the push is real. If it hovers
    # clean -> the drift only exists in flight.
    print('HOVERPROBE start (12 s zero-cmd hover, fastgate as truth)', flush=True)
    _t0 = time.time()
    while time.time() - _t0 < 12.0 and not aborted:
        level_cmd(0.0, 0.0, 0.0)
        acc_ = state['acc']
        _r_acc = math.atan2(acc_[1], -acc_[2])
        _fga = time.time() - state.get('fg_wall', 0.0)
        _fgb = state.get('fg_bear', (9.9, 9.9))
        jlog('hoverprobe', el=round(time.time() - _t0, 2),
             bear=[round(_fgb[0], 4), round(_fgb[1], 4)],
             fga=round(_fga, 2), w=round(state.get('fg_w', 0.0), 0),
             r_est=round(math.degrees(state['roll']), 2),
             r_acc=round(math.degrees(_r_acc), 2),
             vy=round(state['vy_b'], 2))
        aborted = guards('route')
        time.sleep(1 / CMD_HZ)
    land(aborted or 'hoverprobe complete')
    sys.exit(0)

if os.environ.get('STRAIGHTTEST') == '1' and not aborted:
    # SWIVEL ISOLATION (07-12, Alex's eyes-on): climb a little, fly
    # straight forward 5 s, yaw rate pinned to 0 the whole way. No
    # route, no servo, no nose aiming -- if the drone still swivels,
    # the yaw motion is not commanded by us.
    print('STRAIGHTTEST: 5 s forward, yr=0', flush=True)
    t0 = time.time()
    while time.time() - t0 < 5.0 and not aborted:
        level_cmd(0.8, 0, 0, yr=0.0)
        jlog('straight', yaw=round(state['yaw'], 4),
             gz=round(float(state['gyr'][2]), 4),
             p=[round(float(x), 2) for x in KF.p])
        aborted = guards('route')
        time.sleep(1/CMD_HZ)
    land(aborted or 'straight test complete')
    sys.exit(0)

if (os.environ.get('YAWCAL', '1') == '1' and os.environ.get('FGPURSUIT') == '1'
        and not aborted):
    # VISUAL CRAB CALIBRATION -- the vision yaw-relevel (07-15, hp1/st1
    # probes): the takeoff level-off banks ~10-17 deg of est-yaw error
    # (gravity observes roll/pitch, never yaw, so RELEVEL can't touch it);
    # the vy_b velocity loop converts it into a REAL rightward crab
    # ~ vx*sin(err) that the est reads back as vy~0 (st1: 1-1.5 m right
    # over a 4 m pinned-yaw leg while est-y moved 0.03 m). Measure the crab
    # directly against the tracked hole bearing during a short pinned-yaw
    # forward leg, and store the equal-and-opposite lateral ratio; the
    # pursuit applies the constant crab_vy bias everywhere (tracked/coast/punch).
    # Sign chain (verified on fg25/26 logs): drone crabs RIGHT -> hole
    # walks LEFT in frame -> bear[0] slope NEGATIVE -> vlat=-slope*range
    # POSITIVE(right) -> ratio negative -> negative vy = leftward = opposes.
    _cal_t0 = time.time()
    # small nose-down bias throughout the calibration (07-15, fg38 n=0): at
    # hover the hole sits right at the up-tilted camera's FOV edge, and the
    # backward leg's nose-up transient pushes it out entirely -- no samples,
    # no seed. -0.06 rad holds it in view.
    while time.time() - _cal_t0 < 4.0 and not aborted:   # wait for a track
        if time.time() - state.get('fg_wall', 0.0) < 0.3 and state.get('fg_w', 0) >= 40:
            break
        level_cmd(0, 0, 0, pitch_bias=-0.06)
        aborted = guards('climb')
        time.sleep(1 / CMD_HZ)
    _cal_t0 = time.time()
    while time.time() - _cal_t0 < 1.5 and not aborted:   # yaw-center the hole
        _b = state.get('fg_bear', (0.0, 0.0))
        _fresh = time.time() - state.get('fg_wall', 0.0) < 0.3
        # SZ (yaw sign) is defined later in the flow; it is +1 by
        # flight-measured default -- use raw yr here.
        level_cmd(0, 0, 0, pitch_bias=-0.06,
                  yr=(max(-0.4, min(0.4, 1.2 * _b[0])) if _fresh else 0.0))
        aborted = guards('climb')
        time.sleep(1 / CMD_HZ)
    _hold_yaw = state['yaw']
    _samp = []
    # BACKWARD leg (07-15, fg31: the forward leg toward the chute threads the
    # start-light pole corridor at light-box height and hit the right pole
    # mid-calibration). Backing up goes into the open spawn area, keeps the
    # hole in view un-clipped (range grows), and measures the same ratio --
    # crab scales with signed vx.
    _VXL = -0.7
    _cal_t0 = time.time()
    while time.time() - _cal_t0 < 1.6 and not aborted:   # pinned-yaw leg
        _yrh = max(-0.3, min(0.3, 1.0 * wrap(_hold_yaw - state['yaw'])))
        level_cmd(SX * _VXL, 0, 0, pitch_bias=-0.06, yr=_yrh)
        if time.time() - state.get('fg_wall', 0.0) < 0.1:
            _samp.append((time.time(), state['fg_bear'][0], state.get('fg_w', 50.0)))
        aborted = guards('route')
        time.sleep(1 / CMD_HZ)
    _cal_t0 = time.time()
    while time.time() - _cal_t0 < 1.0 and not aborted:   # brake
        level_cmd(0, 0, 0)
        time.sleep(1 / CMD_HZ)
    if len(_samp) >= 12:
        _ts = np.array([s[0] for s in _samp]); _ts -= _ts[0]
        _bs = np.array([s[1] for s in _samp])
        _ws = np.array([s[2] for s in _samp])
        _slope = float(np.polyfit(_ts, _bs, 1)[0])           # rad/s, + = hole right
        _rng = 290.0 / max(20.0, float(np.median(_ws)))      # m from hole width
        _vlat = -_slope * _rng                               # m/s, + = drone RIGHT
        # DIRECTION-INDEPENDENT (07-15, st1 fwd vs fg32 bwd legs): the drift
        # is rightward in BOTH flight directions (~0.1-0.3 m/s when
        # translating, ~0 at hover) -- NOT a yaw-error crab (that would flip
        # with vx). Mechanism open; compensate empirically with a CONSTANT
        # opposite vy bias, re-measured every flight (magnitude varies
        # run-to-run, the historical 0.8-2.2 m spread).
        state['crab_vy'] = max(-0.4, min(0.4, -_vlat))
        # ATTMODE: pre-seed the roll-trim integrator from the measured drift
        # so the trim doesn't need to wind up from zero during the approach.
        state['_att_ri'] = max(-0.15, min(0.15,
            float(os.environ.get('ATT_TRIM_K', '0.5')) * state['crab_vy']))
        print(f'YAWCAL: slope {_slope:+.4f} rad/s @ {_rng:.1f} m -> lateral '
              f'drift {_vlat:+.2f} m/s while translating; vy bias '
              f'{state["crab_vy"]:+.2f}, att roll trim '
              f'{state["_att_ri"]:+.3f} rad', flush=True)
        jlog('yawcal', slope=round(_slope, 5), rng=round(_rng, 2),
             vlat=round(_vlat, 3), n=len(_samp))
    else:
        state['crab_vy'] = 0.0
        print(f'YAWCAL: insufficient samples (n={len(_samp)}), no compensation',
              flush=True)

RUN_PROBES = os.environ.get('PROBES') == '1'
def probe_axis(vx_r, vy_r, dur=1.2):
    v0 = np.array([state['vx_b'], state['vy_b']])
    o0, t_w0 = (None if state['obs'] is None else state['obs'].copy()), state['obs_wall']
    t0 = time.time()
    vpk = v0.copy()
    while time.time() - t0 < dur:
        level_cmd(vx_r, vy_r, 0)
        vpk = np.array([state['vx_b'], state['vy_b']])
        time.sleep(1/CMD_HZ)
    dv = vpk - v0
    t0 = time.time()
    o1 = None
    while time.time() - t0 < 3.5:
        level_cmd(0, 0, 0)
        if state['obs_wall'] > t_w0 and state['obs'] is not None:
            o1 = state['obs'].copy()
            break
        time.sleep(1/CMD_HZ)
    do = None if (o0 is None or o1 is None) else o1 - o0
    return dv, do

SZ = 1.0
if RUN_PROBES:
    dv, do = probe_axis(1.0, 0.0)
    jlog('axis_probe', axis='x', dv=dv.round(2).tolist(), do=None if do is None else do.round(2).tolist())
    print(f'probe X: dv {dv.round(2)} dObs {None if do is None else do.round(2)}', flush=True)
    if do is not None and abs(do[0]) > 0.5:
        SX = 1.0 if do[0] < 0 else -1.0
    dv, do = probe_axis(0.0, 1.0)
    jlog('axis_probe', axis='y', dv=dv.round(2).tolist(), do=None if do is None else do.round(2).tolist())
    print(f'probe Y: dv {dv.round(2)} dObs {None if do is None else do.round(2)}', flush=True)
    if do is not None and abs(do[1]) > 0.5:
        SY = 1.0 if do[1] < 0 else -1.0
    y0 = state['yaw']
    t0 = time.time()
    while time.time() - t0 < 0.5:
        level_cmd(0, 0, 0, yr=0.25); time.sleep(1/CMD_HZ)
    dyaw = state['yaw'] - y0
    t0 = time.time()
    while time.time() - t0 < 1.0:
        level_cmd(0, 0, 0); time.sleep(1/CMD_HZ)
    SZ = 1.0 if dyaw >= 0 else -1.0
    jlog('yaw_probe', dyaw=round(dyaw, 4), sz=SZ)
print(f'sign map: SX {SX} SY {SY} SZ {SZ} (probes {"run" if RUN_PROBES else "skipped -- flight-measured defaults"})', flush=True)

phase = 'route'                 # spline-first: route owns the course; vision approach is terminal aid
if os.environ.get('FGPURSUIT') == '1':
    # FGPURSUIT (07-14): fastgate tracked the hole dead-centered at 30 Hz for
    # 14 s straight (fg2 log) while the legacy GateNet handoff never fired.
    # Skip route/approach/punch: pure pursuit on the live hole bearing from
    # RECENTER to the crossing. The variable push is out-corrected in real
    # time; no handoff lottery, no blind coast.
    phase = 'fgpursuit'
retries = 0
gate_z_off = [0.0, 0.0, 0.0]    # per-gate aperture-height sweep offsets (retry logic)
retreat_pt = np.zeros(3)
phase_t0 = time.time()
punch_t0 = None
route_end_t0 = None
mission_t0 = time.time()
last_route_log = 0.0
while not aborted:
    now = time.time()
    # NO in-flight ZUPT: it froze KF.p mid-route (flight #4) and mid-approach
    # (flight #11, 40 s phantom hover). Drift is bounded by the map-validated
    # landmark fixes and the tick fix instead.
    # DR target = the MAP gate relative to the KF pose (level frame), always
    with KF_LOCK:
        d_w = state['next_gate_w'] - KF.p
    cyw, syw = math.cos(state['yaw']), math.sin(state['yaw'])
    state['tgt'] = np.array([cyw * d_w[0] + syw * d_w[1],
                             -syw * d_w[0] + cyw * d_w[1], d_w[2]])
    obs, age = state['obs'], now - state['obs_wall']
    # vision rules: only fall back to the map-DR target after 2.5 s without a
    # fresh obs -- the KF over-integrates at speed (flight #31/#32), so the DR
    # target goes stale-wrong far faster than the detector does
    if state.get('tgt') is not None and (obs is None or age > 2.5):
        obs = state['tgt']
    if state['gate_idx'] > gate0_idx:
        gate0_idx = state['gate_idx']
        ticks += 1
        print(f'*** GATE TICK {ticks} (idx {gate0_idx}, t={now - mission_t0:.1f} s) ***', flush=True)
        jlog('success', idx=gate0_idx, ticks=ticks)
        # a tick IS a position fix: the judge says we are in the ticked gate's
        # plane, so re-anchor the KF at the map gate (flight #4: the soft-ZUPT
        # had frozen KF.p ~8 m behind truth, stranding the route carrot)
        # tick-as-fix was bled-DR-era logic: it assumes the ticked gate is
        # our aim point. Run-21 proved otherwise (judge gate 1 is at ~[6,0],
        # ticked en route) -- teleporting the KF to the aim point would wreck
        # navigation. Re-anchor ONLY if vision has been stale for 3 s+.
        if os.environ.get('MAPFOLLOW') == '1':
            # MAPFOLLOW re-anchor (07-15, fg49): the punch + clearance dash
            # bank est error hard (fg49 est z -7 m, y -16 within seconds) and
            # the carrot then steers on fantasy. The tick is truth: we are AT
            # the ticked gate's aperture +/-0.75 m, moving ~2 m/s forward,
            # level. Anchor position AND velocity so the gate-2 carrot leg
            # starts clean. (The old run-21 caution no longer applies: 5/5
            # tick-punch correlation proves the ticked gate IS our aim point.)
            with KF_LOCK:
                KF.x[:3] = state['next_gate_w']
                KF.x[3] = 2.0 * math.cos(state['yaw'])
                KF.x[4] = 2.0 * math.sin(state['yaw'])
                KF.x[5] = 0.0
                KF.P[:3, :3] = np.eye(3) * 0.5
            jlog('tick_fix', p=state['next_gate_w'].round(2).tolist(),
                 mode='mapfollow-full')
        elif time.time() - state.get('obs_wall', 0.0) > 3.0 and \
                state.get('fix_count', 0) < 1:
            with KF_LOCK:
                KF.x[:3] = state['next_gate_w']
                KF.P[:3, :3] = np.eye(3) * 0.5
            jlog('tick_fix', p=state['next_gate_w'].round(2).tolist())
        else:
            jlog('tick_fix_skipped', reason='vision healthy')
        state['obs'] = None; state['tgt'] = None; state['obs_wall'] = 0.0
        state['landmark_w'] = None
        _dash_s = 0.4 if os.environ.get('MAPFOLLOW') == '1' else 1.2
        t0c = time.time()
        while time.time() - t0c < _dash_s:   # clearance dash through the plane
            level_cmd(SX * 1.5, 0, 0, pitch_bias=-0.08); time.sleep(1/CMD_HZ)
        if os.environ.get('MAPFOLLOW') == '1':
            # RE-ANCHOR AFTER THE DASH (07-15, fg54): anchoring at the tick
            # and then dashing re-banks est error on bad draws before the
            # staging loop ever runs. Post-dash truth is the ticked gate
            # plus ~1 m of punch residual + dash along heading, ~1.5 m/s
            # forward, level -- start the loop from THAT.
            with KF_LOCK:
                KF.x[0] = float(state['next_gate_w'][0]) \
                    + 1.0 * math.cos(state['yaw'])
                KF.x[1] = float(state['next_gate_w'][1]) \
                    + 1.0 * math.sin(state['yaw'])
                KF.x[2] = float(state['next_gate_w'][2])
                KF.x[3] = 1.5 * math.cos(state['yaw'])
                KF.x[4] = 1.5 * math.sin(state['yaw'])
                KF.x[5] = 0.0
                KF.P[:3, :3] = np.eye(3) * 0.5
            jlog('tick_fix', p=[round(float(x), 2) for x in KF.x[:3]],
                 mode='post-dash')
            # MF_DR seed (07-15): kinematic DR for the staging leg starts
            # HERE -- position = ticked gate + ~1 m of dash along heading.
            # Skip the brake+relevel entirely (fg57 proved it can't settle);
            # the leg flies on commanded-speed x gyro-yaw DR instead.
            state['_mf_p'] = np.array([
                float(state['next_gate_w'][0]) + 1.0 * math.cos(state['yaw']),
                float(state['next_gate_w'][1]) + 1.0 * math.sin(state['yaw']),
                float(state['next_gate_w'][2])])
            state['_mf_lastpb'] = -0.06
            with KF_LOCK:
                state['_mf_z0'] = float(KF.x[2])   # relative-altitude baseline
            state.pop('_mf_zerr', None)
            jlog('mfdr_seed', p=[round(float(x), 2) for x in state['_mf_p']])
        if (os.environ.get('MAPFOLLOW') == '1'
                and os.environ.get('MF_DR') != '1'):
            # BRAKE + MID-FLIGHT RELEVEL (07-15, fg55): the anchor fixes the
            # STATE but the punch banks ATTITUDE error, so est velocity
            # re-explodes within seconds. Hover-brake, re-zero roll/pitch
            # from the accel (exact at hover -- the takeoff RELEVEL move),
            # zero velocity, re-anchor position. The staging loop then runs
            # on a genuinely clean estimator.
            _t0b = time.time()
            _quiet = False
            while time.time() - _t0b < 4.0:
                level_cmd(0, 0, 0.0, att=True)
                _acB = state['acc']
                _an = math.sqrt(_acB[0]**2 + _acB[1]**2 + _acB[2]**2)
                # re-zero ONLY when hover-quiet (fg56: an unsettled accel
                # injects attitude error instead of removing it)
                if time.time() - _t0b > 1.5 and abs(_an - 9.34) < 0.12:
                    _quiet = True
                    break
                time.sleep(1 / CMD_HZ)
            if _quiet:
                _acB = state['acc']
                state['roll'] = math.atan2(_acB[1], -_acB[2])
                state['pitch'] = math.atan2(
                    _acB[0], math.sqrt(_acB[1] ** 2 + _acB[2] ** 2))
            else:
                print('post-tick relevel: not quiet, attitude kept', flush=True)
            with KF_LOCK:
                KF.x[0] = float(state['next_gate_w'][0]) \
                    + 1.7 * math.cos(state['yaw'])
                KF.x[1] = float(state['next_gate_w'][1]) \
                    + 1.7 * math.sin(state['yaw'])
                KF.x[2] = float(state['next_gate_w'][2])
                KF.x[3:6] = 0.0
                KF.P[:3, :3] = np.eye(3) * 0.5
            print('post-tick brake + mid-flight relevel done', flush=True)
            jlog('tick_fix', p=[round(float(x), 2) for x in KF.x[:3]],
                 mode='post-brake-relevel')
        if ticks >= TARGET_TICKS:
            print(f'*** MISSION COMPLETE: {ticks} gates ***', flush=True)
            jlog('mission_complete', ticks=ticks)
            land('mission complete'); break
        nxt = GATES_W[min(ticks, len(GATES_W) - 1)]
        state['next_gate_w'] = nxt + np.array([0, 0, gate_z_off[min(ticks, 2)]])
        retries = 0
        if os.environ.get('FGPURSUIT') == '1':
            # POST-TICK GATE-TO-GATE (07-15, fg36 first tick): stay in the
            # visual pursuit -- the legacy route/obs-DR punch missed gate 2
            # twice after the tick while fg35's pursuit re-acquisition had
            # already reached the ribbon visually. Drop the old track so
            # acquisition picks the next biggest close hole.
            state['fg_wall'] = 0.0
            state['_fg_close'] = 0.0
            state['_fgp_vy'] = state['_fgp_vz'] = 0.0
            state['_fgp_ib'] = 0.0
            state.pop('_punch_rb', None); state.pop('_punch_dthr', None)
            state['_att_bd'] = (0.0, 0.0)
            state['_att_lastb'] = None
            phase, phase_t0, route_end_t0 = 'fgpursuit', now, None
            print(f'FGPURSUIT to gate {ticks} at {state["next_gate_w"].round(2)}',
                  flush=True)
        else:
            phase, phase_t0, route_end_t0 = 'route', now, None
            print(f'ROUTE to gate {ticks} at {state["next_gate_w"].round(2)}', flush=True)
        continue
    if state.get('race_finish_ns', -1) >= 0:
        print(f'*** RACE FINISH *** ticks={ticks}', flush=True)
        jlog('race_finish', ticks=ticks, finish_ns=state['race_finish_ns'])
        land('race finished'); break
    if now - mission_t0 > MISSION_S:
        aborted = f'mission timeout, ticks={ticks}'

    if phase == 'fgpursuit':
        state['trim_ok'] = False
        with KF_LOCK:
            _pp = KF.p.copy()
        _dpvo_control = (os.environ.get('DPVO_ROUTE') == '1'
                         and os.environ.get('DPVO_OBSERVE') != '1')
        if (os.environ.get('DPVO_ROUTE') == '1'
                and select_dpvo_route_position is not None):
            _observe_only = os.environ.get('DPVO_OBSERVE') == '1'
            _dpvo_p, _dpvo_reason = select_dpvo_route_position(
                state, ticks, now, _observe_only)
            if _observe_only:
                if now - state.get('_dpvo_compare_log', 0.0) > 0.5:
                    state['_dpvo_compare_log'] = now
                    jlog('dpvo_compare', reason=_dpvo_reason,
                         dpvo=None if state.get('dpvo_route_p') is None
                         else [round(float(x), 3)
                               for x in state['dpvo_route_p']],
                         control=[round(float(x), 3) for x in _pp])
            elif ticks >= 1:
                if _dpvo_p is not None:
                    _pp = np.asarray(_dpvo_p, dtype=float)
                    state['_route_source'] = 'dpvo'
                else:
                    _tick_age = (time.time_ns()
                                 - int(state.get('gate_tick_ns', 0))) / 1e9
                    _grace = float(os.environ.get('DPVO_START_GRACE', '1.5'))
                    if 0.0 <= _tick_age < _grace:
                        level_cmd(0, 0, 0.0, att=True)
                        time.sleep(1 / CMD_HZ)
                        continue
                    aborted = f'DPVO route unhealthy: {_dpvo_reason}'
                    state['dpvo_abort_reason'] = _dpvo_reason
                    jlog('dpvo_abort', reason=_dpvo_reason)
                    continue
        # FGP_PLANE_X (07-14, fg12 frames): the cyan racing line dives through
        # the STACKED (ribbon) gate at x~11.5, and every historical tick
        # (arch17/23/27) happened THERE -- the chute-plane backstop killed the
        # pursuit at x 10.3 just short of it. Let the pursuit cross the ribbon.
        _plane_x = float(os.environ.get('FGP_PLANE_X', '0') or 0) \
            or float(state['next_gate_w'][0])
        _fwd_dr = _plane_x - float(_pp[0])
        _fgage = now - state.get('fg_wall', 0.0)
        _fgw = state.get('fg_w', 0.0)
        # RANGE FROM THE HOLE, not DR (fg5: DR over-integrated and braked ~2 m
        # short while the bearing was still tracking smoothly). Hole width
        # >=120 px ~= <4 m: mark 'close'. When the hole then vanishes (clipped
        # out at ~1.5 m), punch straight through.
        # PUNCH-CENTERING GATE (07-15, fg15/fg16 frames): post-RECENTER
        # altitude is a +/-1 m est-z lottery -- fg15 skimmed 0.3 m OVER the
        # gate top, fg16 punched with the hole 0.6 rad ABOVE (drone below the
        # aperture) and hit the frame. The punch zeroes vz, so it must only
        # arm when the hole is inside the aperture cone (0.18 rad ~= 0.7 m at
        # the w>=120 trigger range); until then keep steering and let the
        # bearing loop converge.
        # STAGING GATE (07-15, fg50/51): on the gate-2 leg, holes seen
        # obliquely mid-dogleg must not steal the track or arm the punch --
        # vision control is allowed only once the staging point on the
        # gate's crossing normal is reached (head-on geometry, like gate 1).
        _lt_line_fresh = (os.environ.get('LINE_TRANSIT') == '1'
                          and time.time() - state.get('line_wall', 0) < 0.4)
        if os.environ.get('LINE_TRANSIT') == '1':
            # LINE_TRANSIT vision release (v2, 07-19 test15 postmortem):
            # v1 released at the staging thresholds (w>=45, |bear|<0.6) and
            # the pursuit STOLE control from the line on oblique mid-transit
            # holes (first sight at bear 0.83) -- two masters alternated at
            # the freshness boundary and the drone weaved until both went
            # stale. While the line is fresh it OWNS transit; vision takes
            # over only terminal (big AND centered hole -- the line has
            # already aimed us through it). Line lost -> old thresholds, so
            # vision can still rescue.
            _fgb_lt = state.get('fg_bear', (9.9, 9.9))
            # MAP-CONSISTENCY (07-19, test41 FRAMES: the drone hovered at
            # the G2 RIBBON/banner wall converging on its openings while
            # the real scoring gate -- the one the LINE runs through --
            # sat lower and farther ahead. Wrong-structure holes are
            # ~1.5 rad off the map bearing; DR+gyro give ~0.4 rad
            # accuracy at this range): a terminal candidate must agree
            # with the bearing to the next gate's MAP position.
            _lt_map_ok = True
            _gw_m = state.get('next_gate_w')
            _mp_m = state.get('_mf_p')
            if ticks >= 1 and _gw_m is not None and _mp_m is not None:
                _eb_m = wrap(math.atan2(
                    float(_gw_m[1]) - float(_mp_m[1]),
                    float(_gw_m[0]) - float(_mp_m[0])) - state['yaw'])
                _lt_map_ok = abs(wrap(_eb_m - _fgb_lt[0])) < float(
                    os.environ.get('LT_MAP_B', '0.5'))
            state['_lt_map_ok'] = _lt_map_ok
            # LINE-MATCH (07-19, test45 FRAMES: the "dead-center w=113
            # hole" the punch flew into was the NEGATIVE SPACE OF A LETTER
            # on the sponsor banner -- the hole detector reads dark
            # letterforms as apertures, and map/vertical checks cannot
            # separate a banner glyph from the adjacent gate). The one
            # geometric truth: the cyan racing line passes through the
            # SCORING aperture. A terminal candidate must sit where the
            # line's far head points.
            _lh_m = state.get('line_head_off', 9.9)
            _lt_line_match = (time.time() - state.get('line_wall', 0) < 0.6
                              and abs(_lh_m * float(os.environ.get(
                                  'LT_HEAD2RAD', '1.75')) - _fgb_lt[0])
                              < float(os.environ.get('LT_LINE_MATCH',
                                                     '0.28')))
            if _lt_line_fresh:
                # v4 (test16/17 postmortems): BOTH flights chased the same
                # fast-growing non-gate opening ~4 s post-G1 (b1 -> -0.36 =
                # far ABOVE our flight height; G2 was ~68 deg off the nose
                # at that moment -- an arch the line passes UNDER). A real
                # gate approached ON the line at height sits near b1~0, so
                # while the line is fresh vision also needs VERTICAL
                # plausibility, and enough width to punch immediately.
                _allow_track = (os.environ.get('MAPFOLLOW') != '1'
                                or ticks == 0
                                or (_fgage < 0.4 and _lt_map_ok
                                    and _lt_line_match
                                    and _fgw >= float(os.environ.get(
                                        'LT_VIS_W', '95'))
                                    and abs(_fgb_lt[0]) < float(
                                        os.environ.get('LT_VIS_B', '0.25'))
                                    # vertical window opens DOWNWARD too
                                    # (test41 frames: hovering HIGH, the
                                    # real gate is below the axis)
                                    and -float(os.environ.get('LT_VIS_BV',
                                                              '0.2'))
                                    < _fgb_lt[1]
                                    < float(os.environ.get('LT_VIS_BVDN',
                                                           '0.35'))))
                if _allow_track and ticks >= 1:
                    # RELEASE DEBOUNCE (07-19, test42: a single noisy w=67
                    # blip released the latch ~6 m out mid-transit; the
                    # beeline at the gate hit scenery the LINE routes
                    # around). Require the strict release to hold on two
                    # consecutive detections before latching.
                    _rl_w = state.get('_lt_rel_wall', 0.0)
                    _rl_n = state.get('_lt_rel_n', 0)
                    _rl_n = _rl_n + 1 if time.time() - _rl_w < 1.2 else 1
                    state['_lt_rel_n'] = _rl_n
                    state['_lt_rel_wall'] = time.time()
                    if _rl_n < 2:
                        _allow_track = False
                    else:
                        # TERMINAL LATCH (07-19, test36): braking near the
                        # gate while the LINE branch still steered (the
                        # line runs PAST the gate) slid the drone off the
                        # hole, and ownership flapped at the release
                        # thresholds. Once a candidate passes the strict
                        # release TWICE, the pursuit OWNS terminal for
                        # LT_TERM_S seconds -- hover, settle, punch;
                        # fallbacks resume if it expires.
                        state['_lt_term_wall'] = time.time()
            if (ticks >= 1 and time.time() - state.get('_lt_term_wall', 0)
                    < float(os.environ.get('LT_TERM_S', '6.0'))):
                _allow_track = True
            else:
                # line lost: the original staging discipline owns release
                # (stage_done + head-on) -- oblique mid-transit holes must
                # not steal the track (fg50/51).
                _allow_track = (os.environ.get('MAPFOLLOW') != '1'
                                or ticks == 0
                                or (state.get('_mf_stage_done', False)
                                    and _fgage < 0.4 and _fgw >= 45
                                    and abs(_fgb_lt[0]) < 0.6))
        else:
            _allow_track = (os.environ.get('MAPFOLLOW') != '1' or ticks == 0
                            or (state.get('_mf_stage_done', False)
                                # HEAD-ON CONFIRMATION (07-15, fg63):
                                # kinematic DR speed error can fire 'staged'
                                # early/late -- release vision only when the
                                # hole also LOOKS head-on, else keep the
                                # carrot.
                                and _fgage < 0.4 and _fgw >= 45
                                # 0.6, not 0.35 (fg64/65: DR speed error runs
                                # the east leg ~1 m south of the gate line --
                                # the hole sits at 0.4-0.8 rad, visible but
                                # never passing a strict gate; the pursuit
                                # converges 0.75 rad fine)
                                and abs(state.get('fg_bear',
                                                  (9.9, 9.9))[0]) < 0.6))
        _ctr = float(os.environ.get('FGP_PUNCH_CTR', '0.18'))
        _fgb = state.get('fg_bear', (9.9, 9.9))
        if (_allow_track and _fgage < 0.4 and _fgw >= 120
                and abs(_fgb[0]) < _ctr and abs(_fgb[1]) < _ctr):
            state['_fg_close'] = now
        # PUNCH-WHILE-LOCKED (07-15, fg26/fg27 frames): the hole clips out at
        # w 95-108 -- BELOW the 120 vanish-trigger -- so every crossing ended
        # as a 2.5 s blind coast that the rightward push owns (fg26/27 both
        # slid just right of the post). Punch the moment the hole is big AND
        # centered, while still visually locked: no coast, minimal blind time.
        # BELOW-HEIGHT PUNCH CONE (07-19, test28: the terminal climb lifted
        # the drone to hole height and the 20-up camera clipped the hole at
        # w=64 / ~4.5 m -- earliest blindness of the campaign; the fastgate
        # memory's below-height approach is the fix). Asymmetric vertical:
        # a hole up to FGP_PUNCH_BVUP ABOVE the axis is punchable (the
        # punch's vertical steering climbs through it); below stays tight
        # (top-bar strike, fg16).
        _bvup = float(os.environ.get('FGP_PUNCH_BVUP', '0') or 0) or _ctr
        # per-leg punch width (07-19, test34): lowering the width for the
        # gate-2 hover-range punch ALSO fired gate 1's punch 4+ m out and
        # destabilized the proven recipe -- G1 keeps FGP_PUNCH_W, later
        # legs use FGP_PUNCH_W2 (default: same).
        _pw = float(os.environ.get('FGP_PUNCH_W', '95'))
        if ticks >= 1:
            _pw = float(os.environ.get('FGP_PUNCH_W2', '0') or 0) or _pw
        _bvdn = _ctr if ticks == 0 else float(
            os.environ.get('FGP_PUNCH_BVDN', '0.35'))
        _punch_now = (_allow_track and _fgage < 0.4
                      and _fgw >= _pw
                      and abs(_fgb[0]) < _ctr
                      and -_bvup < _fgb[1] < _bvdn
                      and (ticks == 0
                           or os.environ.get('LINE_TRANSIT') != '1'
                           or state.get('_lt_map_ok', True)))
        if _punch_now and os.environ.get('PN_PUNCH') == '1':
            # E1 PN collision-course gate (07-15 handoff, re-applied 07-19
            # after the DPVO rewrite wiped it; test24/25: a centered-only
            # trigger either never fires (ctr 0.10) or fires unconverged
            # into the frame (ctr 0.18)). Two bodies collide iff the LOS
            # stops rotating: require the LATERAL bearing-rate nulled (the
            # push is lateral; vertical rate during a deliberate climb-
            # through is convergence, not miss).
            _bd_pn = state.get('_att_bd', (9.9, 9.9))
            _pnr = float(os.environ.get('PN_LOSRATE', '0.12'))
            _punch_now = abs(_bd_pn[0]) < _pnr
        if _punch_now or (_fgage > 0.5 and now - state.get('_fg_close', 0) < 3.0):
            # GATE-TO-GATE (07-14, Alex): a close hole (w>=120) just left the
            # frame -> punch straight through it, then DROP the track and keep
            # pursuing -- the next course gate is now the biggest close hole and
            # acquisition picks it up. The course flies itself in proximity
            # order; the main-loop tick handler scores as we go.
            print(f'fgpursuit: crossing (w {_fgw:.0f}), punching + continuing', flush=True)
            t0p = time.time()
            _hold_yaw = state['yaw']
            while time.time() - t0p < 1.6 and not aborted:
                _yrh = SZ * max(-0.3, min(0.3, 1.0 * wrap(_hold_yaw - state['yaw'])))
                # STEERED PUNCH (07-15, fg28 frames): the straight punch let
                # the ~0.3-0.5 m/s push slide the drone OUTSIDE the right
                # post in 1.6 s -- while the detector was still SEEING the
                # aperture (w=200 at 0.55 rad left). Keep the lateral loop
                # closed on the hole as long as it stays visible; hold the
                # last correction when it finally clips out.
                if os.environ.get('ATTMODE') == '1':
                    # attitude punch: strong fixed forward pitch; roll/thr
                    # steered on the hole while visible, trim+hold after.
                    if time.time() - state.get('fg_wall', 0.0) < 0.3:
                        _pv = state['fg_bear']
                        # carry the learned push integral through the punch
                        # (07-19): the blind window is exactly when the
                        # push owns the drone
                        # D-term in the punch steering (07-19, tests
                        # 29/30): the line meets the gate slightly from
                        # the right, so the bearing sweeps THROUGH zero at
                        # the punch moment -- steering on b alone lets the
                        # sweep rate carry the drone past the post.
                        _pbd = state.get('_att_bd', (0.0, 0.0))
                        state['_punch_rb'] = max(-0.22, min(0.22,
                            0.4 * _pv[0]
                            + float(os.environ.get('FGP_PUNCH_KD', '0.0'))
                            * _pbd[0]
                            + state.get('_att_ri', 0.0)))
                        state['_punch_dthr'] = max(-0.06, min(0.06,
                            -float(os.environ.get('FGP_PUNCH_KT', '0.15'))
                            * _pv[1]))
                    _ppitch = float(os.environ.get('ATT_PUNCH_PITCH',
                                                   '-0.18'))
                    if ticks >= 1:
                        # per-leg punch pitch (test44: softening the shared
                        # knob broke G1's crossing -- G1 keeps -0.18)
                        _ppitch = float(os.environ.get('ATT_PUNCH_PITCH2',
                                                       '0') or 0) or _ppitch
                    level_cmd(0, 0, state.get('_punch_dthr', 0.0),
                              pitch_bias=_ppitch,
                              yr=_yrh,
                              roll_bias=state.get('_punch_rb',
                                                  state.get('_att_ri', 0.0)),
                              att=True)
                else:
                    if time.time() - state.get('fg_wall', 0.0) < 0.3:
                        _pv = state['fg_bear']
                        state['_punch_vy'] = max(-0.6, min(0.6, 0.9 * _pv[0]))
                        state['_punch_vz'] = max(-0.5, min(0.5, -0.9 * _pv[1]))
                    level_cmd(SX * 1.8,
                              SY * (state.get('_punch_vy', 0.0)
                                    + state.get('crab_vy', 0.0)),
                              state.get('_punch_vz', 0.0), yr=_yrh)
                aborted = guards('punch')
                time.sleep(1 / CMD_HZ)
            state['_punch_vy'] = state['_punch_vz'] = 0.0
            state['fg_wall'] = 0.0          # drop track -> re-acquire next gate
            state['_fg_close'] = 0.0
            state['_fgp_vy'] = state['_fgp_vz'] = 0.0
            state['_fgp_ib'] = 0.0          # new gate, new disturbance integral
            state['_att_ib'] = 0.0          # (ATT push integral likewise)
            if os.environ.get('LINE_TRANSIT') == '1':
                # 07-19 test37 (ib/ri trace): the "global" roll trim
                # entered the G2 terminal at +0.18 -- learned on the
                # PREVIOUS leg -- and dragged the hover rightward faster
                # than the bearing integral could cancel (its cap). The
                # disturbance field is per-LOCATION: relearn the trim on
                # each new leg.
                state['_att_ri'] = 0.0
            state.pop('_punch_rb', None); state.pop('_punch_dthr', None)
            state['_att_bd'] = (0.0, 0.0)   # new gate: derivative discontinuity
            state['_att_lastb'] = None      # (keep _att_ri -- the drift trim
                                            # is global, not per-gate)
            state['_mf_stage_done'] = False  # new gate: stage its normal first
            state['_mf_wps'] = None; state['_mf_wpi'] = 0
            continue
        if os.environ.get('MAPFOLLOW') == '1':
            # x-plane overrun is meaningless on the dogleg (07-15 fg48: the
            # drone exits the chute at x~12, already PAST gate 2's x=11.4,
            # so _fwd_dr starts negative and the brake preempted the carrot
            # instantly). Runaway bound = horizontal distance from the next
            # gate instead.
            _ppo = state['_mf_p'] if (os.environ.get('MF_DR') == '1'
                                      and not _dpvo_control
                                      and state.get('_mf_p') is not None) else _pp
            _dgx = float(state['next_gate_w'][0]) - float(_ppo[0])
            _dgy = float(state['next_gate_w'][1]) - float(_ppo[1])
            _overrun = math.hypot(_dgx, _dgy) > 25.0
        else:
            _overrun = _fwd_dr < -4.0
        if _overrun:
            print('fgpursuit: deep DR overrun, braking', flush=True)
            phase, phase_t0 = 'post', now
            continue
        _vx = 0.9 if _fwd_dr > 2.0 else 1.4
        _yr_t = 0.0
        if _fgage < 0.4:
            state['_fgp_had_track'] = True
            _b = state['fg_bear']
            # RANGE-FROM-HOLE SCALE (07-15, fg19/fg20): max(_fwd_dr, 1.2) was
            # a range-to-gate proxy only while FGP_PLANE_X sat AT the gate;
            # with plane 25 it reads ~20-24 near the pad, so a 0.08 rad
            # bearing saturates vy at +/-0.6 (4-5x hot) -- both runs slid
            # ~1.7 m sideways into the LEFT start-light pole. Hole width is
            # the honest range cue: w=46 px at 6.3 m -> range ~ 290/w.
            _scale = max(1.2, min(8.0, 290.0 / max(_fgw, 20.0)))
            # TURN-TO-TARGET (07-14): yaw toward the hole (a strafe with yaw
            # locked can't fly the 40-deg dogleg to the ribbon gate -- fg13's
            # bearing grew 0.37->0.85 with vy pinned). Small strafe assist for
            # the last-metre fine centering; vz on bearing as before.
            _yr_t = max(-0.6, min(0.6, 1.4 * _b[0]))
            # BEARING INTEGRATOR (07-15, fg23/fg25): the ~0.3 m/s gate-area
            # push turns pure pursuit into a constant-bearing drift and
            # P-gain alone leaves a ~0.5-1 m steady-state miss (fg23 right-
            # side miss at bear -0.014). A fixed APPROACH_VYBIAS feed-forward
            # made the drift 3x WORSE in fg25 (frame convention mismatch) --
            # integrate the bearing instead: pushes vy in whichever direction
            # actually nulls the drift, no sign assumptions.
            _ib = state.get('_fgp_ib', 0.0)
            _ib = max(-0.4, min(0.4, _ib + 1.5 * _b[0] / CMD_HZ))
            state['_fgp_ib'] = _ib
            _vy = max(-0.6, min(0.6, 0.4 * _b[0] * _scale + _ib))
            _vz = max(-0.8, min(0.8, -1.0 * _b[1] * _scale))
            state['_fgp_vy'] = _vy - _ib   # P part only; coast re-adds _fgp_ib
            state['_fgp_vz'] = _vz
            # ATTMODE bearing PID state: derivative from consecutive fresh
            # detections (fastgate is 1.8 ms / 30 Hz -- the D term is usable,
            # unlike GateNet's 450 ms), EMA-smoothed; integrator = the roll
            # trim that nulls steady-state bearing drift (subsumes the crab).
            _fw = state.get('fg_wall', 0.0)
            if _fw != state.get('_att_lastw', 0.0):
                _dtb = _fw - state.get('_att_lastw', _fw)
                _lb = state.get('_att_lastb')
                _rng_n = 290.0 / max(20.0, _fgw)
                if _lb is not None and 0.01 < _dtb < 0.25:
                    _obd = state.get('_att_bd', (0.0, 0.0))
                    state['_att_bd'] = (
                        0.6 * _obd[0] + 0.4 * (_b[0] - _lb[0]) / _dtb,
                        0.6 * _obd[1] + 0.4 * (_b[1] - _lb[1]) / _dtb)
                    _lr = state.get('_att_rng')
                    if _lr is not None:
                        # range rate from hole width: the vision speedometer.
                        # HEAVY smoothing (07-15 fg40 att log): +/-2 px on a
                        # 45 px hole is +/-0.5 m of range -> instantaneous
                        # rates of +/-3 m/s flapped the governor into braking
                        # mid-approach and the drone never closed.
                        state['_att_rrate'] = (0.85 * state.get('_att_rrate', 0.0)
                                               + 0.15 * (_rng_n - _lr) / _dtb)
                state['_att_rng'] = _rng_n
                state['_att_lastw'] = _fw
                state['_att_lastb'] = _b
            if _allow_track:
                # 07-19 (test38 ib/ri trace): this trim integral used to
                # adapt on ANY fresh hole -- during LINE_TRANSIT it ate the
                # +0.6-0.8 rad bearings of holes the line legitimately
                # passes by and wound to +0.18 full-scale in 1.5 s, then
                # dragged every gate-2 hover rightward. Integrate only
                # while the pursuit actually OWNS control.
                _ari = state.get('_att_ri', 0.0)
                _ari = max(-0.18, min(0.18,
                           _ari + float(os.environ.get('ATT_KI', '0.5'))
                           * _b[0] / CMD_HZ))
                state['_att_ri'] = _ari
            if now - state.get('_fgp_log', 0) > 0.5:
                state['_fgp_log'] = now
                jlog('fgp', fwd=round(_fwd_dr, 2), bear=[round(_b[0], 3),
                     round(_b[1], 3)], vy=round(_vy, 2), w=round(_fgw, 0))
        else:
            # hole not in view: hold the last correction, decayed; hold yaw
            _decay = 0.97
            state['_fgp_vy'] = state.get('_fgp_vy', 0.0) * _decay
            state['_fgp_vz'] = state.get('_fgp_vz', 0.0) * _decay
            # carry the integrator (the measured push estimate) UNDECAYED
            # through the blind coast (07-15, fg26: perfect tracked approach
            # to w=95, then the 2.5 s blind coast let the push slide it just
            # right of the post) -- only the P part decays.
            _vy = state['_fgp_vy'] + state.get('_fgp_ib', 0.0) \
                + float(os.environ.get('APPROACH_VYBIAS', '0.0')) * 0.5
            _vz = state['_fgp_vz']
            # DESCEND-UNTIL-ACQUIRE (07-15, fg17 acquisition frame): post-
            # RECENTER the drone sits 2.5-3.5 m up (est-z under-reads the
            # climb) and from there the 20-deg-up camera CANNOT see the near
            # chute hole (~40 deg below axis) while FAR bay gates stay
            # visible -- the blind forward creep then acquired a w=40 far
            # gate and wasted the run. Until the FIRST track of the flight:
            # crawl, don't creep, and descend so the near hole re-enters the
            # FOV band from below gate height (memory: below-height approach).
            if not state.get('_fgp_had_track'):
                _vx = 0.3
                _vz = float(os.environ.get('FGP_SEEK_VZ', '-0.35'))
                if float(_pp[2]) > -0.5:    # est floor margin (alt under-read)
                    _vz = 0.0
        # NOSE-DOWN TILT: rotates the 20-up cam down -> hole in view to ~1.5 m;
        # pitch-forward also accelerates through the crossing.
        # Keyed on HOLE SIZE, not DR (07-15): with FGP_PLANE_X deep, _fwd_dr
        # never reads <4.5 at the chute, the ramp never engaged, and the hole
        # clipped out at w~94 -- short of the w>=120 punch trigger.
        _close_t = (_fgw >= 70 and _fgage < 1.0) or _fwd_dr < 4.5
        if os.environ.get('ATTMODE') == '1':
            # PURE-ATTITUDE PURSUIT: bearing PID -> roll/throttle directly,
            # fixed pitch for speed. No est-velocity feedback anywhere.
            _pb = float(os.environ.get('FGP_PITCH', '-0.15')) if _close_t \
                else float(os.environ.get('ATT_PITCH', '-0.08'))
            # VISION SPEED GOVERNOR (07-15, fg34): a tilt commands
            # acceleration, not speed -- without a governor the drone
            # accelerates indefinitely (est hit the 8 m/s guard). Range rate
            # from hole width is the est-free speedometer: brake tilt when
            # closing faster than ATT_VMAX.
            if (_fgage < 0.4 and _fgw >= 55 and
                    -state.get('_att_rrate', 0.0) > float(os.environ.get('ATT_VMAX', '1.6'))):
                _pb = 0.06
            if _fgage < 0.4 and _allow_track:
                _bd = state.get('_att_bd', (0.0, 0.0))
                _bb = state['fg_bear']
                # CLOSED-LOOP PUSH COMPENSATION (07-19, tests 23-27): the
                # PD pursuit converges the hole then slides RIGHT past the
                # aperture in the last 3-4 m (test27: 0.67 -> -0.03 rad,
                # then w 64->42 while drifting off) -- the ~0.3 m/s gate-
                # area push is a constant disturbance a PD loop can only
                # hold with a steady-state error it cannot afford at 0.7 m
                # aperture margin. Integrate the bearing residual while
                # TRACKED (that IS the measured push), feed it to roll,
                # and hold it UNDECAYED through the punch blind window
                # (the velocity-mode pursuit has done exactly this with
                # _fgp_ib since 07-15; ATTMODE never got it). Reset per
                # gate with the other per-gate state.
                # (the per-gate bearing integral formerly added here was a
                # DUPLICATE of the _att_ri trim integral below, both keyed
                # on ATT_KI -- removed 07-19; _att_ri, now ownership-gated
                # and leg-reset, is the single push integral)
                _rb = (float(os.environ.get('ATT_KP', '0.4')) * _bb[0]
                       + float(os.environ.get('ATT_KD', '0.25')) * _bd[0]
                       + state.get('_att_ri', 0.0))
                _dthr = (-float(os.environ.get('ATT_KT', '0.15')) * _bb[1]
                         - float(os.environ.get('ATT_KTD', '0.10')) * _bd[1])
            else:
                # LEVEL OUT when blind (07-15 fg40): in attitude mode a held
                # tilt is a held ACCELERATION -- coasting at full trim for 8 s
                # ran the drone away leftward. Hold trim briefly (punch-window
                # scale), then fade it and brake gently.
                _ari0 = state.get('_att_ri', 0.0)
                _rb = _ari0 if _fgage < 1.0 else \
                    _ari0 * max(0.0, 1.0 - (_fgage - 1.0) / 1.5)
                if _fgage >= 1.0:
                    _pb = 0.02
                _dthr = -0.02 if not state.get('_fgp_had_track') else 0.0
                if (os.environ.get('MAPFOLLOW') == '1'
                        and (_fgage >= 1.0 or not _allow_track)
                        and state.get('next_gate_w') is not None):
                    # MAP CARROT (07-15, st2 probe: post-roll-fix DR lateral
                    # drift ~0.06 m/s -- navigable): during blind stretches
                    # steer on pose toward the next gate's map position
                    # instead of coasting. The carrot only needs to deliver
                    # the camera into fastgate's acquisition cone; the
                    # moment a hole >=40 px appears, the tracked branch owns
                    # terminal again. Frame conventions per the route phase.
                    _g = state['next_gate_w']
                    # MF_DR (07-15, fg55-57: post-punch KF cannot be
                    # stabilized): kinematic dead reckoning for the staging
                    # leg -- position from COMMANDED speed x GYRO yaw only
                    # (yaw is exact short-term: YAWPROBE 0.2 deg/415 deg),
                    # seeded at the tick. No accel, no KF. +/-20% speed
                    # error over the ~15 m loop lands within fastgate's
                    # acquisition cone.
                    if (os.environ.get('MF_DR') == '1'
                            and not _dpvo_control
                            and state.get('_mf_p') is not None):
                        _vn = 1.2 if state.get('_mf_lastpb', -0.06) < -0.04 \
                            else 0.35
                        state['_mf_p'][0] += _vn * math.cos(state['yaw']) / CMD_HZ
                        state['_mf_p'][1] += _vn * math.sin(state['yaw']) / CMD_HZ
                        # RELATIVE altitude hold (07-15, fg66 frames: zero
                        # vertical control on the corridor drifted the drone
                        # into the CEILING over 20 s). Kinematic DR knows no
                        # z; short-horizon est-z DELTA since the seed is
                        # trustworthy even though absolute z is not.
                        state['_mf_zerr'] = float(_pp[2]) \
                            - state.get('_mf_z0', float(_pp[2]))
                        _pp = state['_mf_p']
                    if (ticks >= 1 and not state.get('_mf_stage_done', False)
                            # LINE_TRANSIT: the line owns transit while it
                            # is trackable (skip staging) -- but when it has
                            # been lost >10 s (the near-gate extinction,
                            # tests 16/18/20), fall back to the PROVEN
                            # teardrop staging (the one historical gate-2
                            # success, fg62 family) instead of a blind
                            # carrot orbit.
                            and (os.environ.get('LINE_TRANSIT') != '1'
                                 or time.time() - state.get('line_wall', 0)
                                 > 10.0)):
                        # STAGING VIA A FORWARD-ONLY LOOP (07-15, fg52/53):
                        # the chute exit (x~12.5) is PAST the staging x, and
                        # brake-turn-in-place destabilizes the est (rotation
                        # across IMU gaps). Instead: a teardrop of waypoints
                        # -- forward-right, over the top north of the gate,
                        # down onto the staging point heading +x. Always
                        # translating, every leg turn within the yaw cap.
                        if state.get('_mf_wps') is None:
                            _gx, _gy, _gz = (float(_g[0]), float(_g[1]),
                                             float(_g[2]))
                            _stgb = float(os.environ.get('MF_STAGE_BACK', '3.0'))
                            if float(_pp[0]) < _gx - _stgb + 0.5:
                                # already WEST of the staging plane (short
                                # dash, 07-15 fg60): direct +y corridor to
                                # the staging point -- hugs x~8, clear of
                                # the pillar row (x>=10). No loop needed.
                                state['_mf_wps'] = [
                                    np.array([_gx - _stgb, _gy, _gz])]
                            else:
                                state['_mf_wps'] = [
                                    np.array([_gx + 2.5, _gy - 2.3, _gz]),
                                    np.array([_gx - 0.2, _gy + 2.0, _gz]),
                                    np.array([_gx - _stgb - 1.7, _gy + 1.3, _gz]),
                                    np.array([_gx - _stgb, _gy, _gz]),
                                ]
                            state['_mf_wpi'] = 0
                        _wps = state['_mf_wps']
                        _wpi = state['_mf_wpi']
                        if math.hypot(float(_wps[_wpi][0]) - float(_pp[0]),
                                      float(_wps[_wpi][1]) - float(_pp[1])) < 1.4:
                            _wpi += 1
                            state['_mf_wpi'] = _wpi
                            if _wpi >= len(_wps):
                                state['_mf_stage_done'] = True
                                print('MAPFOLLOW: staged on gate normal, '
                                      'releasing vision', flush=True)
                            else:
                                print(f'MAPFOLLOW: waypoint {_wpi}', flush=True)
                        if not state.get('_mf_stage_done', False):
                            _g = _wps[min(_wpi, len(_wps) - 1)]
                    _dx = float(_g[0]) - float(_pp[0])
                    _dyw = float(_g[1]) - float(_pp[1])
                    _cyw = math.cos(state['yaw']); _syw = math.sin(state['yaw'])
                    _exb = _cyw * _dx + _syw * _dyw
                    _eyb = -_syw * _dx + _cyw * _dyw
                    # DPVO-friendly transit (07-15): CONSTANT pitch (drag
                    # equilibrium ~1-1.5 m/s -- no hover, no lunges) and a
                    # tight yaw-rate cap (smooth rotation for DPVO tracking
                    # AND for our own gyro integration across the bursty
                    # IMU gaps).
                    _yrmax = float(os.environ.get('MF_YRMAX', '0.2'))
                    _brg = math.atan2(_eyb, _exb)   # TRUE bearing, +/-pi
                    if abs(_brg) > 1.2:
                        # target well off the nose (fg52: the staging point
                        # sits BEHIND the post-punch position) -- brake and
                        # turn in place at a faster cap; do not fly away
                        # while slowly yawing.
                        _yr_t = math.copysign(
                            max(_yrmax, 0.45), _brg)
                        _rb = state.get('_att_ri', 0.0)
                        _pb = 0.03
                    else:
                        _yr_t = max(-_yrmax, min(_yrmax, 1.0 * _brg))
                        _rb = max(-0.12, min(0.12, 0.08 * _eyb)) \
                            + state.get('_att_ri', 0.0)
                        _pb = -float(os.environ.get('MF_PITCH', '0.06')) \
                            if _exb > 1.0 else 0.02
                    state['_mf_lastpb'] = _pb   # MF_DR speed inference
                    if (os.environ.get('MF_DR') == '1'
                            and not _dpvo_control
                            and '_mf_zerr' in state):
                        # climbed since seed (zerr negative) -> descend
                        _dthr = max(-0.035, min(0.035,
                                0.06 * state['_mf_zerr']))
                    else:
                        _dthr = max(-0.04, min(0.04,
                                0.08 * (float(_pp[2]) - float(_g[2]))))
                    if os.environ.get('LINEFOLLOW') == '1':
                        # E3 height-hold (07-15 handoff): cyan-line image row
                        # is a REAL height observation -- overrides the est-z
                        # channel that ceiling-drifts. Lateral is OPT-IN
                        # (LINE_LAT=1): line-centering fought the map carrot
                        # in fg75 (the corridor deliberately leaves the line).
                        _line_age = time.time() - state.get('line_wall', 0)
                        if _line_age < 0.4:
                            if os.environ.get('LINE_LAT') == '1':
                                _rb = max(-0.2, min(0.2,
                                          _rb + 0.35 * state['line_off']))
                            # gain raised 0.08 -> 0.25 after test3: the old
                            # gain maxed at ~-0.012 dthr against a multi-metre
                            # climb (row err saturates at ~0.15).
                            _dthr = max(-0.05, min(0.05, _dthr
                                        + float(os.environ.get('LINE_KZ',
                                                               '0.25'))
                                        * (float(os.environ.get(
                                            'LINE_ROW_REF', '0.80'))
                                            - state['line_row'])))
                        elif (os.environ.get('LINE_LOST_DESCEND', '1') == '1'
                                and state.get('line_n', 0) > 0
                                and _line_age < 10.0):
                            # LINE_LOST_DESCEND (07-17, test3 frames): the
                            # climb pushes the line out the BOTTOM of the
                            # 20-deg-up camera -- the height observation
                            # self-extinguishes exactly when it's needed most
                            # (row 0.73->0.955->gone, then blind into the
                            # roof trusses). The line's ABSENCE is itself the
                            # height signal: firm descend bias until it
                            # reacquires (reacquisition resumes the row
                            # consumer above, so it cannot run away; >10 s
                            # stale means something else is wrong -- stop).
                            _dthr = max(-0.055, min(0.055, _dthr
                                        - float(os.environ.get('LINE_LOST_DZ',
                                                               '0.03'))))
                    if (os.environ.get('LINE_TRANSIT') == '1' and ticks >= 1
                            and _fgage < 0.4 and _fgw >= 60
                            and abs(_fgb_lt[0]) < 0.3
                            and -0.45 < _fgb_lt[1] < -0.05):
                        # TERMINAL CLIMB (07-19, test20): the low transit
                        # (line visible ahead) approaches the gate from
                        # BELOW the aperture; when a laterally-centered
                        # candidate hole shows moderately above the axis,
                        # bias a climb so b1 closes toward the release
                        # gate. The b1 floor (-0.45) keeps the high arch
                        # (b1 ~ -0.72 at close range) from dragging us up.
                        _dthr = max(-0.055, min(0.055, _dthr
                                    + float(os.environ.get('LT_CLIMB',
                                                           '0.035'))))
                    _lt_on = 0
                    _lt_age = time.time() - state.get('line_wall', 0)
                    _lt_latched = (ticks >= 1
                                   and time.time()
                                   - state.get('_lt_term_wall', 0)
                                   < float(os.environ.get('LT_TERM_S',
                                                          '6.0')))
                    # DIRECTION PRIOR (07-19, test18: after a scan the drone
                    # locked the line pointing BACK the way it came -- the
                    # painted line has no arrow). v5 used bearing-to-gate
                    # off the DR position and misfired mid-curve (test19:
                    # DR under-rotates the track, the gate tripped while
                    # the line-lock was GOOD). v6: pure gyro yaw (0.2 deg /
                    # 415 deg measured) vs the fixed course heading of the
                    # leg (judge anchors G1->G2: atan2(5.2,5.4) ~ 0.77
                    # rad); backward-following is ~180 deg off, so a wide
                    # +-100 deg window never trips on curvature.
                    _lt_aligned = abs(wrap(
                        state['yaw'] - float(os.environ.get(
                            'LT_COURSE_YAW', '0.77')))) \
                        < float(os.environ.get('LT_DIR_GATE', '1.75'))
                    if (os.environ.get('LINE_TRANSIT') == '1' and ticks >= 1
                            and not _lt_latched
                            and state.get('line_n', 0) > 0
                            and ((1.0 < _lt_age < 10.0)
                                 or (_lt_age < 0.4 and not _lt_aligned))):
                        # LINE-LOST SCAN (07-19, test16: line stale at t+38,
                        # the DR carrot then walked a 70 s phantom orbit
                        # around its imagined gate while the real drone
                        # wandered). The line is painted through the WHOLE
                        # course: hold position-ish, yaw-scan (toward the
                        # gate side when misaligned), and let
                        # LINE_LOST_DESCEND (E3 above) sink until it
                        # reacquires; >10 s stale falls through to the
                        # carrot as last resort.
                        _yr_t = math.copysign(
                            float(os.environ.get('LT_SCAN_YR', '0.18')),
                            wrap(float(os.environ.get('LT_COURSE_YAW',
                                                      '0.77'))
                                 - state['yaw']) if not _lt_aligned else 1.0)
                        _pb = 0.03      # brake-ish: scan in place, don't
                                        # wander (test20 scans arced ~3 m)
                        _rb = state.get('_att_ri', 0.0)
                        _lt_on = 2
                    elif (os.environ.get('LINE_TRANSIT') == '1'
                            and ticks >= 1 and _lt_age < 0.4
                            and _lt_aligned and not _lt_latched):
                        # LINE_TRANSIT (07-19, queued in the test4
                        # postmortem): the blind est/DR carrot owns the
                        # lateral channel in transit and drifts (test4:
                        # est walked a clean dogleg while the real track
                        # hit the parked airplane). The painted line IS
                        # the route: while it is fresh, yaw-to-line +
                        # roll-centering REPLACE the carrot's lateral
                        # commands -- never both at once (the fg75
                        # failure); the carrot above remains the stale
                        # fallback. Height stays with the E3 row consumer;
                        # the fgpursuit/punch machinery owns terminal.
                        # v2: yaw at the line HEAD (lookahead -- where the
                        # line goes), roll centers the centroid. v1 yawed at
                        # the centroid and parked on top of the line.
                        _lo = float(state['line_off'])
                        _lho = float(state.get('line_head_off', _lo))
                        _yr_t = max(-0.35, min(0.35,
                                float(os.environ.get('LT_KYAW', '0.9'))
                                * _lho))
                        _rb = max(-0.15, min(0.15,
                                float(os.environ.get('LT_KLAT', '0.25'))
                                * _lo)) + state.get('_att_ri', 0.0)
                        _pb = -float(os.environ.get('LT_PITCH', '0.06'))
                        if (_fgage < 0.4
                                and _fgw >= float(os.environ.get(
                                    'LT_BRAKE_W', '58'))
                                # 58, not 45 (test39: braking at w=45
                                # (~6.4 m) stalled the drone OUTSIDE the
                                # w=60 release -- it hovered there, never
                                # released, never punched. Brake just
                                # under the release width.)
                                and state.get('_lt_map_ok', True)
                                and abs(_fgb_lt[0]) < 0.4
                                and -0.35 < _fgb_lt[1] < 0.4):
                            # TERMINAL BRAKE (07-19, tests 27-32): every
                            # terminal so far was a geometric fly-by --
                            # the pursuit only got 0.2-2 s of authority
                            # after release and could not null the LOS
                            # sweep at line-follow speed (the punch then
                            # fires mid-sweep and misses left). A
                            # candidate gate hole ahead => drop transit
                            # pitch: slow closing = seconds of steering =
                            # a real collision course before the punch.
                            _pb = -float(os.environ.get('LT_PITCH_NEAR',
                                                        '0.0'))
                        _lt_on = 1
                    if now - state.get('_mf_log', 0) > 0.5:
                        state['_mf_log'] = now
                        jlog('mapfollow', ex=round(_exb, 2), ey=round(_eyb, 2),
                             rb=round(_rb, 3), pb=round(_pb, 3),
                             p=[round(float(x), 2) for x in _pp],
                             lr=round(state.get('line_row', -1.0), 3),
                             ln=state.get('line_n', 0),
                             lt=_lt_on,
                             lo=round(state.get('line_off', 0.0), 3),
                             lho=round(state.get('line_head_off', 0.0), 3))
            _rb = max(-0.22, min(0.22, _rb))
            _dthr = max(-0.055, min(0.055, _dthr))
            if now - state.get('_att_log', 0) > 0.5:
                state['_att_log'] = now
                jlog('att', rb=round(_rb, 3), dthr=round(_dthr, 3),
                     ib=round(state.get('_att_ib', 0.0), 3),
                     ri=round(state.get('_att_ri', 0.0), 3),
                     pb=round(_pb, 3),
                     rrate=round(state.get('_att_rrate', 0.0), 2),
                     r_est=round(state['roll'], 3),
                     p_est=round(state['pitch'], 3),
                     yaw=round(state['yaw'], 3),
                     gz=round(float(state['gyr'][2]), 3),
                     yrc=round(_yr_t, 3))
            level_cmd(0, 0, _dthr, pitch_bias=_pb, yr=SZ * _yr_t,
                      roll_bias=_rb, att=True)
        else:
            _pb = float(os.environ.get('FGP_PITCH', '-0.15')) if _close_t else -0.05
            # crab compensation (YAWCAL): constant measured lateral drift while
            # translating -- apply uniformly (tracked, coast, seek).
            _vy = max(-0.6, min(0.6, _vy + state.get('crab_vy', 0.0)))
            level_cmd(SX * _vx, SY * _vy, _vz, pitch_bias=_pb, yr=SZ * _yr_t)
        aborted = aborted or guards('fgpursuit')
        if now - phase_t0 > 90:
            aborted = 'fgpursuit timeout'
        time.sleep(1 / CMD_HZ)
        continue
    elif phase == 'acquire':
        print(f'flying pad lock: {state["tgt"].round(2)}', flush=True)
        phase, phase_t0 = 'approach', now
        continue
    elif phase == 'approach':
        state['trim_ok'] = False    # maneuvering: real a_y breaks the trim
        det_age = now - state.get('det_wall', 0.0)
        if det_age > 20.0:
            aborted = 'no detections > 20 s'
        elif age > 3.0:
            d = obs / max(1e-6, np.linalg.norm(obs[:2]))
            with KF_LOCK:
                pz = float(KF.p[2])
            vz_creep = max(-0.7, min(0.7, 1.1 * (pz - float(state['next_gate_w'][2]))))
            level_cmd(SX * 1.2 * d[0], SY * 1.2 * d[1], vz_creep, pitch_bias=-0.10)
        else:
            fwd, lat, dwn = obs[0], obs[1], obs[2]
            # SERVO APERTURE BIAS (07-14, nt8 frames): every servo crossing
            # misses RIGHT of the visible gate -- the judge-calibrated 07-11
            # offset (true aperture ~2 m LEFT of the PnP origin, AIMBIAS_Y)
            # was applied to the blind anchor but never to the live servo lat.
            # Steer on the aperture, not the PnP origin.
            lat = lat + float(os.environ.get('AIMBIAS_Y', '0.0'))
            # FASTGATE OVERRIDE (07-14): live 30 Hz HOLE-center bearing beats
            # the 2 Hz / 450 ms GateNet lat -- and it needs no aperture bias
            # (it tracks the hole itself). fwd stays GateNet/DR (fg range is
            # glow-corrupted); lat/dwn = live bearing scaled by fwd.
            _fgw = state.get('fg_wall', 0.0)
            if os.environ.get('FASTGATE') == '1' and now - _fgw < 0.35:
                _fb = state['fg_bear']
                lat = _fb[0] * fwd
                dwn = _fb[1] * fwd
                age = now - _fgw          # live again: unfreeze stale logic
            # misalignment speed governor: off-axis (lat) OR off-height (KF-z
            # error) -> slow down, buy correction time (flight #3 clipped a post
            # at 2.5 m/s with 2.5 m lat error; #5 clipped the top bar high)
            with KF_LOCK:
                pz = float(KF.p[2])
            # terminal vision servo flies AT gate height -- the G1 obstacle
            # overfly now lives in the route spline (hump waypoint), and a high
            # perch blinds the 20-deg-up camera (flight #10)
            z_tgt = float(state['next_gate_w'][2])
            dz_err = pz - z_tgt
            if now - state.get('_ap_log', 0) > 0.5:
                state['_ap_log'] = now
                jlog('approach', fwd=round(float(fwd), 2), lat=round(float(lat), 2),
                     dz=round(float(dz_err), 2), age=round(float(age), 2))
            lat_slow = max(0.4, 1.0 - 0.3 * min(abs(lat), 2.0))
            dwn_slow = max(0.45, 1.0 - 0.5 * min(abs(dz_err), 1.5))
            vx_ref = min(1.4, max(0.6, 0.4 * (fwd - 1.0))) * lat_slow * dwn_slow   # slow: the detector drops the gate ~9 m out above ~1.5 m/s (flight #33)
            lat_gain = 0.35 if fwd < 8.0 else 0.2   # halved: servo sway through the crossing (07-10)
            if age > 0.6:
                # STALE-LAT FREEZE (07-14, servo_nt4): the detector's last close
                # obs freezes; steering on frozen lat for 2.5 s double-corrects
                # and drives the drone off-line sideways. Hold the line instead.
                lat_gain = 0.0
            vy_ref = max(-1.0, min(1.0, lat_gain * lat))
            # APPROACH_VYBIAS (07-14, nt15): a real ~0.3 m/s rightward push acts
            # through the terminal area (lat grew -0.57 -> -1.05 across fresh
            # obs faster than the 2 Hz servo corrects; same 0.34 m/s measured in
            # the nt8/nt13 coasts). Velocity-level, invisible to the IMU (est
            # never sees it) -- counter it with a constant feed-forward.
            vy_ref += float(os.environ.get('APPROACH_VYBIAS', '0.0'))
            # align-then-shoot: the start-light poles flank the course at ~x 5
            # with a ~+-1 m corridor (flights #3/#5/#6/#8 clipped them arriving
            # 0.3-2 m off-axis; #4 threaded it dead-center). Center FIRST, then
            # accelerate through.
            if fwd > 3.5 and abs(lat) > 0.25 and age <= 0.6:
                vx_ref = 0.5
                vy_ref = max(-1.2, min(1.2, 1.0 * lat))
            # altitude: hold the MAP gate height on KF z (pure-IMU + xy-only
            # fixes, so unbiased). The obs z channel is corrupted by pitch
            # transients AND a per-run spawn artifact -- chasing it climbed
            # #3/#5/#6 into the ~2 m obstacle (id 1001) and floored #7.
            vz_ref = max(-0.7, min(0.7, 1.1 * dz_err))
            yr_cmd = SZ * max(-0.35, min(0.35, 0.10 * lat))
            level_cmd(SX * vx_ref, SY * vy_ref, vz_ref, pitch_bias=-0.12, yr=yr_cmd)
            rng = float(np.linalg.norm(obs))
            if fwd < ORIGIN_OFFSET + 0.2 and rng < 4.5 and abs(lat) < 2.0:
                print(f'passed target plane (DR, rng {rng:.1f}); braking', flush=True)
                phase, phase_t0 = 'post', now
                continue
            if fwd < -1.0 and age < 2.5:
                # only drop on a FRESH obs that says behind. Near the gate the
                # DR target (obs fallback when stale) reports "behind" while the
                # drone is still ~3 m out -- that dropped a centered lock and
                # reverted to blind route -> clipped the top (servo_nt run).
                print('target behind at range -- dropping lock, back to route', flush=True)
                state['obs'] = None; state['tgt'] = None; state['obs_wall'] = 0.0
                state['landmark_w'] = None
                phase, phase_t0, route_end_t0 = 'route', now, None
                continue
            fresh = age < 2.5
            fwd_ap = fwd - ORIGIN_OFFSET   # distance to the APERTURE plane
            if ARCHTEST:
                # COAST, not punch: with corrected ranges the blind zone is
                # only the last ~1.5 m -- servo all the way in, then cross
                # straight at approach speed. The 2 m/s blind lunge from
                # 2.8 m (pursuit-oblique + stale-lat steering) missed the
                # 1.5 m aperture left on arch34-36.
                yaw_ok = abs(wrap(0.0 - state['yaw'])) < 0.15
                if fresh and fwd_ap < 1.5 and abs(lat) < 0.25 and abs(dz_err) < 0.5 and yaw_ok:
                    punch_t0, punch_dur = now, (fwd_ap + 1.5) / 1.0
                    phase = 'punch'
                    print(f'COAST from {fwd_ap:.1f} m (lat {lat:.2f} dwn {dwn:.2f})', flush=True)
            elif fresh and fwd_ap < 2.6 and abs(lat) < 0.35 and abs(dz_err) < 0.6:
                punch_t0, punch_dur = now, max(0.8, fwd_ap / 2.0 + 1.2)
                state['_punch_lat0'] = float(lat)
                phase = 'punch'
                print(f'PUNCH from {fwd_ap:.1f} m to aperture (lat {lat:.2f} dwn {dwn:.2f})', flush=True)
        # PUNCH-ON-RECENT-OBS: the detector reliably dies ~6 m out (gate slides
        # under the 20-deg-up camera), so a fresh-obs-at-2.6m trigger never
        # fires (#41/#42). Short-horizon DR from the last fresh obs is
        # cm-accurate right after a vision-velocity fix -- fire on that.
        lk = state.get('_last_fresh')
        if (phase == 'approach' and lk is not None
                and now - lk[4] < float(os.environ.get('PUNCH_WIN', '2.5'))):
            with KF_LOCK:
                dp = KF.p - lk[3]
            fwd_est = lk[0] - float(np.hypot(dp[0], dp[1]))
            trig = 1.5 if ARCHTEST else float(os.environ.get('PUNCH_TRIG', '2.8')) + ORIGIN_OFFSET
            lat_ok = 0.25 if ARCHTEST else float(os.environ.get('PUNCH_LAT', '0.4'))
            lk_lat = lk[1] + float(os.environ.get('AIMBIAS_Y', '0.0'))  # aperture-ref
            if fwd_est < trig and abs(lk_lat) < lat_ok and (
                    not ARCHTEST or abs(wrap(0.0 - state['yaw'])) < 0.15):
                if ARCHTEST:
                    punch_t0, punch_dur = now, (max(0.3, fwd_est) + 1.5) / 1.0
                    print(f'COAST (obs-DR) est {fwd_est:.1f} m (last lat {lk[1]:.2f}, obs age {now-lk[4]:.1f}s)', flush=True)
                else:
                    _pvx0 = float(os.environ.get('PUNCH_VX', '2.0'))
                    punch_t0 = now
                    punch_dur = (max(0.6, fwd_est) + 1.5) / _pvx0
                    state['_punch_lat0'] = float(lk_lat)
                    print(f'PUNCH (obs-DR) est {fwd_est:.1f} m (last lat {lk[1]:.2f}, obs age {now-lk[4]:.1f}s)', flush=True)
                phase = 'punch'
        if now - phase_t0 > 60:
            print('approach timeout -- dropping lock, back to route', flush=True)
            state['obs'] = None; state['tgt'] = None; state['obs_wall'] = 0.0
            state['landmark_w'] = None
            phase, phase_t0, route_end_t0 = 'route', now, None
    elif phase == 'punch':
        if ARCHTEST:
            # coast v2 (07-10 frames: bare slow coast angled ~1 m right of
            # the aperture -- the oblique nose plus 5 s of blind drift):
            # yaw-aligned trigger, 1 m/s, and hold yaw on the chute axis
            # through the crossing
            yr_hold = SZ * max(-0.3, min(0.3, 1.0 * wrap(0.0 - state['yaw'])))
            level_cmd(SX * 1.0, 0.0, 0, yr=yr_hold)
        else:
            # hold the lateral line through the blind drive (#32 crossed drifting)
            lat_p = float(obs[1]) if obs is not None else 0.0
            if os.environ.get('PUNCH_STRAIGHT') == '1':
                # FEED-FORWARD COAST (07-14): null the remembered lateral offset
                # exactly once over the punch duration (vy = lat0/dur), instead
                # of drifting with it (nt8 crossed on the edge) or pushing on a
                # frozen lat with no feedback (overshoots). Hold yaw on-axis.
                # FF coast (07-14): null the remembered lat over the coast, PLUS
                # a constant counter-RATE for the ~0.3 m/s rightward push (the
                # terminal 'inner assist', measured nt8/nt13/nt15 -- a rate, so
                # it compensates correctly for any coast duration). Faster coast
                # (PUNCH_VX) = less blind time = less push variance.
                _lat0 = state.get('_punch_lat0', 0.0)
                _vy_ff = (max(-0.5, min(0.5, _lat0 / max(punch_dur, 0.5)))
                          + float(os.environ.get('PUNCH_VYBIAS', '0.0')))
                _pvx = float(os.environ.get('PUNCH_VX', '1.2'))
                # FASTGATE closed-loop crossing: while the hole is in view,
                # steer on its LIVE bearing (out-corrects the variable push);
                # fall back to the FF when it finally leaves the frame (<1.5 m).
                _fgw = state.get('fg_wall', 0.0)
                if os.environ.get('FASTGATE') == '1' and now - _fgw < 0.35:
                    _rem = max(0.5, _pvx * (punch_dur - (now - punch_t0)))
                    _lat_live = state['fg_bear'][0] * _rem
                    _vy_ff = max(-0.8, min(0.8, 0.9 * _lat_live))
                yrh = SZ * max(-0.3, min(0.3, 1.0 * wrap(0.0 - state['yaw'])))
                level_cmd(SX * _pvx, SY * _vy_ff, 0, yr=yrh)
            else:
                level_cmd(SX * 2.0, SY * max(-0.5, min(0.5, 0.5 * lat_p)), 0)
        if now - punch_t0 > punch_dur:
            phase, phase_t0 = 'post', now
            print('punch window over, braking', flush=True)
    elif phase == 'post':
        level_cmd(0, 0, 0)
        if now - phase_t0 > 2.0:
            print('no tick -- back to route', flush=True)
            state['obs'] = None; state['tgt'] = None; state['obs_wall'] = 0.0
            state['landmark_w'] = None
            phase, phase_t0, route_end_t0 = 'route', now, None
    elif phase == 'route':
        state['trim_ok'] = True     # steady cruise: force-balance trim valid
        # map-spline carrot on KF dead reckoning, nose along the carrot
        with KF_LOCK:
            p = KF.p.copy()
        # WINDOWED CARROT (07-11/12, g2test runs 1-4): with the G2TEST leg the
        # spline passes within ~2 m of the start chute and the raw global
        # nearest-s projection leaps onto the turn leg (run 3: 2.99 -> 12.61;
        # run 4 showed rate-limiting the advance only CREEPS to the same wrong
        # leg). Search only a window around the previous s so the other leg is
        # never a candidate, and keep the advance monotonic + rate-limited.
        s_prev = state.get('_s_prev', 0.0)
        s_raw = TRAJ.nearest_s_window(p, s_prev - 1.0, s_prev + 2.0)
        s_here = min(s_raw, s_prev + CARROT_DS) if s_raw > s_prev else s_prev
        state['_s_prev'] = s_here
        # carrot stops just past the next UN-TICKED gate: #12 blew through G2's
        # plane blind because the carrot ran the whole spline while validation
        # still expected G1
        s_stop = (S_GATES[ticks] if ticks < len(S_GATES) else TRAJ.s_max) + 2.0
        if os.environ.get('G2FULL') == '1':
            # judge-gate-identity probe (07-12): fly the WHOLE two-gate route
            # regardless of ticks -- if the judge's gate 1 is the ribbon gate
            # (07-08 forensics: arch17/23/27 all ticked THERE), the tick
            # arrives at the second crossing, not the chute gate
            s_stop = TRAJ.s_max
        s_ref = min(s_here + LEAD, s_stop, TRAJ.s_max)
        ref = TRAJ.sample(s_ref)
        if (os.environ.get('STRAIGHTCHUTE') == '1' and ticks == 0
                and os.environ.get('NOFIX') == '1'):
            # STRAIGHT CHUTE (07-14, nt11 frames): RECENTER parks the drone ON
            # the biased aperture line, but the spline curves from (0,0) and
            # the carrot DRAGS the drone back off the line (est crossed -1.38
            # aiming -1.98 -> clipped the right edge). Pre-tick: fly a straight
            # line at the aperture's y/z from wherever we are.
            _ngw = state['next_gate_w']
            ref = {'pos': np.array([p[0] + 2.0, float(_ngw[1]), float(_ngw[2])]),
                   'v': ref['v'], 'tang': np.array([1.0, 0.0, 0.0])}
        d = ref['pos'] - p
        # route-phase forensics: live KF estimate vs spline carrot, so a
        # missed gate can be diagnosed as estimate drift vs tracking error
        jlog('route', p=np.round(p, 3).tolist(), s=round(float(s_here), 2),
             ref=np.round(ref['pos'], 3).tolist())
        dh = float(np.hypot(d[0], d[1]))
        v_ref = min(ref['v'], V_ROUTE_MAX)
        if s_here > s_stop:
            v_ref *= 0.5            # past the stop point: ease back, no lunge
        # descend/climb-priority: on the near-vertical dive (high gate -> red
        # gate) cap horizontal speed so the clipped vz can keep up with the path
        v_ref *= max(0.3, 1.0 - 0.6 * min(abs(float(d[2])), 2.0))
        if dh > 1e-3:
            v_w = v_ref * d[:2] / dh
        else:
            v_w = np.zeros(2)
        cyw, syw = math.cos(state['yaw']), math.sin(state['yaw'])
        vx_b_ref = cyw * v_w[0] + syw * v_w[1]
        vy_b_ref = -syw * v_w[0] + cyw * v_w[1]
        vz_ref = max(-0.8, min(0.8, 1.0 * (p[2] - float(ref['pos'][2]))))
        # yaw follows the spline TANGENT, not the carrot bearing: bearing
        # swings with cross-track error and flips 180 deg when the carrot ends
        # up behind (flight #14's spin into structure); tangent is stable and
        # keeps the camera downcourse for re-acquisition
        yaw_ref_t = math.atan2(float(ref['tang'][1]), float(ref['tang'][0]))
        if os.environ.get('NOFIX') == '1' and ticks == 0:
            # HEADING LOCK on the blind chute (07-12 recenter traces): the
            # obs-stale scan sweep oscillated yaw the entire chute (obs are
            # ALWAYS stale under NOFIX) and yaw rotation across the 28 ms
            # IMU gaps banks yaw error that rotates the whole DR frame --
            # est crossed at a perfect -2.06 while the frames show the
            # truth 2-3 m right into the top bar. Camera is unused
            # pre-tick; hold heading, full speed.
            yaw_ref_t = 0.0
        elif ARCHTEST and ARCH_SWEEP_S < s_here < s_stop - 1.0:
            # perception-aware yaw (bench, arch15 corridor: nose a median
            # 72 deg off the est-expected gate under tangent yaw -- the
            # detector starves because the camera never faces the target).
            # Aim the camera at the KF-expected gate; the residual bearing
            # error is estimate drift only (~10-20 deg at 5 m, in FOV).
            rel_g = HIGH_W - p
            yaw_ref_t = math.atan2(float(rel_g[1]), float(rel_g[0]))
            if state['obs'] is None or now - state['obs_wall'] > 1.5:
                # narrow scan about the expected bearing until a lock
                if state.get('_sweep_t0') is None:
                    state['_sweep_t0'] = now
                yaw_ref_t += 0.3 * math.sin(0.9 * (now - state['_sweep_t0']))
                vx_b_ref *= 0.3; vy_b_ref *= 0.3
        yr_cmd = SZ * max(-0.5, min(0.5, 1.2 * wrap(yaw_ref_t - state['yaw'])))
        if os.environ.get('CTRLV') == '1':
            # HYBRID (RATEPROBE verdict): the armed-race vehicle obeys
            # rate commands linearly to 0.02 rad/s yet damps translation
            # (inner assist) -- 100 s hovers under commanded tilt. Route
            # translation goes through the sim's velocity interface,
            # which tracked laterally in every probe. NED z down = spawn
            # z down (verified); xy mapping assumed spawn-aligned --
            # first flight verifies via the position fixes.
            send_vel(v_w[0], v_w[1], -vz_ref)
        else:
            level_cmd(SX * vx_b_ref, SY * vy_b_ref, vz_ref, pitch_bias=-0.08, yr=yr_cmd)
        state['_viz_ref'] = ref['pos']
        if now - last_route_log > 0.5:
            last_route_log = now
            jlog('route', p=p.round(2).tolist(), s=round(s_here, 1),
                 ref=ref['pos'].round(2).tolist(), yaw=round(state['yaw'], 3))
        # vision handoff: det_loop only pins map-validated landmarks now
        # vision handoff for EVERY gate: KF world position over-integrates at
        # speed (+3 m by the gate, flight #31 dashboard), so the terminal leg
        # must servo on RELATIVE obs -- the only recipe that ever ticked
        if (state['obs'] is not None and now - state['obs_wall'] < 0.8
                and float(state['obs'][0]) < 7.5
                and abs(float(state['obs'][1])) < 2.5
                and (not ARCHTEST or s_here > 3.5)
                and (not os.environ.get('DIRTEST') or ticks >= 1)):
            # DIRTEST blocks the handoff only PRE-tick (blind calibrated
            # chute for gate 1); after tick 1 the servo takes gate 2 --
            # hybrid config for the fresh-sim tick budget (07-12)
            # HANDOFF RESTORED for ARCHTEST (07-10): route-only flying
            # reproduces the est-rotation drift every run (v3385 canary:
            # est crossed 0.25 m off-center, frames show 2-3 m right at
            # Station 21's pillar). All four ticks ever scored came via
            # this relative-obs servo, which rotation error cannot touch.
            # s_here > 1.5 keeps it from firing over the pad (arch37).
            # handoff only INSIDE the furniture radius: an early lock pulls the
            # drone off the dogleg straight into the x~5-7 structure (flight #36).
            # |lat| bound: arch26 locked a junk solve 4.3 m off-axis and the
            # approach chased it into a wall -- the route target is AHEAD
            print(f'vision lock on route: {state["obs"].round(2)}', flush=True)
            phase, phase_t0 = 'approach', now
        if s_here > s_stop - 2.6:
            # past the gate without a tick: HEIGHT SWEEP retry -- back up and
            # re-cross 0.4 m higher (the aperture height is only known to
            # ~0.5 m; #17 crossed at 1.3, #18 at 2.24, no tick; #4 ticked ~2.7)
            route_end_t0 = route_end_t0 or now
            # G2TEST: judge flip lags the crossing ~2 s and run 3's tick
            # arrived during the landing -- give it 8 s before giving up
            if now - route_end_t0 > (8.0 if os.environ.get('G2TEST') == '1' else 4.0):
                retries += 1
                if retries > 3:
                    aborted = f'gate not ticked after {retries - 1} height retries, ticks={ticks}'
                else:
                    gidx = min(ticks, 2)
                    gate_z_off[gidx] = +0.35 if retries == 1 else -0.35
                    gh = HIGH_W + np.array([0, 0, gate_z_off[0]])
                    g1 = G1_W + np.array([0, 0, gate_z_off[1]])
                    g2 = G2_W + np.array([0, 0, gate_z_off[2]])
                    TRAJ, S_GATES = build_traj(gh, g1, g2)
                    gate_w = (gh, g1, g2)[gidx]
                    state['next_gate_w'] = gate_w.copy()
                    retreat_pt = gate_w - 7.0 * THRU[gidx]
                    if gidx == 0:
                        retreat_pt[1] -= 2.2   # retreat via the dogleg lane, not through the x~5-7 furniture (flight #37)
                    retreat_pt[2] = gate_w[2]
                    if ARCHTEST:
                        land('archtest pass complete (no tick)'); break
                    print(f'no tick -- retry {retries}: gate {gidx} aperture z {gate_w[2]:.2f}, retreating', flush=True)
                    jlog('retry', n=retries, gate=gidx, gate_z=float(gate_w[2]))
                    _viz_static()          # spline changed: re-log the reference path
                    phase, phase_t0, route_end_t0 = 'retreat', now, None
        else:
            route_end_t0 = None
        if now - phase_t0 > 90:
            aborted = f'route timeout, ticks={ticks}'
    elif phase == 'retreat':
        # fly straight back to the re-approach point, then re-run the route
        with KF_LOCK:
            p = KF.p.copy()
        d = retreat_pt - p
        dist = float(np.linalg.norm(d[:2]))
        if dist < 1.2:
            print('retreat done -- re-running route', flush=True)
            phase, phase_t0 = 'route', now
        else:
            v_w = 1.3 * d[:2] / max(dist, 1e-3)
            cyw, syw = math.cos(state['yaw']), math.sin(state['yaw'])
            vx_b_ref = cyw * v_w[0] + syw * v_w[1]
            vy_b_ref = -syw * v_w[0] + cyw * v_w[1]
            vz_ref = max(-0.6, min(0.6, 0.9 * (p[2] - float(retreat_pt[2]))))
            level_cmd(SX * vx_b_ref, SY * vy_b_ref, vz_ref, pitch_bias=-0.06, yr=0.0)
        if now - phase_t0 > 25:
            aborted = f'retreat timeout, ticks={ticks}'
    g = guards(phase)
    if g: aborted = g
    with KF_LOCK:
        _pv = KF.p.copy()
    viz_tick(_pv, state.get('_viz_ref'))
    time.sleep(1/CMD_HZ)

if aborted:
    land(aborted)
state['stop'] = True
time.sleep(1)
log_f.close()
print(f'DONE: {aborted or "SUCCESS"} | TICKS: {ticks}', flush=True)
