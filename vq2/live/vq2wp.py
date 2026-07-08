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

HOVER = 0.2675
CMD_HZ = 50.0
TILT_ABORT = math.radians(55)
RATE_GAIN = 1.93
SIGN_R, SIGN_P = -1.0, +1.0
KP = 1.8
K_V = 0.12
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
ORIGIN_OFFSET = 1.9   # gate-frame origin sits this far behind the aperture plane
# QUALIFIER COURSE RE-MAP (07-07, VQ2-COURSE-01): the ribbon's gate 1 is at
# [26.9, 8.6] (191 georeferenced obs, dominant cluster; confirmed visually --
# the ribbon threads it). The [11, 0] object is a DECOY the ribbon bypasses;
# every prior 'crossing' threaded it perfectly for zero points.
# CORNER-TRIANGULATED (tick corpus, 79 obs, 8.2 px): the ribbon gate's
# PHYSICAL structure is at [11.7, 5.2] -- the [26.9,8.6] cluster was the
# PnP origin's far-projection (origin-offset illusion at range). Aperture
# centroid [11.68, 5.17, -1.06], face along y => through-normal ~ +x.
N1 = np.array([0.996, -0.087, 0.0]); N1 /= np.linalg.norm(N1)
G1_W = np.array([26.9, 8.6, -1.3])    # obs-ORIGIN anchor (matching only)
G1_AP = np.array([11.68, 5.17, -1.1]) # measured aperture centroid
HIGH_W = G1_AP.copy()                 # route/punch aim point
G2_W = np.array([44.8, 1.9, -1.5])    # provisional downstream cluster
DECOY_W = np.array([10.97, -0.11, -1.3])  # non-course gate: NAV LANDMARK ONLY
#                     (122-obs cluster; reliable close-range position fixes
#                      on the way out -- never a target, never sets tgt/lock)
N2 = N1.copy()
GATES_W = [HIGH_W, G2_W, G2_W]
THRU = [N1, N2, N2]

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
        pts = np.array([
            [0.0, 0.0, -1.3],
            [4.0, 0.7, -1.3],
            [9.4, 1.7, -1.3],       # arch center (replay-measured)
            [12.5, 2.3, -1.3],
        ])
        tr = GateTrajectory(pts, v_cruise=1.2, phi_max_deg=15.0,
                            tilt_budget_deg=12.0, vz_max=0.55)
        return tr, [tr.nearest_s(np.array([9.4, 1.7, -1.3])), tr.s_max, tr.s_max]

TRAJ, S_GATES = build_traj(HIGH_W, G1_W, G2_W)

# ---- Rerun live dashboard (best-effort: never raises into the control loop) ----
MAC_VIEWER = 'rerun+http://100.101.13.126:9876/proxy'   # Mac over Tailscale (VQ1 convention)
VIZ = os.environ.get('NOVIZ') != '1'
_rr = None
if VIZ:
    try:
        import rerun as rr
        rr.init('vq2wp')
        rr.connect_grpc(MAC_VIEWER)
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
        jp = state.get('jpeg')
        if jp is not None and now - _viz_last[1] > 0.2:
            _viz_last[1] = now
            rr.set_time('t', duration=now)
            rr.log('camera/fpv', rr.EncodedImage(contents=jp, media_type='image/jpeg'))
    except Exception:
        pass
LEAD = 2.5                      # carrot lead (m); drone-locked s_ref cannot run away
V_ROUTE_MAX = float(os.environ.get('VMAX', '1.0'))  # slow everywhere: keep the detector locked to punch range
ACCEPT_R = 3.0                  # legacy radius gate (POLICY=radius rollback + hysteresis bound)
OBS_POLICY = os.environ.get('POLICY', 'huber_area')  # 'huber_area' | 'radius'
def _load_npy(p):
    try:
        return np.load(p)
    except Exception:
        return None
DECOY_C = _load_npy(r'C:\Users\alexj\g1_corners_world.npy')
G2RIB_C = _load_npy(r'C:\Users\alexj\g2rib_corners_world.npy')
IDENT_ON = OBS_POLICY in ('huber_area', 'huber') and \
    os.environ.get('NOIDENT', '0') != '1'
HUBER_DELTA = 1.5               # m: miss below this = full-weight fix
RANGE_RATIO_MIN = 0.55          # identity gate (see det_loop comment)
RANGE_SCALE = 1.00              # gates are VQ1-size (Alex): PnP ranges are TRUE; the KF under-integrates instead

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
         'next_gate_w': HIGH_W.copy(), 'nav_ready': False}
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
            if last_us is not None and us > last_us:
                dt = (us - last_us) / 1e6
                state['roll'] += gyr[0] * dt
                state['pitch'] += (-gyr[1]) * dt      # wfix
                state['yaw'] += (-gyr[2]) * dt        # wfix (yaw mirrored too)
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
                if state.get('airborne') and state.get('trim_ok'):
                    gm = max(abs(gyr[0]), abs(gyr[1]), abs(gyr[2]))
                    an = math.sqrt(acc[0]**2 + acc[1]**2 + acc[2]**2)
                    if gm < 0.4 and abs(an - 9.81) < 1.0 and abs(gyr[2]) < 0.03:
                        # |gz| gate: on an ARC the bank is real lateral accel
                        # (run-20: trims clamped +-2 deg oscillating on the
                        # curved route) -- only trim on straight segments
                        TRIM_WIN.append((us / 1e6, acc[1]))
                    else:
                        TRIM_WIN.clear()
                    if TRIM_WIN and TRIM_WIN[-1][0] - TRIM_WIN[0][0] >= 1.0:
                        ay = sum(w[1] for w in TRIM_WIN) / len(TRIM_WIN)
                        roll_true = math.asin(max(-1.0, min(1.0, -ay / 9.81)))
                        err = roll_true - state['roll']
                        err = max(-0.035, min(0.035, err))
                        state['roll'] += err
                        jlog('roll_trim', err_deg=round(math.degrees(err), 2))
                        TRIM_WIN.clear()
                a_lvl = accel_level(acc, state['roll'], state['pitch'])
                cyw, syw = math.cos(state['yaw']), math.sin(state['yaw'])
                a_w = np.array([cyw * a_lvl[0] - syw * a_lvl[1],
                                syw * a_lvl[0] + cyw * a_lvl[1], a_lvl[2]])
                with KF_LOCK:
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
            jlog('collision', **msg.to_dict())
        elif t == 'ENCAPSULATED_DATA':
            d = bytes(msg.data)
            if d and d[0] == 1:
                try:
                    rs = struct.unpack('<BQqqIq', d[:37])
                    if rs[4] != state['gate_idx']:
                        jlog('gate_tick', idx=rs[4])
                    state['gate_idx'] = rs[4]
                    state['race_finish_ns'] = rs[3]
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
    print('detector ready', flush=True)
    last_ns = 0
    while not state['stop']:
        img, ns = state['frame'], state['frame_ns']
        f_wall = state.get('frame_wall', 0.0)
        if img is None or ns == last_ns:
            time.sleep(0.01); continue
        last_ns = ns
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
            best = (best[0], best[1] * RANGE_SCALE, best[2], best[3])   # GateNet range bias: PnP assumes VQ1 gate size;
                                                      # qualifier gates are smaller (Alex free-cam 07-05)
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
            ident_full = False
            miss, match_g = 1e9, None
            for gw_map in (G1_W, G2_W, DECOY_W):
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
            _cmap = (G2RIB_C if match_g is G1_W else
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
                    uv_ = np.array([rel_[k_, 0] / rel_[k_, 2] * 226.0 + 319.5,
                                    rel_[k_, 1] / rel_[k_, 2] * 226.0 + 179.5])
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
                if bad:
                    dists_ = []
                    for k_ in range(min(len(best[2]), len(_cmap))):
                        if rel_[k_, 2] <= 0.2:
                            dists_.append(None); continue
                        uv_ = np.array([rel_[k_, 0] / rel_[k_, 2] * 226.0 + 319.5,
                                        rel_[k_, 1] / rel_[k_, 2] * 226.0 + 179.5])
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
                # xy-only fix: zero the z innovation (obs z carries a per-run
                # spawn-tilt bias of -0.6..-4 m; the map owns z)
                g_w_upd = g_w.copy()
                with KF_LOCK:
                    g_w_upd[2] = float(match_g[2]) - KF.p[2]
                    ok = KF.update_position(match_g - g_w_upd,
                                            rng=rng_meas, r_scale=r_scale)
                    nis = getattr(KF, 'last_nis', None)
                    p_now = KF.p.round(2).tolist(); v_now = KF.v.round(2).tolist()
                state['fix_count'] = state.get('fix_count', 0) + 1
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
                        and np.linalg.norm((g_w - prev[0])[:2]) < 1.5):
                    # consistency gate: consecutive obs must be the SAME physical
                    # gate (two neighbors matched to one map slot injected an
                    # 8 m/s phantom velocity, flight #38 runaway)
                    v_meas = -(g_w - prev[0]) / (t_now - prev[1])
                    v_meas[2] = 0.0     # z channel corrupt; xy only
                    if np.linalg.norm(v_meas[:2]) < 4.0:
                        with KF_LOCK:
                            KF.update_velocity(np.array([v_meas[0], v_meas[1], KF.v[2]]),
                                               sigma=0.7)
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

def send_rate(rr, pr, yr, thr):
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

def level_cmd(vx_ref=0.0, vy_ref=0.0, vz_ref=0.0, thr_base=HOVER, pitch_bias=0.0, yr=0.0):
    roll_ref = max(-0.25, min(0.25, -K_V * (state['vy_b'] - vy_ref)))
    pitch_ref = max(-0.35, min(0.35, K_V * (state['vx_b'] - vx_ref) + pitch_bias))
    rr = SIGN_R * (KP * (roll_ref - state['roll'])) / RATE_GAIN
    pr = SIGN_P * (KP * (pitch_ref - state['pitch'])) / RATE_GAIN
    # RATE DISCIPLINE (07-06 collapse root cause): commanded transients hit
    # 356 deg/s measured; HIGHRES_IMU drops ~24% of samples (14 ms gaps), and
    # gyro integration across gaps at those rates accrues 1-3 deg PERMANENT
    # attitude error -> gravity leak -> estimate runaway. The attitude chain
    # is proven clean below ~1 rad/s actual; RATE_GAIN 1.93 means +-0.6
    # commanded ~ +-1.2 actual worst case. Do not raise without re-deriving
    # the gap-error budget.
    RATE_MAX = float(os.environ.get('RATE_MAX', '0.6'))
    rr = max(-RATE_MAX, min(RATE_MAX, rr)); pr = max(-RATE_MAX, min(RATE_MAX, pr))
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

for th in (rx_loop, cam_loop, det_loop):
    threading.Thread(target=th, daemon=True).start()
time.sleep(2.0)

print('sim HARD reset (param1=1: restarts the RACE + countdown)', flush=True)
m.mav.command_long_send(m.target_system, m.target_component, 31000, 0, 1, 0, 0, 0, 0, 0, 0)
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
    HIGH_W = G1_AP.copy()
    GATES_W[0] = HIGH_W
    TRAJ, S_GATES = build_traj(HIGH_W, G1_W, G2_W)
    state['next_gate_w'] = HIGH_W.copy()
    print(f'spline anchored to pad lock: aperture {G1_AP.round(2)} origin {G1_W.round(2)}', flush=True)
else:
    print('no course-gate pad obs -- flying the map', flush=True)

m.mav.command_long_send(m.target_system, m.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
time.sleep(0.5)
print('armed', flush=True)

def guards(phase):
    if tilt() > TILT_ABORT: return f'tilt abort ({phase})'
    c = state['collision']
    imp_lim = 4.0 if phase == 'punch' else 2.5
    if c and c.get('threat_level', 0) >= 2 and c.get('horizontal_minimum_delta', 0) > imp_lim:
        return f'collision ({phase}): imp {c.get("horizontal_minimum_delta"):.1f}'
    if abs(state['vx_b']) > 8 or abs(state['vy_b']) > 8: return f'velocity runaway ({phase})'
    return None

_viz_static()
aborted = None
# rotate level at LOW thrust, then climb gently
t0 = time.time()
while time.time() - t0 < 0.7:
    level_cmd(0, 0, 0, thr_base=0.22); time.sleep(1/CMD_HZ)
t0 = time.time()
while time.time() - t0 < 1.6 and not aborted:
    level_cmd(0, 0, 1.0); aborted = guards('climb'); time.sleep(1/CMD_HZ)
state['airborne'] = True   # loosens the det z-guard (in-flight obs-z bias)
print(f'airborne tilt {math.degrees(tilt()):.1f} vz {state["vz_up"]:.2f}', flush=True)

# AXIS PROBES (frame tripwires -- physical defaults, flip only on strong evidence).
# Flight 2026-07-05 #3 measured SX/SY/SZ all +1 AND the probe shove (2 m off-axis,
# +16 deg yaw) misaligned the G1 approach into a gate-post clip. Probes now
# opt-in via PROBES=1; the measured physical signs are the default.
SX, SY = 1.0, 1.0
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
        if time.time() - state.get('obs_wall', 0.0) > 3.0 and                 state.get('fix_count', 0) < 1:
            with KF_LOCK:
                KF.x[:3] = state['next_gate_w']
                KF.P[:3, :3] = np.eye(3) * 0.5
            jlog('tick_fix', p=state['next_gate_w'].round(2).tolist())
        else:
            jlog('tick_fix_skipped', reason='vision healthy')
        state['obs'] = None; state['tgt'] = None; state['obs_wall'] = 0.0
        state['landmark_w'] = None
        t0c = time.time()
        while time.time() - t0c < 1.2:   # clearance dash through the plane
            level_cmd(SX * 1.5, 0, 0, pitch_bias=-0.08); time.sleep(1/CMD_HZ)
        if ticks >= TARGET_TICKS:
            print(f'*** MISSION COMPLETE: {ticks} gates ***', flush=True)
            jlog('mission_complete', ticks=ticks)
            land('mission complete'); break
        nxt = GATES_W[min(ticks, len(GATES_W) - 1)]
        state['next_gate_w'] = nxt + np.array([0, 0, gate_z_off[min(ticks, 2)]])
        retries = 0
        phase, phase_t0, route_end_t0 = 'route', now, None
        print(f'ROUTE to gate {ticks} at {state["next_gate_w"].round(2)}', flush=True)
        continue
    if state.get('race_finish_ns', -1) >= 0:
        print(f'*** RACE FINISH *** ticks={ticks}', flush=True)
        jlog('race_finish', ticks=ticks, finish_ns=state['race_finish_ns'])
        land('race finished'); break
    if now - mission_t0 > MISSION_S:
        aborted = f'mission timeout, ticks={ticks}'

    if phase == 'acquire':
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
            lat_slow = max(0.4, 1.0 - 0.3 * min(abs(lat), 2.0))
            dwn_slow = max(0.45, 1.0 - 0.5 * min(abs(dz_err), 1.5))
            vx_ref = min(1.4, max(0.6, 0.4 * (fwd - 1.0))) * lat_slow * dwn_slow   # slow: the detector drops the gate ~9 m out above ~1.5 m/s (flight #33)
            lat_gain = 0.6 if fwd < 8.0 else 0.25
            vy_ref = max(-1.0, min(1.0, lat_gain * lat))
            # align-then-shoot: the start-light poles flank the course at ~x 5
            # with a ~+-1 m corridor (flights #3/#5/#6/#8 clipped them arriving
            # 0.3-2 m off-axis; #4 threaded it dead-center). Center FIRST, then
            # accelerate through.
            if fwd > 3.5 and abs(lat) > 0.25:
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
            if fwd < -1.0:
                print('target behind at range -- dropping lock, back to route', flush=True)
                state['obs'] = None; state['tgt'] = None; state['obs_wall'] = 0.0
                state['landmark_w'] = None
                phase, phase_t0, route_end_t0 = 'route', now, None
                continue
            fresh = age < 2.5
            fwd_ap = fwd - ORIGIN_OFFSET   # distance to the APERTURE plane
            if fresh and fwd_ap < 2.6 and abs(lat) < 0.35 and abs(dz_err) < 0.6:
                punch_t0, punch_dur = now, max(0.8, fwd_ap / 2.0 + 1.2)
                phase = 'punch'
                print(f'PUNCH from {fwd_ap:.1f} m to aperture (lat {lat:.2f} dwn {dwn:.2f})', flush=True)
        # PUNCH-ON-RECENT-OBS: the detector reliably dies ~6 m out (gate slides
        # under the 20-deg-up camera), so a fresh-obs-at-2.6m trigger never
        # fires (#41/#42). Short-horizon DR from the last fresh obs is
        # cm-accurate right after a vision-velocity fix -- fire on that.
        lk = state.get('_last_fresh')
        if phase == 'approach' and lk is not None and now - lk[4] < 2.5:
            with KF_LOCK:
                dp = KF.p - lk[3]
            fwd_est = lk[0] - float(np.hypot(dp[0], dp[1]))
            if fwd_est < 2.8 + ORIGIN_OFFSET and abs(lk[1]) < 0.4:
                punch_t0, punch_dur = now, max(0.6, fwd_est) / 2.0 + 1.2
                phase = 'punch'
                print(f'PUNCH (obs-DR) est {fwd_est:.1f} m (last lat {lk[1]:.2f}, obs age {now-lk[4]:.1f}s)', flush=True)
        if now - phase_t0 > 60:
            print('approach timeout -- dropping lock, back to route', flush=True)
            state['obs'] = None; state['tgt'] = None; state['obs_wall'] = 0.0
            state['landmark_w'] = None
            phase, phase_t0, route_end_t0 = 'route', now, None
    elif phase == 'punch':
        # hold the lateral line through the blind drive (#32 crossed drifting)
        lat_p = float(obs[1]) if obs is not None else 0.0
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
        s_here = TRAJ.nearest_s(p)
        # carrot stops just past the next UN-TICKED gate: #12 blew through G2's
        # plane blind because the carrot ran the whole spline while validation
        # still expected G1
        s_stop = (S_GATES[ticks] if ticks < len(S_GATES) else TRAJ.s_max) + 2.0
        s_ref = min(s_here + LEAD, s_stop, TRAJ.s_max)
        ref = TRAJ.sample(s_ref)
        d = ref['pos'] - p
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
        yr_cmd = SZ * max(-0.5, min(0.5, 1.2 * wrap(yaw_ref_t - state['yaw'])))
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
                and float(state['obs'][0]) < 7.5):
            # handoff only INSIDE the furniture radius: an early lock pulls the
            # drone off the dogleg straight into the x~5-7 structure (flight #36)
            print(f'vision lock on route: {state["obs"].round(2)}', flush=True)
            phase, phase_t0 = 'approach', now
        if s_here > s_stop - 2.6:
            # past the gate without a tick: HEIGHT SWEEP retry -- back up and
            # re-cross 0.4 m higher (the aperture height is only known to
            # ~0.5 m; #17 crossed at 1.3, #18 at 2.24, no tick; #4 ticked ~2.7)
            route_end_t0 = route_end_t0 or now
            if now - route_end_t0 > 4.0:
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
