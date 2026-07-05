"""VQ2 gate thread: GateNet vision + IMU-only state + velocity-PD servoing.

Fusion v0: vision supplies gate-relative position (band-midplane center, cam
frame -> body frame via the canonical mount geometry); the flight-verified v9
IMU stack (wfix gyro attitude + integrated velocity) supplies the 50 Hz state;
the velocity damper's references come from vision instead of zeros.
Success signal = RACE_STATUS active_gate_index tick (judge signal, never our
own geometry). Training mode.
"""
import sys, os, socket, struct, time, json, math, threading
from collections import deque

sys.path.insert(0, r'C:\Users\alexj\Documents\algo_src_main')
import numpy as np
import cv2
import torch
from pymavlink import mavutil
from scripts.dcl import vq1_detect_overlay as OV
from perception.training import train_gatenet as TG
from perception.decode import associate as AS

OUT = 'C:/Users/alexj/vq2_survey'
os.makedirs(OUT + '/frames', exist_ok=True)
CKPT = r'C:\Users\alexj\gatenet_b2_cov.pt'
CFG = r'C:\Users\alexj\Documents\algo_src_main\configs\perception\gatenet_b2_multi_pb_cov.yaml'

HOVER = 0.2675
CMD_HZ = 50.0
TILT_ABORT = math.radians(55)
RATE_GAIN = 1.93
SIGN_R, SIGN_P = -1.0, +1.0
KP = 1.8
K_V = 0.12
CAM_TILT = math.radians(20.0)   # VQ2 CONFIRMED ~20 up: hover frame horizon in bottom third + pad geometry closes
CAM_YAW = math.radians(0.0)     # VQ1 value; unverified on VQ2 -- servo self-corrects small bias

# camera axes in body frame (FRD; camera looks along -x, pitched up, yawed)
ct, st = math.cos(CAM_TILT), math.sin(CAM_TILT)
cy, sy = math.cos(CAM_YAW), math.sin(CAM_YAW)
C_Z = np.array([-ct * cy, -ct * sy, -st])   # boresight
C_Y = np.array([-st * cy, -st * sy, ct])    # image down
C_X = np.cross(C_Y, C_Z)                    # image right
M_BODY_CAM = np.stack([C_X, C_Y, C_Z], axis=1)

ATT_BUF = deque(maxlen=600)   # (wall_t, roll, pitch) at IMU rate ~116 Hz
state = {'acc': (0, 0, -9.81), 'gyr': (0, 0, 0), 't_us': 0,
         'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'vx_b': 0.0, 'vy_b': 0.0, 'vz_up': 0.0,
         'collision': None, 'gate_idx': 0, 'stop': False, 'flying': False,
         'obs': None, 'obs_wall': 0.0, 'frame': None, 'frame_ns': 0}
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
        if t == 'HIGHRES_IMU':
            acc = (msg.xacc, msg.yacc, msg.zacc)
            gyr = (msg.xgyro, msg.ygyro, msg.zgyro)
            us = msg.time_usec
            if last_us is not None and us > last_us:
                dt = (us - last_us) / 1e6
                state['roll'] += gyr[0] * dt
                state['pitch'] += (-gyr[1]) * dt      # wfix
                state['yaw'] += gyr[2] * dt
                sr, cr = math.sin(state['roll']), math.cos(state['roll'])
                sp, cp = math.sin(state['pitch']), math.cos(state['pitch'])
                a_up = -(acc[0]*(-sp) + acc[1]*(sr*cp) + acc[2]*(cr*cp)) - 9.81
                state['vz_up'] += a_up * dt
                state['vx_b'] += (acc[0] + 9.81*(-sp)) * dt
                state['vy_b'] += (acc[1] + 9.81*(sr*cp)) * dt
                state['vz_up'] *= (1 - 0.02*dt)
                state['vx_b'] *= (1 - 0.05*dt)
                state['vy_b'] *= (1 - 0.05*dt)
                state['vx2'] = state.get('vx2', 0.0) + (acc[0] + 9.81*(-sp)) * dt
                state['vy2'] = state.get('vy2', 0.0) + (acc[1] + 9.81*(sr*cp)) * dt
                state['vz2'] = state.get('vz2', 0.0) + a_up * dt
            elif last_us is None:
                p0 = math.atan2(acc[0], math.sqrt(acc[1]**2 + acc[2]**2))
                r0 = math.atan2(acc[1], -acc[2])
                state['roll'], state['pitch'] = r0, p0
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
                    state['frame'], state['frame_ns'] = img, ns   # drop-to-latest
                    state['frame_wall'] = time.time()
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
        all_solved = []
        best = None
        for dec in insts:
            ip = AS.solve_instance(dec, image_size=(int(x.shape[-1]), int(x.shape[-2])),
                                   direct_pose_vec=None, **sk)
            if not ip.solved or ip.t is None or ip.low_confidence:
                continue
            z = float(ip.t[2])
            if not (1.5 < z < 45.0):        # range gate (start-arch/pillar FPs)
                continue
            # bearing-weighted nearest: course gates sit near the camera axis;
            # an off-axis nearer gate (parallel row) must not outrank a centered one
            tx = float(ip.t[0])
            key = ((z + 2.0 * abs(tx)) if z < 22.0 else 100.0 + z, -float(dec.center_score))
            if best is None or key < best[0]:
                best = (key, np.asarray(ip.t, float))
            R9 = None
            if ip.pnp is not None and getattr(ip.pnp, 'R_cam_gate', None) is not None:
                R9 = np.asarray(ip.pnp.R_cam_gate).reshape(-1).round(5).tolist()
            all_solved.append({'t_cam': np.asarray(ip.t, float).round(3).tolist(),
                               'R': R9, 'score': float(dec.center_score),
                               'lc': bool(ip.low_confidence)})
        if best is not None:
            g_b = M_BODY_CAM @ best[1]
            # rotate into the LEVEL frame with the attitude AT FRAME CAPTURE TIME
            # (inference is ~300 ms; using live attitude corrupts z by meters
            # whenever the frame straddles a pitch transient)
            r, p = state['roll'], state['pitch']
            best_dt = 1e9
            if f_wall <= 0:
                ATT_ITER = []
            else:
                ATT_ITER = ATT_BUF
            for (tw, rb, pb) in reversed(ATT_ITER):
                d = abs(tw - f_wall)
                if d < best_dt:
                    best_dt, r, p = d, rb, pb
                elif tw < f_wall - 0.5:
                    break
            sr, cr = math.sin(r), math.cos(r)
            sp, cp = math.sin(-p), math.cos(p)   # aero pitch sign vs RH y-rotation
            Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
            g_lvl = Ry @ (Rx @ g_b)
            # target lock: reject jumps > 3.5 m from the dead-reckoned target
            tgt = state.get('tgt')
            state['det_wall'] = time.time()
            rejected = tgt is not None and np.linalg.norm(g_lvl - tgt) > 5.0
            if rejected:
                # streak breaker: 3 consecutive mutually-consistent rejects
                # overthrow the DR ghost -- vision is absolute, DR interpolates
                buf = state.setdefault('rej_buf', [])
                buf.append((time.time(), g_lvl.copy()))
                buf[:] = [(t_, g_) for (t_, g_) in buf if time.time() - t_ < 6.0][-4:]
                if len(buf) >= 3 and all(np.linalg.norm(g_ - g_lvl) < 2.5 for (_, g_) in buf):
                    jlog('relock', ns=ns, g_lvl=g_lvl.round(3).tolist(),
                         ghost=tgt.round(3).tolist())
                    print(f'RELOCK: vision cluster overthrows DR ghost', flush=True)
                    buf.clear()
                    rejected = False
                else:
                    jlog('obs_rejected', ns=ns, g_lvl=g_lvl.round(3).tolist(),
                         tgt=tgt.round(3).tolist())
            if rejected:
                pass
            else:
                state['rej_buf'] = []
                state['obs'] = g_lvl
                state['obs_wall'] = time.time()
                state['tgt'] = g_lvl.copy()
                jlog('obs', ns=ns, score=best[0], t_cam=best[1].round(3).tolist(),
                     g_lvl=g_lvl.round(3).tolist())
        if all_solved:
            jlog('multi_obs', ns=ns, f_wall=f_wall,
                 roll=round(r, 4), pitch=round(p, 4), insts=all_solved)
        cv2.imwrite(f'{OUT}/frames/{ns}.jpg', img)

def send_rate(rr, pr, yr, thr):
    m.mav.set_attitude_target_send(
        int(time.time()*1000) & 0xFFFFFFFF, m.target_system, m.target_component,
        mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE,
        [1.0, 0, 0, 0], rr, pr, yr, thr)

def level_cmd(vx_ref=0.0, vy_ref=0.0, vz_ref=0.0, thr_base=HOVER, pitch_bias=0.0, yr=0.0):
    roll_ref = max(-0.25, min(0.25, -K_V * (state['vy_b'] - vy_ref)))
    pitch_ref = max(-0.35, min(0.35, K_V * (state['vx_b'] - vx_ref) + pitch_bias))
    rr = SIGN_R * (KP * (roll_ref - state['roll'])) / RATE_GAIN
    pr = SIGN_P * (KP * (pitch_ref - state['pitch'])) / RATE_GAIN
    rr = max(-1.5, min(1.5, rr)); pr = max(-1.5, min(1.5, pr))
    dthr = max(-0.06, min(0.06, 0.10 * (vz_ref - state['vz_up'])))
    send_rate(rr, pr, yr, max(0.05, min(0.6, thr_base + dthr)))

def tilt(): return math.sqrt(state['roll']**2 + state['pitch']**2)

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

print('sim reset (training)', flush=True)
m.mav.command_long_send(m.target_system, m.target_component, 31000, 0, 0, 0, 0, 0, 0, 0, 0)
time.sleep(6.0)
state['collision'] = None
time.sleep(1.0)
acc = state['acc']
state['roll'] = math.atan2(acc[1], -acc[2])
state['pitch'] = math.atan2(acc[0], math.sqrt(acc[1]**2 + acc[2]**2))
state['vx_b'] = state['vy_b'] = state['vz_up'] = 0.0
state['obs'] = None          # discard pre-reset ghost observations
state['tgt'] = None
state['obs_wall'] = 0.0
gate0_idx = state['gate_idx']
print(f'post-reset pitch {math.degrees(state["pitch"]):.1f} gate_idx {gate0_idx}', flush=True)

# PAD ACQUISITION: the ~18 deg spawn tilt frames gate 0 (VQ1 lesson) -- lock it
pad_t0 = time.time()
while state['obs'] is None and time.time() - pad_t0 < 15.0:
    time.sleep(0.1)
if state['obs'] is None:
    print('NO PAD ACQUISITION -- aborting before takeoff', flush=True)
    sys.exit(1)
print(f'pad lock: g_lvl {state["obs"].round(2)} range {np.linalg.norm(state["obs"]):.1f} m', flush=True)

m.mav.command_long_send(m.target_system, m.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
time.sleep(0.5)
state['flying'] = True
print('armed', flush=True)

def guards(phase):
    if tilt() > TILT_ABORT: return f'tilt abort ({phase})'
    c = state['collision']
    imp_lim = 4.0 if phase == 'punch' else 2.5
    if c and c.get('threat_level', 0) >= 2 and c.get('horizontal_minimum_delta', 0) > imp_lim:
        return f'collision ({phase}): imp {c.get("horizontal_minimum_delta"):.1f}'
    if abs(state['vx_b']) > 6 or abs(state['vy_b']) > 6: return f'velocity runaway ({phase})'
    return None

aborted = None
# rotate level at LOW thrust (v9 lesson: hover thrust while tilted 18 deg = slide)
t0 = time.time()
while time.time() - t0 < 0.7:
    level_cmd(0, 0, 0, thr_base=0.22); time.sleep(1/CMD_HZ)
# climb gently ~1.2 m
t0 = time.time()
while time.time() - t0 < 1.6 and not aborted:
    level_cmd(0, 0, 1.0); aborted = guards('climb'); time.sleep(1/CMD_HZ)
print(f'airborne tilt {math.degrees(tilt()):.1f} vz {state["vz_up"]:.2f}', flush=True)

STARE = os.environ.get('STARE') == '1'
if STARE:
    print('STARE mode: hover + log obs 25 s', flush=True)
    alt = 0.0
    t0 = time.time(); last = t0
    while time.time() - t0 < 25 and not aborted:
        now = time.time()
        alt += state['vz_up'] * (now - last); last = now
        level_cmd(0, 0, 0)
        if int((now - t0) * 2) != int((now - t0 - 0.02) * 2):
            o = state['obs']
            jlog('stare', alt_est=round(alt, 2),
                 obs=o.round(2).tolist() if o is not None else None,
                 age=round(now - state['obs_wall'], 2))
        aborted = guards('stare')
        time.sleep(1/CMD_HZ)
    land(aborted or 'stare done')
    state['stop'] = True
    log_f.close()
    print('DONE stare:', aborted or 'ok', flush=True)
    sys.exit(0)

# ---- SURVEY MISSION: pitched straight traverse, bank ALL detections ----
SX = -1.0   # confirmed sign map
SURVEY_S = 95.0
CRUISE = 2.0

# det_loop already logs every ACCEPTED obs; for the survey we want EVERY solved
# instance with the flight state at frame time -- patch via a live hook:
state['survey'] = True

pos = np.zeros(3)      # level-frame position by IMU integration (survey truth)
last_p = time.time()

def step_pos():
    global last_p
    now = time.time()
    dt = now - last_p
    last_p = now
    v = np.array([state['vx_b'], state['vy_b'], -state['vz_up']])
    # rotate body-horizontal velocity into the survey frame by yaw
    c, s = math.cos(state['yaw']), math.sin(state['yaw'])
    vw = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])
    globals()['pos'] = pos + vw * dt
    v2 = np.array([state.get('vx2', 0.0), state.get('vy2', 0.0), -state.get('vz2', 0.0)])
    c2, s2 = math.cos(state['yaw']), math.sin(state['yaw'])
    vw2 = np.array([c2 * v2[0] - s2 * v2[1], s2 * v2[0] + c2 * v2[1], v2[2]])
    globals()['pos2'] = globals().get('pos2', np.zeros(3)) + vw2 * (now - last_p if False else 0.02)
    jlog('pose', p=pos.round(3).tolist(), p2=globals()['pos2'].round(3).tolist(),
         yaw=round(state['yaw'], 4),
         roll=round(state['roll'], 4), pitch=round(state['pitch'], 4))

t0 = time.time()
aborted = None
# climb to survey altitude ~2.5 m
while time.time() - t0 < 1.5 and not aborted:
    level_cmd(0, 0, 1.0); step_pos(); aborted = guards('climb'); time.sleep(1/CMD_HZ)
print('survey altitude, cruising', flush=True)
t0 = time.time()
LAT = float(os.environ.get('SURVEY_LAT', '0'))
lat_done = False
while time.time() - t0 < SURVEY_S and not aborted:
    vy = SX * 0.8 * (1 if LAT > 0 else -1) if (abs(LAT) > 0.1 and time.time() - t0 < abs(LAT) / 0.8) else 0.0
    level_cmd(SX * -CRUISE, vy, 0, pitch_bias=0.16)
    step_pos()
    aborted = guards('survey')
    time.sleep(1/CMD_HZ)
land(aborted or 'survey complete')
state['stop'] = True
time.sleep(1)
log_f.close()
print('DONE survey:', aborted or 'ok', flush=True)
