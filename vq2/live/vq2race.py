"""VQ2 ribbon racer: follow the rendered cyan course line through the gates.

Route = the in-world course ribbon (image-space servo at frame rate).
Altitude = KF-z hold at aperture height. Order/score = RACE_STATUS ticks.
"""
import sys, os, socket, struct, time, json, math, threading
from collections import deque

sys.path.insert(0, r'C:\Users\alexj')
import numpy as np
import cv2
from pymavlink import mavutil
from eskf import PosVelKF, accel_level

OUT = 'C:/Users/alexj/vq2_race'
os.makedirs(OUT + '/frames', exist_ok=True)

HOVER = 0.2675
CMD_HZ = 50.0
RATE_GAIN = 1.93
SIGN_R, SIGN_P = -1.0, +1.0
KP = 1.8
K_V = 0.12
TILT_ABORT = math.radians(55)
FX, CX, CY = 320.0, 320.0, 180.0
Z_TARGET = -0.9        # ~1.1 m above floor; tick flights flew ~1.2 and scored
V_CRUISE = 1.6
MISSION_S = 300.0

KF = PosVelKF()
KF_LOCK = threading.Lock()

state = {'acc': (0, 0, -9.81), 'gyr': (0, 0, 0), 'roll': 0.0, 'pitch': 0.0,
         'yaw': 0.0, 'collision': None, 'col_wall': 0.0, 'gate_idx': 0,
         'race_finish_ns': -1, 'stop': False,
         'frame': None, 'frame_ns': 0,
         'rib_near': None, 'rib_far': None, 'rib_wall': 0.0, 'rib_frac': 0.0}
log_f = open(OUT + '/log.jsonl', 'w')
llock = threading.Lock()

def jlog(kind, **kw):
    kw['kind'] = kind; kw['t'] = time.time()
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
                state['pitch'] += (-gyr[1]) * dt          # wfix
                state['yaw'] += (-gyr[2]) * dt            # wfix (yaw mirrored too)
                a_lvl = accel_level(acc, state['roll'], state['pitch'])
                cyw, syw = math.cos(state['yaw']), math.sin(state['yaw'])
                a_w = np.array([cyw * a_lvl[0] - syw * a_lvl[1],
                                syw * a_lvl[0] + cyw * a_lvl[1], a_lvl[2]])
                with KF_LOCK:
                    KF.predict(a_w, dt)
            elif last_us is None:
                p0 = math.atan2(acc[0], math.sqrt(acc[1]**2 + acc[2]**2))
                state['roll'] = math.atan2(acc[1], -acc[2])
                state['pitch'] = p0
            last_us = us
            state['acc'], state['gyr'] = acc, gyr
        elif t == 'COLLISION':
            d = msg.to_dict()
            state['collision'] = d; state['col_wall'] = time.time()
            jlog('collision', imp=d.get('horizontal_minimum_delta'), tl=d.get('threat_level'))
        elif t == 'ENCAPSULATED_DATA':
            d = bytes(msg.data)
            if d and d[0] == 1:
                try:
                    rs = struct.unpack('<BQqqIq', d[:37])
                    if rs[4] != state['gate_idx']:
                        jlog('gate_tick', idx=rs[4])
                        print(f'*** TICK {rs[4]} ***', flush=True)
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
                seen.add(ns)
            del frames[fid]

def ribbon_loop():
    """HSV ribbon tracker at frame rate: near/far band centroids."""
    last_ns = 0
    n_saved = 0
    while not state['stop']:
        img, ns = state['frame'], state['frame_ns']
        if img is None or ns == last_ns:
            time.sleep(0.005); continue
        last_ns = ns
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (80, 70, 110), (110, 255, 255))
        frac = float(mask.mean()) / 255.0
        near = mask[240:360, :]      # bottom third
        far = mask[150:240, :]       # mid band
        rn = rf = None
        ys, xs = np.nonzero(near)
        if len(xs) > 40:
            rn = float(np.median(xs))
        ys, xs = np.nonzero(far)
        if len(xs) > 25:
            rf = float(np.median(xs))
        state['rib_near'], state['rib_far'] = rn, rf
        state['rib_frac'] = frac
        if rn is not None or rf is not None:
            state['rib_wall'] = time.time()
        if n_saved % 4 == 0:
            cv2.imwrite(f'{OUT}/frames/{ns}.jpg', img)
        n_saved += 1
        jlog('rib', ns=ns, near=rn, far=rf, frac=round(frac, 4))

def send_rate(rr, pr, yr, thr):
    m.mav.set_attitude_target_send(
        int(time.time()*1000) & 0xFFFFFFFF, m.target_system, m.target_component,
        mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE,
        [1.0, 0, 0, 0], rr, pr, yr, thr)

def level_cmd(vx_ref=0.0, vy_ref=0.0, vz_up_ref=0.0, pitch_bias=0.0, yr=0.0, thr_base=HOVER):
    with KF_LOCK:
        vw = KF.v.copy()
    cyw, syw = math.cos(state['yaw']), math.sin(state['yaw'])
    vx_b = cyw * vw[0] + syw * vw[1]
    vy_b = -syw * vw[0] + cyw * vw[1]
    vz_up = -vw[2]
    roll_ref = max(-0.28, min(0.28, -K_V * (vy_b - vy_ref)))
    pitch_ref = max(-0.35, min(0.35, K_V * (vx_b - vx_ref) + pitch_bias))
    rr = SIGN_R * (KP * (roll_ref - state['roll'])) / RATE_GAIN
    pr = SIGN_P * (KP * (pitch_ref - state['pitch'])) / RATE_GAIN
    dthr = max(-0.07, min(0.07, 0.10 * (vz_up_ref - vz_up)))
    send_rate(max(-1.5, min(1.5, rr)), max(-1.5, min(1.5, pr)), yr,
              max(0.05, min(0.6, thr_base + dthr)))

def tilt():
    return math.sqrt(state['roll']**2 + state['pitch']**2)

def land(reason):
    print('LAND:', reason, flush=True); jlog('land', reason=reason)
    t0 = time.time()
    while time.time() - t0 < 5.0:
        level_cmd(0, 0, -0.7); time.sleep(1/CMD_HZ)
    while time.time() - t0 < 7.0:
        send_rate(0, 0, 0, 0.10); time.sleep(1/CMD_HZ)
    m.mav.command_long_send(m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0, 0, 0, 0, 0, 0, 0)

for th in (rx_loop, cam_loop, ribbon_loop):
    threading.Thread(target=th, daemon=True).start()
time.sleep(2.0)

print('sim reset (training)', flush=True)
m.mav.command_long_send(m.target_system, m.target_component, 31000, 0, 0, 0, 0, 0, 0, 0, 0)
time.sleep(6.0)
acc = state['acc']
state['roll'] = math.atan2(acc[1], -acc[2])
state['pitch'] = math.atan2(acc[0], math.sqrt(acc[1]**2 + acc[2]**2))
state['yaw'] = 0.0
with KF_LOCK:
    KF.reset_at_rest()
state['collision'] = None
gate0 = state['gate_idx']
print(f'post-reset pitch {math.degrees(state["pitch"]):.1f} gate_idx {gate0}', flush=True)

m.mav.command_long_send(m.target_system, m.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
time.sleep(0.5)
print('armed', flush=True)

aborted = None
# rotate at low thrust, then climb to line altitude
t0 = time.time()
while time.time() - t0 < 0.7:
    level_cmd(0, 0, 0, thr_base=0.22); time.sleep(1/CMD_HZ)
t0 = time.time()
while time.time() - t0 < 0.9 and not aborted:
    level_cmd(0, 0, 1.0)
    if tilt() > TILT_ABORT: aborted = 'tilt in climb'
    time.sleep(1/CMD_HZ)
print('airborne, rolling straight into the race (accel pitches the road into view)', flush=True)

SY_RIB = 1.0    # physical: nose camera image-right = +y_body; tripwire DISABLED (flapped on acquisition transients)
lat_hist = deque(maxlen=60)
last_side = 1.0     # which side the line was last seen drifting toward
mission_t0 = time.time()
ticks0 = gate0
last_tick_wall = 0.0
while not aborted:
    now = time.time()
    if state['race_finish_ns'] >= 0:
        dtt = now - mission_t0
        print(f'*** RACE FINISH *** gates={state["gate_idx"]} wall={dtt:.1f}s', flush=True)
        jlog('race_finish', gates=state['gate_idx'], wall_s=dtt)
        land('race finished'); break
    if now - mission_t0 > MISSION_S:
        aborted = f'mission timeout, ticks={state["gate_idx"]}'
        break
    if state['gate_idx'] != ticks0:
        ticks0 = state['gate_idx']
        last_tick_wall = now
    # guards (grind-suppressed 3 s after a tick)
    if tilt() > TILT_ABORT:
        aborted = f'tilt abort, ticks={state["gate_idx"]}'; break
    c = state['collision']
    if c and state['col_wall'] > now - 0.5 and c.get('threat_level', 0) >= 2:
        imp = c.get('horizontal_minimum_delta', 0)
        lim = 6.0 if now - last_tick_wall < 3.0 else 3.5
        if imp > lim:
            aborted = f'collision imp {imp:.1f}, ticks={state["gate_idx"]}'; break
    with KF_LOCK:
        pz = KF.p[2]; spd = float(np.linalg.norm(KF.v[:2]))
    if spd > 6.0:
        aborted = f'speed runaway, ticks={state["gate_idx"]}'; break

    vz_ref = max(-0.8, min(0.8, 0.9 * (pz - Z_TARGET)))
    age = now - state['rib_wall']
    rn, rf = state['rib_near'], state['rib_far']
    if age < 0.6 and (rf is not None or rn is not None):
        guide = rf if rf is not None else rn
        err_far = (guide - CX) / FX
        err_near = ((rn - CX) / FX) if rn is not None else err_far
        lat_hist.append(abs(err_near))
        vy_ref = SY_RIB * max(-1.2, min(1.2, 1.6 * err_near))
        yr_cmd = max(-0.5, min(0.5, 1.1 * err_far))
        if abs(err_far) > 0.08:
            last_side = 1.0 if err_far > 0 else -1.0
        slow = max(0.0, 1.0 - 2.0 * abs(err_far))
        vmax_now = 1.0 if now - last_tick_wall < 2.0 else V_CRUISE   # bends follow ticks
        vx_ref = 0.6 + (vmax_now - 0.6) * slow
        level_cmd(vx_ref, vy_ref, vz_ref, pitch_bias=-0.10, yr=yr_cmd)
        # sign tripwire: sustained growth in |lateral error| while servoing => flip once
        if False:
            a = np.mean(list(lat_hist)[:20]); b = np.mean(list(lat_hist)[-20:])
            if b > a + 0.15 and b > 0.35:
                SY_RIB = -SY_RIB
                lat_hist.clear()
                print('SY_RIB tripwire: flipped lateral servo sign', flush=True)
                jlog('sy_flip')
    elif age < 8.0:
        if now - mission_t0 < 5.0:
            # opening straight: the road is dead ahead -- just roll forward, camera down
            level_cmd(1.0, 0, max(-0.3, vz_ref - 0.2), pitch_bias=-0.14)
        else:
            # seek: slow arc TOWARD the side the line was last drifting
            level_cmd(0.5, SY_RIB * 0.5 * last_side, max(-0.3, vz_ref - 0.2),
                      pitch_bias=-0.14, yr=0.35 * last_side)
    elif age < 14.0:
        # widen: reverse scan direction
        level_cmd(0.3, -SY_RIB * 0.4 * last_side, -0.15, pitch_bias=-0.12,
                  yr=-0.3 * last_side)
    else:
        aborted = f'line lost > 14 s, ticks={state["gate_idx"]}'; break
    time.sleep(1/CMD_HZ)

if aborted:
    land(aborted)
state['stop'] = True
time.sleep(1)
log_f.close()
print(f'DONE: {aborted or "FINISH"} | gates={state["gate_idx"]}', flush=True)
