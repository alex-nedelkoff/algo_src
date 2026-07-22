"""VQ2 acceptance-corpus flight: slow forward creep toward gate 1, full recording.

Fork of vq2fly.py (open-loop-safe: gyro attitude + accel damping, no position
control, no KF) with the excitation schedule replaced by a gate-height forward
creep -- the exact blind-leg regime the VIO acceptance gate needs vision-locked
data for. Gate 1 is dead ahead of spawn at ~11 m; yaw held at spawn heading.
"""
import socket, struct, time, json, os, math, threading
from pymavlink import mavutil

OUT = os.environ.get('RECORD', 'C:/Users/alexj/vq2_accept')
os.makedirs(OUT + '/frames', exist_ok=True)

HOVER = 0.2675            # sysID thr_hover_pred
TAKEOFF_THR = 0.37
TAKEOFF_S = 1.8
CMD_HZ = 50.0
TILT_ABORT = math.radians(55)
RATE_GAIN = 1.93          # |sim rate amplification|; sign probed live

m = mavutil.mavlink_connection('udpin:127.0.0.1:14550')
m.wait_heartbeat(timeout=8)
print('hb ok', flush=True)

state = {'acc': (0.0, 0.0, -9.81), 'gyr': (0.0, 0.0, 0.0), 't_us': 0,
         'roll': 0.0, 'pitch': 0.0, 'vz_up': 0.0, 'vx_b': 0.0, 'vy_b': 0.0, 'collision': None, 'stop': False, 'flying': False}
log_mav = open(OUT + '/mavlink.jsonl', 'w')
log_cmd = open(OUT + '/cmds.jsonl', 'w')

def rx_loop():
    last_us = None
    while not state['stop']:
        msg = m.recv_match(blocking=True, timeout=0.5)
        if not msg: continue
        t = msg.get_type()
        if t == 'BAD_DATA': continue
        d = msg.to_dict()
        d['_rx_wall'] = time.time()
        if t == 'ENCAPSULATED_DATA':
            d['data'] = bytes(msg.data)[:64].hex()
        log_mav.write(json.dumps(d, default=str) + '\n')
        if t == 'HIGHRES_IMU':
            acc = (msg.xacc, msg.yacc, msg.zacc)
            gyr = (msg.xgyro, msg.ygyro, msg.zgyro)
            us = msg.time_usec
            if last_us is not None and us > last_us:
                dt = (us - last_us) / 1e6
                # gyro integrate
                state['roll'] += gyr[0] * dt
                state['pitch'] += (-gyr[1]) * dt   # wfix: msg pitch-rate is mirrored
                # accel tilt correction (quasi-static)
                an = math.sqrt(sum(a*a for a in acc))
                if (not state['flying']) and 8.3 < an < 11.3:
                    r_acc = math.atan2(acc[1], -acc[2])
                    p_acc = math.atan2(acc[0], math.sqrt(acc[1]**2 + acc[2]**2))
                    a = 0.02
                    state['roll'] = (1-a)*state['roll'] + a*r_acc
                    state['pitch'] = (1-a)*state['pitch'] + a*p_acc
                # world-z specific force + gravity -> vz_up integration
                sr, cr = math.sin(state['roll']), math.cos(state['roll'])
                sp, cp = math.sin(state['pitch']), math.cos(state['pitch'])
                a_up = -(acc[0]*(-sp) + acc[1]*(sr*cp) + acc[2]*(cr*cp)) - 9.81
                state['vz_up'] += a_up * dt
                # body-frame linear accel = specific force + gravity-in-body
                gx_b = 9.81 * (-sp)
                gy_b = 9.81 * (sr * cp)
                state['vx_b'] += (acc[0] + gx_b) * dt
                state['vy_b'] += (acc[1] + gy_b) * dt
                # bleed toward 0 slowly (bounds integration error)
                state['vz_up'] *= (1.0 - 0.02*dt)
                state['vx_b'] *= (1.0 - 0.05*dt)
                state['vy_b'] *= (1.0 - 0.05*dt)
            last_us = us
            state['acc'], state['gyr'], state['t_us'] = acc, gyr, us
        elif t == 'COLLISION':
            state['collision'] = d

def cam_loop():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('127.0.0.1', 5600)); s.settimeout(0.5)
    H = struct.Struct('<IHHIIQ')
    frames, seen = {}, set()
    meta = open(OUT + '/frames_dedup.jsonl', 'w')
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
                open(f'{OUT}/frames/{ns}.jpg', 'wb').write(data)
                meta.write(json.dumps({'sim_ns': ns, 'fid': fid, 'rx_wall': time.time()}) + '\n')
                seen.add(ns)
            del frames[fid]
    meta.close()

def send_rate(rr, pr, yr, thr):
    m.mav.set_attitude_target_send(
        int(time.time()*1000) & 0xFFFFFFFF, m.target_system, m.target_component,
        mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE,
        [1.0, 0, 0, 0], rr, pr, yr, thr)
    log_cmd.write(json.dumps({'t': time.time(), 'rr': rr, 'pr': pr, 'yr': yr, 'thr': thr}) + '\n')

rx = threading.Thread(target=rx_loop); rx.start()
cam = threading.Thread(target=cam_loop); cam.start()
time.sleep(2.0)
r0, p0 = state['roll'], state['pitch']
print(f'baseline attitude roll {math.degrees(r0):.1f} pitch {math.degrees(p0):.1f}', flush=True)

print('sim reset (training)', flush=True)
m.mav.command_long_send(m.target_system, m.target_component, 31000, 0, 0, 0, 0, 0, 0, 0, 0)
time.sleep(6.0)
state['collision'] = None
time.sleep(1.0)
acc = state['acc']
state['roll'] = math.atan2(acc[1], -acc[2])
state['pitch'] = math.atan2(acc[0], math.sqrt(acc[1]**2 + acc[2]**2))
time.sleep(1.0)
print(f"post-reset attitude roll {math.degrees(state['roll']):.1f} pitch {math.degrees(state['pitch']):.1f}", flush=True)

m.mav.command_long_send(m.target_system, m.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
time.sleep(0.5)
state['flying'] = True   # accel-CF correction OFF from here: pure gyro attitude
print('armed (gyro-only attitude from here)', flush=True)

def tilt():
    return math.sqrt(state['roll']**2 + state['pitch']**2)

def hard_collision():
    c = state['collision']
    if not c: return False
    if c.get('threat_level', 0) >= 2 and c.get('horizontal_minimum_delta', 0) > 2.5:
        return True
    return False

def land(reason):
    print('LAND:', reason, flush=True)
    t0 = time.time()
    while time.time() - t0 < 6.0:
        level_cmd(0, 0, 0, HOVER, vz_ref=-0.8); time.sleep(1/CMD_HZ)
    while time.time() - t0 < 8.0:
        send_rate(0, 0, 0, 0.10); time.sleep(1/CMD_HZ)
    m.mav.command_long_send(m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0, 0, 0, 0, 0, 0, 0)

# roll sign measured (cmd +0.3 -> gyro -0.674); pitch probed in-air (wfix warns pitch may differ)
sign_r = -1.0
sign_p = +1.0  # physical frame: probe cmd +0.25 -> msg -0.549 -> physical +0.549 (wfix)

KP = 1.8
K_V = 0.12
def level_cmd(extra_rr=0.0, extra_pr=0.0, extra_yr=0.0, thr=HOVER, vz_ref=0.0):
    roll_ref = max(-0.25, min(0.25, -K_V * state['vy_b']))
    pitch_ref = max(-0.25, min(0.25, K_V * state['vx_b']))
    rr = sign_r * (KP * (roll_ref - state['roll'])) / RATE_GAIN + extra_rr
    pr = sign_p * (KP * (pitch_ref - state['pitch'])) / RATE_GAIN + extra_pr
    rr = max(-1.5, min(1.5, rr)); pr = max(-1.5, min(1.5, pr))
    dthr = 0.10 * (vz_ref - state['vz_up'])
    dthr = max(-0.06, min(0.06, dthr))
    send_rate(rr, pr, extra_yr, max(0.05, min(0.6, thr + dthr)))

# takeoff phase 1: rotate level at low thrust (minimal lift => minimal fwd impulse)
t0 = time.time()
while time.time() - t0 < 0.7:
    level_cmd(0, 0, 0, 0.22, vz_ref=0.0); time.sleep(1/CMD_HZ)
acc = state['acc']
p_acc = math.degrees(math.atan2(acc[0], math.sqrt(acc[1]**2 + acc[2]**2)))
print('rotate done, est tilt %.1f deg (accel-implied pitch %.1f) vx %.2f' % (math.degrees(tilt()), p_acc, state['vx_b']), flush=True)
# takeoff phase 2: climb to gate-aperture height (~1.4 m); the creep needs the
# camera on the gate banner, not a safety altitude
t0 = time.time()
while time.time() - t0 < 1.2:
    level_cmd(0, 0, 0, HOVER, vz_ref=1.2); time.sleep(1/CMD_HZ)
print('takeoff done, tilt %.1f deg vz %.2f vx %.2f' % (math.degrees(tilt()), state['vz_up'], state['vx_b']), flush=True)
# pitch sign confirmed from flight-4 recorded probe: cmd +0.25 -> gyro y -0.549 (inverts, same as roll)

# creep schedule: settle, then hold vx ~ VCREEP toward the gate. pitch_ref in
# level_cmd is a velocity damper (drives vx->0); creep replaces its target so
# the same loop drives vx->vx_ref instead.
VCREEP = float(os.environ.get('VCREEP', '0.7'))
CREEP_S = float(os.environ.get('CREEP_S', '22.0'))
# Hover-only survey mode: turn in place before any forward creep so a
# side-pillar can be brought toward the camera centre while the launch gate
# remains in view.  Defaults preserve the historical straight-capture run.
YAW_SCAN_RATE = float(os.environ.get('YAW_SCAN_RATE', '0.0'))
YAW_SCAN_S = float(os.environ.get('YAW_SCAN_S', '0.0'))

def creep_cmd(vx_ref):
    roll_ref = max(-0.25, min(0.25, -K_V * state['vy_b']))
    pitch_ref = max(-0.12, min(0.12, K_V * (state['vx_b'] - vx_ref)))
    rr = sign_r * (KP * (roll_ref - state['roll'])) / RATE_GAIN
    pr = sign_p * (KP * (pitch_ref - state['pitch'])) / RATE_GAIN
    rr = max(-1.5, min(1.5, rr)); pr = max(-1.5, min(1.5, pr))
    dthr = 0.10 * (0.0 - state['vz_up'])
    dthr = max(-0.06, min(0.06, dthr))
    send_rate(rr, pr, 0.0, max(0.05, min(0.6, HOVER + dthr)))

blocks = [
    ('settle', 2.0, lambda t: (0, 0, 0, HOVER)),
    ('yaw_scan', YAW_SCAN_S, lambda t: (0, 0, YAW_SCAN_RATE, HOVER)),
    ('creep',  CREEP_S, None),   # None -> creep_cmd path below
    ('settle', 2.0, lambda t: (0, 0, 0, HOVER)),
]
aborted = False
for name, dur, fn in blocks:
    print('block %s tilt %.1f vx %.2f vy %.2f vz %.2f' % (name, math.degrees(tilt()), state['vx_b'], state['vy_b'], state['vz_up']), flush=True)
    t0 = time.time()
    while time.time() - t0 < dur:
        if tilt() > TILT_ABORT: land(f'tilt abort in {name}'); aborted = True; break
        if abs(state['vx_b']) > 6.0 or abs(state['vy_b']) > 6.0: land(f'velocity runaway in {name}'); aborted = True; break
        if hard_collision(): land(f'collision in {name}: {state["collision"]}'); aborted = True; break
        if fn is None:
            creep_cmd(VCREEP)
        else:
            out = fn(time.time() - t0)
            er, ep, ey, thr = out[:4]
            vzr = out[4] if len(out) > 4 else 0.0
            level_cmd(er, ep, ey, thr, vz_ref=vzr)
        time.sleep(1/CMD_HZ)
    if aborted: break

if not aborted:
    land('schedule complete')
time.sleep(1)
state['stop'] = True
rx.join(timeout=3); cam.join(timeout=3)
log_mav.close(); log_cmd.close()
print('DONE aborted=%s final tilt %.1f' % (aborted, math.degrees(tilt())), flush=True)
