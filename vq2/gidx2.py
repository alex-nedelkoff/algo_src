"""Judge decode v2: only post-reset gate_idx transitions count.

A recording starts before the hard reset, so its head rows carry the
PREVIOUS race's gate_idx/tickstamp. A real tick is a 0->1 transition
AFTER the race clock resets (drops by >5 s). Prints every transition
with the race clock so stale state is visible instead of miscounted.

07-15 upgrade: a corpus can contain MULTIPLE resets (the sim auto-
restarts the race a few seconds after a drone wreck -- verified in
vq2_servo_fg47: one 31000 ACK at flight start, a second commandless
clock-zeroing right before disarm). Report EVERY post-reset segment,
not just the last; the headline REAL TICKS is the max over segments
that follow a flight-start-style reset, with all segments printed so
nothing is hidden.
"""
import json, struct, sys

for run in sys.argv[1:]:
    rows = []
    try:
        for line in open(r'C:/Users/alexj/' + run + '/mavlink.jsonl'):
            m = json.loads(line)
            if m.get('mavpackettype') == 'ENCAPSULATED_DATA':
                d = bytes.fromhex(m['data'])
                if len(d) >= 37 and d[0] == 1:
                    f = struct.unpack_from('<BQqqIq', d, 0)
                    rows.append((f[1], f[4], f[5]))   # clock_ms, gate_idx, tickstamp
    except Exception as e:
        print(run, 'ERR', repr(e)); continue
    if not rows:
        print(run, 'no RACE_STATUS rows'); continue
    resets = [i for i in range(1, len(rows))
              if rows[i][0] < rows[i-1][0] - 5000]
    print(f'{run}: {len(rows)} rows, {len(resets)} reset(s) at rows {resets}'
          if resets else f'{run}: {len(rows)} rows, NO reset (stale corpus?)')
    best = 0
    bounds = [0] + resets + [len(rows)]
    for s in range(len(bounds) - 1):
        a, b = bounds[s], bounds[s + 1]
        seg = rows[a:b]
        is_post = s > 0 or (seg[0][0] < 5000)   # segment starts at a reset
        tag = f'segment {s}' + (' (pre-reset head, STALE -- not scored)'
                                if not is_post else '')
        print(f'  {tag}: rows {a}..{b-1}, clock {seg[0][0]/1e3:.2f}s ->'
              f' {seg[-1][0]/1e3:.2f}s, idx {seg[0][1]} -> {seg[-1][1]}')
        if not is_post:
            continue
        seen = seg[0][1]
        real = 0
        for clock, idx, ts in seg:
            if idx != seen:
                print(f'    TRANSITION idx {seen} -> {idx} at clock '
                      f'{clock/1e3:.2f} s ts {ts}')
                if idx > seen and seen == 0 and idx == 1:
                    real += 1
                elif idx > seen:
                    real += 1
                seen = idx
        print(f'    segment REAL TICKS: {real}, final idx {seen}')
        best = max(best, real)
    print(f'  REAL TICKS: {best}')
