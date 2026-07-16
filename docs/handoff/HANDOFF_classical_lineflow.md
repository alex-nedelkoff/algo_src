# HANDOFF — Classical line-flow / adjacent-field gate-2 experiments (2026-07-15)

**Status: code was WIPED from vq2wp.py by the overnight DPVO-scale work
(vq2wp.py rewritten 07-16 08:07; `grep LINEFOLLOW` = 0 hits). This file has
the full snippets to RE-APPLY. Notes also in `EXPERIMENTS_gate2.md` +
obsidian_outbox.md (07-15 evening entries).**

## Why this path exists
Gate-1 tick ~90% (vision punch). Gate-2 leg = 1 success (fg62) in ~11 tries —
eleven patch-layers on a mode-switched vision/DR handoff. Research pass
(racing SOTA + adjacent fields) picked replacements. SOTA racing = continuous
EKF fusion of gate PnP against a known map, noise ∝ d²/N², Mahalanobis gating,
perception-aware yaw (keep next gate in FOV). Adjacent-field ideas below.
All CPU-only, VO-independent by design (VO = accuracy upgrade, not dependency).

## E1 — PN collision-course punch (missile guidance)
Two bodies collide iff the LOS **stops rotating**. Punch when hole is big AND
**bearing-RATE** nulled (not bearing angle) → course locked, terminal blind
range harmless. We already compute ḃ (`state['_att_bd']`) for the D-term.
Env `PN_PUNCH=1`, `PN_LOSRATE=0.12`. In the punch-trigger block (where
`_punch_now` is set):
```
_bd_pn = state.get('_att_bd', (9.9, 9.9))
_punch_now = (_allow_track and _fgage < 0.4
    and _fgw >= float(os.environ.get('FGP_PUNCH_W','95'))
    and abs(_fgb[0]) < 0.35 and abs(_fgb[1]) < 0.35
    and abs(_bd_pn[0]) < float(os.environ.get('PN_LOSRATE','0.12'))
    and abs(_bd_pn[1]) < float(os.environ.get('PN_LOSRATE','0.12')))
```
fg74: PN correctly REFUSED to punch (no collision course ever formed) → no
crash, but timeout. Behaved as designed.

## E2 — Go-around (aviation stabilized-approach doctrine)
Close to gate but not on collision course → never salvage; abort to staging,
retry (max 3). Env `GO_AROUND=1`. After `_punch_now` is computed, gate ≥1,
staged, `_fgw>=80`, `abs(_fgb[0])>0.3`, `_ga_count<3`:
```
state['_ga_count'] = state.get('_ga_count',0)+1
state['_mf_stage_done'] = False; state['_mf_wps']=None; state['_mf_wpi']=0
state['fg_wall']=0.0
print(f'GO-AROUND {state["_ga_count"]}: unstabilized ...')
```
fg74: never armed (hole never reached w≥80 while in view). Untested-positive.

## E3 — Cyan-line following (autonomous-driving lane keeping)  ← MOST PROMISING
The sim PAINTS the racing line cyan through every gate, visible in nearly every
frame. Its image position is a FREE observation: **column = lateral error,
row = height over course** (est-z can't see climbs — THE recurring gate-2
killer: ceiling drift). CPU color segmentation, ~fastgate cost. Env
`LINEFOLLOW=1`, `LINE_LAT=1` (lateral on, default OFF), `LINE_ROW_REF=0.80`.

**Detector** — in `fastgate_loop()`, right after `last_ns = ns`:
```
if os.environ.get('LINEFOLLOW') == '1':
    try:
        _h,_w2 = img.shape[:2]
        _roi = img[int(_h*0.55):,:]                 # lower 45% = floor
        _b_,_g_,_r_ = (_roi[:,:,0].astype('int16'),
                       _roi[:,:,1].astype('int16'), _roi[:,:,2].astype('int16'))
        _cy = ((_b_>120)&(_g_>100)&(_b_-_r_>40)&(_g_-_r_>20))
        _n_ = int(_cy.sum())
        if _n_ > 150:
            import numpy as _np2
            _ys,_xs = _np2.nonzero(_cy)
            state['line_off'] = (float(_xs.mean())-_w2/2.0)/_w2
            state['line_row'] = (int(_h*0.55)+float(_ys.mean()))/_h
            state['line_n'] = _n_; state['line_wall'] = time.time()
    except Exception: pass
```
**Consumer** — in the MAPFOLLOW carrot block, after `_pb`/`_dthr` computed,
before the `_mf_log` jlog:
```
if (os.environ.get('LINEFOLLOW')=='1'
        and time.time()-state.get('line_wall',0) < 0.4):
    if os.environ.get('LINE_LAT')=='1':          # lateral OFF by default —
        _rb = max(-0.2,min(0.2,_rb+0.35*state['line_off']))  # fights carrot
    _dthr = max(-0.035,min(0.035, _dthr
        + 0.08*(float(os.environ.get('LINE_ROW_REF','0.80')) - state['line_row'])))
```
fg75 (lateral ON): crashed — lateral line-centering fought the map carrot
(corridor deliberately deviates from the line). **Fix applied but UNFLOWN:**
made lateral opt-in (`LINE_LAT`), height-only by default. **fg76 was staged
with height-only + PN + go-around, sim freshly restarted & pad-verified, NEVER
FLOWN** (session ended). fly_servo_fg76.bat exists but its vq2wp.py base is
gone — re-apply snippets first.

## KEY INSIGHT (drove E3)
Every gate-2 failure = the SAME root cause on a different axis: kinematic-DR /
est error in x (early stage), y (offset), z (CEILING — est-z under-reads
climbs ~40%, so est-relative z-hold is satisfied while truth ascends). Vertical
needs a REAL observation. Two sources: VO (the overnight path), or the cyan
line's image row (E3, CPU-only, no lietorch dependency). **E3 height-hold is
the cheapest fix for the dominant (ceiling) failure mode.**

## Bigger idea not yet tried
Full **line-FOLLOWING** for transit (not just height-hold): replace the whole
map/carrot/kinematic-DR/staging stack with "keep the glowing line centered +
at row-ref," gates become fastgate/punch terminal events when a hole blooms on
the line. Generalizes to all ~20 gates with ZERO course-mapping. This is the
fresh-session project if height-hold alone gets gate-2 repeatable. Needs a
line-following PATH (yaw toward line vanishing-point) so lateral centering
stops fighting the carrot.

## Next actions (fresh session)
1. Re-apply E1/E2/E3 snippets to current vq2wp.py (or branch off pre-DPVO
   vq2wp). 2. Fly fg76 (height-only): does ceiling drift die? 3. If yes →
   3-run gate-2 repeatability. 4. If promising → full line-following transit.
Protocol: fresh sim (ESC→DOWN→ENTER, screenshot-verify pad + green lights),
`spawn_att_probe.py` NOSE-DOWN −17.8, gidx2 scoring ONLY, frames-before-theory.
Gate-1 recipe untouched (fly_servo_fg62.bat family = ATTMODE + punch-while-
locked). fg62 = the one verified TICKS=2 (gidx2 0→1→2).
