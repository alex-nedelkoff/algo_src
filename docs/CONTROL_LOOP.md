# How the drone is flown — control loop I/O primer

Audience: someone new to the stack (hi Janahan) who needs to know what
the control loop consumes, what it emits, and how a gate crossing is
actually commanded. Everything lives in one file:
`vq2/live/vq2wp.py` — a single-process **50 Hz** loop (`CMD_HZ`)
talking MAVLink to the AI-GP simulator on localhost.

## Inputs

**1. MAVLink telemetry** — `udpin:127.0.0.1:14550` (pymavlink):

- `HIGHRES_IMU` @ ~144 Hz nominal, **bursty** (unique samples arrive at
  7/14/28 ms spacing; ~38 % are byte-identical re-sends — dedupe before
  integrating). Gyro is near-noiseless (0.2°/415° drift measured);
  accel is used for gravity releveling at rest. Attitude
  (`state['roll']/['pitch']/['yaw']`) is integrated gyro, relevelled.
- `ENCAPSULATED_DATA` type 1 = **RACE_STATUS**, packed `<BQqqIq`:
  race clock (ms), `active_gate_index` (the judge's gate-tick truth —
  this is how we know a gate scored; it lags the physical crossing by
  ~0.75 s), last gate time, collision count.
- `COLLISION` messages (threat_level, horizontal/altitude deltas) —
  consumed by the wedge detector and guards.

**2. Camera** — UDP `127.0.0.1:5600`, chunked JPEG (header struct
`<IHHIIQ`), **640×360 @ 30 fps**, camera pitched **20° UP** (this one
fact explains half the vision failure modes: at/above aperture height
the hole exits the bottom of the FOV).

**3. Derived perception** (in-process threads, written into `state`):

- `fastgate.detect` (~2-4 ms/frame): gate-aperture candidates.
  Track state: `fg_bear = (lat, down)` bearings in rad (+lat = hole
  right of nose, −down = hole above axis), `fg_w` = hole width px
  (range ≈ 290 / w meters), `fg_wall` = freshness timestamp.
- Cyan racing-line detector: `line_off` (centroid offset),
  `line_head_off` (far-head offset — where the line is going),
  `line_row` (image row ≈ height cue), `line_n` (pixel count),
  `line_wall` freshness. The line passes through the SCORING aperture
  only — it is the ground truth the map/DR can never be.
- Optic flow (`FLOWOBS=1`): observe-only velocity; trustworthy only
  near hover (< ~1.2 m/s).
- Kinematic DR (`MF_DR=1`, post-tick legs): position from COMMANDED
  speed × gyro yaw, seeded at each tick. No accelerometer, no KF.
  Good for a short staging leg; lateral error grows to 1-2 m with
  time + maneuvering — do not trust it for terminal geometry.

## Output — the only actuator interface

Every control branch funnels into `level_cmd(...)` →
`send_rate(rr, pr, yr, thr)` → MAVLink **`SET_ATTITUDE_TARGET`** with
`ATTITUDE_IGNORE` typemask, i.e. the sim receives **body rates +
collective throttle**, 50 times a second:

- `rr`, `pr`, `yr` — roll/pitch/yaw body rates, rad/s, clamped to
  `RATE_MAX` 0.6 (commanded ≈ ×1.93 actual via `RATE_GAIN`; do NOT
  raise — gyro integration across the bursty IMU gaps accrues
  permanent attitude error above ~1 rad/s actual).
- `thr` — collective throttle, clamped [0.05, 0.6]; hover ≈ `HOVER`
  0.2675 (constant across the campaign — the vehicle never changed).

`level_cmd(vx_ref, vy_ref, vz_ref, thr_base, pitch_bias, yr,
roll_bias, att)` has two personalities:

- **Velocity mode** (`att=False`): refs are m/s; roll/pitch references
  are closed on estimated body velocity (`K_V` 0.07) plus biases; a
  vertical-speed loop maps `vz_ref` to a throttle delta (±0.06).
  Mistrust it after aggressive maneuvers — the velocity estimate banks
  ~1 m/s offsets per maneuver.
- **Pure-attitude mode** (`att=True`, `ATTMODE=1` — the mode that
  ticks gates): `roll_bias`/`pitch_bias` ARE the attitude references
  (rad, clamps ±0.25/±0.35), `vz_ref` is a raw throttle delta
  (±0.06), `yr` passes through. No estimator feedback anywhere — refs
  come straight from vision bearings. Inner P loop: rate =
  `KP` 1.8 × (attitude ref − integrated attitude) / `RATE_GAIN`.

So: **what you "pass into the control loop" each cycle is a small set
of reference biases** — `roll_bias` (lateral steering), `pitch_bias`
(speed: −0.06 ≈ 1.2 m/s cruise, −0.18 punch, +0.03 brake), `yr`
(yaw rate), `vz_ref`/throttle delta (height) — computed from vision
bearings, and the loop turns them into rate commands.

## How a gate is flown (the proven chain)

1. **Reset + arm**: `command_long` 31000 (hard race reset), wait for
   countdown, `RACE GO` from the race clock, arm, take off.
2. **Pad lock**: fastgate acquires the first hole (w≈45 px at ~6 m),
   spawn pitch must be −17.8° nose-down (verify with `simprobe.py`
   BEFORE flying; 0.0° = poisoned restart, the up-camera misses the
   pad gate).
3. **Pursuit** (`FGPURSUIT` + `ATTMODE`): each cycle with a fresh
   detection — `yr = 1.4·b_lat` (turn to target), `roll_bias` = PID on
   `b_lat` (P + EMA'd derivative + a band-limited integral trim `ri`
   that learns the gate-area push), throttle delta on `b_down`,
   `pitch_bias` fixed for speed with a vision range-rate governor
   (range from hole width — never from DR).
4. **Punch** (the crossing): when the hole is big enough
   (`FGP_PUNCH_W` 95 px ≈ <4 m) AND centered (±0.18 rad cone) AND the
   lateral LOS-rate is nulled (`PN_PUNCH`), command a fixed strong
   forward pitch (−0.18) for ~1.6 s and steer roll on the hole while
   it stays visible. When it clips out (blind last meter): G1 holds
   the last correction; G2+ applies a fixed LEFT counter
   (`FGP_PUNCH_PUSH`) against the measured ~0.3 m/s rightward push.
5. **Tick**: RACE_STATUS `active_gate_index` increments (~0.75 s after
   the physical crossing). Per-leg state resets, DR re-seeds, the next
   leg's machinery (line-follow transit → settle → punch) takes over.
6. **Safety envelope** (always on): tilt abort, collision guards,
   wedge detector (continuous-contact back-off), runaway bounds,
   `land()` on any abort — throttle down, disarm.

## Where to look / poke

- `vq2/live/vq2wp.py` — the loop (search `level_cmd`, `send_rate`,
  `SETTLE PHASE`, `LINE_TRANSIT`, `STAGE-PRI`).
- `fastgate.py` — the hole detector. `simprobe.py` — passive
  pre-flight validation of every input listed above (run it first).
- `fly_test*.bat` — env-gated flight configs (every behavior above has
  an env knob; the parked config is named in the handoff).
- `docs/handoff/HANDOFF_line_transit.md` — the living campaign log:
  what is proven, what failed, and why.

## Two control methods (how we decide WHERE to steer)

The loop above is the actuator plumbing. On top of it the stack now
has two distinct ways of turning what the camera sees into steering
references. Both are live in the code; they answer different
questions and are used together.

### Method 1 — bearing servoing (IBVS): "point at the hole"

The proven workhorse (every gate tick so far). Steer directly on the
aperture's image bearings:

    roll_bias  = Kp * b_lat + Kd * d(b_lat)/dt + trim
    yaw_rate   = k * b_lat
    thr_delta  = k * b_down
    speed      = governor on range-rate (hole width -> range 290/w)

Strengths: no state estimate anywhere, millisecond latency, robust to
everything the estimator gets wrong. This is what pursues, settles,
and punches.

Blind spot (measured, not theoretical): **pointing at the hole is not
the same as being on the gate's axis.** In corpus vq2_test90 the
drone held b_lat = +0.09 "converged" for 3.5 s while physically
sliding 0.9 m off the gate normal — the crossing came in 25 deg
oblique and struck the frame. Bearing servoing cannot see that
failure coming, by construction.

### Method 2 — gate-relative pose (PBVS): "know where you are in the
### gate's frame"

`_quad_pose()` in `fastgate.py` (this branch): fit a 4-corner quad to
the hole contour, `cv2.solvePnP(IPPE_SQUARE)` against the known
1.5 m square aperture -> per detection:

- **obliquity** (deg): angle between the line of sight and the gate
  plane's normal — "how sideways would I cross right now";
- **lateral** (m): the camera's offset from the gate's center axis —
  a waypoint error in GATE coordinates, no global state needed.

Each detection tuple is now `(t_cam, w_px, area, clipped, obl, lat)`;
the flight loop EMAs the tracked candidate's values (`fg_obl_f`,
`fg_lat_f`, signed `fg_lat_s`) under env `FG_POSE=1`. Cost: <1 ms on
top of detect(). Caveat: planar PnP has a hemisphere sign ambiguity —
resolved by temporal consistency; magnitudes are stable.

Validation (offline truth in `vq2/tools/gaterel_offline.py`, run
against recorded corpora): the ticked flight held obl 12-22 deg /
|lat| <= 0.25 m through its terminal; the oblique miss read
obl 17->32 deg and |lat| 0.36->0.9 m growing — cleanly separable, and
visible seconds before impact. Flight-proven consumers (July 19
flights 93-98): settle entry APPROVE/VETO on measured obliquity
(binding when a fresh measurement exists), and an OFF-NORMAL ABORT
(|lat| > 0.5 m or obl > 45 deg) that twice broke off doomed
approaches the bearing rules could not see.

### How they compose

Bearing servoing FLIES (fast, estimator-free tracking of the hole);
gate-relative pose JUDGES (is this approach geometrically sound?) and
is beginning to STEER (a lateral term driving `fg_lat_s -> 0` in the
settle — i.e. flying to a waypoint expressed in the gate's own frame,
which is the practical answer to "can we just fly to a coordinate"
when no trustworthy global position exists). Open item as of this
commit: the lateral term does not yet overcome the known ~0.3 m/s
gate-area push (starved when the hole clips out of frame near the
gate) — the abort covers that gap in the meantime.
