# Multi-Gate Lookahead Design

## Goal

Extend the RL observation space so the drone policy sees N upcoming gates (configurable, default 1) instead of just the next gate. This enables anticipatory flight through gate sequences — the policy can plan turns earlier when it knows what's coming after the next gate.

## Architecture

Flat observation extension: append additional 4-dim blocks (3D relative position + yaw delta, in current gate's yaw frame) for each extra lookahead gate. Gate indices wrap around via modulo (per-env, since each env may have a different track length) since tracks are closed loops. The extra lookahead count is configurable via Hydra (`sim.n_lookahead_gates`, default 1 = current behavior).

## Observation Layout

### Current layout (24 dims)

```
[0:3]   position drone → current gate (gate-yaw frame)
[3:6]   velocity (gate-yaw frame)
[6:8]   roll, pitch
[8]     yaw relative to current gate
[9:12]  body angular rates
[12:16] motor speeds (normalized)
[16:20] next gate: relative pos (3) + yaw delta (1)
[20:24] previous action
```

### Proposed layout (n_lookahead_gates=1, default — 24 dims)

The prev_action and gate lookahead blocks swap positions so that the variable-length gate block sits at the end of the observation vector:

```
[0:3]   position drone → current gate (gate-yaw frame)
[3:6]   velocity (gate-yaw frame)
[6:8]   roll, pitch
[8]     yaw relative to current gate
[9:12]  body angular rates
[12:16] motor speeds (normalized)
[16:20] previous action
[20:24] lookahead gate 1: relative pos (3) + yaw delta (1)
```

### Proposed layout (n_lookahead_gates=2 — 28 dims)

```
[0:20]  same base dims as above
[20:24] lookahead gate 1: relative pos (3) + yaw delta (1)
[24:28] lookahead gate 2: relative pos (3) + yaw delta (1)
```

General formula: `OBS_DIM = 20 + 4 * n_lookahead_gates`

Where `n_lookahead_gates` is the number of future gates visible beyond the current gate. Default 1 = current behavior (just the next gate). Setting to 2 means seeing the next 2 gates, etc.

**Note:** The default (n_lookahead_gates=1) contains the same *information* as the current 24-dim obs, but the ordering of prev_action and gate block is swapped. This is a breaking change for saved models, which is acceptable since we're actively iterating on the policy.

### Lookahead gate encoding

Each lookahead gate k (k=1,2,...,n_lookahead_gates) is encoded as 4 dims:
- **Relative position (3D)**: Vector from current gate to gate k, rotated into current gate's yaw frame
- **Yaw delta (1D)**: Heading difference between gate k and current gate, normalized to [-pi, pi]

This matches the existing encoding for the "next gate" (currently at obs[16:20] in the old layout), generalized to a loop.

### Magnitude scaling

Lookahead gates further away will have larger relative position magnitudes. For typical procedural tracks (gate spacing 1.5-4.0m), even gate k=3 will be at most ~12m away — well within the same order of magnitude as the current next-gate distance. No additional normalization or clipping is needed for N<=3. For larger N values in future experiments, observation normalization could be revisited.

## Files Modified

### `sim/envs/gate_race_env.py`
- Add `n_lookahead_gates: int = 1` constructor parameter (1 = current behavior)
- Replace hardcoded `OBS_DIM = 24` with computed `self._obs_dim = 20 + 4 * n_lookahead_gates`
- Refactor `_compute_obs_batched()`: move prev_action to [16:20], replace the hardcoded next-gate block with a loop over `range(1, n_lookahead_gates + 1)` that fills dims `[20 + 4*(k-1) : 20 + 4*k]` for each lookahead gate k
- Gate indices wrap per-env via modulo: `(gate_idx + k) % self._tracks[i].num_gates`

### `sim/envs/numpy_quad_factory.py`
- Pass `n_lookahead_gates` from factory config to `GateRaceEnv` constructor
- Add `n_lookahead_gates: int = 1` to factory constructor signature

### `configs/train.yaml`
- Add `n_lookahead_gates: 1` under `sim:` section (explicit default)

## Files NOT Modified

- **Policy architecture**: MLP auto-sizes to observation dimension via SB3
- **Reward function**: No reward changes needed
- **Training loop** (`training/loops/rl.py`): No changes
- **Trajectory recorder**: Records obs as-is, no schema change
- **Perception wrapper**: Identity wrapper passes through unchanged
- **Procedural track generator**: Generates tracks independently of obs space

## Backwards Compatibility

The default `n_lookahead_gates=1` contains the same information as the current 24-dim obs, but with prev_action and gate-lookahead blocks swapped in position. This means saved models are NOT compatible — a fresh training run is required. This is acceptable since we're actively iterating on the policy and the reorder gives a cleaner layout (fixed dims first, variable gate block last).

## Constraints

- Minimum `n_lookahead_gates` is 1 (just the next gate — matches current behavior, 24 dims)
- Maximum is unbounded thanks to per-env modulo wrapping: `(gate_idx + k) % self._tracks[i].num_gates`
- With 4-8 gate procedural tracks, n_lookahead_gates=2 means seeing 25-50% of the course ahead

## Performance

The inner loop in `_compute_obs_batched` already iterates per-env (100 envs). Adding a nested loop over `range(1, n_lookahead_gates + 1)` for each env is negligible for typical values (1-3). Vectorization across envs is a future optimization if needed.

## Testing

- Unit test: verify obs dim matches `20 + 4 * n_lookahead_gates` for n=1,2,3
- Unit test: verify gate wrapping with n_lookahead > num_gates (e.g., n=5 on a 4-gate track)
- Unit test: verify lookahead obs values match manual computation — e.g., 3 gates in a line along X-axis, verify lookahead[1] position is the gate-to-gate vector rotated by negative current-gate yaw
- Integration test: training smoke test with n_lookahead_gates=2 runs without errors
