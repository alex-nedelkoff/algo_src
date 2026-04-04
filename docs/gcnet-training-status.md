# G&CNet Training Status

> Last updated: 2026-04-04. Tracking issue: [COR-87](https://linear.app/corvidx-drone-grand-prix/issue/COR-87)

## Best Models

| Model | Checkpoint | Tier 1 | Tier 2 | Overall | Old Benchmark | Download |
|---|---|---|---|---|---|---|
| **MoE+GRU 100M** | `outputs/2026-04-03/16-40-03/final_model.zip` | 108% | **58%** | **88%** | 4.1 gates | [R2](https://pub-030397d1ed024b568ceff9508ce0bd30.r2.dev/checkpoints/best-gru-overall-88pct.zip) |
| MoE MLP 33dim 20M | `outputs/2026-04-02/01-58-43/final_model.zip` | **93%** | 25% | 80% | 5.8 gates | [R2](https://pub-030397d1ed024b568ceff9508ce0bd30.r2.dev/checkpoints/best-mlp-tier1-93pct.zip) |
| MoE MLP 33dim 50M | `outputs/2026-04-02/12-08-03/final_model.zip` | 72% | 32% | 56% | 6.3 gates | [R2](https://pub-030397d1ed024b568ceff9508ce0bd30.r2.dev/checkpoints/best-mlp-tier2-32pct.zip) |

**Use the GRU model** for new work. It's the best overall and the architecture path forward.

## Architecture

**MoE+GRU(64)**: 5-expert Mixture of Experts with a GRU temporal context layer.

```
obs(33) → GRU(64) → [obs, gru_context](97) → Router → top-2 Experts → action(4)
                                              → Value Net → value(1)
```

- **Observation (33 dims)**: gate_rel_pos(3) + vel(3) + attitude(3) + body_rates(3) + motors(4) + prev_action(4) + 2x lookahead gates (6 each: rel_pos + yaw_delta + width + height) + arena_extent(1)
- **Action**: TRPY (thrust, roll_rate, pitch_rate, yaw_rate) in [-1, 1]
- **Params**: ~160K (5 experts x 128 hidden + GRU + router + value_net)

Key files:
- `control/policies/moe_policy.py` — MoE+GRU policy (set `gru_hidden_dim: 64` to enable GRU, `0` for MLP-only)
- `control/algorithms/ppo.py` — PPO wrapper, passes `gru_hidden_dim` through
- `control/expand_obs.py` — checkpoint surgery for obs dim expansion
- `control/expand_moe.py` — checkpoint surgery for adding experts

## Benchmarks

### 57-Track Golden Benchmark (PR #2)

57 hand-crafted tracks across 5 tiers. This is the primary eval for generalization. Run with:

```bash
python -m evaluation.golden_eval --checkpoint <path> --tier 1 --tier 2 -o results.json
```

Tiers:
1. **Atomic maneuvers** (18 tracks): straights, turns, climbs, dives, slaloms
2. **Composed courses** (12 tracks): multi-maneuver circuits with 5-14 gates
3. Obstacle courses (9 tracks) — not yet targeted
4. Perceptual challenges (10 tracks) — not yet targeted
5. Dynamic environments (8 tracks) — not yet targeted

### Old Golden Benchmark (10 layouts)

10 layouts x 5 variants = 50 environments. Results feed the [leaderboard](https://pub-030397d1ed024b568ceff9508ce0bd30.r2.dev/leaderboard/index.html).

```bash
python -m evaluation.benchmark run --checkpoint <path> --sim-config <hydra_config> --golden-set-dir configs/golden_set
python -m evaluation.benchmark leaderboard-refresh  # update leaderboard on R2
```

## Training Infrastructure

### Composable Track Segments

8 segment primitives that chain into diverse training tracks:

| Segment | Gates | Description |
|---------|-------|-------------|
| `straight` | 1-2 | Along heading, 3-6m spacing |
| `turn_90` | 1-2 | 90-degree arc |
| `turn_180` | 2-3 | Hairpin reversal |
| `slalom` | 3-5 | Alternating left-right |
| `chicane` | 2-3 | Tight S-curve |
| `climb` | 1-2 | Elevation gain |
| `dive` | 1-2 | Elevation loss |
| `orbit` | 3-5 | Sustained 180-270 deg arc |

`ComposedTrackGenerator` chains 3-5 segments with automatic closure. Used at 45% of training tracks.

Key files: `sim/track_segments.py`, `sim/composed_tracks.py`

### Experiment Configs

| Config | Description |
|---|---|
| `moe_gru.yaml` | MoE+GRU from scratch, 33-dim obs, composable tracks |
| `moe_33dim.yaml` | MoE MLP with 33-dim obs (gate geometry + arena extent) |
| `moe_diverse_tracks.yaml` | MoE MLP with composable tracks, 28-dim obs |
| `moe_spline_speed.yaml` | Original MoE with spline speed reward |

### Checkpoint Surgery

Expand obs dim (e.g., 28→33) without retraining from scratch:

```bash
python -m control.expand_obs --checkpoint <old>.zip --old-obs-dim 28 --new-obs-dim 33 --output <new>.zip
```

Zero-pads new input columns so existing weights are preserved.

## Key Learnings

### What Worked
1. **Composable track segments** — tier 1: 44% → 79%. The policy needs to see diverse maneuver patterns during training.
2. **Gate geometry in obs** (width + height per gate) — tier 1: 79% → 93%. The policy can adapt approach strategy per gate.
3. **Arena extent in obs** — crash rate: 20% → 10%. Gives the policy a track-scale signal.
4. **GRU temporal context** — tier 2: 28% → 58%, crash rate: 89% → 39%. Memory is essential for multi-maneuver circuits.
5. **Freeze+warmup for new experts** — router must stay trainable during warmup.

### What Didn't Work
1. **Reward cranking** (gate_passage 100→150, progress 3→5) — always regresses. Don't repeat.
2. **Longer segments** (4-6 at 55% ratio) — too aggressive, policy forgot fundamentals.
3. **Speed cap** (v_max=3.5) — tier 2 didn't improve. Geometry not speed is the bottleneck.
4. **80M+ continuation** on same config — plateaus at ~60M.

### Architecture Decision

MoE+GRU is the path forward. The MLP policy fundamentally can't reason about maneuver sequences. The GRU's temporal memory allows planning through transitions, which is exactly what tier 2 requires.

## Links

- [Leaderboard](https://pub-030397d1ed024b568ceff9508ce0bd30.r2.dev/leaderboard/index.html)
- [Track Viewer](https://pub-030397d1ed024b568ceff9508ce0bd30.r2.dev/leaderboard/track-viewer.html)
- [W&B Project](https://wandb.ai/janahanr-corvidx/corvidx-drone-racing)
- [COR-87 (tracking issue)](https://linear.app/corvidx-drone-grand-prix/issue/COR-87)
- [NotebookLM Research](https://notebooklm.google.com/notebook/e8224c7a-5ef8-4568-94eb-153696f760a9) — drone racing papers
