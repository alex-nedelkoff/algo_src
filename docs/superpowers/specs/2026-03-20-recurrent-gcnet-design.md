# Recurrent G&CNet Design Spec

**Date:** 2026-03-20
**Status:** Approved
**Related:** COR-64 (GRU/LSTM integration), COR-65 (n_lookahead_gates=3)

## Problem

The memoryless 3×64 MLP policy cannot:
- Disambiguate crossing points on figure-8 tracks (75% gate passage vs 100% on procedural)
- Plan multi-step approach trajectories through gate sequences
- Implicitly estimate wind/disturbances from velocity history
- Smooth out perception noise over time

The CRL paper (Sun et al., ICRA 2026) ablation showed removing recurrent context drops success rate from 100% to 77%.

## Solution

Replace the MLP-only policy with MLP encoder + LSTM + Actor/Critic heads, using sb3-contrib's `RecurrentPPO` which handles hidden state management, recurrent rollout buffers, and episode boundary resets automatically.

## Architecture

```
obs (32-dim) → MLP encoder [64, 64, ReLU] → LSTM (128 hidden, 1 layer)
                                                    ↓
                                            Actor:  [64, ReLU] → 4 (TRPY)
                                            Critic: [64, ReLU] → 1 (value)
```

- **Observation**: 32-dim (20 base + 4×3 lookahead gates)
- **LSTM hidden**: 128 units, 1 layer
- **Total params**: ~25K (up from ~10K MLP-only)
- **Inference**: <1ms on Jetson Orin NX at 500 Hz

Note: sb3-contrib RecurrentPPO uses LSTM (not GRU). LSTM provides equivalent temporal context with an additional cell state. The CRL paper's ablation results apply to either architecture.

## Components

### 1. RecurrentGCNetPolicy

**File:** `control/policies/recurrent_gcnet.py`

Subclasses sb3-contrib's `RecurrentActorCriticPolicy`. Injects a custom MLP feature extractor (2×64 ReLU layers, features_dim=64). The LSTM layer and Actor/Critic heads are managed by sb3-contrib internally.

Configuration via policy_kwargs:
- `features_extractor_class`: Custom MLP encoder
- `lstm_hidden_size`: 128
- `n_lstm_layers`: 1
- `net_arch`: `{"pi": [64], "vf": [64]}` (post-LSTM heads)

### 2. PPO Wrapper Extension

**File:** `control/algorithms/ppo.py`

New `recurrent: bool` parameter (default False). When True:
- Uses `RecurrentPPO` from `sb3_contrib` instead of `PPO` from `stable_baselines3`
- Passes `lstm_hidden_size` and `n_lstm_layers` to the algorithm
- Sets `policy_type` to `RecurrentGCNetPolicy`

The rest of the PPO wrapper (train, save, load) works identically — sb3-contrib's RecurrentPPO has the same interface as SB3 PPO.

### 3. ONNX Export

**File:** `control/algorithms/ppo.py` (modified `export_onnx`)

For deployment, LSTM hidden state becomes explicit I/O:
```
Inputs:  observation (1, 32), lstm_h (1, 1, 128), lstm_c (1, 1, 128)
Outputs: action (1, 4), lstm_h_out (1, 1, 128), lstm_c_out (1, 1, 128)
```

The Jetson inference loop maintains (h, c) between calls, resets to zeros at episode start.

### 4. Observation Space Change

**Config:** `n_lookahead_gates: 3`

Obs dim changes from 28 (20 base + 4×2) to 32 (20 base + 4×3). The 3rd lookahead gate provides trajectory direction context at figure-8 crossing points.

This requires training from scratch — cannot resume from 28-dim checkpoints.

## What Does NOT Change

- **GateRaceEnv** — no modifications needed
- **VecEnvAdapter** — no modifications (RecurrentPPO manages hidden state internally)
- **ARPOActionWrapper** — no modifications (acts on actions only)
- **Reward functions** — all 12 components work as-is
- **Curriculum callback** — works with RecurrentPPO's logger interface
- **Multi-scene callback** — no hidden state interaction
- **Perception noise wrapper** — observation-level, no hidden state

sb3-contrib's `RecurrentRolloutBuffer` handles LSTM state storage, sequence chunking, and episode boundary resets. No env-side changes required.

## Training Configuration

Fresh from-scratch run (~40M steps):
- RecurrentPPO with LSTM(128)
- `n_lookahead_gates: 3` (32-dim obs)
- α-RPO bootstrap (base policy provides consistent trajectories for LSTM to learn from)
- 40% tight figure-8 mix, `gate_collision: false`
- All rewards: centering(4.0), boundary(5.0), speed_bonus(0.3), approach(0.3)
- Perception noise enabled
- Curriculum: conservative → moderate → speed push
- `ent_coef: 0.005` (higher than recent runs since training from scratch)

## Dependencies

Add to pyproject.toml `control` extras:
```toml
"sb3-contrib",
```

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| RecurrentPPO slower than PPO (sequence processing) | LSTM is tiny (128 units). Overhead is ~10-20% based on sb3-contrib benchmarks |
| 40M steps not enough from scratch | Can extend; α-RPO bootstrap accelerates early training |
| LSTM overfits to specific track patterns | Multi-scene + figure-8 mix provides diversity |
| ONNX export with LSTM state | Well-documented pattern; PyTorch ONNX supports LSTM natively |
