# COR-127 Pod-Scale RL Plan — VQ2 racing competitiveness

Goal: a policy that **transfers live to gate-0 capture** (lateral < 1.5 m), not just in-env.
Honest motivation: VQ1 has analytic fallback (`vq_waypoint2`, `vq_course`); **VQ2 racing speed needs RL** because analytic controllers cap at ~9.4 m/s sideslip (WAYPOINT-02) and brittle to course geometry. RL is the only path to scale.

What tonight proved (TRANSFER-01..10):
- **Rate-path control transfers live**: cap6brake = 30 s controlled flight, true tilt 35–48°, no tumble. The 8-day "PPO destroys warm-start" wall is broken via `log_std=-2.5 ent=0.005` (POD-SWEEP-01: PPO 4.4 → 4.9 fin 4/16 vs prior 5.0 → 0.7 erosion).
- **Reward shaping responds live** (sideslip pen drops beta 154 → 72) but doesn't crack lateral capture at 1M Mac.
- **Sim-to-sim divergence is the binding constraint**: every in-env improvement specializes to matched-plant precision live plant doesn't share.
- Champion `ft_cap6brake_v6c_best` (live 8.7 m closest, controlled 30 s). Bar to beat.

## What pod scale needs (all three, not in series)

### 1. Wider plant DR (the sim-to-sim cure)
Measured live divergences from this campaign (numbers to match):
- Mid-band drag: live ~20% faster (TRANSFER-03, refit refuted but measured)
- Rate-loop ring: live ~11 Hz at higher Q than fit (RING-04/05)
- Latency: 12–22 ms cmd, 22 ms obs (step-input ID, 06-08)
- Thrust-floor + low-thr regime (LOWTHR-01)

Current DR is ±30% on rate_loop_mimo cross terms, ±20% on diag, ±30% on B; weathervane ±50%. Per MonoRace (arxiv 2601.15222, the 2025 Abu Dhabi champion): **15–55% wide DR every episode over all 30+ params** is what made their PPO transfer with no deploy→refit loop.

**Implement (cheap, ≤1h)**:
- Bump diag scatter ±20% → ±40%
- Bump cross scatter ±30% → ±50%
- Add rate_loop_2nd pole-magnitude scatter ±20% (currently ±10%)
- Add drag Dx/qx scatter ±30% across the band
- Latency scatter 0–30 ms (currently fixed 0)
- Per-episode resample (currently per-DAgger-round) — needs an env hook

### 2. LSTM/temporal architecture (the hidden-state cure)
NeuroBEM lesson: single-step MLP cannot resolve sideslip/wake/wind from instantaneous obs. Need ~50 ms / 20-sample window. Repo has `RecurrentPPO` import path; `--recurrent` flag exists in rl_finetune but never run successfully on VQ.

**Implement (≤2h to first signal)**:
- Verify `--recurrent` path works at the C-config (log_std=-2.5 ent=0.005)
- Test budget: 5M PPO single seed, compare LSTM vs MLP-baseline
- If LSTM > MLP at 5M → scale to 30M multi-seed
- If LSTM = MLP → fall back to MLP + history-stacked obs (manual 20-step action history in obs vector)

### 3. 10–30M multi-seed PPO (the trajectory budget)
1M Mac is too short to see PPO past initial DAgger erosion/recovery. POD-SWEEP-02 showed 1M ≈ 3M (plateau). Real PPO trajectories climb 5–30M on similar tasks (MonoRace 100M+ but they have wider DR).

**Implement**:
- 4 seeds × winning arch (LSTM or MLP) × 30M steps each, parallel on A4500. ~10 h × $0.25/hr × 4 = ~$10 if serial; 4 parallel on one A4500 = same wall-clock but uses GPU/CPU contention. Likely run as 2 parallel × 2 sequential.
- Eval every 1M; save best-by-eval (existing mechanism, but **with eval_radius bug fixed first**).

## Bugs to fix before scale (queued)

1. **`eval_gates(rad)` param** (TRANSFER-10): eval_gates currently builds fresh env at RAD_START, so curriculum eval doesn't match training rad. Add `rad` param + pass current curriculum rad from callback.
2. **`make_env(train=False)` eval radius mismatch**: even without curriculum, eval should use the same radius as training. Audit.
3. **PPO eval frequency**: currently 500k steps. Drop to 250k for finer signal at scale.

## Step-by-step pod plan

### Phase 0 — preflight (Mac, before pod resume; ~1 h)
- [ ] Fix eval_gates rad param + commit
- [ ] Audit make_env for train-vs-eval radius parity
- [ ] Patch DR widths in rl_finetune dr_model_path() per §1 above
- [ ] Bump PPO eval every→250k
- [ ] Smoke test on Mac: `rl_finetune --recurrent --log_std -2.5 --ent 0.005 --steps 500_000 --tag mac_lstm_smoke` — does LSTM warm-start + run without errors?

### Phase 1 — single-config baseline (pod, ~2 h, ~$0.50)
- [ ] Resume pod (`runpodctl pod start ur5fncrpamh05u`)
- [ ] Push commits, pull on pod
- [ ] Run 2 single-seed configs in parallel, 5M each:
  - **MLP**: `--maxw 6 --thrmax 0.6 --slew 40 --scatter --dr --log_std -2.5 --ent 0.005 --steps 5_000_000 --tag pscale_mlp_5M`
  - **LSTM**: same + `--recurrent --tag pscale_lstm_5M`
- [ ] Compare eval trajectories. Winner = arch for scale.

### Phase 2 — winning-arch multi-seed scale (pod, ~10 h, ~$3)
- [ ] 4 seeds × winning arch × 30M steps. 4 parallel on A4500 (CPU-bound rollouts share 48 vCPU fine; GPU shared too but MLP-PPO is tiny).
- [ ] Eval every 1M; save best-by-eval per seed.
- [ ] Pull winners, grid on live-branch.

### Phase 3 — live test winner (laptop, ~30 min)
- [ ] Refly top 2 winners on the gate-0 course.
- [ ] capture_geom on each.
- [ ] Pass criterion: closest_dist < 1.5 m on at least one run → first live gate pass.
- [ ] Failure modes:
  - Closest < cap6brake's 8.7 m but > 1.5 m → reward iteration round (sideslip + aperture + curriculum)
  - Closest > 8.7 m → DR was too wide (over-conservative) or arch insufficient (escalate to asymmetric privileged critic)

### Phase 4 — reward+curriculum iteration on winning arch (pod, ~6 h, ~$1.50)
Only if Phase 3 doesn't pass a gate. Sweep on winning arch:
- ws ∈ {0, 0.05, 0.10}
- wa ∈ {0, 1.0, 2.0}
- rad_curr ∈ {1.0 const, 3→1.5}
- 3×3×2 = 18 configs at 5M each in parallel × 2 (9 parallel)
- Pick best, scale to 30M, refly.

### Phase 5 — VQ2 readiness (after first live gate)
Once gate-0 passes live:
- Full-course battery (cap6thr06-style 18-cell grid + live cap6rec3 recovery)
- 50% trajectory threading consistency = qualification-grade
- 90% = racing-grade
- Pose for VQ2 course handover

## Stop conditions / honest fallbacks

- Phase 1 + 2 + 3 with no live gate ≈ $4 total. If no gate → reward iteration is right move.
- Phase 1 → 4 with no live gate ≈ $5.50. If still no gate, then **the bottleneck isn't training** — it's the live↔sim drag/rate-loop mismatch we measured but never refit successfully. Reopen TRANSFER-03 with a smarter calibration (segment-band drag fit, not global lstsq).
- $80 spend limit gives ~30 h of pod runs total — plenty of budget for 2–3 full iteration loops.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| LSTM doesn't train (BC → LSTM is unsupported in rl_finetune) | Run LSTM from scratch (warm=0) OR stack action history into MLP obs (manual temporal) |
| Wider DR breaks DAgger demos (teacher fails on extreme DR samples) | Cap DAgger DR at current width (±30%), expand only PPO DR (split flag) |
| 30M takes longer than estimated | 4 seeds × 10M each first, then extend best to 30M |
| A4500 host still full at resume | Fall back to RTX 4090 in different DC; copy /workspace/algo_src_cor127 first via tar over ssh |

## Code touched (committed)
- `--ws --wa --wc` reward weights, `sideslip_penalty` + `aperture_proximity_reward`
- `--log_std --lr --ent --epo --clip --target_kl` PPO flags
- `--rad_start --rad_end --rad_steps` curriculum (eval-rad bug pending)

## Champion baseline (bar to beat)
- `ft_cap6brake_v6c_best.zip` — live 8.7 m closest, 30 s controlled flight, true tilt 35–48°, no tumble
- Lateral closest -5.0 m (capture_geom on TRANSFER-01 recording)
- Grid on live branch: 17/18 (lag-0), 18/18 (lag plant)
