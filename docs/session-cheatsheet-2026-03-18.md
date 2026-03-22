# Session Cheat Sheet — 2026-03-18

**Project:** Autonomous drone racing for the Anduril AI Grand Prix
**Team:** Corvidx (3 people)
**Hardware target:** NVIDIA Jetson Orin NX onboard computer, single camera + IMU sensor

---

## What is this project?

We are building a fully autonomous racing drone. The drone has no human pilot — an AI (a neural network trained with Reinforcement Learning) learns to fly as fast as possible through a series of gates on a track. The target competition is the Anduril AI Grand Prix.

**Key acronyms you'll see everywhere:**

- **RL (Reinforcement Learning):** A way to train an AI by letting it take actions, observe outcomes, and learn from rewards and penalties — like training a dog with treats, but for software.
- **PPO (Proximal Policy Optimization):** The specific RL algorithm we use. It's one of the most reliable and widely-used algorithms for training continuous control agents. "Proximal" means it takes careful, not-too-large learning steps to avoid catastrophic forgetting.
- **SB3 (Stable Baselines 3):** A Python library that provides ready-to-use implementations of RL algorithms including PPO. We build on top of it instead of writing PPO from scratch.
- **W&B (Weights & Biases):** An experiment tracking platform — it logs metrics, plots graphs, and stores model checkpoints every time we run a training job. Think of it as a lab notebook that fills itself in automatically.
- **OOB (Out Of Bounds):** When the drone flies outside the allowed track area and the simulation terminates the episode as a failure.
- **GCNet (Guidance and Control Network):** The name for our policy network — the neural network that watches sensor data and outputs drone commands.
- **MLP (Multi-Layer Perceptron):** The basic type of neural network; a stack of fully-connected layers. Our policy is an MLP.
- **EKF (Extended Kalman Filter):** A classical algorithm for estimating position and velocity from noisy sensor readings. It fuses camera and IMU data to give the drone a best-guess of where it is.
- **IMU (Inertial Measurement Unit):** A chip containing an accelerometer and gyroscope. It measures acceleration and rotation rate many times per second.

---

## 1. We Investigated a Crashed Training Run

**Run name:** `vivid-leaf-35`

A previous training run was going well — the drone was actively improving — but the laptop applied a Windows update mid-run and killed the process at **12 million training steps** (steps = total decisions the agent made across all parallel simulations).

We pulled the logged data from W&B to understand the state of training when it died:

- **59% of episode endings were OOB crashes** — the drone was leaving the track often.
- There was a growing **train/eval gap**: performance in the training environment was better than in the evaluation environment. This is like a student who memorizes practice problems but struggles on the actual test.
- Despite the crashes, the drone was clearly in an "acceleration phase" — it was starting to complete full laps, which is a good sign.

**Takeaway:** The run was not a failure — it had learned real skills. But it needed better techniques to reduce crashes and generalize better. That's what the rest of the session addressed.

---

## 2. We Reviewed Three Research Papers

We read three academic papers and identified techniques to borrow. Here's a plain-English summary of each:

### TraD-RL (Trajectory-guided Deep Reinforcement Learning)

**Core idea:** Instead of just rewarding the drone for passing through gates, fit a smooth curve (a **cubic spline** — a mathematical curve that passes smoothly through a set of points) through all the gate centers. Then reward the drone for staying close to this curve.

Think of it like a highway: the gates are the exits, and the spline is the road itself. The drone is rewarded for staying on the road, not just for reaching exits.

**Why this helps:** Without a path reward, the drone might pass through gates via wild, inefficient trajectories. The spline shapes behavior between gates, encouraging smoother, faster flight.

### α-RPO (Alpha Residual Policy Optimization)

- **PD (Proportional-Derivative) controller:** A classical (non-AI) control algorithm that computes corrections proportional to how far off you are (P) and how fast the error is growing (D). Like cruise control in a car.
- **Residual policy:** Instead of training from scratch, the RL agent learns a *correction* on top of a working baseline controller.

**Core idea:** Start training with a classical PD controller doing most of the work. The RL agent adds a small correction. Over time, gradually reduce the classical controller's influence and let the RL policy take full control. The parameter **alpha (α)** controls the blend: alpha=1 means 100% classical, alpha=0 means 100% learned policy.

**Why this helps:** Pure RL from scratch often spends millions of steps just learning "don't crash immediately." With a bootstrap controller, early training is already useful — the agent starts from a working baseline rather than random noise.

### CRL (Curriculum Reinforcement Learning for drone racing)

- **GRU (Gated Recurrent Unit):** A type of neural network layer with memory. Unlike an MLP which only sees the current frame, a GRU remembers recent history. This helps when the drone can't perceive all the information it needs in a single snapshot.
- **Curriculum learning:** Training in stages, starting with easier tasks and progressively introducing harder ones — like how school teaches arithmetic before calculus.
- **Domain Randomization (DR):** Randomly varying simulation parameters (wind, mass, sensor noise) during training so the policy learns to handle variation, not just one fixed scenario.

**Core ideas borrowed:**
1. **Curriculum staging:** Move to harder tasks based on performance, not just after a fixed number of steps.
2. **Multi-scene track regeneration:** Each rollout (batch of data collection), different groups of parallel environments get different procedurally-generated tracks. This prevents overfitting to a single track layout.
3. **Heading alignment reward:** Reward the drone for *pointing toward* the next gate, not just for being close to it.

---

## 3. New Action Space: TRPY

**What changed:**

- **Old system:** The policy directly output 4 motor speeds in **RPM (Revolutions Per Minute)** — one command per motor.
- **New system:** The policy outputs **TRPY** — [Thrust, Roll rate, Pitch rate, Yaw rate].

**What TRPY means:**
- **Thrust:** How hard all motors collectively push upward (total lift).
- **Roll rate:** How fast the drone rotates left/right (tipping sideways).
- **Pitch rate:** How fast the drone rotates forward/backward (nose up/down).
- **Yaw rate:** How fast the drone spins around its vertical axis (turning left/right while staying level).

**Why the change:** The Anduril Grand Prix competition requires TRPY output to interface with their proprietary **ESCs (Electronic Speed Controllers)** — the hardware boards that translate commands into actual motor voltages. The competition organizers provide the ESC interface, and it expects TRPY, not raw RPMs.

**What we built:** A `TRPYMixer` class that converts TRPY commands into individual motor speeds using a **quadrotor allocation matrix** — a fixed mathematical transform that describes how each motor contributes to each of the four axes of control.

We also created an `ActionMode` enum (a named set of constants) to cleanly switch between RPM and TRPY modes instead of using raw strings like `"trpy"`, which are easy to mistype.

---

## 4. Cubic Spline Reward Shaping

This implements the core idea from the TraD-RL paper.

**What we built:**
1. At the start of each episode, fit a cubic spline through all gate center positions.
2. Each step, compute how far the drone is from the nearest point on the spline.
3. Reward the drone for being close to the spline (soft tube constraint).
4. Also add a reward for **heading alignment** — the dot product between the drone's forward direction and the spline's tangent direction at the nearest point. This rewards the drone for pointing the right way, not flying sideways.

**Why batch queries:** The spline is sampled at many points (e.g. 1000), and we query it every step for every parallel environment. We batch these queries using NumPy vectorized operations so it doesn't slow down training.

**Result:** OOB crashes dropped from **59% to 0.5%**. The drone stopped flying out of bounds because staying near the spline naturally keeps it on the track.

---

## 5. α-RPO Bootstrap

**What we built:**

A **VecEnv wrapper** — a class that wraps around the vectorized (parallel) simulation environment and intercepts actions before they reach the simulation.

- **VecEnv (Vectorized Environment):** SB3 runs many simulation instances in parallel to collect data faster. A VecEnv groups them together and provides a single interface.
- **Wrapper pattern:** Instead of modifying the environment itself, we wrap it — the wrapper sits in between SB3 and the environment, like a middleman.

**Why a wrapper, not a callback:**

SB3 supports **callbacks** — functions that fire at specific points during training. We initially tried to use a callback to blend the PD controller output. The problem: SB3 callbacks fire *after* `env.step()` has already been called, so the action has already been applied. A VecEnv wrapper intercepts actions *before* they reach the environment.

**The "sync trick":** Alpha (the blend parameter) needs to be updated in coordination with the training loop. We update alpha *between* rollout collection and the optimization step, ensuring consistent behavior within a single rollout batch.

**The PD tracker:** The classical controller computes a target velocity pointing toward the current gate, then applies PD control to drive actual velocity toward that target. It uses **privileged information** (exact gate positions) during training — information the policy won't have at test time. This is fine because during actual flight the learned policy takes over.

**Result:** At 2 million training steps, the bootstrap approach achieved **53% gate passage rate** vs **0%** for training from scratch. The policy had a strong starting point.

---

## 6. CRL Cherry-Picks

Three specific ideas borrowed from the CRL paper:

### Performance-Based Curriculum

Instead of advancing to harder training stages after a fixed number of steps, we advance when the drone actually meets a performance threshold (e.g. "pass gates at 75% rate for 5 consecutive evaluations"). This avoids rushing the agent into harder scenarios before it has mastered easier ones.

### Multi-Scene Track Regeneration

Each time we collect a batch of training data (a "rollout"), different groups of parallel environments get freshly generated tracks with different gate layouts. This is a form of **Domain Randomization** — it forces the policy to generalize to new tracks rather than memorizing one specific layout.

### Heading Alignment Reward

A small bonus reward each step based on how well the drone is pointed toward the next gate. Combined with the spline heading reward, this consistently nudges the drone to orient correctly throughout the track, not just near gates.

---

## 7. Training Runs: What Happened

We ran four training jobs in sequence. Each builds on the previous.

### Run 1: `toasty-resonance-40` (20 million steps)

All new features enabled together for the first time.

| Metric | Value |
|--------|-------|
| Eval gate passage rate | 99% |
| Avg speed | 3.0 m/s |
| Avg laps per episode | 1.36 |
| OOB crashes | ~0.5% |

This was the proof-of-concept run — everything worked together.

### Run 2: Speed Fine-tune (20 million more steps)

Adjusted reward weights: higher reward for gate passage, lower penalty for crashing. This encouraged the drone to be more aggressive.

| Metric | Value |
|--------|-------|
| Eval gate passage rate | 100% |
| Avg speed | 4.5 m/s |
| Avg laps per episode | 2.2 |
| Best lap time | 2.0 s |

### Run 3: `speed_perception` (20 million steps)

Added **perception noise** (simulated sensor errors in gate position estimates) and a **speed bonus** reward. Episodes were also lengthened so the drone could demonstrate many laps.

| Metric | Value |
|--------|-------|
| Eval gate passage rate | 99.5% |
| Avg speed | 5.0 m/s |
| Avg laps per episode | 7.64 |
| Best lap time | 1.8 s |
| Max laps in one episode | 15 laps in 30 seconds |

The drone became robust to imperfect sensor readings — a critical property for real hardware where the camera and EKF estimates are never perfect.

### Run 4: `quick_wins` (20 million steps, 2026-03-19)

Added **boundary penalty** (quadratic penalty near arena walls), **gate approach reward** (reward for velocity aligned with gate normal), **entropy annealing** (0.005 → 0.001), and **cosine LR decay** (3e-4 → 5e-5).

| Metric | Value |
|--------|-------|
| Eval gate passage rate | 99% |
| Avg speed | 5.2 m/s |
| Avg laps per episode | 7.38 |
| Best lap time | 1.78 s |
| Max laps in one episode | 16 |
| Max gates in one episode | 77 |
| Eval reward | +3193 |

Speed edged up and new best-ever records were set (77 gates, 16 laps in one episode). However, gate collisions remain the dominant failure mode at 73.5% of training terminations.

---

## 8. Before vs. After Summary

| Metric | Old baseline (12M steps, crashed run) | Final (80M total steps) |
|--------|:---:|:---:|
| Avg speed | 3.4 m/s | 5.2 m/s |
| Best lap time | 2.38 s | 1.78 s |
| Gate passage (train / eval) | 80.5% / 55.5% | 97.5% / 99% |
| Laps per episode (train / eval) | 0.38 / 0.015 | 3.11 / 7.38 |
| Best laps in one episode | 4 | 16 |
| OOB crashes | 59% | 17% |
| Gate collisions | unknown | 73.5% (current main problem) |
| Success rate (train / eval) | 7% / 0% | 5.5% / 80% |

The drone is now doing 16-lap runs at 5.2 m/s with noisy perception — a massive improvement from a drone that couldn't complete a single lap. The main remaining challenge is gate collisions (see Section 10 below).

---

## 9. Key Technical Decisions (with Reasoning)

These are the subtle choices that shaped the implementation:

| Decision | Why |
|----------|-----|
| VecEnv wrapper for α-RPO, not SB3 callback | Callbacks fire post-step; wrapper intercepts pre-step |
| PD tracker targets velocity toward gate (not zero velocity) | Targeting zero causes oscillation near gate; targeting a pass-through velocity is smoother |
| Linearized TRPY mixer (antisymmetric) | Ensures equal and opposite motor responses for roll/pitch — physically correct for a symmetric quadrotor frame |
| Batch spline queries with NumPy | Per-step, per-env queries in a Python loop would be too slow; vectorized operations are ~100x faster |
| `ActionMode` enum instead of strings | Prevents silent bugs from typos like `"TRPY"` vs `"trpy"`; makes switch statements exhaustive |
| Alpha updates between rollout and optimize | Ensures a full batch of data is collected at one alpha before changing it — cleaner training signal |

---

## 10. The Gate Collision Problem — What the Literature Says

Our biggest remaining issue: **73.5% of training terminations are gate collisions**. The drone crosses the gate plane but *outside* the gate opening radius. It's reaching gates but clipping the edges instead of flying cleanly through the center.

We queried our research notebook (43 papers on autonomous drone racing) for insights. Here's what the literature says:

### Why It Happens

1. **Conflicting reward gradients:** The crash penalty pushes the drone *away* from the gate frame, while the gate passage reward pulls it *toward* the center. At high speed, these opposing forces can cancel out or create "dead zones" where the policy gets no useful learning signal. The **DiffRacing** paper specifically studied this problem.

2. **No continuous centering feedback:** Our current system only penalizes off-center passage *at the moment of crossing*. There's no reward signal guiding the drone toward center as it approaches. By the time it knows it's off-center, it's too late to correct.

3. **Memoryless policy:** Our 3×64 MLP sees one snapshot at a time. It can't plan a smooth approach trajectory over multiple timesteps. **Swift** (the world champion system) looks 2-3 gates ahead. The **CRL paper** showed adding a GRU (recurrent memory) improved success rate from 77% to 100%.

4. **Actuator saturation:** At 5+ m/s, the drone may not have enough control authority to correct lateral error in time. If it commits to a slightly off-center approach, the physics prevents last-moment correction.

### What the Papers Recommend

**Most promising for our setup (roughly ordered by expected impact):**

| Technique | Source Paper | Core Idea |
|-----------|-------------|-----------|
| **Gate proximity centering reward** | Song et al. (2021) | Continuous penalty for lateral offset when near gate plane — creates "pressure" toward center before crossing |
| **Hourglass safety region (Gate-SDF)** | DiffRacing | Traversable space narrows as drone approaches gate — a "funnel" effect |
| **GRU temporal context** | CRL (Sun et al.) | Add recurrent memory so policy can plan multi-step approach trajectories |
| **Attractive Vector Fields** | DiffRacing | Replace distance-based rewards with field-based guidance that naturally spirals into gate center |
| **Spherical coordinate observations** | Multiple | Represent gate position as (distance, azimuth, elevation) instead of (x, y, z) — better separates "how far" from "which direction" |

### What We Implemented (gate_fix run)

We implemented the top recommendation — a **gate centering reward** that gives continuous feedback about lateral offset as the drone approaches each gate. The formula `r = -(offset/radius) * 1/(1+d²)` creates a "funnel" that gets stronger near the gate plane. Combined with forced low entropy (`ent_coef=0.001`), this dramatically reduced gate collisions in eval from ~73% to **8.5%**.

### Run 5: `gate_fix` (20 million steps, 2026-03-19)

| Metric | Before (quick_wins) | After (gate_fix) |
|--------|:---:|:---:|
| Eval gate collision | ~73% | **8.5%** |
| Eval success rate | 80% | **90.5%** |
| Eval avg speed | 5.2 m/s | **5.7 m/s** |
| Eval laps/ep | 7.38 | **8.41** |
| Best lap time | 1.78s | **1.18s** |
| Best laps in one ep | 16 | **18** |

---

## 11. Figure-Eight Track Evaluation — Transfer Test

After achieving strong results on procedural tracks, we tested the gate_fix policy on a hardcoded **figure-eight track** — a track shape the policy had **never seen during training**.

### Why This Matters

All training used **procedurally generated closed loops** — simple oval/kidney shapes with 4-8 gates where the path never crosses itself. A figure-eight is fundamentally different: the path crosses itself at the center, meaning the drone passes through the same spatial region going in *opposite directions* on alternating halves of the lap.

### Results: Significant Performance Drop

| Metric | Procedural tracks (eval) | Figure-8 track |
|--------|:---:|:---:|
| Gate passage | 100% | **55%** |
| Lap completion | 99% | **30%** |
| Success rate | 90.5% | **35%** |
| Gates/ep | 49 | **4.2** |
| Laps/ep | 8.4 | **0.3** |
| Avg speed | 5.7 m/s | **2.65 m/s** |
| Gate collisions | 8.5% | **40%** |
| Ground crashes | 1% | **20%** |

### Why the Figure-Eight Is Hard

1. **The crossing point is confusing.** Gates 3 and 7 sit just 0.6 meters apart at the center of the eight. The drone's observation includes the next 2 gates — at the crossing point, gates from both loops are nearby, creating ambiguity about which direction to fly.

2. **Much tighter geometry.** The figure-eight fits within ±3 meters, while procedural tracks spread to ±10m. The 180° turns required for the loops are beyond the 120° maximum the procedural generator uses.

3. **Speed drops by half.** The drone slows to 2.65 m/s — it's cautious and uncertain on unfamiliar geometry.

4. **Ground crashes increase 20×.** The tight turns may cause altitude loss that the policy hasn't learned to recover from.

### Key Insight

This is a **generalization gap**, not a capability gap. The policy learned excellent skills (gate finding, centering, speed) but only on one family of track shapes. It would likely recover quickly with a few million steps of fine-tuning on figure-eight tracks, or by mixing figure-eights into the procedural training distribution.

---

## 12. Figure-Eight Training Attempt (2026-03-20)

### What We Built

A `Figure8TrackGenerator` that creates randomized figure-eights with varying loop radius (1.5-4m), gates per loop (3-5), crossing offset (0.3-0.8m), and elevation. A `MixedTrackGenerator` wraps it with the existing procedural generator, selecting figure-8 tracks 30% of the time.

### Run 6: `figure8_training` (20M steps)

Mixed training: 30% randomized figure-eights, 70% procedural loops. Resumed from gate_fix checkpoint.

**Procedural track eval (mixed training distribution):**

| Metric | gate_fix (before) | figure8_training (after) |
|--------|:---:|:---:|
| Eval gate passage | 100% | **99%** |
| Eval success rate | 90.5% | **89.5%** |
| Eval avg speed | 5.7 m/s | **5.4 m/s** |
| Eval laps/ep | 8.41 | **8.95** |
| Best laps in one ep | 18 | **19** |
| Best gates in one ep | 75 | **107** |

Procedural track performance held — slightly slower (5.4 vs 5.7 m/s) but more laps per episode. The 107 gates in a single episode is a new all-time record.

**Figure-8 specific eval (hardcoded figure-8 track, gate_radius=1.0):**

| Metric | Before (gate_fix) | After (figure8_training) |
|--------|:---:|:---:|
| Gate passage | 55% | **55%** (no change) |
| Lap completion | 30% | **10%** (worse) |
| Success rate | 35% | **20%** (worse) |
| Gates/ep | 4.2 | **2.0** (worse) |
| OOB crashes | 5% | **35%** (much worse) |

### Why It Didn't Help

The figure-8 training **did not improve** performance on the hardcoded figure-8 track, and actually degraded it on some metrics:

1. **The randomized figure-eights are different from the hardcoded one.** The generator creates elliptical loops with 3-5 gates per loop spread over a larger area (±4m radius + ±3m translation). The hardcoded figure-8 is very tight (±3m) with 4 gates per loop, all at the same height, and a tiny 0.6m crossing offset. The randomized versions are easier — wider loops, more space between gates.

2. **OOB went from 5% to 35%.** The drone learned to fly faster on the wider randomized tracks but the hardcoded figure-8's tight geometry punishes that speed. It overshoots turns and flies out of bounds.

3. **The crossing point is still the core problem.** With only 2 lookahead gates, the drone can't reliably distinguish which loop it's on at the crossing point. The randomized figure-8s have a wider crossing offset (0.3-0.8m vs 0.6m), giving more room, but the hardcoded track's tight crossing remains confusing.

### Diagnosis

Simply mixing figure-eights into training doesn't solve the hardcoded figure-8 — the randomized versions are too different. The real bottleneck is:
- **2-gate lookahead is insufficient** at the crossing point
- **The tight geometry (±3m)** requires precision the policy hasn't learned on wider tracks
- The drone needs **direct exposure to the specific hardcoded layout** or very similar tight configurations

### Run 7: `figure8_finetune` — CATASTROPHIC FAILURE

Attempted a 5M step fine-tune directly on the hardcoded figure-8 track with high centering reward and low LR. **The policy collapsed completely**: 5% gate passage (down from 55%), zero laps, crashes in ~100 steps every episode.

**What went wrong:** The tight figure-8 with gate_collision=True creates a near-impossible environment for a policy trained on wider tracks. Every episode ends in a gate collision, so the reward signal is overwhelmingly negative. Over 5M steps, the policy learned "don't fly toward gates" — the opposite of what we wanted. This is **catastrophic forgetting**: intense single-task fine-tuning destroyed the general skills.

**Lesson learned:** Never fine-tune a general policy on a single hard track with harsh penalties. Instead:
- Mix the hard track into a broader distribution (low ratio)
- Disable gate_collision termination on the hard track initially
- Use very short fine-tune passes (500K, not 5M) with even lower LR

### Run 8: `figure8_gentle` — THE FIX THAT WORKED

Key insight: **train with `gate_collision=false`** (don't terminate on gate clips) so the policy gets centering reward signal instead of immediate death. Combined with tighter figure-8 generator params (loop_radius_max=2.5, gates_per_loop_max=4, crossing_offset_max=0.6).

**Figure-8 eval (strict gate_collision=true):**

| Metric | Before (gate_fix) | figure8_finetune (FAILED) | **figure8_gentle** |
|--------|:---:|:---:|:---:|
| Gate passage | 55% | 5% | **75%** |
| Lap completion | 30% | 0% | **50%** |
| Success rate | 35% | 0% | **50%** |
| Gates/ep | 4.2 | 0.1 | **8.2** |
| Laps/ep | 0.3 | 0 | **0.8** |

The successful episodes complete **2 full figure-8 laps in 30 seconds**. General procedural performance also held: 100% eval gate passage, 5.8 m/s, 93.5% success.

**Lesson:** Disable harsh termination on hard tracks during training. Let the policy learn from near-misses via continuous centering reward, not binary death.

---

## 13. LSTM Integration Attempt — Failed (2026-03-20/21)

### What We Built

Integrated sb3-contrib's `RecurrentPPO` with a custom `RecurrentGCNetExtractor` (MLP encoder → LSTM(128) → Actor/Critic). Also installed CUDA PyTorch (was CPU-only) for a 3.4× training speedup.

### Three Training Attempts — All Failed

**Attempt 1 (fast attenuation, CPU):** Base policy removed at 6M steps. LSTM collapsed immediately — 0% gate passage, 98% ground crashes. The LSTM hadn't learned to fly during the bootstrap phase.

**Attempt 2 (slow attenuation, CPU):** Base policy active until 32M of 40M steps. At 24.5M: training 92% gate passage (with base policy helping), but eval 1% (LSTM alone). Killed — LSTM was "riding along" without learning.

**Attempt 3 (slow attenuation, GPU):** Same config on CUDA. At 18.4M: training 84% gate passage, eval 1%. Same pattern — killed.

### Root Cause: Chicken-and-Egg Problem

The LSTM needs successful multi-gate trajectories to learn temporal patterns. But it can't generate those trajectories without the base policy. With the base policy active, the LSTM "rides along" — it receives good actions but doesn't internalize them. Without it, it crashes immediately.

The MLP policy succeeded from scratch because MLPs learn reactive control (one obs → one action) faster. LSTMs need consistent sequential data which they can't generate early in training.

### What Was Gained

- **CUDA PyTorch** — all future runs 3.4× faster (was CPU-only!)
- **sb3-contrib integration** — RecurrentPPO infrastructure tested and working
- **The code is ready** if a better training approach emerges (MLP→LSTM distillation, DAgger, etc.)

---

## 14. Golden Set Benchmark — Standardized Evaluation (2026-03-21)

Janahan built a **golden set benchmark suite** that tests policies across 10 diverse track types with 5 starting variants each (50 episodes total). We cherry-picked this from origin/main and ran our best policy through it.

### Track Types

| Category | Tracks | Description |
|----------|--------|-------------|
| **Handcrafted** | figure8, oval, hairpin, elevation_climb | Specific geometric challenges |
| **Procedural** | gen_seed_42/77/123/256/314/999 | Randomly generated with fixed seeds for reproducibility |

### Our Results vs Janahan's Best (60M steps)

| Track | Our policy | Janahan's |
|-------|:---:|:---:|
| **oval** | **19g, 3L, 0% crash** | 3.8g, 0L, 100% crash |
| **figure8** | 0g, 100% crash | **23.8g, 2.8L** |
| **hairpin** | 1g, 100% crash | 3.8g, 100% crash |
| **elevation_climb** | 0g, 100% crash | 2g, 100% crash |
| **gen_seed_123** | **5g, 0% crash** | 2.8g |
| **gen_seed_999** | **10g, 1L** | 1.4g |
| **Aggregate** | 3.8g, 0.4L, 80% crash | 4.4g, 0.3L, 90% crash |

*g=gates passed, L=laps completed per episode*

### Key Insight

Both policies have **significant generalization gaps**. Our policy dominates on smooth loops (oval, some seeds) but fails on hairpins and elevation changes. Janahan's dominates on figure-8 but crashes on ovals. Neither generalizes well to all track types.

The procedural track generator only creates smooth closed loops — it doesn't produce hairpins, elevation climbs, or figure-8 crossings. The policy overfits to the training distribution.

---

## 15. Generalist Training Attempts — Two Failures (2026-03-21)

Attempted to train a generalist policy matching golden set parameters (5m arena, 0.75m gate radius, diverse tracks with ±170° turns and ±1.5m elevation).

### Attempt 1: α-RPO bootstrap + diverse tracks (generalist v1)

At 24.8M steps, α-RPO base policy fully removed → policy collapsed to 0% gate passage, 82% OOB. Same failure pattern as the LSTM attempts — the learned policy "rides along" during bootstrap but doesn't internalize skills.

### Attempt 2: Spline-bootstrapped, no α-RPO (generalist v2)

Replaced α-RPO with heavy spline reward (3.0 weight) as the primary bootstrap signal. The idea: "the spline IS the teacher." At 37M steps:
- Eval spline proximity reward was 54.9 — **the drone learned to follow the spline**
- But **0% gate passage, 83% OOB** — following the spline wasn't enough to pass gates in a 5m arena
- Curriculum auto-advanced to stage 3 at 25M (timestep trigger), removing spline reward too early
- Episodes only 0.4 seconds — crashes almost immediately

### Root Cause

The 5m arena + 0.75m gate radius + diverse tracks (±170° turns, ±1.5m elevation) is too hard to learn from scratch in one shot. The drone can't survive long enough to accumulate meaningful experience. Even with dense spline rewards, the tight arena kills episodes before the drone reaches gates.

### Next Approach: Progressive Difficulty

Instead of starting at the hard environment, gradually increase difficulty:
1. Start with easy params (10m arena, 1.5m gates, ±120° turns) — policy learns to fly and pass gates
2. Progressively tighten: shrink arena, narrow gates, widen turn angles, increase elevation
3. End at golden set params (5m arena, 0.75m gates, ±170° turns)

This is the CRL paper's core insight applied to environment difficulty, not just reward weights.

---

## 16. Generalist v4 — Training on Tighter Gates Than Benchmark (2026-03-22)

### The Idea

Train on 0.5m gate radius (tighter than golden set's 0.75m) so the benchmark feels easy by comparison. Progressive difficulty: 5M easy (10m/1.5m) → 5M medium (7m/1.0m) → 30M hard (5m/0.5m).

### Result: Complete Failure

0 gates passed, 100% crash on golden benchmark. The policy learned to fly at 4.2 m/s but **never learned to pass through any gates at all**. 30M steps at 0.5m gate radius wasn't enough — the precision requirement was simply too high.

The curriculum auto-advanced from medium to hard after just 300K steps (the easy → medium performance trigger fired immediately). So the policy spent almost all 40M steps at the impossible difficulty.

### Lessons Learned Across All Generalist Attempts

| Attempt | Approach | Result | Problem |
|---------|----------|--------|---------|
| v1 | α-RPO + diverse tracks, golden set params | 0% gates at 24.8M | α-RPO attenuation cliff |
| v2 | Spline bootstrap, golden set params | 0% gates at 37M | 5m arena too hard from scratch |
| v3 | Progressive difficulty 10m→7m→5m/0.75m | 0% gates on benchmark | Only 15M steps at hard stage |
| **v4** | **Progressive + 0.5m gates** | **0% gates** | **0.5m too tight, never learned** |

**Core finding:** Training from scratch on golden set difficulty doesn't work with 40M steps. The 5m arena + tight gates + diverse tracks (hairpins, elevation) requires either:
1. Much more training (100M+ steps)
2. Resume from a capable policy and fine-tune to tighter params
3. Match training conditions more closely to golden set tracks

**Best policy remains `figure8_gentle`** — 3.8 gates, 0.4 laps on golden benchmark, trained on easier params (10m/1.5m). Our best approach would be to resume from this policy and gradually tighten parameters.

---

## Full Acronym Glossary

| Acronym | Full name | What it means in plain English |
|---------|-----------|-------------------------------|
| RL | Reinforcement Learning | Training AI by reward/penalty feedback |
| PPO | Proximal Policy Optimization | The specific RL algorithm we use |
| SB3 | Stable Baselines 3 | Python library providing PPO and other RL algorithms |
| TRPY | Thrust, Roll rate, Pitch rate, Yaw rate | The four control axes our policy outputs |
| RPM | Revolutions Per Minute | Motor spin speed |
| ESC | Electronic Speed Controller | Hardware that converts commands to motor voltages |
| α-RPO | Attenuated Residual Policy Optimization | RL bootstrapped from a classical controller, gradually removed |
| CRL | Curriculum Reinforcement Learning | Training in progressive difficulty stages |
| TraD-RL | Trajectory-guided Deep Reinforcement Learning | Reward shaping using a path fitted through gate centers |
| PD | Proportional-Derivative | Classical control law that reacts to error and its rate of change |
| VecEnv | Vectorized Environment | Many parallel simulation instances grouped as one interface |
| MLP | Multi-Layer Perceptron | Standard fully-connected neural network |
| GRU | Gated Recurrent Unit | Neural network layer with memory of recent history |
| GCNet | Guidance and Control Network | Our policy network name |
| OOB | Out Of Bounds | Episode ended because drone left the track area |
| W&B | Weights & Biases | Experiment tracking and visualization platform |
| EKF | Extended Kalman Filter | Algorithm for estimating position from noisy sensors |
| IMU | Inertial Measurement Unit | Accelerometer + gyroscope chip |
| DR | Domain Randomization | Varying simulation parameters during training for robustness |
| ONNX | Open Neural Network Exchange | Standard format for exporting/deploying trained models |
| FPS | Frames Per Second | Simulation throughput speed |
| GAE | Generalized Advantage Estimation | Method for computing how good each action was in hindsight |
| CBF | Control Barrier Function | Safety constraint that guarantees the drone stays in a safe region |
| HMM | Hidden Markov Model | A probabilistic sequence model (not used this session) |

---

*Generated 2026-03-18, updated 2026-03-19 — Corvidx drone racing team*
