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

### Recommended Next Steps

1. **Gate proximity centering reward (easiest, highest impact):** When the drone is within ~2 meters of a gate, add a continuous reward proportional to how centered it is on the gate plane. Formula: `r = -lambda * lateral_offset * (1 / distance_to_gate_plane)`. This gives the policy gradient information *before* the binary pass/fail event.

2. **GRU integration (medium effort, large potential):** Replace the memoryless MLP with MLP + GRU (128 hidden). The CRL paper's ablation showed this is the difference between 77% and 100% success. Requires handling hidden state resets across episode boundaries in SB3.

3. **Lower entropy / tighter exploration:** The policy's action standard deviation is 3.1 — very high. The drone is exploring aggressively, which causes crashes. Reducing entropy more aggressively (start at 0.003, anneal to 0.0005) would tighten the policy around its learned behavior.

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
