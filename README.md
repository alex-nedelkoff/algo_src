# algo_src

Autonomous drone racing algorithm stack for the Anduril AI Grand Prix.

## Directory Structure

```
algo_src/
├── perception/        — Gate detection CNN pipeline (GateNet-equivalent)
├── control/           — G&CNet RL control policy (PPO training, motor RPM output)
├── state_estimation/  — EKF / VIO state estimation (camera + IMU fusion)
└── sim/               — Quadrotor dynamics simulator and training environments
```

### perception/
Gate detection CNN pipeline. Responsible for detecting racing gates from monocular camera frames, producing bounding boxes and relative pose estimates (GateNet-equivalent architecture).

### control/
Guidance and control network (G&CNet) reinforcement learning policy. Trains via PPO to output motor RPM commands from perceived gate state and vehicle state estimates.

### state_estimation/
Extended Kalman Filter (EKF) and Visual-Inertial Odometry (VIO) for robust state estimation. Fuses monocular camera and IMU data to produce position, velocity, and attitude estimates.

### sim/
Quadrotor dynamics simulator and training environments. Provides the physics model and reward-shaped environments used for offline RL policy training before sim-to-real transfer.

## Target Platform

- **Sim platform:** DCL (Drone Champions League) simulator
- **Onboard compute:** NVIDIA Jetson Orin NX 16GB
- **Sensors:** Single monocular camera + IMU
