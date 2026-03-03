# algo_src

Autonomous drone racing algorithm stack for the Anduril AI Grand Prix.

## Architecture

```
algo_src/
├── perception/        — Gate detection CNN pipeline (GateNet-equivalent)
├── control/           — G&CNet RL control policy (PPO, motor RPM output)
├── state_estimation/  — EKF / VIO (camera + IMU fusion)
└── sim/               — Quadrotor dynamics simulator and training envs
```

## Target Platform

- **Onboard compute**: NVIDIA Jetson Orin NX 16GB
- **Sensors**: Single monocular camera + IMU
- **Sim platform**: DCL (Drone Champions League) simulator

## Project Management

- Linear workspace: **corvidx-drone-grand-prix** (team key: `COR`)
