"""Custom SB3 training callbacks for gate racing metrics."""

from __future__ import annotations

import time
from collections import deque

import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback

    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False

# Termination reason codes (mirror gate_race_env.py)
TERM_NAMES = {
    0: "none",
    1: "ground",
    2: "ceiling",
    3: "quat",
    4: "nan",
    5: "arena_oob",
    6: "body_rate",
    7: "gate_collision",
    8: "timeout",
}


class GateMetricsCallback(BaseCallback):
    """Log gate passage, lap completion, and termination statistics.

    Reads ``gates_passed``, ``laps_completed``, and ``termination_reason``
    from SB3's episode info dicts (populated by VecEnvAdapter) and logs
    rolling statistics to TensorBoard / stdout.

    Args:
        log_freq: Log summary every N env steps (not timesteps).
        window_size: Rolling window size for statistics.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        log_freq: int = 1000,
        window_size: int = 200,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.log_freq = log_freq
        self.window_size = window_size

        # Rolling buffers
        self._gates: deque[int] = deque(maxlen=window_size)
        self._laps: deque[int] = deque(maxlen=window_size)
        self._ep_lens: deque[int] = deque(maxlen=window_size)
        self._ep_rewards: deque[float] = deque(maxlen=window_size)
        self._term_reasons: deque[int] = deque(maxlen=window_size)
        self._lap_times: deque[float] = deque(maxlen=window_size)

        # Reward component buffers
        self._rew_progress: deque[float] = deque(maxlen=window_size)
        self._rew_body_rate: deque[float] = deque(maxlen=window_size)
        self._rew_action_smooth: deque[float] = deque(maxlen=window_size)
        self._rew_gate_passage: deque[float] = deque(maxlen=window_size)
        self._rew_gate_offset: deque[float] = deque(maxlen=window_size)
        self._rew_crash_penalty: deque[float] = deque(maxlen=window_size)

        # Speed and first-gate buffers
        self._avg_speed: deque[float] = deque(maxlen=window_size)
        self._first_gate_step: deque[int] = deque(maxlen=window_size)

        # Tracking for lap time estimation
        self._total_episodes = 0
        self._total_gates = 0
        self._total_laps = 0
        self._last_log_time = 0.0

    def _on_step(self) -> bool:
        """Called after each env.step(). Extract episode metrics from infos."""
        infos = self.locals.get("infos", [])

        for info in infos:
            ep = info.get("episode")
            if ep is None:
                continue

            gates = ep.get("gates_passed", 0)
            laps = ep.get("laps_completed", 0)
            ep_len = ep.get("l", 0)
            ep_rew = ep.get("r", 0.0)
            term = ep.get("termination_reason", 0)

            self._gates.append(gates)
            self._laps.append(laps)
            self._ep_lens.append(ep_len)
            self._ep_rewards.append(ep_rew)
            self._term_reasons.append(term)
            self._total_episodes += 1
            self._total_gates += gates
            self._total_laps += laps

            # Estimate lap time: if laps > 0, time per lap = ep_len * dt / laps
            # dt is 0.01s, so ep_len steps = ep_len * 0.01 seconds
            if laps > 0:
                lap_time = (ep_len * 0.01) / laps
                self._lap_times.append(lap_time)

            # Reward component breakdown
            rc = ep.get("reward_components")
            if rc is not None:
                self._rew_progress.append(rc.get("progress", 0.0))
                self._rew_body_rate.append(rc.get("body_rate", 0.0))
                self._rew_action_smooth.append(rc.get("action_smooth", 0.0))
                self._rew_gate_passage.append(rc.get("gate_passage", 0.0))
                self._rew_gate_offset.append(rc.get("gate_offset", 0.0))
                self._rew_crash_penalty.append(rc.get("crash_penalty", 0.0))

            # Average speed
            avg_spd = ep.get("avg_speed")
            if avg_spd is not None:
                self._avg_speed.append(float(avg_spd))

            # First gate step (only record when a gate was actually passed)
            fgs = ep.get("first_gate_step")
            if fgs is not None:
                self._first_gate_step.append(int(fgs))

        # Log at frequency
        if self.n_calls % self.log_freq == 0 and len(self._gates) > 0:
            self._log_metrics()

        return True

    def _log_metrics(self) -> None:
        """Write rolling metrics to SB3 logger."""
        gates = np.array(self._gates)
        laps = np.array(self._laps)
        terms = np.array(self._term_reasons)

        # Gate metrics
        self.logger.record("racing/gates_per_ep", float(gates.mean()))
        self.logger.record("racing/gates_per_ep_max", int(gates.max()))
        self.logger.record("racing/laps_per_ep", float(laps.mean()))
        self.logger.record("racing/total_gates", self._total_gates)
        self.logger.record("racing/total_laps", self._total_laps)

        # Success rate: episodes that reached timeout without crashing
        timeouts = (terms == 8).sum()
        success_rate = timeouts / len(terms) if len(terms) > 0 else 0.0
        self.logger.record("racing/success_rate", success_rate)

        # Gate passage rate: fraction of episodes with at least 1 gate
        gate_passers = (gates > 0).sum()
        self.logger.record("racing/gate_passage_rate", gate_passers / len(gates))

        # Lap completion rate: fraction of episodes with at least 1 lap
        lap_completers = (laps > 0).sum()
        self.logger.record("racing/lap_completion_rate", lap_completers / len(laps))

        # Lap times
        if self._lap_times:
            lt = np.array(self._lap_times)
            self.logger.record("racing/lap_time_mean", float(lt.mean()))
            self.logger.record("racing/lap_time_best", float(lt.min()))

        # Reward component breakdown
        if self._rew_gate_passage:
            self.logger.record(
                "racing/reward_gate_passage",
                float(np.mean(self._rew_gate_passage)),
            )
            self.logger.record(
                "racing/reward_progress",
                float(np.mean(self._rew_progress)),
            )
            self.logger.record(
                "racing/reward_body_rate",
                float(np.mean(self._rew_body_rate)),
            )
            self.logger.record(
                "racing/reward_action_smooth",
                float(np.mean(self._rew_action_smooth)),
            )
            self.logger.record(
                "racing/reward_gate_offset",
                float(np.mean(self._rew_gate_offset)),
            )
            self.logger.record(
                "racing/reward_crash_penalty",
                float(np.mean(self._rew_crash_penalty)),
            )

        # Average speed
        if self._avg_speed:
            self.logger.record(
                "racing/avg_speed", float(np.mean(self._avg_speed))
            )

        # Steps to first gate
        if self._first_gate_step:
            self.logger.record(
                "racing/steps_to_first_gate",
                float(np.mean(self._first_gate_step)),
            )

        # Termination breakdown
        for code, name in TERM_NAMES.items():
            count = (terms == code).sum()
            frac = count / len(terms) if len(terms) > 0 else 0.0
            self.logger.record(f"termination/{name}", frac)

        # Print to stdout
        if self.verbose >= 1:
            now = time.time()
            if now - self._last_log_time >= 10.0:  # stdout every 10s max
                self._last_log_time = now
                lt_str = ""
                if self._lap_times:
                    lt = np.array(self._lap_times)
                    lt_str = f"  lap_time={lt.mean():.1f}s (best {lt.min():.1f}s)"
                print(
                    f"[Racing] gates/ep={gates.mean():.1f}  "
                    f"laps/ep={laps.mean():.2f}  "
                    f"success={success_rate:.0%}  "
                    f"total_laps={self._total_laps}"
                    f"{lt_str}"
                )
