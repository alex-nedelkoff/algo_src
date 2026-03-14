"""Custom SB3 training callbacks for gate racing metrics."""

from __future__ import annotations

import time
from collections import Counter, deque

import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback

    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False


class GateMetricsCallback(BaseCallback):
    """Log gate passage, lap completion, and termination statistics.

    Reads contract-compliant episode info dicts from VecEnvAdapter:
    ``termination`` (str), ``success`` (bool), ``effective_dt`` (float),
    ``reward_components`` (dict), etc.

    Args:
        log_freq: Log summary every N env steps (not timesteps).
        window_size: Rolling window size for statistics.
        prefix: Metric prefix (default "racing" for train, use "eval_racing" for eval).
        verbose: Verbosity level.
    """

    def __init__(
        self,
        log_freq: int = 1000,
        window_size: int = 200,
        prefix: str = "racing",
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.log_freq = log_freq
        self.window_size = window_size
        self.prefix = prefix

        # Rolling buffers
        self._gates: deque[int] = deque(maxlen=window_size)
        self._laps: deque[int] = deque(maxlen=window_size)
        self._ep_lens: deque[int] = deque(maxlen=window_size)
        self._ep_rewards: deque[float] = deque(maxlen=window_size)
        self._term_reasons: deque[str] = deque(maxlen=window_size)
        self._lap_times: deque[float] = deque(maxlen=window_size)
        self._successes: deque[bool] = deque(maxlen=window_size)

        # Reward component buffers (populated dynamically from first episode)
        self._reward_component_buffers: dict[str, deque[float]] = {}

        # Speed and first-gate buffers
        self._avg_speed: deque[float] = deque(maxlen=window_size)
        self._first_gate_step: deque[int] = deque(maxlen=window_size)

        # Tracking for lap time estimation
        self._total_episodes = 0
        self._total_gates = 0
        self._total_laps = 0
        self._last_log_time = 0.0

        # All-time bests
        self._best_lap_time: float = float("inf")
        self._best_gates_per_ep: int = 0
        self._best_laps_per_ep: int = 0
        self._best_ep_reward: float = float("-inf")

        # Contract fields
        self._effective_dt: float | None = None
        self._success_criterion_logged: bool = False

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
            term = ep.get("termination", "unknown")
            success = ep.get("success", False)

            self._gates.append(gates)
            self._laps.append(laps)
            self._ep_lens.append(ep_len)
            self._ep_rewards.append(ep_rew)
            self._term_reasons.append(term)
            self._successes.append(bool(success))
            self._total_episodes += 1
            self._total_gates += gates
            self._total_laps += laps

            # Track all-time bests
            if gates > self._best_gates_per_ep:
                self._best_gates_per_ep = gates
            if laps > self._best_laps_per_ep:
                self._best_laps_per_ep = laps
            if ep_rew > self._best_ep_reward:
                self._best_ep_reward = ep_rew

            # Capture effective_dt on first episode
            if self._effective_dt is None:
                self._effective_dt = ep.get("effective_dt", 0.01)

            # Log success_criterion as wandb config once
            if not self._success_criterion_logged:
                criterion = ep.get("success_criterion", "unknown")
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.config.update(
                            {"success_criterion": criterion},
                            allow_val_change=True,
                        )
                except ImportError:
                    pass
                self._success_criterion_logged = True

            # Lap time using effective_dt from contract
            if laps > 0:
                dt = self._effective_dt or 0.01
                lap_time = (ep_len * dt) / laps
                self._lap_times.append(lap_time)
                if lap_time < self._best_lap_time:
                    self._best_lap_time = lap_time

            # Reward component breakdown (dynamic names)
            rc = ep.get("reward_components")
            if rc is not None:
                # Coerce to dict if needed (numpy object arrays, lists of tuples, etc.)
                if not isinstance(rc, dict):
                    try:
                        rc = dict(rc)
                    except (TypeError, ValueError):
                        rc = None
                if rc is not None:
                    for name, val in rc.items():
                        if name not in self._reward_component_buffers:
                            self._reward_component_buffers[name] = deque(maxlen=self.window_size)
                        self._reward_component_buffers[name].append(float(val))

            # Average speed
            avg_spd = ep.get("avg_speed")
            if avg_spd is not None:
                self._avg_speed.append(float(avg_spd))

            # First gate step (only record when a gate was actually passed)
            fgs = ep.get("first_gate_step")
            if fgs is not None and int(fgs) >= 0:
                self._first_gate_step.append(int(fgs))

        # Log at frequency
        if self.n_calls % self.log_freq == 0 and len(self._gates) > 0:
            self._log_metrics()

        return True

    def _log_metrics(self) -> None:
        """Write rolling metrics to SB3 logger."""
        p = self.prefix
        gates = np.array(self._gates)
        laps = np.array(self._laps)

        # Gate metrics
        self.logger.record(f"{p}/gates_per_ep", float(gates.mean()))
        self.logger.record(f"{p}/gates_per_ep_max", int(gates.max()))
        self.logger.record(f"{p}/laps_per_ep", float(laps.mean()))
        self.logger.record(f"{p}/total_gates", self._total_gates)
        self.logger.record(f"{p}/total_laps", self._total_laps)

        # Success rate from contract bool
        success_rate = sum(self._successes) / len(self._successes) if self._successes else 0.0
        self.logger.record(f"{p}/success_rate", success_rate)

        # Gate passage rate: fraction of episodes with at least 1 gate
        gate_passers = (gates > 0).sum()
        self.logger.record(f"{p}/gate_passage_rate", gate_passers / len(gates))

        # Lap completion rate: fraction of episodes with at least 1 lap
        lap_completers = (laps > 0).sum()
        self.logger.record(f"{p}/lap_completion_rate", lap_completers / len(laps))

        # Lap times
        if self._lap_times:
            lt = np.array(self._lap_times)
            self.logger.record(f"{p}/lap_time_mean", float(lt.mean()))
            self.logger.record(f"{p}/lap_time_best", float(lt.min()))

        # All-time bests
        if self._best_lap_time < float("inf"):
            self.logger.record(f"{p}/best_lap_time_ever", self._best_lap_time)
        self.logger.record(f"{p}/best_gates_per_ep_ever", self._best_gates_per_ep)
        self.logger.record(f"{p}/best_laps_per_ep_ever", self._best_laps_per_ep)
        self.logger.record(f"{p}/best_ep_reward_ever", self._best_ep_reward)

        # Max laps in rolling window
        self.logger.record(f"{p}/laps_per_ep_max", int(laps.max()))
        self.logger.record(f"{p}/total_episodes", self._total_episodes)

        # Reward component breakdown (dynamic)
        for name, buf in self._reward_component_buffers.items():
            if buf:
                self.logger.record(
                    f"{p}/reward_{name}",
                    float(np.mean(buf)),
                )

        # Average speed
        if self._avg_speed:
            self.logger.record(
                f"{p}/avg_speed", float(np.mean(self._avg_speed))
            )

        # Steps to first gate
        if self._first_gate_step:
            self.logger.record(
                f"{p}/steps_to_first_gate",
                float(np.mean(self._first_gate_step)),
            )

        # Termination breakdown (dynamic from string names)
        term_prefix = "eval_termination" if p != "racing" else "termination"
        term_counts = Counter(self._term_reasons)
        total = len(self._term_reasons)
        for name, count in term_counts.items():
            frac = count / total if total > 0 else 0.0
            self.logger.record(f"{term_prefix}/{name}", frac)

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


class EvalRacingCallback(BaseCallback):
    """Collect racing metrics from eval episodes.

    Attach as ``callback_after_eval`` on SB3's ``EvalCallback``.
    After each eval round, reads buffered episode dicts from the eval
    env's ``_completed_episodes`` list and logs under ``eval_racing/``.

    The eval env (VecEnvAdapter) must have ``_completed_episodes``
    buffering enabled — call ``eval_env.enable_episode_buffer()``
    before training starts.
    """

    def __init__(self, eval_env, window_size: int = 200, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self._metrics = GateMetricsCallback(
            log_freq=1, window_size=window_size, prefix="eval_racing", verbose=0,
        )

    def init_callback(self, model) -> None:
        super().init_callback(model)
        self._metrics.init_callback(model)

    def _on_step(self) -> bool:
        """Called by EvalCallback after eval episodes complete."""
        # Drain buffered episodes from the eval env
        episodes = getattr(self.eval_env, "_completed_episodes", [])
        if not episodes:
            return True

        for ep in episodes:
            self._metrics.locals = {"infos": [{"episode": ep}]}
            self._metrics.n_calls += 1
            self._metrics._on_step()

        # Force log and clear buffer
        if self._metrics._gates:
            self._metrics._log_metrics()
        self.eval_env._completed_episodes.clear()

        return True
