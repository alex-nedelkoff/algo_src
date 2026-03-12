# TensorBoard Log Checker — Design Spec

## Summary

A CLI diagnostic module (`python -m utils.tb_check`) that parses TensorBoard event files using `tbparse`, displays metric summaries, and runs automated diagnostic rules to detect common RL training failures. Supports single-run analysis and two-run comparison.

## Motivation

Claude Code cannot open a browser to view TensorBoard visually. A CLI tool that reads event files directly enables mid-training monitoring, post-training comparison, and debugging — all from the terminal. Diagnosis is as important as display: the tool should flag problems, not just dump numbers.

## Module Structure

```
utils/
└── tb_check/
    ├── __init__.py
    ├── __main__.py      # CLI entrypoint
    ├── loader.py        # tbparse wrapper, load event files -> DataFrames
    ├── diagnostics.py   # diagnostic rules engine
    └── formatter.py     # terminal output (tables, findings)
```

Invocation: `python -m utils.tb_check <log_dir> [options]`

## CLI Interface

```bash
# Single run summary + diagnostics
python -m utils.tb_check outputs/2026-03-11/20-34-56/tb_logs

# Compare two runs
python -m utils.tb_check outputs/run1/tb_logs --compare outputs/run2/tb_logs

# Filter to last N steps
python -m utils.tb_check outputs/.../tb_logs --last 500000

# Show only diagnostics (skip metric table)
python -m utils.tb_check outputs/.../tb_logs --diag-only
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `log_dir` | Yes | Path to TB log directory (finds event files recursively) |
| `--compare <dir>` | No | Second TB log directory for side-by-side comparison |
| `--last <N>` | No | Only analyze the last N timesteps |
| `--diag-only` | No | Skip the metric summary table, show only diagnostic findings |

## Components

### loader.py

Wraps `tbparse.SummaryReader` to load event files into pandas DataFrames.

**Responsibilities:**
- Recursively find event files under a given directory
- Load all scalar metrics into a single DataFrame with columns: `step`, `tag`, `value`
- Handle multiple event files (SB3 creates new ones on resume)
- Expose a `load_run(path: str) -> pd.DataFrame` function

### diagnostics.py

A rules engine that analyzes metric timeseries and produces findings.

**Finding dataclass:**
```python
@dataclasses.dataclass
class Finding:
    severity: Literal["INFO", "WARN", "CRITICAL"]
    rule: str           # rule name
    message: str        # human-readable description
    metric: str | None  # tag that triggered it
    step: int | None    # step where detected
```

**Diagnostic rules (default thresholds):**

| Rule | Trigger | Severity |
|------|---------|----------|
| Reward plateau | Mean reward flat (< 1% change) over last 20% of training | WARN |
| Reward collapse | Mean reward drops > 30% from peak | CRITICAL |
| KL spike | `approx_kl` exceeds 0.05 | WARN |
| Entropy collapse | Entropy loss drops below 10% of initial value | WARN |
| High crash rate | Any termination reason > 50% of episodes | WARN |
| NaN detected | Any NaN in any metric | CRITICAL |
| No gate progress | `gates_per_ep` stuck at 0 after 500K steps | CRITICAL |
| Value loss explosion | Value loss > 10x its rolling average | WARN |
| Clip fraction saturation | `clip_fraction` consistently > 0.3 | WARN |

Each rule is a function `(df: pd.DataFrame) -> list[Finding]`. The engine collects all rules and runs them against the loaded data.

Thresholds are hardcoded defaults for now. Configurable thresholds can be added later if needed.

### formatter.py

Formats output for terminal display.

**Summary mode output:**
```
══ TB Check: outputs/2026-03-11/20-34-56/tb_logs ══
Steps: 5,200,000 | Episodes: ~12,400

 METRIC                          LATEST    MEAN     MIN      MAX      TREND
 racing/gates_per_ep              3.2      2.1      0.0      4.8       ↑
 racing/lap_time_best            12.4s    14.1s    12.4s    99.9s      ↓
 reward/reward_gate_passage       0.82     0.54     0.00     0.92      ↑
 train/policy_loss               -0.012   -0.008   -0.015    0.002    →
 train/approx_kl                  0.018    0.015    0.003    0.042    →
 termination/ground               0.12     0.18     0.05     0.45     ↓
 ...

 DIAGNOSTICS
 ⚠ WARN   Clip fraction saturation: clip_fraction=0.32 (last 500K steps)
 ✓ No critical issues detected
```

**Trend calculation:** Compare mean of last 20% of data points to mean of prior 80%. Up (↑) if > 5% increase, down (↓) if > 5% decrease, flat (→) otherwise.

**Compare mode:** Adds a `DELTA` column showing the difference between run2 and run1 latest values, with percentage change.

**Metric ordering:** Group by prefix (`racing/`, `reward/`, `train/`, `termination/`) and sort alphabetically within groups.

## Dependency

Add `tbparse` to Docker dependencies. Specifically, add to `pyproject.toml` under `[project.optional-dependencies]` in the `control` extras group (alongside existing `tensorboard`).

## Scope Boundaries

**In scope:**
- Scalar metric parsing and display
- Automated diagnostic rules for common PPO/racing failures
- Two-run comparison
- Step-range filtering

**Out of scope (future):**
- Histogram or image event parsing
- Configurable threshold files
- HTML/image report generation
- Integration into training callbacks (inline diagnostics)
- W&B API integration (this tool reads local event files only)

## Testing

All tests run in Docker per project convention.

- **loader**: Test against a small synthetic event file (write a few scalars with `torch.utils.tensorboard.SummaryWriter`, then load with `tbparse`)
- **diagnostics**: Unit test each rule with crafted DataFrames (e.g., flat reward series triggers plateau warning)
- **formatter**: Snapshot test of formatted output strings
- **CLI integration**: End-to-end test: write event file, run `python -m utils.tb_check`, assert output contains expected metrics and findings
