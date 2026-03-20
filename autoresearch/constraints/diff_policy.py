"""Edit surface enforcement: denylist/allowlist per scope with max diff limits."""

from __future__ import annotations

from dataclasses import dataclass, field

DENYLIST = [
    "sim/dynamics/",
    "sim/rewards.py",
    "sim/rewards_mavlab.py",
    "metrics/",
    "training/callbacks.py",
    "training/trajectory_recorder.py",
    "autoresearch/",
    ".claude/",
    "artifacts/",
    "docker/",
]

SCOPE_ALLOWLIST: dict[str, list[str]] = {
    "hyperparameter": [],
    "algorithm": [
        "control/algorithms/", "control/policies/", "perception/wrappers/", "configs/",
    ],
    "architecture": [
        "control/algorithms/", "control/policies/", "perception/wrappers/", "configs/",
        "perception/detectors/", "state_estimation/", "control/",
    ],
    "system": [
        "control/algorithms/", "control/policies/", "perception/wrappers/", "configs/",
        "perception/detectors/", "state_estimation/", "control/",
        "sim/envs/", "training/loops/", "sim/",
    ],
}

MAX_DIFF_LINES: dict[str, int] = {
    "hyperparameter": 0,
    "algorithm": 500,
    "architecture": 1000,
    "system": 2000,
}


@dataclass(frozen=True)
class FileChange:
    """A single file changed in a diff."""
    path: str
    added: int
    removed: int


@dataclass
class DiffValidationResult:
    """Result of diff policy validation."""
    passed: bool
    violations: list[str] = field(default_factory=list)
    file_summary: dict[str, dict[str, int]] = field(default_factory=dict)


class DiffPolicy:
    """Validates proposed code changes against denylist/allowlist/size rules."""

    def __init__(
        self,
        denylist: list[str] | None = None,
        scope_allowlist: dict[str, list[str]] | None = None,
        max_diff_lines: dict[str, int] | None = None,
    ) -> None:
        self.denylist = denylist or DENYLIST
        self.scope_allowlist = scope_allowlist or SCOPE_ALLOWLIST
        self.max_diff_lines = max_diff_lines or MAX_DIFF_LINES

    def validate(self, changes: list[FileChange], scope: str) -> DiffValidationResult:
        violations: list[str] = []
        file_summary: dict[str, dict[str, int]] = {}

        if scope == "hyperparameter" and changes:
            violations.append(
                f"Hyperparameter scope does not allow file edits, "
                f"but {len(changes)} files were changed"
            )
            return DiffValidationResult(passed=False, violations=violations, file_summary=file_summary)

        total_added = 0
        total_removed = 0
        allowlist = self.scope_allowlist.get(scope, [])

        for change in changes:
            file_summary[change.path] = {"added": change.added, "removed": change.removed}
            total_added += change.added
            total_removed += change.removed

            # Special case: tests/ allows additions only
            if change.path.startswith("tests/"):
                if change.removed > 0:
                    violations.append(
                        f"Test file {change.path}: only additive changes allowed, "
                        f"but {change.removed} lines were removed"
                    )
                continue  # tests are handled, skip denylist/allowlist

            # Check denylist
            if self._is_denylisted(change.path):
                violations.append(f"Denylist violation: {change.path} is off-limits")
                continue

            # Check allowlist
            if not self._is_allowlisted(change.path, allowlist):
                violations.append(f"Scope '{scope}' does not allow changes to {change.path}")

        # Check total diff size
        total_lines = total_added + total_removed
        max_lines = self.max_diff_lines.get(scope, 0)
        if max_lines > 0 and total_lines > max_lines:
            violations.append(
                f"Diff size {total_lines} lines exceeds {scope} limit of {max_lines} lines"
            )

        return DiffValidationResult(
            passed=len(violations) == 0, violations=violations, file_summary=file_summary
        )

    def _is_denylisted(self, path: str) -> bool:
        for pattern in self.denylist:
            if pattern.endswith("/"):
                if path.startswith(pattern):
                    return True
            else:
                if path == pattern:
                    return True
        return False

    def _is_allowlisted(self, path: str, allowlist: list[str]) -> bool:
        for pattern in allowlist:
            if pattern.endswith("/"):
                if path.startswith(pattern):
                    return True
            else:
                if path == pattern:
                    return True
        return False
