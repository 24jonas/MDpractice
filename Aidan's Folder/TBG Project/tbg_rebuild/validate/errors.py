# tbg_rebuild/validate/errors.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    passed: bool
    severity: str  # "ERROR" | "WARN"
    message: str
    details: Dict[str, Any]


@dataclass(frozen=True, slots=True)
class ValidationReport:
    stage: str
    job_name: str
    n_atoms: int
    checks: List[CheckResult]

    @property
    def passed(self) -> bool:
        # any ERROR failure blocks the gate
        return all(c.passed or c.severity != "ERROR" for c in self.checks)

    def summary(self) -> str:
        lines = []
        lines.append(f"ValidationReport(stage={self.stage!r}, job={self.job_name!r}, N={self.n_atoms})")
        for c in self.checks:
            status = "OK" if c.passed else ("FAIL" if c.severity == "ERROR" else "WARN")
            lines.append(f"- [{status}] {c.name}: {c.message}")
        return "\n".join(lines)

    def failures(self) -> List[CheckResult]:
        return [c for c in self.checks if (not c.passed and c.severity == "ERROR")]


class ValidationError(RuntimeError):
    def __init__(self, report: ValidationReport):
        super().__init__(report.summary())
        self.report = report
