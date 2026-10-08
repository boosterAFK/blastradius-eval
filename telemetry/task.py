"""Explicit task paths and independently checked outcomes for benchmark scoring."""

from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Any

from injector.faults.timeout_fault import TimeoutFault
from injector.schedulers.base import FaultScheduler


@dataclass(frozen=True)
class Outcome:
    status: Literal["valid", "invalid", "indeterminate"]
    reason: str


@dataclass(frozen=True)
class Baseline:
    calls: int | None
    path: tuple[str, ...]
    reason: str


class TaskSpec:
    """Candidate minimal valid paths and a scenario-specific outcome oracle.

    Paths are declared candidate valid tool-name sequences, not paths inferred
    from the LLM graph. A blocking fault forces a retry of the same call; effects
    requiring alternative calls or partial-data reconstruction remain unmodeled.
    """

    def __init__(
        self,
        paths: tuple[tuple[str, ...], ...],
        oracle: Callable[[Mapping[str, Any]], Outcome],
    ) -> None:
        if not paths or any(not path for path in paths):
            raise ValueError("at least one nonempty valid tool path is required")
        self.paths = paths
        self.oracle = oracle

    def baseline(self, scheduler: FaultScheduler | None = None) -> Baseline:
        candidates: list[tuple[int, tuple[str, ...]]] = []
        unknown: list[str] = []
        for path in self.paths:
            if scheduler is None:
                candidates.append((len(path), path))
                continue
            state = scheduler.prediction_state()
            attempts = 0
            expanded: list[str] = []
            uncertain = False
            for name in path:
                while True:
                    if attempts >= 100:
                        unknown.append("prediction exceeded 100 calls")
                        uncertain = True
                        break
                    state, fault = scheduler.predict(state, name)
                    attempts += 1
                    expanded.append(name)
                    if fault is None:
                        break
                    if not isinstance(fault, TimeoutFault) or not fault.raise_error:
                        unknown.append(f"fault effect {type(fault).__name__} is not modeled")
                        uncertain = True
                        break
                if uncertain:
                    break
            if uncertain:
                continue
            candidates.append((attempts, tuple(expanded)))
        if unknown:
            # An uncertain path might be shorter than every modeled path.
            return Baseline(None, (), "; ".join(sorted(set(unknown))))
        count, best = min(candidates, key=lambda item: item[0])
        return Baseline(count, best, "shortest declared valid path under schedule")