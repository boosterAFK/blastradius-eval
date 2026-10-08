from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Mapping, Protocol, Sequence, TypeVar, cast

from opentelemetry.sdk.trace import ReadableSpan


@dataclass(frozen=True)
class MetricData:
    """Telemetry captured for one run, after the runner and spans have finished."""

    trajectory: Mapping[str, Any]
    spans: Sequence[ReadableSpan] = field(default_factory=tuple)
    trace_id: int | None = None
    _results: dict[int, "ReadableResult"] = field(default_factory=dict, init=False, repr=False, compare=False)
    _active: set[int] = field(default_factory=set, init=False, repr=False, compare=False)

    def evaluate_metric(self, metric: "Metric[ResultT]") -> "ResultT":
        """Evaluate each metric instance once per run; reject recursive dependencies."""
        key = id(metric)
        if key in self._active:
            raise ValueError(f"metric dependency cycle at {metric.name}")
        if key not in self._results:
            self._active.add(key)
            try:
                self._results[key] = metric.evaluate(self)
            finally:
                self._active.remove(key)
        return cast("ResultT", self._results[key])


class ReadableResult(Protocol):
    def readable(self) -> str:
        """Render this result for a human-readable report."""


ResultT = TypeVar("ResultT", bound=ReadableResult, covariant=True)


class Metric(ABC, Generic[ResultT]):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def evaluate(self, data: MetricData) -> ResultT:
        """Evaluate one run without discarding metric-specific result details."""
