"""Run-level metric registry and telemetry snapshot boundary."""

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from opentelemetry.sdk.trace import ReadableSpan

from telemetry.metrics.base import Metric, MetricData, ReadableResult


class FlushableProvider(Protocol):
    def force_flush(self) -> bool: ...


class SpanSource(Protocol):
    provider: FlushableProvider

    def get_finished_spans(self) -> list[ReadableSpan]: ...


@dataclass(frozen=True)
class EvaluationResult:
    results: Mapping[str, ReadableResult]

    def readable(self) -> str:
        if not self.results:
            return "metrics: (no metrics)"
        lines = "\n".join(result.readable() for result in self.results.values())
        return "metrics:\n" + "\n".join(f"  {line}" for line in lines.splitlines())


class Evaluator:
    """Register root metrics, then distribute one run snapshot to all of them."""

    def __init__(self, instrumentation: SpanSource):
        self.instrumentation = instrumentation
        self._metrics: dict[str, Metric[ReadableResult]] = {}

    def register(self, metric: Metric[ReadableResult]) -> None:
        if not isinstance(metric, Metric):
            raise TypeError("registered metrics must implement Metric")
        if metric.name in self._metrics:
            raise ValueError(f"duplicate metric name: {metric.name}")
        self._metrics[metric.name] = metric

    def evaluate(self, trajectory: Mapping[str, Any], *, trace_id: int | None = None) -> EvaluationResult:
        """Snapshot finished spans once per run, then evaluate registered roots."""
        self.instrumentation.provider.force_flush()
        data = MetricData(trajectory, tuple(self.instrumentation.get_finished_spans()), trace_id)
        return EvaluationResult({
            name: data.evaluate_metric(metric) for name, metric in self._metrics.items()
        })