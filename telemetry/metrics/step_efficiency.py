from dataclasses import dataclass
from openinference.semconv.trace import SpanAttributes
from langchain_core.messages import ToolMessage

from telemetry.metrics.base import Metric, MetricData
from telemetry.task import Outcome, TaskSpec


@dataclass(frozen=True)
class StepEfficiencyResult:
    name: str
    score: float | None
    baseline_calls: int | None
    actual_calls: int
    outcome: Outcome
    reason: str
    baseline_path: tuple[str, ...] = ()

    def readable(self) -> str:
        score = f"{self.score:.4f}" if self.score is not None else "unavailable"
        return f"{self.name}: {score} ({self.baseline_calls} / {self.actual_calls} tool calls; {self.outcome.status}: {self.reason})"


class StepEfficiencyMetric(Metric[StepEfficiencyResult]):
    """
    Tool-call efficiency for an independently validated task outcome.
    """

    def __init__(self, task: TaskSpec, scheduler=None, baseline: str = "clean"):
        if baseline not in ("clean", "fault_aware"):
            raise ValueError("baseline must be clean or fault_aware")
        if baseline == "fault_aware" and scheduler is None:
            raise ValueError("fault_aware requires a scheduler")
        super().__init__(name=f"step_efficiency/{baseline}")
        self.task = task
        self.baseline = task.baseline(scheduler if baseline == "fault_aware" else None)

    def evaluate(self, data: MetricData) -> StepEfficiencyResult:
        outcome = self.task.oracle(data.trajectory)
        if data.trace_id is None:
            return StepEfficiencyResult(self.name, None, self.baseline.calls, 0, outcome,
                                        "trace_id required", self.baseline.path)
        seen: set[tuple[int, int]] = set()
        for span in data.spans:
            context = getattr(span, "context", None)
            attributes = getattr(span, "attributes", None) or {}
            if context is None:
                continue
            if (context.trace_id == data.trace_id
                and attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "TOOL"):
                seen.add((context.trace_id, context.span_id))
        actual = len(seen)
        tool_messages = sum(isinstance(message, ToolMessage)
                            for message in data.trajectory.get("messages", ()))
        reason = outcome.reason if outcome.status != "valid" else self.baseline.reason
        score = (self.baseline.calls / actual
                 if outcome.status == "valid" and self.baseline.calls is not None
                 and actual >= self.baseline.calls and actual > 0 else None)
        if actual == 0:
            reason = "no TOOL spans for run"
            score = None
        elif tool_messages != actual:
            reason = f"TOOL spans ({actual}) differ from tool results ({tool_messages})"
            score = None
        elif outcome.status == "valid" and self.baseline.calls is not None and actual < self.baseline.calls:
            reason = "observed calls below baseline; trace or task definition incomplete"
        return StepEfficiencyResult(
            name=self.name,
            score=score, baseline_calls=self.baseline.calls, actual_calls=actual,
            outcome=outcome, reason=reason, baseline_path=self.baseline.path,
        )
