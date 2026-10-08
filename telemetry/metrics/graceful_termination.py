from dataclasses import dataclass

from langchain_core.messages import ToolMessage

from telemetry.metrics.base import Metric, MetricData


@dataclass(frozen=True)
class GracefulTerminationResult:
    score: float

    def readable(self) -> str:
        return f"Graceful Termination: {self.score:.4f}"


class GracefulTerminationMetric(Metric[GracefulTerminationResult]):
    def __init__(self):
        super().__init__("graceful_termination")

    def evaluate(self, data: MetricData) -> GracefulTerminationResult:
        last_tool = next(
            (message for message in reversed(data.trajectory.get("messages", []))
             if isinstance(message, ToolMessage)), None
        )
        return GracefulTerminationResult(float(
            last_tool is not None
            and last_tool.additional_kwargs.get("terminal", False)
            and last_tool.status != "error"
        ))
