from dataclasses import dataclass

from telemetry.metrics.base import Metric, MetricData


@dataclass(frozen=True)
class RecoveryRateResult:
    score: float

    def readable(self) -> str:
        return f"Recovery Rate: {self.score:.4f}"


class RecoveryRateMetric(Metric[RecoveryRateResult]):
    def __init__(self):
        super().__init__("recovery_rate")

    def evaluate(self, data: MetricData) -> RecoveryRateResult:
        messages = data.trajectory.get("messages", [])
        errors = any(getattr(message, "status", None) == "error" for message in messages)
        return RecoveryRateResult(
            0.0 if errors and getattr(messages[-1], "status", None) == "error" else 1.0
        )