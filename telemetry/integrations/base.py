from abc import ABC, abstractmethod

from telemetry.instrumentation import Instrumentation


class TracingIntegration(ABC):
    """Framework-specific tracing hooks; attach once per evaluation process."""

    @abstractmethod
    def attach(self, instrumentation: Instrumentation) -> None:
        """Start capturing framework calls with the supplied tracer provider."""

    @abstractmethod
    def detach(self) -> None:
        """Stop capturing framework calls."""