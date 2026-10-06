from openinference.instrumentation.langchain import LangChainInstrumentor

from telemetry.instrumentation import Instrumentation
from telemetry.integrations.base import TracingIntegration


class LangChainTracingIntegration(TracingIntegration):
    """Capture LangChain model calls without modifying workflow definitions."""

    def __init__(self) -> None:
        self._instrumentor = LangChainInstrumentor()
        self._instrumentation: Instrumentation | None = None

    def attach(self, instrumentation: Instrumentation) -> None:
        if self._instrumentation is instrumentation:
            return
        if self._instrumentation is not None:
            raise RuntimeError("Detach the integration before changing tracer providers")
        # LangChainInstrumentor patches process-global callbacks; taking over
        # someone else's active integration would silently use the wrong provider.
        if self._instrumentor.is_instrumented_by_opentelemetry:
            raise RuntimeError("LangChain is already instrumented; detach the existing integration first")
        self._instrumentor.instrument(tracer_provider=instrumentation.provider)
        self._instrumentation = instrumentation

    def detach(self) -> None:
        if self._instrumentation is not None:
            self._instrumentor.uninstrument()
            self._instrumentation = None