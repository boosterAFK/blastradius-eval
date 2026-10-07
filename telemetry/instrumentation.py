from contextlib import contextmanager
import time
import uuid
from typing import Any, Dict, Optional
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from openinference.semconv.trace import SpanAttributes

from telemetry.span import SpanKind
from telemetry.llm import NormalizingSpanExporter
class Instrumentation:
    def __init__(self, service_name: str = "blastradius-eval", otlp_endpoint: str = "http://localhost:4317", enable_otlp: bool = True) -> None:
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint
        self.enable_otlp = enable_otlp

        self.resource = Resource.create({"service.name": self.service_name})
        self.provider = TracerProvider(resource=self.resource)

        self._memory = InMemorySpanExporter()
        self.provider.add_span_processor(SimpleSpanProcessor(NormalizingSpanExporter(self._memory)))

        if enable_otlp:
            self.provider.add_span_processor(SimpleSpanProcessor(NormalizingSpanExporter(
                OTLPSpanExporter(endpoint=self.otlp_endpoint, insecure=True)
            )))

        trace.set_tracer_provider(self.provider)
        self.tracer = self.provider.get_tracer(service_name)

    # UPDATED: Use a context manager to automatically handle span hierarchy (Parent/Child relationships)
    @contextmanager
    def span_context(self, name: str, span_kind: SpanKind, **attrs):
        """
        Creates a managed span that automatically nests inside the currently active span.
        Requires passing standard openinference span kinds (e.g., "AGENT", "TOOL", "LLM").
        """
        # Enforce GenAI Semantic Conventions for span kind
        attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] = span_kind.value.value
        
        with self.tracer.start_as_current_span(
            name, attributes=attrs, record_exception=False, set_status_on_exception=False
        ) as span:
            try:
                yield span
            except Exception as e:
                self.record_error(span, e)
                raise

    # UPDATED: Added context passing if you must manage span state manually (e.g., across async callbacks)
    def start_span(self, name: str, context: Optional[trace.Context] = None, **attrs) -> trace.Span:
        span = self.tracer.start_span(name, context=context, attributes=attrs)
        return span

    def end_span(self, span: trace.Span, status: str = "OK", **attrs) -> None:
        if attrs:
            span.set_attributes(attrs)
        
        span.set_status(trace.Status(trace.StatusCode.OK if status.lower() == "ok" else trace.StatusCode.ERROR))
        span.end()

    # NEW: Standardized exception recording to populate 'error.type' for chaos metrics
    def record_error(self, span: trace.Span, exception: Exception, **attrs) -> None:
        span.record_exception(exception)
        span.set_status(trace.Status(trace.StatusCode.ERROR, description=str(exception)))
        span.set_attributes({
            "error.type": type(exception).__name__,
            **attrs
        })

    def get_finished_spans(self) -> list:
        return self._memory.get_finished_spans()

    def make_tool_observer(self):
        """Provide the injector's optional observer without exposing OTel to it."""
        from telemetry.tool_observer import OpenTelemetryToolObserver

        return OpenTelemetryToolObserver(self)
