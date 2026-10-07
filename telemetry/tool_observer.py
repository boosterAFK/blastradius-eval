import json
from contextlib import contextmanager
from typing import Any, TYPE_CHECKING

from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace

from telemetry.span import SpanKind

if TYPE_CHECKING:
    from telemetry.instrumentation import Instrumentation


class SpanToolObservation:
    def __init__(self, span: trace.Span) -> None:
        self.span = span

    def record_fault(self, fault: object | None) -> None:
        if fault is None:
            self.span.set_attribute("fault.injected", False)
        else:
            self.span.set_attributes({"fault.injected": True, "fault.type": type(fault).__name__})


class OpenTelemetryToolObserver:
    """Own tool span attributes and exception behavior for both tracing paths."""

    def __init__(self, instrumentation: "Instrumentation") -> None:
        self.instrumentation = instrumentation

    @contextmanager
    def observe(
        self,
        tool_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        context: object | None = None,
    ):
        attrs = {
            SpanAttributes.TOOL_NAME: tool_name,
            "gen_ai.tool.name": tool_name,
            SpanAttributes.INPUT_VALUE: json.dumps({"args": args, "kwargs": kwargs}, default=str),
            SpanAttributes.INPUT_MIME_TYPE: "application/json",
        }
        if context is None:
            # Own the standalone span, including its error event and closure.
            with self.instrumentation.span_context(
                name=f"chaos_tool_proxy.{tool_name}", span_kind=SpanKind.TOOL, **attrs
            ) as span:
                yield SpanToolObservation(span)
        else:
            # The framework owns this span and records its own exception event.
            # Make it active so nested operations become its children.
            if not isinstance(context, trace.Span) or not context.is_recording():
                raise ValueError("Framework tool context must be a recording OpenTelemetry span")
            context.set_attributes(attrs)
            with trace.use_span(context, end_on_exit=False, record_exception=False, set_status_on_exception=False):
                try:
                    yield SpanToolObservation(context)
                except Exception as exc:
                    context.set_attribute("error.type", type(exc).__name__)
                    raise