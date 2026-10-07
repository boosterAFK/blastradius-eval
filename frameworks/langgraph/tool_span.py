from langchain_core.callbacks.manager import CallbackManager
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace


def get_current_tool_span(callbacks: CallbackManager | None) -> trace.Span | None:
    """Resolve the in-flight tool run by ID; never match by name or time."""
    if callbacks is None or callbacks.parent_run_id is None:
        return None
    instrumentor = LangChainInstrumentor()
    if not instrumentor.is_instrumented_by_opentelemetry:
        return None
    span = instrumentor.get_span(callbacks.parent_run_id)
    return span if span is not None and span.is_recording() else None