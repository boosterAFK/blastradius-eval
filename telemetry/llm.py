from openinference.semconv.trace import SpanAttributes
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter


# Canonical LLM fields consumed by metrics. Missing provider data stays missing.
LLM_MODEL = "gen_ai.request.model"
INPUT_TOKENS = "gen_ai.usage.input_tokens"
OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

OPENINFERENCE_LLM_FIELDS = {
    SpanAttributes.LLM_MODEL_NAME: LLM_MODEL,
    SpanAttributes.LLM_TOKEN_COUNT_PROMPT: INPUT_TOKENS,
    SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: OUTPUT_TOKENS,
}


def normalize_llm_span(span: ReadableSpan) -> ReadableSpan:
    """Return a copy of an LLM span with aliases for observed values only."""
    attrs = span.attributes
    if attrs.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) != "LLM":
        return span
    aliases = {
        target: attrs[source]
        for source, target in OPENINFERENCE_LLM_FIELDS.items()
        if source in attrs and target not in attrs and attrs[source] is not None
    }
    if not aliases:
        return span
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes={**attrs, **aliases},
        events=span.events,
        links=span.links,
        kind=span.kind,
        status=span.status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


class NormalizingSpanExporter(SpanExporter):
    """Normalize a readable copy consistently for memory and OTLP export."""

    def __init__(self, exporter: SpanExporter) -> None:
        self._exporter = exporter

    def export(self, spans):
        return self._exporter.export(tuple(normalize_llm_span(span) for span in spans))

    def shutdown(self) -> None:
        self._exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._exporter.force_flush(timeout_millis)