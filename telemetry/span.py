
import enum

from openinference.semconv.trace import OpenInferenceSpanKindValues


class SpanKind(enum.Enum):
    AGENT = OpenInferenceSpanKindValues.AGENT
    TOOL = OpenInferenceSpanKindValues.TOOL
    CHAIN = OpenInferenceSpanKindValues.CHAIN
    LLM = OpenInferenceSpanKindValues.LLM
    RETRIEVER = OpenInferenceSpanKindValues.RETRIEVER