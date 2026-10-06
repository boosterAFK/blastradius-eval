
import enum


class SpanKind(enum.Enum):
    AGENT = "AGENT"
    TOOL = "TOOL"
    CHAIN = "CHAIN"
    LLM = "LLM"