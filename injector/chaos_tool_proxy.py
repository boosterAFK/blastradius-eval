from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol

from openinference.semconv.trace import SpanAttributes
from telemetry.instrumentation import Instrumentation
from telemetry.span import SpanKind

if TYPE_CHECKING:
    from injector.base import Fault


class FaultDispatcher(Protocol):
    """
    Minimal contract the ChaosToolProxy needs from the orchestrator: given
    an observed tool call, return the fault to apply or None.
    """

    def dispatch(self, tool_name: str) -> Optional[Fault]: ...

    @property
    def instrumentation(self) -> Optional[Instrumentation]: ...


class ChaosToolProxy:
    def __init__(self, tool_name: str, original_callable: Callable[..., Any], dispatcher: FaultDispatcher) -> None:
        self.tool_name = tool_name
        self.original_callable = original_callable
        self.dispatcher = dispatcher
        self.instrumentation = dispatcher.instrumentation

    def __call__(self, *args, **kwargs):
        if self.instrumentation is None:
            fault = self.dispatcher.dispatch(self.tool_name)
            if fault is not None:
                return fault.apply(self.tool_name, self.original_callable, *args, **kwargs)
            return self.original_callable(*args, **kwargs)

        with self.instrumentation.span_context(
            name=f"chaos_tool_proxy.{self.tool_name}",
            span_kind=SpanKind.TOOL,
            **{
                SpanAttributes.TOOL_NAME: self.tool_name,
                "gen_ai.tool.name": self.tool_name,
                SpanAttributes.INPUT_VALUE: json.dumps(
                    {"args": args, "kwargs": kwargs},
                    default=str,
                ),
                SpanAttributes.INPUT_MIME_TYPE: "application/json",
            },
        ) as span:
            fault = self.dispatcher.dispatch(self.tool_name)
            if fault is not None:
                span.set_attributes({"fault.injected": True, "fault.type": type(fault).__name__})
                return fault.apply(self.tool_name, self.original_callable, *args, **kwargs)

            span.set_attribute("fault.injected", False)
            return self.original_callable(*args, **kwargs)

    def __repr__(self) -> str:
        return f"ChaosToolProxy(tool_name={self.tool_name!r}, original={self.original_callable!r})"
