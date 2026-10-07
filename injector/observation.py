from contextlib import AbstractContextManager
from typing import Any, Protocol


class ToolObservation(Protocol):
    """The fault information a tool proxy can report without knowing about spans."""

    def record_fault(self, fault: object | None) -> None: ...


class ToolObserver(Protocol):
    """Telemetry-independent boundary for observing one tool invocation."""

    def observe(
        self,
        tool_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        context: object | None = None,
    ) -> AbstractContextManager[ToolObservation]: ...